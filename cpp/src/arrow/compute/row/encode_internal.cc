// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "arrow/compute/row/encode_internal.h"
#include "arrow/util/checked_cast.h"

namespace arrow {
namespace compute {

void RowTableEncoder::Init(const std::vector<KeyColumnMetadata>& cols, int row_alignment,
                           int string_alignment) {
  row_metadata_.FromColumnMetadataVector(cols, row_alignment);
  uint32_t num_cols = row_metadata_.num_cols();
  uint32_t num_varbinary_cols = row_metadata_.num_varbinary_cols();
  batch_all_cols_.resize(num_cols);
  batch_varbinary_cols_.resize(num_varbinary_cols);
  batch_varbinary_cols_base_offsets_.resize(num_varbinary_cols);
}

void RowTableEncoder::PrepareKeyColumnArrays(int64_t start_row, int64_t num_rows,
                                             const std::vector<KeyColumnArray>& cols_in) {
  const auto num_cols = static_cast<uint32_t>(cols_in.size());
  DCHECK(batch_all_cols_.size() == num_cols);

  uint32_t num_varbinary_visited = 0;
  for (uint32_t i = 0; i < num_cols; ++i) {
    const KeyColumnArray& col = cols_in[row_metadata_.column_order[i]];
    KeyColumnArray col_window = col.Slice(start_row, num_rows);

    batch_all_cols_[i] = col_window;
    if (!col.metadata().is_fixed_length) {
      DCHECK(num_varbinary_visited < batch_varbinary_cols_.size());
      // If start row is zero, then base offset of varbinary column is also zero.
      if (start_row == 0) {
        batch_varbinary_cols_base_offsets_[num_varbinary_visited] = 0;
      } else {
        if (col.metadata().is_large_binary()) {
          batch_varbinary_cols_base_offsets_[num_varbinary_visited] =
              col.large_offsets()[start_row];
        } else {
          batch_varbinary_cols_base_offsets_[num_varbinary_visited] =
              col.offsets()[start_row];
        }
      }
      batch_varbinary_cols_[num_varbinary_visited++] = col_window;
    }
  }
}

// 只影响hash grouper
void RowTableEncoder::DecodeFixedLengthBuffers(int64_t start_row_input,
                                               int64_t start_row_output, int64_t num_rows,
                                               const RowTableImpl& rows,
                                               std::vector<KeyColumnArray>* cols,
                                               int64_t hardware_flags,
                                               util::TempVectorStack* temp_stack) {
  // Prepare column array vectors
  PrepareKeyColumnArrays(start_row_output, num_rows, *cols);

  LightContext ctx;
  ctx.hardware_flags = hardware_flags;
  ctx.stack = temp_stack;

  // Create two temp vectors with 16-bit elements
  auto temp_buffer_holder_A =
      util::TempVectorHolder<uint16_t>(ctx.stack, static_cast<uint32_t>(num_rows));
  auto temp_buffer_A = KeyColumnArray(
      KeyColumnMetadata(true, sizeof(uint16_t)), num_rows, nullptr,
      reinterpret_cast<uint8_t*>(temp_buffer_holder_A.mutable_data()), nullptr);
  auto temp_buffer_holder_B =
      util::TempVectorHolder<uint16_t>(ctx.stack, static_cast<uint32_t>(num_rows));
  auto temp_buffer_B = KeyColumnArray(
      KeyColumnMetadata(true, sizeof(uint16_t)), num_rows, nullptr,
      reinterpret_cast<uint8_t*>(temp_buffer_holder_B.mutable_data()), nullptr);

  bool is_row_fixed_length = row_metadata_.is_fixed_length;
  if (!is_row_fixed_length) {
    EncoderOffsets::Decode(static_cast<uint32_t>(start_row_input),
                           static_cast<uint32_t>(num_rows), rows, &batch_varbinary_cols_,
                           batch_varbinary_cols_base_offsets_, &ctx);
  }

  // Process fixed length columns
  const auto num_cols = static_cast<uint32_t>(batch_all_cols_.size());
  for (uint32_t i = 0; i < num_cols;) {
    if (!batch_all_cols_[i].metadata().is_fixed_length ||
        batch_all_cols_[i].metadata().is_null_type) {
      i += 1;
      continue;
    }
    bool can_process_pair =
        (i + 1 < num_cols) && batch_all_cols_[i + 1].metadata().is_fixed_length &&
        EncoderBinaryPair::CanProcessPair(batch_all_cols_[i].metadata(),
                                          batch_all_cols_[i + 1].metadata());
    if (!can_process_pair) {
      EncoderBinary::Decode(static_cast<uint32_t>(start_row_input),
                            static_cast<uint32_t>(num_rows),
                            row_metadata_.column_offsets[i], rows, &batch_all_cols_[i],
                            &ctx, &temp_buffer_A);
      i += 1;
    } else {
      EncoderBinaryPair::Decode(
          static_cast<uint32_t>(start_row_input), static_cast<uint32_t>(num_rows),
          row_metadata_.column_offsets[i], rows, &batch_all_cols_[i],
          &batch_all_cols_[i + 1], &ctx, &temp_buffer_A, &temp_buffer_B);
      i += 2;
    }
  }

  // Process nulls
  EncoderNulls::Decode(static_cast<uint32_t>(start_row_input),
                       static_cast<uint32_t>(num_rows), rows, &batch_all_cols_);
}

void RowTableEncoder::DecodeVaryingLengthBuffers(
    int64_t start_row_input, int64_t start_row_output, int64_t num_rows,
    const RowTableImpl& rows, std::vector<KeyColumnArray>* cols, int64_t hardware_flags,
    util::TempVectorStack* temp_stack) {
  // Prepare column array vectors
  PrepareKeyColumnArrays(start_row_output, num_rows, *cols);

  LightContext ctx;
  ctx.hardware_flags = hardware_flags;
  ctx.stack = temp_stack;

  bool is_row_fixed_length = row_metadata_.is_fixed_length;
  if (!is_row_fixed_length) {
    for (size_t i = 0; i < batch_varbinary_cols_.size(); ++i) {
      // Memcpy varbinary fields into precomputed in the previous step
      // positions in the output row buffer.
      EncoderVarBinary::Decode(static_cast<uint32_t>(start_row_input),
                               static_cast<uint32_t>(num_rows), static_cast<uint32_t>(i),
                               rows, &batch_varbinary_cols_[i], &ctx);
    }
  }
}

void RowTableEncoder::PrepareEncodeSelected(int64_t start_row, int64_t num_rows,
                                            const std::vector<KeyColumnArray>& cols) {
  // Prepare column array vectors
  PrepareKeyColumnArrays(start_row, num_rows, cols);
}

Status RowTableEncoder::EncodeSelected(RowTableImpl* rows, uint32_t num_selected,
                                       const uint16_t* selection) {
  rows->Clean();
  RETURN_NOT_OK(
      rows->AppendEmpty(static_cast<uint32_t>(num_selected)));

  for (size_t icol = 0; icol < batch_all_cols_.size(); ++icol) {
    if (batch_all_cols_[icol].metadata().is_fixed_length) {
      uint32_t offset_within_row = rows->metadata().column_offsets[icol];
      EncoderBinary::EncodeSelected(offset_within_row, rows, batch_all_cols_[icol],
                                    num_selected, selection);
    }
  }

  // EncoderOffsets::EncodeSelected(rows, batch_varbinary_cols_, num_selected, selection);

  for (size_t icol = 0; icol < batch_varbinary_cols_.size(); ++icol) {
    bool is_large_binary = (!batch_varbinary_cols_[icol].metadata().is_fixed_length) 
                          && batch_varbinary_cols_[icol].metadata().is_large_binary();
    if (is_large_binary) {
      EncoderVarBinary::EncodeSelected<true>(static_cast<uint32_t>(icol), rows,
                                      batch_varbinary_cols_[icol], num_selected,
                                      selection);
    } else {
      EncoderVarBinary::EncodeSelected<false>(static_cast<uint32_t>(icol), rows,
                                      batch_varbinary_cols_[icol], num_selected,
                                      selection);
    }
    
  }

  EncoderNulls::EncodeSelected(rows, batch_all_cols_, num_selected, selection);

  return Status::OK();
}

namespace {
struct TransformBoolean {
  static KeyColumnArray ArrayReplace(const KeyColumnArray& column,
                                     const KeyColumnArray& temp) {
    // Make sure that the temp buffer is large enough
    DCHECK(temp.length() >= column.length() && temp.metadata().is_fixed_length &&
           temp.metadata().fixed_length >= sizeof(uint8_t));
    KeyColumnMetadata metadata;
    metadata.is_fixed_length = true;
    metadata.fixed_length = sizeof(uint8_t);
    constexpr int buffer_index = 1;
    return column.WithBufferFrom(temp, buffer_index).WithMetadata(metadata);
  }

  static void PostDecode(const KeyColumnArray& input, KeyColumnArray* output,
                         LightContext* ctx) {
    // Make sure that metadata and lengths are compatible.
    DCHECK(output->metadata().is_fixed_length == input.metadata().is_fixed_length);
    DCHECK(output->metadata().fixed_length == 0 && input.metadata().fixed_length == 1);
    DCHECK(output->length() == input.length());
    constexpr int buffer_index = 1;
    DCHECK(input.data(buffer_index) != nullptr);
    DCHECK(output->mutable_data(buffer_index) != nullptr);

    util::bit_util::bytes_to_bits(
        ctx->hardware_flags, static_cast<int>(input.length()), input.data(buffer_index),
        output->mutable_data(buffer_index), output->bit_offset(buffer_index));
  }
};

}  // namespace

bool EncoderInteger::IsBoolean(const KeyColumnMetadata& metadata) {
  return metadata.is_fixed_length && metadata.fixed_length == 0 && !metadata.is_null_type;
}

bool EncoderInteger::UsesTransform(const KeyColumnArray& column) {
  return IsBoolean(column.metadata());
}

KeyColumnArray EncoderInteger::ArrayReplace(const KeyColumnArray& column,
                                            const KeyColumnArray& temp) {
  if (IsBoolean(column.metadata())) {
    return TransformBoolean::ArrayReplace(column, temp);
  }
  return column;
}

void EncoderInteger::PostDecode(const KeyColumnArray& input, KeyColumnArray* output,
                                LightContext* ctx) {
  if (IsBoolean(output->metadata())) {
    TransformBoolean::PostDecode(input, output, ctx);
  }
}

void EncoderInteger::Decode(uint32_t start_row, uint32_t num_rows,
                            uint32_t offset_within_row, const RowTableImpl& rows,
                            KeyColumnArray* col, LightContext* ctx,
                            KeyColumnArray* temp) {
  KeyColumnArray col_prep;
  if (UsesTransform(*col)) {
    col_prep = ArrayReplace(*col, *temp);
  } else {
    col_prep = *col;
  }

  // When we have a single fixed length column we can just do memcpy
  if (rows.metadata().is_fixed_length &&
      col_prep.metadata().fixed_length == rows.metadata().row_length()) {
    DCHECK_EQ(offset_within_row, 0);
    uint32_t row_size = rows.metadata().row_length();
    memcpy(col_prep.mutable_data(1), rows.data(1) + start_row * row_size,
           num_rows * row_size);
  } else {
    uint32_t row_size = rows.metadata().row_length();
    const uint8_t* row_base = rows.data(1) + start_row * row_size;
    row_base += offset_within_row;
    uint8_t* col_base = col_prep.mutable_data(1);
    switch (col_prep.metadata().fixed_length) {
      case 1:
        for (uint32_t i = 0; i < num_rows; ++i) {
          col_base[i] = row_base[i * row_size];
        }
        break;
      case 2:
        for (uint32_t i = 0; i < num_rows; ++i) {
          reinterpret_cast<uint16_t*>(col_base)[i] =
              *reinterpret_cast<const uint16_t*>(row_base + i * row_size);
        }
        break;
      case 4:
        for (uint32_t i = 0; i < num_rows; ++i) {
          reinterpret_cast<uint32_t*>(col_base)[i] =
              *reinterpret_cast<const uint32_t*>(row_base + i * row_size);
        }
        break;
      case 8:
        for (uint32_t i = 0; i < num_rows; ++i) {
          reinterpret_cast<uint64_t*>(col_base)[i] =
              *reinterpret_cast<const uint64_t*>(row_base + i * row_size);
        }
        break;
      default:
        DCHECK(false);
    }
  }

  if (UsesTransform(*col)) {
    PostDecode(col_prep, col, ctx);
  }
}

template <class COPY_FN, class SET_NULL_FN>
void EncoderBinary::EncodeSelectedImp(uint32_t offset_within_row, RowTableImpl* rows,
                                      const KeyColumnArray& col, uint32_t num_selected,
                                      const uint16_t* selection, COPY_FN copy_fn,
                                      SET_NULL_FN set_null_fn) {
  uint32_t row_width = rows->metadata().row_length();
  const uint8_t* src_base = col.data(1);
  uint8_t* dst = rows->mutable_data(1) + offset_within_row;
  for (uint32_t i = 0; i < num_selected; ++i) {
    copy_fn(dst, src_base, selection[i]);
    dst += row_width;
  }
  if (col.data(0)) {
    const uint8_t* non_null_bits = col.data(0);
    uint8_t* dst = rows->mutable_data(1) + offset_within_row;
    for (uint32_t i = 0; i < num_selected; ++i) {
      bool is_null = !bit_util::GetBit(non_null_bits, selection[i] + col.bit_offset(0));
      if (is_null) {
        set_null_fn(dst);
      }
      dst += row_width;
    }
  }
}

void EncoderBinary::EncodeSelected(uint32_t offset_within_row, RowTableImpl* rows,
                                   const KeyColumnArray& col, uint32_t num_selected,
                                   const uint16_t* selection) {
  if (col.metadata().is_null_type) {
    return;
  }
  uint32_t col_width = col.metadata().fixed_length;
  if (col_width == 0) {
    int bit_offset = col.bit_offset(1);
    EncodeSelectedImp(
        offset_within_row, rows, col, num_selected, selection,
        [bit_offset](uint8_t* dst, const uint8_t* src_base, uint16_t irow) {
          *dst = bit_util::GetBit(src_base, irow + bit_offset) ? 0xff : 0x00;
        },
        [](uint8_t* dst) { *dst = 0xae; });
  } else if (col_width == 1) {
    EncodeSelectedImp(
        offset_within_row, rows, col, num_selected, selection,
        [](uint8_t* dst, const uint8_t* src_base, uint16_t irow) {
          *dst = src_base[irow];
        },
        [](uint8_t* dst) { *dst = 0xae; });
  } else if (col_width == 2) {
    EncodeSelectedImp(
        offset_within_row, rows, col, num_selected, selection,
        [](uint8_t* dst, const uint8_t* src_base, uint16_t irow) {
          *reinterpret_cast<uint16_t*>(dst) =
              reinterpret_cast<const uint16_t*>(src_base)[irow];
        },
        [](uint8_t* dst) { *reinterpret_cast<uint16_t*>(dst) = 0xaeae; });
  } else if (col_width == 4) {
    EncodeSelectedImp(
        offset_within_row, rows, col, num_selected, selection,
        [](uint8_t* dst, const uint8_t* src_base, uint16_t irow) {
          *reinterpret_cast<uint32_t*>(dst) =
              reinterpret_cast<const uint32_t*>(src_base)[irow];
        },
        [](uint8_t* dst) {
          *reinterpret_cast<uint32_t*>(dst) = static_cast<uint32_t>(0xaeaeaeae);
        });
  } else if (col_width == 8) {
    EncodeSelectedImp(
        offset_within_row, rows, col, num_selected, selection,
        [](uint8_t* dst, const uint8_t* src_base, uint16_t irow) {
          *reinterpret_cast<uint64_t*>(dst) =
              reinterpret_cast<const uint64_t*>(src_base)[irow];
        },
        [](uint8_t* dst) { *reinterpret_cast<uint64_t*>(dst) = 0xaeaeaeaeaeaeaeaeULL; });
  } else {
    EncodeSelectedImp(
        offset_within_row, rows, col, num_selected, selection,
        [col_width](uint8_t* dst, const uint8_t* src_base, uint16_t irow) {
          memcpy(dst, src_base + col_width * irow, col_width);
        },
        [col_width](uint8_t* dst) { memset(dst, 0xae, col_width); });
  }
}

bool EncoderBinary::IsInteger(const KeyColumnMetadata& metadata) {
  if (metadata.is_null_type) {
    return false;
  }
  bool is_fixed_length = metadata.is_fixed_length;
  auto size = metadata.fixed_length;
  return is_fixed_length &&
         (size == 0 || size == 1 || size == 2 || size == 4 || size == 8);
}

void EncoderBinary::Decode(uint32_t start_row, uint32_t num_rows,
                           uint32_t offset_within_row, const RowTableImpl& rows,
                           KeyColumnArray* col, LightContext* ctx, KeyColumnArray* temp) {
  if (IsInteger(col->metadata())) {
    EncoderInteger::Decode(start_row, num_rows, offset_within_row, rows, col, ctx, temp);
  } else {
    KeyColumnArray col_prep;
    if (EncoderInteger::UsesTransform(*col)) {
      col_prep = EncoderInteger::ArrayReplace(*col, *temp);
    } else {
      col_prep = *col;
    }

    bool is_row_fixed_length = rows.metadata().is_fixed_length;

#if defined(ARROW_HAVE_RUNTIME_AVX2)
    if (ctx->has_avx2()) {
      DecodeHelper_avx2(start_row, num_rows, offset_within_row, rows,
                        col);
    } else {
#endif
      DecodeImp(start_row, num_rows, offset_within_row, rows, col);
#if defined(ARROW_HAVE_RUNTIME_AVX2)
    }
#endif

    if (EncoderInteger::UsesTransform(*col)) {
      EncoderInteger::PostDecode(col_prep, col, ctx);
    }
  }
}

void EncoderBinary::DecodeImp(uint32_t start_row, uint32_t num_rows,
                              uint32_t offset_within_row, const RowTableImpl& rows,
                              KeyColumnArray* col) {
  DecodeHelper(
      start_row, num_rows, offset_within_row, &rows, nullptr, col, col,
      [](uint8_t* dst, const uint8_t* src, int64_t length) {
        for (uint32_t istripe = 0; istripe < bit_util::CeilDiv(length, 8); ++istripe) {
          auto dst64 = reinterpret_cast<uint64_t*>(dst);
          auto src64 = reinterpret_cast<const uint64_t*>(src);
          util::SafeStore(dst64 + istripe, src64[istripe]);
        }
      });
}

void EncoderBinaryPair::Decode(uint32_t start_row, uint32_t num_rows,
                               uint32_t offset_within_row, const RowTableImpl& rows,
                               KeyColumnArray* col1, KeyColumnArray* col2,
                               LightContext* ctx, KeyColumnArray* temp1,
                               KeyColumnArray* temp2) {
  DCHECK(CanProcessPair(col1->metadata(), col2->metadata()));

  KeyColumnArray col_prep[2];
  if (EncoderInteger::UsesTransform(*col1)) {
    col_prep[0] = EncoderInteger::ArrayReplace(*col1, *temp1);
  } else {
    col_prep[0] = *col1;
  }
  if (EncoderInteger::UsesTransform(*col2)) {
    col_prep[1] = EncoderInteger::ArrayReplace(*col2, *temp2);
  } else {
    col_prep[1] = *col2;
  }

  uint32_t col_width1 = col_prep[0].metadata().fixed_length;
  uint32_t col_width2 = col_prep[1].metadata().fixed_length;
  int log_col_width1 = col_width1 == 8   ? 3
                       : col_width1 == 4 ? 2
                       : col_width1 == 2 ? 1
                                         : 0;
  int log_col_width2 = col_width2 == 8   ? 3
                       : col_width2 == 4 ? 2
                       : col_width2 == 2 ? 1
                                         : 0;

  bool is_row_fixed_length = rows.metadata().is_fixed_length;

  uint32_t num_processed = 0;

#if defined(ARROW_HAVE_RUNTIME_AVX2)
  if (ctx->has_avx2() && col_width1 == col_width2) {
    num_processed =
        DecodeHelper_avx2(col_width1, start_row, num_rows,
                          offset_within_row, rows, &col_prep[0], &col_prep[1]);
  }
#endif
  if (num_processed < num_rows) {
    using DecodeImp_t = void (*)(uint32_t, uint32_t, uint32_t, uint32_t,
                                 const RowTableImpl&, KeyColumnArray*, KeyColumnArray*);
    static const DecodeImp_t DecodeImp_fn[] = {
        DecodeImp<uint8_t, uint8_t>,   DecodeImp<uint16_t, uint8_t>,
        DecodeImp<uint32_t, uint8_t>,  DecodeImp<uint64_t, uint8_t>,
        DecodeImp<uint8_t, uint16_t>,  DecodeImp<uint16_t, uint16_t>,
        DecodeImp<uint32_t, uint16_t>, DecodeImp<uint64_t, uint16_t>,
        DecodeImp<uint8_t, uint32_t>,  DecodeImp<uint16_t, uint32_t>,
        DecodeImp<uint32_t, uint32_t>, DecodeImp<uint64_t, uint32_t>,
        DecodeImp<uint8_t, uint64_t>,  DecodeImp<uint16_t, uint64_t>,
        DecodeImp<uint32_t, uint64_t>, DecodeImp<uint64_t, uint64_t>};
    int dispatch_const =
        (log_col_width2 << 2) | log_col_width1;
    DecodeImp_fn[dispatch_const](num_processed, start_row, num_rows, offset_within_row,
                                 rows, &(col_prep[0]), &(col_prep[1]));
  }

  if (EncoderInteger::UsesTransform(*col1)) {
    EncoderInteger::PostDecode(col_prep[0], col1, ctx);
  }
  if (EncoderInteger::UsesTransform(*col2)) {
    EncoderInteger::PostDecode(col_prep[1], col2, ctx);
  }
}

template <typename col1_type, typename col2_type>
void EncoderBinaryPair::DecodeImp(uint32_t num_rows_to_skip, uint32_t start_row,
                                  uint32_t num_rows, uint32_t offset_within_row,
                                  const RowTableImpl& rows, KeyColumnArray* col1,
                                  KeyColumnArray* col2) {
  DCHECK(rows.length() >= start_row + num_rows);
  DCHECK(col1->length() == num_rows && col2->length() == num_rows);

  uint8_t* dst_A = col1->mutable_data(1);
  uint8_t* dst_B = col2->mutable_data(1);

  uint32_t row_width = rows.metadata().row_length();
  const uint8_t* src_base;

  src_base = rows.data(1) + row_width * start_row + offset_within_row;

  using col1_type_const = typename std::add_const<col1_type>::type;
  using col2_type_const = typename std::add_const<col2_type>::type;

  const uint8_t* src = src_base + num_rows_to_skip * row_width;
  for (uint32_t i = num_rows_to_skip; i < num_rows; ++i) {
    reinterpret_cast<col1_type*>(dst_A)[i] = *reinterpret_cast<col1_type_const*>(src);
    reinterpret_cast<col2_type*>(dst_B)[i] =
        *reinterpret_cast<col2_type_const*>(src + sizeof(col1_type));
    src += row_width;
  }
}

void EncoderOffsets::Decode(uint32_t start_row, uint32_t num_rows,
                            const RowTableImpl& rows,
                            std::vector<KeyColumnArray>* varbinary_cols,
                            const std::vector<uint64_t>& varbinary_cols_base_offset,
                            LightContext* ctx) {
  DCHECK(!varbinary_cols->empty());
  DCHECK(varbinary_cols->size() == varbinary_cols_base_offset.size());

  DCHECK(!rows.metadata().is_fixed_length);
  DCHECK(rows.length() >= start_row + num_rows);
  for (const auto& col : *varbinary_cols) {
    // Rows and columns must all be varying-length
    DCHECK(!col.metadata().is_fixed_length);
    // The space in columns must be exactly equal to a subset of rows selected
    DCHECK(col.length() == num_rows);
  }

  // Offsets of varbinary columns data within each encoded row are stored
  // in the same encoded row as an array of 32-bit integers.
  // This array follows immediately the data of fixed-length columns.
  // There is one element for each varying-length column.
  // The Nth element is the sum of all the lengths of varbinary columns data in
  // that row, up to and including Nth varbinary column.

  // const uint32_t* row_offsets = rows.offsets() + start_row;
  const uint32_t row_length = rows.metadata().row_length();
  const uint8_t* row_base = rows.data(1) + start_row * row_length;

  // Set the base offset for each column
  for (size_t col = 0; col < varbinary_cols->size(); ++col) {
    bool is_large_binary = (*varbinary_cols)[col].metadata().is_large_binary();
    if (is_large_binary) {
      uint64_t* col_offsets = (*varbinary_cols)[col].mutable_large_offsets();
      col_offsets[0] = varbinary_cols_base_offset[col];
    } else {
      uint32_t* col_offsets = (*varbinary_cols)[col].mutable_offsets();
      col_offsets[0] = varbinary_cols_base_offset[col];
    }
  }

  int string_alignment = rows.metadata().string_alignment;

  for (uint32_t i = 0; i < num_rows; ++i) {
    // Find the beginning of cumulative lengths array for next row
    // const uint8_t* row = rows.data(2) + row_offsets[i];
    const uint8_t* row = row_base + i * row_length;
    // const uint32_t* varbinary_ends = rows.metadata().varbinary_end_array(row);

    // Update the offset of each column
    uint32_t offset_within_row = rows.metadata().fixed_length;
    for (size_t col = 0; col < varbinary_cols->size(); ++col) {
      const BinaryView* varbinary_view = rows.metadata().nth_varbinary_ptr(row, col);
      // offset_within_row +=
      //     RowTableMetadata::padding_for_alignment(offset_within_row, string_alignment);
      uint32_t length = varbinary_view->length();
      // offset_within_row = varbinary_ends[col];
      bool is_large_binary = (*varbinary_cols)[col].metadata().is_large_binary();
      if (is_large_binary) {
        uint64_t* col_offsets = (*varbinary_cols)[col].mutable_large_offsets();
        col_offsets[i + 1] = col_offsets[i] + length;
      } else {
        uint32_t* col_offsets = (*varbinary_cols)[col].mutable_offsets();
        col_offsets[i + 1] = col_offsets[i] + length;
      }
    }
  }
}

void EncoderVarBinary::Decode(uint32_t start_row, uint32_t num_rows,
                              uint32_t varbinary_col_id, const RowTableImpl& rows,
                              KeyColumnArray* col, LightContext* ctx) {
  // Output column varbinary buffer needs an extra 32B
  // at the end in avx2 version and 8B otherwise.
#if defined(ARROW_HAVE_RUNTIME_AVX2)
  if (ctx->has_avx2()) {
    DecodeHelper_avx2(start_row, num_rows, varbinary_col_id, rows, col);
  } else {
#endif
    DecodeImp(start_row, num_rows, varbinary_col_id, rows, col);
#if defined(ARROW_HAVE_RUNTIME_AVX2)
  }
#endif
}

void EncoderVarBinary::DecodeImp(uint32_t start_row, uint32_t num_rows,
                                 uint32_t varbinary_col_id, const RowTableImpl& rows,
                                 KeyColumnArray* col) {
  bool is_large_binary = col->metadata().is_large_binary();
  if (is_large_binary) {
    DecodeHelper<true>(
        start_row, num_rows, varbinary_col_id, &rows, nullptr, col, col,
        [](uint8_t* dst, const uint8_t* src, int64_t length) {
          for (uint32_t istripe = 0; istripe < bit_util::CeilDiv(length, 8); ++istripe) {
            auto dst64 = reinterpret_cast<uint64_t*>(dst);
            auto src64 = reinterpret_cast<const uint64_t*>(src);
            util::SafeStore(dst64 + istripe, src64[istripe]);
          }
        });
  } else {
    DecodeHelper<false>(
        start_row, num_rows, varbinary_col_id, &rows, nullptr, col, col,
        [](uint8_t* dst, const uint8_t* src, int64_t length) {
          for (uint32_t istripe = 0; istripe < bit_util::CeilDiv(length, 8); ++istripe) {
            auto dst64 = reinterpret_cast<uint64_t*>(dst);
            auto src64 = reinterpret_cast<const uint64_t*>(src);
            util::SafeStore(dst64 + istripe, src64[istripe]);
          }
        });
  }
  
}

void EncoderNulls::Decode(uint32_t start_row, uint32_t num_rows, const RowTableImpl& rows,
                          std::vector<KeyColumnArray>* cols) {
  // Every output column needs to have a space for exactly the required number
  // of rows. It also needs to have non-nulls bit-vector allocated and mutable.
  DCHECK_GT(cols->size(), 0);
  for (auto& col : *cols) {
    DCHECK(col.length() == num_rows);
    DCHECK(col.mutable_data(0) || col.metadata().is_null_type);
  }

  const uint8_t* null_masks = rows.null_masks();
  uint32_t null_masks_bytes_per_row = rows.metadata().null_masks_bytes_per_row;
  for (size_t col = 0; col < cols->size(); ++col) {
    if ((*cols)[col].metadata().is_null_type) {
      continue;
    }
    uint8_t* non_nulls = (*cols)[col].mutable_data(0);
    const int bit_offset = (*cols)[col].bit_offset(0);
    DCHECK_LT(bit_offset, 8);
    non_nulls[0] |= 0xff << (bit_offset);
    if (bit_offset + num_rows > 8) {
      int bits_in_first_byte = 8 - bit_offset;
      memset(non_nulls + 1, 0xff, bit_util::BytesForBits(num_rows - bits_in_first_byte));
    }
    for (uint32_t row = 0; row < num_rows; ++row) {
      uint32_t null_masks_bit_id =
          (start_row + row) * null_masks_bytes_per_row * 8 + static_cast<uint32_t>(col);
      bool is_set = bit_util::GetBit(null_masks, null_masks_bit_id);
      if (is_set) {
        bit_util::ClearBit(non_nulls, bit_offset + row);
      }
    }
  }
}

template<bool is_large_binary>
void EncoderVarBinary::EncodeSelected(uint32_t ivarbinary, RowTableImpl* rows,
                                      const KeyColumnArray& cols, uint32_t num_selected,
                                      const uint16_t* selection) {
  uint8_t* row_base = rows->mutable_data(1);
  const uint64_t* col_offsets64 = cols.large_offsets();
  const uint32_t* col_offsets32 = cols.offsets();
  const uint8_t* col_base = cols.data(2);

  const RowTableMetadata metadata = rows->metadata();
  uint32_t row_width = metadata.row_length();
  for (uint32_t i = 0; i < num_selected; ++i) {
    int32_t irow = selection[i];
    uint8_t* row = row_base + row_width * i;
    BinaryView* binary_view = metadata.nth_mutable_varbinary_ptr(row, ivarbinary);
    
    uint64_t col_offset;
    uint32_t length;
    if constexpr (is_large_binary) {
      col_offset = col_offsets64[irow];
      length = static_cast<uint32_t>(col_offsets64[irow + 1] - col_offset);
    } else {
      col_offset = col_offsets32[irow];
      length = col_offsets32[irow + 1] - static_cast<uint32_t>(col_offset);
    }
    *binary_view = BinaryView(col_base + col_offset, length);
  }
}

void EncoderNulls::EncodeSelected(RowTableImpl* rows,
                                  const std::vector<KeyColumnArray>& cols,
                                  uint32_t num_selected, const uint16_t* selection) {
  uint8_t* null_masks = rows->null_masks();
  uint32_t null_mask_num_bytes = rows->metadata().null_masks_bytes_per_row;
  memset(null_masks, 0, null_mask_num_bytes * num_selected);
  for (size_t icol = 0; icol < cols.size(); ++icol) {
    const uint8_t* non_null_bits = cols[icol].data(0);
    if (non_null_bits) {
      for (uint32_t i = 0; i < num_selected; ++i) {
        uint32_t irow = selection[i];
        bool is_null = !bit_util::GetBit(non_null_bits, irow + cols[icol].bit_offset(0));
        if (is_null) {
          bit_util::SetBit(null_masks, i * null_mask_num_bytes * 8 + icol);
        }
      }
    }
  }
}

}  // namespace compute
}  // namespace arrow
