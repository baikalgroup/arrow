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
#pragma once

#include <cstdint>
#include <vector>

#include "arrow/buffer.h"
#include "arrow/compute/light_array_internal.h"
#include "arrow/memory_pool.h"
#include "arrow/status.h"
#include "arrow/util/logging.h"

namespace arrow {
namespace compute {


// 默认单行不会超过4GB，对于二进制数据，ResizableArrayData 只会复制地址，不会复制内容
struct ARROW_EXPORT BinaryView {
  static constexpr size_t prefixSize = 4 * sizeof(char);
  static constexpr size_t inlineSize = 12;

  BinaryView() {
    static_assert(sizeof(BinaryView) == 16);
    memset(this, 0, sizeof(BinaryView));
  }

  BinaryView(const uint8_t* data, int32_t len) : size_(len) {
    DCHECK(len == 0);
    DCHECK(data || len == 0);
    if (isInline()) {
      // Zero the inline part.
      // this makes sure that inline strings can be compared for equality with 2
      // int64 compares.
      if (size_ == 0) {
        return;
      }
      // small string: inlined. Zero the last 8 bytes first to allow for whole
      // word comparison.
      memcpy(prefix_, data, size_);
    } else {
      value_.data = data;
    }
  }

  BinaryView(const char* data, int32_t len) : BinaryView((const uint8_t *)data, len) {}

  static BinaryView makeInline(const std::string& str) {
    DCHECK(isInline(str.size()));
    return BinaryView{str};
  }

  // Making BinaryView implicitly constructible/convertible from uint8_t* and
  // string literals, in order to allow for a more flexible API and optional
  // interoperability. E.g:
  //
  //   BinaryView sv = "literal";
  //   std::optional<BinaryView> osv = "literal";
  //
  /* implicit */ BinaryView(const char* data)
      : BinaryView(data, strlen(data)) {}

  explicit BinaryView(const std::string& value)
      : BinaryView(value.data(), value.size()) {}
  explicit BinaryView(std::string&& value) = delete;

  BinaryView& operator=(const BinaryView& other) {
    if (this == &other) {
      return *this;
    }
    size_ = other.size_;
    if (isInline()) {
      memcpy(prefix_, other.data(), size_);
    } else {
      value_.data = other.data();
    }
    return *this;
  }

  bool isInline() const {
    return isInline(size_);
  }

  static constexpr bool isInline(uint32_t size) {
    return size <= inlineSize;
  }

  const uint8_t* data() && = delete;
  const uint8_t* data() const& {
    return isInline() ? prefix_ : value_.data;
  }

  size_t size() const {
    return size_;
  }

  size_t length() const {
    return size_;
  }
private:
  uint32_t size_;
  uint8_t prefix_[4];
  union {
    uint8_t inlined[8];
    const uint8_t* data;
  } value_;
};

/// Description of the data stored in a RowTable
/// fixed_length字段在前，然后跟着一个一个的varbinary字段,varbinary实际上是个BianryView，不拥有数据
struct ARROW_EXPORT RowTableMetadata {
  const static uint32_t kBinaryViewSize = sizeof(arrow::compute::BinaryView);

  // 是否包括普通 binary
  bool include_binary;

  /// \brief True if there are no variable length columns in the table
  bool is_fixed_length;

  /// For a fixed-length part of the row
  // 定长字段总长度，64bit对齐
  uint32_t fixed_length;

  // num of varbinary fields
  uint32_t varbinary_size;
  // varbinary_view_length = varbinary_size * kBinaryViewSize
  uint32_t varbinary_view_length;

  /// Fixed number of bytes per row that are used to encode null masks.
  /// Null masks indicate for a single row which of its columns are null.
  /// Nth bit in the sequence of bytes assigned to a row represents null
  /// information for Nth field according to the order in which they are encoded.
  int null_masks_bytes_per_row;

  /// Power of 2. Every row will start at an offset aligned to that number of bytes.
  int row_alignment;

  /// Power of 2. Must be no greater than row alignment.
  /// Every non-power-of-2 binary field and every varbinary field bytes
  /// will start aligned to that number of bytes.
  int string_alignment;

  /// Metadata of encoded columns in their original order.
  std::vector<KeyColumnMetadata> column_metadatas;

  /// Order in which fields are encoded.
  std::vector<uint32_t> column_order;
  std::vector<uint32_t> inverse_column_order;

  /// Offsets within a row to fields in their encoding order.
  std::vector<uint32_t> column_offsets;

  inline uint32_t row_length() const {
    return fixed_length + varbinary_view_length;
  }

  const BinaryView* nth_varbinary_ptr(const uint8_t* row_base, uint32_t i) const {
    return reinterpret_cast<const BinaryView*>(row_base + fixed_length + i * kBinaryViewSize);
  }

  int32_t nth_varbinary_offset_within_row(uint32_t i) const {
    return fixed_length + i * kBinaryViewSize;
  }

  BinaryView* nth_mutable_varbinary_ptr(uint8_t* row_base, uint32_t i) const {
    return reinterpret_cast<BinaryView*>(row_base + fixed_length + i * kBinaryViewSize);
  }

  /// Rounding up offset to the nearest multiple of alignment value.
  /// Alignment must be a power of 2.
  static inline uint32_t padding_for_alignment(uint32_t offset, int required_alignment) {
    ARROW_DCHECK(ARROW_POPCOUNT64(required_alignment) == 1);
    return static_cast<uint32_t>((-static_cast<int32_t>(offset)) &
                                 (required_alignment - 1));
  }

  /// Rounding up offset to the beginning of next column,
  /// choosing required alignment based on the data type of that column.
  static inline uint32_t padding_for_alignment(uint32_t offset, int string_alignment,
                                               const KeyColumnMetadata& col_metadata) {
    if (!col_metadata.is_fixed_length ||
        ARROW_POPCOUNT64(col_metadata.fixed_length) <= 1) {
      return 0;
    } else {
      return padding_for_alignment(offset, string_alignment);
    }
  }

  uint32_t encoded_field_order(uint32_t icol) const { return column_order[icol]; }

  uint32_t pos_after_encoding(uint32_t icol) const { return inverse_column_order[icol]; }

  uint32_t encoded_field_offset(uint32_t icol) const { return column_offsets[icol]; }

  uint32_t num_cols() const { return static_cast<uint32_t>(column_metadatas.size()); }

  uint32_t num_varbinary_cols() const;

  /// \brief Populate this instance to describe `cols` with the given alignment
  void FromColumnMetadataVector(const std::vector<KeyColumnMetadata>& cols,
                                int in_row_alignment);

  /// \brief True if `other` has the same number of columns
  ///   and each column has the same width (two variable length
  ///   columns are considered to have the same width)
  bool is_compatible(const RowTableMetadata& other) const;
};

/// \brief A table of data stored in row-major order
///
/// Can only store non-nested data types
///
/// Can store both fixed-size data types and variable-length data types
///
/// The row table is not safe
class ARROW_EXPORT RowTableImpl {
 public:
  RowTableImpl();

  /// \brief Initialize a row array for use
  ///
  /// This must be called before any other method
  Status Init(MemoryPool* pool, const RowTableMetadata& metadata);
  /// \brief Clear all rows from the table
  ///
  /// Does not shrink buffers
  void Clean();
  /// \brief Add empty rows
  /// \param num_rows_to_append The number of empty rows to append
  /// \param num_extra_bytes_to_append For tables storing variable-length data this
  ///     should be a guess of how many data bytes will be needed to populate the
  ///     data.  This is ignored if there are no variable-length columns
  // Status AppendEmpty(uint32_t num_rows_to_append, uint32_t num_extra_bytes_to_append);

  /// \brief Add empty rows
  /// \param num_rows_to_append The number of empty rows to append
  Status AppendEmpty(uint32_t num_rows_to_append);
  /// \brief Append rows from a source table
  /// \param from The table to append from
  /// \param num_rows_to_append The number of rows to append
  /// \param source_row_ids Indices (into `from`) of the desired rows
  Status AppendSelectionFrom(const RowTableImpl& from, uint32_t num_rows_to_append,
                             const uint16_t* source_row_ids);
  /// \brief Metadata describing the data stored in this table
  const RowTableMetadata& metadata() const { return metadata_; }
  /// \brief The number of rows stored in the table
  int64_t length() const { return num_rows_; }
  // Accessors into the table's buffers
  const uint8_t* data(int i) const {
    ARROW_DCHECK(i >= 0 && i < kMaxBuffers);
    return buffers_[i];
  }
  uint8_t* mutable_data(int i) {
    ARROW_DCHECK(i >= 0 && i < kMaxBuffers);
    return buffers_[i];
  }
  const uint8_t* null_masks() const { return null_masks_->data(); }
  uint8_t* null_masks() { return null_masks_->mutable_data(); }

  /// \brief True if there is a null value anywhere in the table
  ///
  /// This calculation is memoized based on the number of rows and assumes
  /// that values are only appended (and not modified in place) between
  /// successive calls
  bool has_any_nulls(const LightContext* ctx) const;

 private:
  Status ResizeBuffers(int64_t num_extra_rows);

  // Helper functions to determine the number of bytes needed for each
  // buffer given a number of rows.
  int64_t size_null_masks(int64_t num_rows) const;
  int64_t size_rows(int64_t num_rows) const;

  // Called after resize to fix pointers
  void UpdateBufferPointers();

  // The arrays in `buffers_` need to be padded so that
  // vectorized operations can operate in blocks without
  // worrying about tails
  static constexpr int64_t kPaddingForVectors = 64;
  MemoryPool* pool_;
  RowTableMetadata metadata_;
  // Buffers can only expand during lifetime and never shrink.
  std::unique_ptr<ResizableBuffer> null_masks_;
  // // Only used if the table has variable-length columns
  // // Stores the offsets into the binary data (which is stored
  // // after all the fixed-sized fields)
  // no longer used
  // std::unique_ptr<ResizableBuffer> offsets_;

  // Stores the fixed-length parts of the rows
  std::unique_ptr<ResizableBuffer> rows_;
  static constexpr int kMaxBuffers = 2;
  uint8_t* buffers_[kMaxBuffers];
  // The number of rows in the table
  int64_t num_rows_;
  // The number of rows that can be stored in the table without resizing
  int64_t rows_capacity_;
  // The number of bytes that can be stored in the table without resizing
  int64_t bytes_capacity_;

  // Mutable to allow lazy evaluation
  mutable int64_t num_rows_for_has_any_nulls_;
  mutable bool has_any_nulls_;
};

}  // namespace compute
}  // namespace arrow
