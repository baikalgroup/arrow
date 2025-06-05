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

#include <immintrin.h>

#include "arrow/acero/swiss_join_internal.h"
#include "arrow/util/bit_util.h"

namespace arrow {
namespace acero {
// AVX2-only：8 × uint32 → uint64 = 两个 __m256i 输出
inline void avx2_mul_u32_to_u64(
    const __m256i& input_left,
    const __m256i& input_right,
    __m256i& out_low,
    __m256i& out_high) {
  alignas(32) uint32_t a[8], b[8];
  alignas(32) uint64_t result[8];

  _mm256_store_si256((__m256i*)a, input_left);
  _mm256_store_si256((__m256i*)b, input_right);

  for (int i = 0; i < 8; ++i) {
    result[i] = static_cast<uint64_t>(a[i]) * b[i];
  }

  // 打包输出（注意 set_epi64x 顺序是逆序的）
  out_low = _mm256_set_epi64x(result[3], result[2], result[1], result[0]);
  out_high = _mm256_set_epi64x(result[7], result[6], result[5], result[4]);
}

// 这些函数只被定义未被使用 包括arrow 13.0.0, 15.0.0, 16.0.0版本，或许直接删除更好？
template <class PROCESS_8_VALUES_FN>
int RowArrayAccessor::Visit_avx2(const RowTableImpl& rows, int column_id, int num_rows,
                                 const uint32_t* row_ids,
                                 PROCESS_8_VALUES_FN process_8_values_fn) {
  // Number of rows processed together in a single iteration of the loop (single
  // call to the provided processing lambda).
  //
  constexpr int unroll = 8;

  bool is_fixed_length_column =
      rows.metadata().column_metadatas[column_id].is_fixed_length;

  // There are 2 cases, each requiring different steps:
  // 1. Varying length column 
  // 2. Fixed length column 

  if (!is_fixed_length_column) {
    int varbinary_column_id = VarbinaryColumnId(rows.metadata(), column_id);
    const uint8_t* row_ptr_base = rows.data(1);
    __m256i row_length = _mm256_set1_epi32(rows.metadata().row_length());

    __m256i field_offset_within_row = 
        _mm256_set1_epi64x(
            rows.metadata().nth_varbinary_offset_within_row(varbinary_column_id));
    for (int i = 0; i < num_rows / unroll; ++i) {
        __m256i row_id =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row_ids) + i);
        __m256i row_offset_lo, row_offset_hi;
        avx2_mul_u32_to_u64(row_id, row_length, row_offset_lo, row_offset_hi);
        row_offset_lo = _mm256_add_epi64(row_offset_lo, field_offset_within_row);
        row_offset_hi = _mm256_add_epi64(row_offset_hi, field_offset_within_row);
        process_8_values_fn(i * unroll, row_ptr_base, row_offset_lo, row_offset_hi);
    }
  }

  if (is_fixed_length_column) {
    __m256i field_offset_within_row =
        _mm256_set1_epi32(rows.metadata().encoded_field_offset(
            rows.metadata().pos_after_encoding(column_id)));
    __m256i row_length =
        _mm256_set1_epi64x(rows.metadata().row_length());

    // Case 3: This is a fixed length column
    //
    const uint8_t* row_ptr_base = rows.data(1);
    for (int i = 0; i < num_rows / unroll; ++i) {
      __m256i row_id =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row_ids) + i);
      __m256i row_offset_lo, row_offset_hi;
      avx2_mul_u32_to_u64(row_id, row_length, row_offset_lo, row_offset_hi);
      row_offset_lo = _mm256_add_epi64(row_offset_lo, field_offset_within_row);
      row_offset_hi = _mm256_add_epi64(row_offset_hi, field_offset_within_row);
      process_8_values_fn(i * unroll, row_ptr_base, row_offset_lo, row_offset_hi);
    }
  }

  return num_rows - (num_rows % unroll);
}

template <class PROCESS_8_VALUES_FN>
int RowArrayAccessor::VisitNulls_avx2(const RowTableImpl& rows, int column_id,
                                      int num_rows, const uint32_t* row_ids,
                                      PROCESS_8_VALUES_FN process_8_values_fn) {
  // Number of rows processed together in a single iteration of the loop (single
  // call to the provided processing lambda).
  //
  constexpr int unroll = 8;

  const uint8_t* null_masks = rows.null_masks();
  __m256i null_bits_per_row =
      _mm256_set1_epi32(8 * rows.metadata().null_masks_bytes_per_row);
  for (int i = 0; i < num_rows / unroll; ++i) {
    __m256i row_id = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row_ids) + i);
    __m256i bit_id = _mm256_mullo_epi32(row_id, null_bits_per_row);
    bit_id = _mm256_add_epi32(bit_id, _mm256_set1_epi32(column_id));
    __m256i bytes = _mm256_i32gather_epi32(reinterpret_cast<const int*>(null_masks),
                                           _mm256_srli_epi32(bit_id, 3), 1);
    __m256i bit_in_word = _mm256_sllv_epi32(
        _mm256_set1_epi32(1), _mm256_and_si256(bit_id, _mm256_set1_epi32(7)));
    __m256i result =
        _mm256_cmpeq_epi32(_mm256_and_si256(bytes, bit_in_word), bit_in_word);
    uint64_t null_bytes = static_cast<uint64_t>(
        _mm256_movemask_epi8(_mm256_cvtepi32_epi64(_mm256_castsi256_si128(result))));
    null_bytes |= static_cast<uint64_t>(_mm256_movemask_epi8(
                      _mm256_cvtepi32_epi64(_mm256_extracti128_si256(result, 1))))
                  << 32;

    process_8_values_fn(i * unroll, null_bytes);
  }

  return num_rows - (num_rows % unroll);
}

}  // namespace acero
}  // namespace arrow
