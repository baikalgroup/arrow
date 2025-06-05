// Copyright (c) 2018 Baidu, Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>
#include <climits>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <arrow/ipc/writer.h>
#include <arrow/ipc/reader.h>
#include <arrow/api.h>
#include <arrow/buffer.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/dictionary.h>
#include <arrow/result.h>
#include <arrow/compute/function.h>
#include "arrow/compute/api_aggregate.h"
#include "arrow/compute/api_scalar.h"
#include "arrow/compute/api_vector.h"
#include "arrow/compute/cast.h"
#include "arrow/compute/function_internal.h"
#include "arrow/compute/kernel.h"
#include "arrow/compute/registry.h"
#include <arrow/acero/options.h>
#include <arrow/acero/exec_plan.h>
#include <arrow/status.h>
#include <vector>
#include "arrow/type_fwd.h"
#include <chrono>
#include <unistd.h>

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

inline void CHECK_OK(const arrow::Status status,const std::string& msg){
    if(!status.ok()) {                              
        std::cout<< "status not ok! msg: " << msg << std::endl;  
        exit(-1);
    } 
}

namespace arrow_test {
const int const_build_len = 1030;
const int const_match_keys = 11;
const int const_batch_size = 10000;
int basic_test_len = 2000 * 10000;

// 4KB
//const std::string long_prefix = "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";
 const std::string long_prefix = "123456";
const std::string short_prefix = "short_prefix_";
const std::string key_prefix = long_prefix;

uint64_t milliseconds_now(){
    auto now = std::chrono::system_clock::now();
 
    // 将时间戳转换为毫秒数
    auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
    auto value = now_ms.time_since_epoch().count();
    return value;
}

class SourceReader : public arrow::RecordBatchReader {
private:
    std::shared_ptr<arrow::Schema> schema_;
    int64_t batch_size_;
    int64_t batch_count_ = 0;
    int64_t total_len_ = 0;
    
    std::string prefix_;
    int64_t offset_;
    bool with_null_;

public:
    std::string loooooong_prefix;
    SourceReader(const std::string& prefix, 
                std::vector<std::shared_ptr<arrow::DataType>>& types, 
                int64_t batch_size, 
                int total_len, 
                bool with_null = false,
                std::string prefix_binary = long_prefix)
        : batch_size_{batch_size}, 
          prefix_(prefix), 
          total_len_(total_len), 
          offset_(0),
          with_null_(with_null),
          loooooong_prefix(prefix_binary) {
            int i = 0;
            std::vector<std::shared_ptr<arrow::Field>> fields;
            for(std::shared_ptr<arrow::DataType> &type : types){
                std::string field_name = prefix + "_" + std::to_string(i);
                fields.emplace_back(arrow::field(field_name, type));
                ++i;
            }
            schema_ = std::make_shared<arrow::Schema>(fields);
        }

    std::shared_ptr<arrow::Schema> schema() const override { return schema_; }

    // 每次吐一个recordBatch, 流式执行
    arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* out) override {
        // 模拟rocksdb iterator, 每扫描batch_size_行构建一个recordBatch, 流式执行，吐给下游执行
        if (total_len_ - offset_ <= 0) {
            out->reset();
            return arrow::Status::OK();
        }
        int64_t batch_size = std::min(batch_size_, total_len_ - offset_);
        auto status = build_row_batch(batch_size, out);
        CHECK_OK(status, "next batch: ");
        offset_ += batch_size;
        std::cout << prefix_ << ": " << offset_ << std::endl;
        return arrow::Status::OK();
    }

private:
    arrow::Status build_row_batch(int64_t batch_size, std::shared_ptr<arrow::RecordBatch>* out){
        arrow::Status status;
        std::vector<std::shared_ptr<arrow::Array>> arrays(schema_->num_fields());
        for(auto array: arrays){
            array = nullptr;
        }
 
        for(int i = 0 ; i < schema_->num_fields(); ++i){
            status = build_column(batch_size, arrays[i], i);
            CHECK_OK(status, "build_row_batch: ");
        }
        *out = arrow::RecordBatch::Make(schema_, batch_size, std::move(arrays));
        return arrow::Status::OK();
    }

    arrow::Status build_large_binary_key_col(
                std::unique_ptr<arrow::ArrayBuilder>& builder, 
                int64_t batch_size, 
                std::shared_ptr<arrow::Array>& output){
        auto type_builder = dynamic_cast<arrow::LargeBinaryBuilder*>((builder).get());
        for(size_t i = 0 ; i < batch_size ; ++i){
            arrow::Status status;
            if(with_null_ && i % 10 == 9){
                status = type_builder->AppendNull();
            }else{
                status = type_builder->Append(key_prefix + std::to_string(offset_ + i));
            }
            CHECK_OK(status, "build_key_column: Append: row: " + std::to_string(offset_ + i));
        }
        auto status = builder->Finish(&output);
        return status;
    }

    arrow::Status build_binary_key_col(
                std::unique_ptr<arrow::ArrayBuilder>& builder, 
                int64_t batch_size, 
                std::shared_ptr<arrow::Array>& output){
        auto type_builder = dynamic_cast<arrow::BinaryBuilder*>((builder).get());
        for(size_t i = 0 ; i < batch_size ; ++i){
            arrow::Status status;
            if(with_null_ && i % 10 == 9){
                status = type_builder->AppendNull();
            }else{
                status = type_builder->Append(key_prefix + std::to_string(offset_ + i));
            }
            CHECK_OK(status, "build_key_column: Append: row: " + std::to_string(offset_ + i));
        }
        auto status = builder->Finish(&output);
        return status;
    }


    template<typename T>
    arrow::Status build(std::unique_ptr<arrow::ArrayBuilder>& builder, int64_t batch_size, std::shared_ptr<arrow::Array>& output){
        auto type_builder = dynamic_cast<T*>(builder.get());
        for(size_t i = 0 ; i < batch_size ; ++i){
            auto status = type_builder->Append(static_cast<typename T::value_type>(offset_ + i));
            CHECK_OK(status, "build_key_column: Append: row: " + std::to_string(offset_ + i));
        }
        auto status = builder->Finish(&output);
        return status;
    }

    arrow::Status build_column(int64_t batch_size, std::shared_ptr<arrow::Array>& array, int col_index);
};

template< >
arrow::Status SourceReader::build<arrow::BinaryBuilder>(
            std::unique_ptr<arrow::ArrayBuilder>& builder, 
            int64_t batch_size, 
            std::shared_ptr<arrow::Array>& output){
    auto type_builder = dynamic_cast<arrow::BinaryBuilder*>((builder).get());
    for(size_t i = 0 ; i < batch_size ; ++i){
        arrow::Status status;
        if(with_null_ && i % 10 == 9){
            status = type_builder->AppendNull();
        }else{
            status = type_builder->Append(loooooong_prefix + std::to_string(offset_ + i));
        }
    }
    auto status = builder->Finish(&output);
    return status;
}

template< >
arrow::Status SourceReader::build<arrow::LargeBinaryBuilder>(
            std::unique_ptr<arrow::ArrayBuilder>& builder, 
            int64_t batch_size, 
            std::shared_ptr<arrow::Array>& output){
    auto type_builder = dynamic_cast<arrow::LargeBinaryBuilder  *>((builder).get());
    for(size_t i = 0 ; i < batch_size ; ++i){
        arrow::Status status;
        if(with_null_ && i % 10 == 9){
            status = type_builder->AppendNull();
        }else{
            status = type_builder->Append(loooooong_prefix + std::to_string(offset_ + i));
        }
        CHECK_OK(status, "build_key_column: Append: row: " + std::to_string(offset_ + i));
    }
    auto status = builder->Finish(&output);
    return status;
}

arrow::Status SourceReader::build_column(int64_t batch_size, std::shared_ptr<arrow::Array>& array, int col_index) {
    auto type = schema_->field(col_index)->type();
    std::unique_ptr<arrow::ArrayBuilder> builder = nullptr;
    arrow::MakeBuilder(arrow::default_memory_pool(), type, &builder);

    if(col_index == 0){
        if(type->id() == arrow::Type::type::BINARY){
            return build_binary_key_col(builder, batch_size, array);
        }else if(type->id() == arrow::Type::type::LARGE_BINARY){
            return build_large_binary_key_col(builder, batch_size, array);
        }
    }
    
    auto typeId = type->id();
    switch(typeId){
        case arrow::Type::type::INT32:
            return build<arrow::Int32Builder>(builder, batch_size, array);
        case arrow::Type::type::UINT64:
            return build<arrow::UInt64Builder>(builder, batch_size, array);
        case arrow::Type::type::INT64:
            return build<arrow::Int64Builder>(builder, batch_size, array);
        case arrow::Type::type::FLOAT:
            return build<arrow::FloatBuilder>(builder, batch_size, array);
        case arrow::Type::type::DOUBLE:
            return build<arrow::DoubleBuilder>(builder, batch_size, array);
        case arrow::Type::type::BINARY:
            return build<arrow::BinaryBuilder>(builder, batch_size, array);
        case arrow::Type::type::LARGE_BINARY:
            return build<arrow::LargeBinaryBuilder>(builder, batch_size, array);
        default:
            return arrow::Status::Invalid("unsupport type");
    }
}

void wide_join_test(int left_len, int right_len) {
	auto cup_e = arrow::internal::GetCpuThreadPool();

    std::vector<std::shared_ptr<arrow::DataType>> left_types{arrow::int32(), arrow::int64(), arrow::int64(),arrow::int64(),arrow::int64(),arrow::int64(),arrow::int64(),arrow::int64(),arrow::int64(),arrow::int64(),arrow::int64(),arrow::int64(),arrow::int64(),arrow::int64(),arrow::int64(),arrow::int64(),arrow::int64(), arrow::large_binary(), arrow::large_binary(),arrow::large_binary(),arrow::large_binary(),arrow::large_binary(),arrow::large_binary(),arrow::large_binary(),arrow::large_binary(),arrow::large_binary(),arrow::large_binary(),arrow::large_binary(),arrow::large_binary()};
    std::vector<std::shared_ptr<arrow::DataType>> right_types{arrow::int32(), arrow::int64(), arrow::int64(),arrow::int64(),arrow::int64(),arrow::int64(),arrow::int64(),arrow::int64(),arrow::int64(),arrow::int64(),arrow::int64(),arrow::int64(),arrow::int64(),arrow::int64(),arrow::int64(),arrow::int64(),arrow::int64(), arrow::large_binary(), arrow::large_binary(),arrow::large_binary(),arrow::large_binary(),arrow::large_binary(),arrow::large_binary(),arrow::large_binary(),arrow::large_binary(),arrow::large_binary(),arrow::large_binary(),arrow::large_binary(),arrow::large_binary()};
    std::shared_ptr<SourceReader> reader_left =
            std::make_shared<SourceReader>("left", left_types, const_batch_size, left_len);
    std::shared_ptr<SourceReader> reader_right =
            std::make_shared<SourceReader>("right", right_types, const_batch_size, right_len);

    arrow::acero::Declaration left{"record_batch_reader_source", arrow::acero::RecordBatchReaderSourceNodeOptions(reader_left)};
    arrow::acero::Declaration right{"record_batch_reader_source", arrow::acero::RecordBatchReaderSourceNodeOptions(reader_right)};
    arrow::acero::HashJoinNodeOptions join_options{arrow::acero::JoinType::INNER,
                                    {arrow::FieldRef("left_0")},
                                    {arrow::FieldRef("right_0")},
                                    arrow::compute::literal(true)};
    arrow::acero::Declaration join{"hashjoin", {std::move(left), std::move(right)}, join_options};
    uint64_t join_start_time = milliseconds_now();

    auto result = arrow::acero::DeclarationToTable(std::move(join), /*use_threads=*/true);
    std::shared_ptr<arrow::Table> final_table = result.ValueOrDie();
	int64_t len = final_table->num_rows();
    uint64_t join_end_time = milliseconds_now();
}

void basic_join_test(int left_len, int right_len, bool is_large_binary, bool check_result = true) {

    auto cup_e = arrow::internal::GetCpuThreadPool();

    std::vector<std::shared_ptr<arrow::DataType>> left_types{arrow::int32(), arrow::int64()};
    std::vector<std::shared_ptr<arrow::DataType>> right_types{arrow::int32(), is_large_binary ? arrow::large_binary() : arrow::binary()};

    std::shared_ptr<SourceReader> reader_left = 
            std::make_shared<SourceReader>("left", left_types, const_batch_size, left_len);
    std::shared_ptr<SourceReader> reader_right = 
            std::make_shared<SourceReader>("right", right_types, const_batch_size, right_len, /*with_null*/ true);

    arrow::acero::Declaration left{"record_batch_reader_source", arrow::acero::RecordBatchReaderSourceNodeOptions(reader_left)};
    arrow::acero::Declaration right{"record_batch_reader_source", arrow::acero::RecordBatchReaderSourceNodeOptions(reader_right)};
    arrow::acero::HashJoinNodeOptions join_options{arrow::acero::JoinType::INNER,
                                    {arrow::FieldRef("left_0")},
                                    {arrow::FieldRef("right_0")},
                                    arrow::compute::literal(true)};
    arrow::acero::Declaration join{"hashjoin", {std::move(left), std::move(right)}, join_options};

    uint64_t join_start_time = milliseconds_now();

    auto result = arrow::acero::DeclarationToTable(std::move(join), /*use_threads=*/true);
    std::shared_ptr<arrow::Table> final_table = result.ValueOrDie();

    uint64_t join_end_time = milliseconds_now();
    std::cout << "join time: " << (join_end_time - join_start_time) / 1000.0 << "s" << std::endl;
    std::shared_ptr<arrow::ChunkedArray> right_col1 = final_table->GetColumnByName("right_1");
    std::shared_ptr<arrow::ChunkedArray> left_col1 = final_table->GetColumnByName("left_1");

    std::cout << "join output " << final_table->num_rows() << " rows" << std::endl;
    ASSERT_EQ(final_table->num_rows(), left_len);
    std::cout << "after join, table row: "  << final_table->num_rows() << std::endl;
    
    if(check_result){
        for(int i = 0 ; i < left_len; ++i){
            auto str = left_col1->GetScalar(i).ValueOrDie()->ToString();
            int row_index = std::stoi(str);
            
            if(row_index % 10 == 9){
                ASSERT_TRUE(!right_col1->GetScalar(i).ValueOrDie()->is_valid);
            }
            else{
                auto str1 = reader_left->loooooong_prefix + str;
                auto str2 = right_col1->GetScalar(i).ValueOrDie()->ToString();
                ASSERT_EQ(str1, str2);
                // if(str1 != str2){
                //     std::cout << "str1: " << str1 << std::endl;
                //     std::cout << "str2: " << str2 << std::endl;
                //     ASSERT_EQ(str1, str2);
                // }
            }
        }
        std::cout << "each row match!" << std::endl;
    }
}

// left_len should equal to right_len
void join_key_as_string_test(int left_len, int right_len, bool is_large_binary, bool left_with_null, bool right_with_null) {

    auto cup_e = arrow::internal::GetCpuThreadPool();

    std::vector<std::shared_ptr<arrow::DataType>> left_types{is_large_binary ? arrow::large_binary() : arrow::binary(), arrow::int64()};
    std::vector<std::shared_ptr<arrow::DataType>> right_types{is_large_binary ? arrow::large_binary() : arrow::binary(), is_large_binary ? arrow::large_binary() : arrow::binary()};

    std::shared_ptr<SourceReader> reader_left = 
            std::make_shared<SourceReader>("left", left_types, const_batch_size, left_len, left_with_null);
    std::shared_ptr<SourceReader> reader_right = 
            std::make_shared<SourceReader>("right", right_types, const_batch_size, right_len, /*with_null*/ right_with_null);

    arrow::acero::Declaration left{"record_batch_reader_source", arrow::acero::RecordBatchReaderSourceNodeOptions(reader_left)};
    arrow::acero::Declaration right{"record_batch_reader_source", arrow::acero::RecordBatchReaderSourceNodeOptions(reader_right)};
    arrow::acero::HashJoinNodeOptions join_options{arrow::acero::JoinType::INNER,
                                    {arrow::FieldRef("left_0")},
                                    {arrow::FieldRef("right_0")},
                                    arrow::compute::literal(true)};
    arrow::acero::Declaration join{"hashjoin", {std::move(left), std::move(right)}, join_options};

    uint64_t join_start_time = milliseconds_now();

    auto result = arrow::acero::DeclarationToTable(std::move(join), /*use_threads=*/true);
    std::shared_ptr<arrow::Table> final_table = result.ValueOrDie();

    uint64_t join_end_time = milliseconds_now();
    std::cout << "join time: " << (join_end_time - join_start_time) / 1000.0 << "s" << std::endl;
    std::shared_ptr<arrow::ChunkedArray> right_col1 = final_table->GetColumnByName("right_1");
    std::shared_ptr<arrow::ChunkedArray> left_col1 = final_table->GetColumnByName("left_0");
    std::shared_ptr<arrow::ChunkedArray> id_col = final_table->GetColumnByName("left_1");

    std::cout << "join output " << final_table->num_rows() << " rows" << std::endl;
    int expected_len = left_with_null ? left_len / 10 * 9 + left_len % 10 : left_len;
    int expected_right = right_with_null ? right_len / 10 * 9 + right_len % 10 : right_len;
    expected_len = expected_len > expected_right ? expected_right : expected_len;
    ASSERT_EQ(final_table->num_rows(), expected_len);
    
    std::cout << "after join, table row: "  << final_table->num_rows() << std::endl;

    for(int i = 0 ; i < expected_len; ++i){
        auto id = id_col->GetScalar(i).ValueOrDie()->ToString();
        int row_index = std::stoi(id);

        auto str1 = reader_left->loooooong_prefix + id;
        auto str2 = right_col1->GetScalar(i).ValueOrDie()->ToString();
        ASSERT_EQ(str1, str2);
        // if(str1 != str2){
        //     std::cout << "str1: " << str1 << std::endl;
        //     std::cout << "str2: " << str2 << std::endl;
        //     ASSERT_EQ(str1, str2);
        // }
    }
    std::cout << "each row match!" << std::endl;
    //std::cout << "Results : " << final_table->ToString() << std::endl;
}

void join_key_as_two_int_test(int left_len, int right_len, bool is_large_binary) {

    auto cup_e = arrow::internal::GetCpuThreadPool();

    std::vector<std::shared_ptr<arrow::DataType>> left_types{arrow::int32(), arrow::int32()};
    std::vector<std::shared_ptr<arrow::DataType>> right_types{arrow::int32(), arrow::int32(), is_large_binary ? arrow::large_binary() : arrow::binary()};

    std::shared_ptr<SourceReader> reader_left = 
            std::make_shared<SourceReader>("left", left_types, const_batch_size, left_len);
    std::shared_ptr<SourceReader> reader_right = 
            std::make_shared<SourceReader>("right", right_types, const_batch_size, right_len, /*with_null*/ true);

    arrow::acero::Declaration left{"record_batch_reader_source", arrow::acero::RecordBatchReaderSourceNodeOptions(reader_left)};
    arrow::acero::Declaration right{"record_batch_reader_source", arrow::acero::RecordBatchReaderSourceNodeOptions(reader_right)};
    arrow::acero::HashJoinNodeOptions join_options{arrow::acero::JoinType::INNER,
                                    {arrow::FieldRef("left_0"), arrow::FieldRef("left_1")},
                                    {arrow::FieldRef("right_0"), arrow::FieldRef("right_1")},
                                    arrow::compute::literal(true)};
    arrow::acero::Declaration join{"hashjoin", {std::move(left), std::move(right)}, join_options};

    uint64_t join_start_time = milliseconds_now();

    auto result = arrow::acero::DeclarationToTable(std::move(join), /*use_threads=*/true);
    std::shared_ptr<arrow::Table> final_table = result.ValueOrDie();

    uint64_t join_end_time = milliseconds_now();
    std::cout << "join time: " << (join_end_time - join_start_time) / 1000.0 << "s" << std::endl;
    std::shared_ptr<arrow::ChunkedArray> right_col1 = final_table->GetColumnByName("right_2");
    std::shared_ptr<arrow::ChunkedArray> left_col1 = final_table->GetColumnByName("left_1");

    std::cout << "join output " << final_table->num_rows() << " rows" << std::endl;
    ASSERT_EQ(final_table->num_rows(), left_len);
    std::cout << "after join, table row: "  << final_table->num_rows() << std::endl;

    for(int i = 0 ; i < left_len; ++i){
        auto str = left_col1->GetScalar(i).ValueOrDie()->ToString();
        int row_index = std::stoi(str);
        
        if(row_index % 10 == 9){
            ASSERT_TRUE(!right_col1->GetScalar(i).ValueOrDie()->is_valid);
        }
        else{
            auto str1 = reader_left->loooooong_prefix + str;
            auto str2 = right_col1->GetScalar(i).ValueOrDie()->ToString();
            ASSERT_EQ(str1, str2);
            // if(str1 != str2){
            //     std::cout << "str1: " << str1 << std::endl;
            //     std::cout << "str2: " << str2 << std::endl;
            //     ASSERT_EQ(str1, str2);
            // }
        }
    }
    std::cout << "each row match!" << std::endl;
    //std::cout << "Results : " << final_table->ToString() << std::endl;
}

void join_key_as_int_and_binary_test(int left_len, int right_len, bool is_large_binary) {

    auto cup_e = arrow::internal::GetCpuThreadPool();

    std::vector<std::shared_ptr<arrow::DataType>> left_types{arrow::int32(), is_large_binary ? arrow::large_binary() : arrow::binary()};
    std::vector<std::shared_ptr<arrow::DataType>> right_types{arrow::int32(), is_large_binary ? arrow::large_binary() : arrow::binary(), is_large_binary ? arrow::large_binary() : arrow::binary()};

    std::shared_ptr<SourceReader> reader_left = 
            std::make_shared<SourceReader>("left", left_types, const_batch_size, left_len);
    std::shared_ptr<SourceReader> reader_right = 
            std::make_shared<SourceReader>("right", right_types, const_batch_size, right_len, /*with_null*/ true);

    arrow::acero::Declaration left{"record_batch_reader_source", arrow::acero::RecordBatchReaderSourceNodeOptions(reader_left)};
    arrow::acero::Declaration right{"record_batch_reader_source", arrow::acero::RecordBatchReaderSourceNodeOptions(reader_right)};
    arrow::acero::HashJoinNodeOptions join_options{arrow::acero::JoinType::INNER,
                                    {arrow::FieldRef("left_0"), arrow::FieldRef("left_1")},
                                    {arrow::FieldRef("right_0"), arrow::FieldRef("right_1")},
                                    arrow::compute::literal(true)};
    arrow::acero::Declaration join{"hashjoin", {std::move(left), std::move(right)}, join_options};

    uint64_t join_start_time = milliseconds_now();

    auto result = arrow::acero::DeclarationToTable(std::move(join), /*use_threads=*/true);
    std::shared_ptr<arrow::Table> final_table = result.ValueOrDie();

    uint64_t join_end_time = milliseconds_now();
    std::cout << "join time: " << (join_end_time - join_start_time) / 1000.0 << "s" << std::endl;
    std::shared_ptr<arrow::ChunkedArray> right_col1 = final_table->GetColumnByName("right_2");
    std::shared_ptr<arrow::ChunkedArray> left_col1 = final_table->GetColumnByName("left_0");

    std::cout << "join output " << final_table->num_rows() << " rows" << std::endl;
    int expected_len = left_len / 10 * 9 + left_len % 10;
    ASSERT_EQ(final_table->num_rows(), expected_len);
    std::cout << "after join, table row: "  << final_table->num_rows() << std::endl;

    for(int i = 0 ; i < expected_len; ++i){
        auto str = left_col1->GetScalar(i).ValueOrDie()->ToString();
        int row_index = std::stoi(str);
        
        if(row_index % 10 == 9){
            ASSERT_TRUE(!right_col1->GetScalar(i).ValueOrDie()->is_valid);
        }
        else{
            auto str1 = reader_left->loooooong_prefix + str;
            auto str2 = right_col1->GetScalar(i).ValueOrDie()->ToString();
            ASSERT_EQ(str1, str2);
            // if(str1 != str2){
            //     std::cout << "str1: " << str1 << std::endl;
            //     std::cout << "str2: " << str2 << std::endl;
            //     ASSERT_EQ(str1, str2);
            // }
        }
    }
    std::cout << "each row match!" << std::endl;
    //std::cout << "Results : " << final_table->ToString() << std::endl;
}

void run_agg_without_join_key_as_binary(int table_len, bool is_large_binary, bool check_result = true) {
    auto cup_e = arrow::internal::GetCpuThreadPool();

    // 三列，第一列是id，后两列为4kb字符串（> 4kb）
    std::vector<std::shared_ptr<arrow::DataType>> types{is_large_binary ? arrow::large_binary() :arrow::binary(), 
                                                        arrow::int64()};
    std::shared_ptr<SourceReader> reader = std::make_shared<SourceReader>("table", types, const_batch_size, table_len);
    
    arrow::acero::Declaration data_source{"record_batch_reader_source", arrow::acero::RecordBatchReaderSourceNodeOptions(reader)};

    std::vector<arrow::compute::Aggregate> aggregates;
    aggregates.emplace_back("hash_max" , /*options*/nullptr, arrow::FieldRef("table_1"), /*new field name*/"col1");
    // aggregates.emplace_back("hash_max" , /*options*/nullptr, arrow::FieldRef("table_2"), /*new field name*/"col2");

    std::vector<arrow::FieldRef> group_by_fields;
    group_by_fields.emplace_back(arrow::FieldRef("table_0"));

    arrow::acero::AggregateNodeOptions agg_options{aggregates, group_by_fields};

    arrow::acero::Declaration agg{"aggregate", {std::move(data_source)}, std::move(agg_options)};
    uint64_t start_time = milliseconds_now();
    auto result = arrow::acero::DeclarationToTable(std::move(agg), /*use_threads=*/true);
    uint64_t end_time = milliseconds_now();
    auto final_table = result.ValueOrDie();

    std::cout << "agg time: " << (end_time - start_time) / 1000.0 << "s" << std::endl;

    std::shared_ptr<arrow::ChunkedArray> col0 = final_table->GetColumnByName("table_0");
    std::shared_ptr<arrow::ChunkedArray> col1 = final_table->GetColumnByName("col1");
    // std::shared_ptr<arrow::ChunkedArray> col2 = final_table->GetColumnByName("col2");

    if(check_result){
        for(int i = 0 ; i < table_len; ++i){
            auto id_str = col1->GetScalar(i).ValueOrDie()->ToString();
            
            auto str1 = key_prefix + id_str;
            auto str0 = col0->GetScalar(i).ValueOrDie()->ToString();
            ASSERT_EQ(str1, str0);
            // if(str1 != str2){
            //     std::cout << "str1: " << str1 << std::endl;
            //     std::cout << "str2: " << str2 << std::endl;
            //     ASSERT_EQ(str1, str2);
            // }
        }
    }
    std::cout << "after agg rows size: " << col1->length() << std::endl;
    int64_t bytes_size = 0;

    std::cout << "total date in Byte: " << bytes_size << std::endl;
}

void run_agg_without_join_key_as_int(int table_len, bool is_large_binary, bool check_result = true) {
    auto cup_e = arrow::internal::GetCpuThreadPool();

    // 三列，第一列是id，后两列为4kb字符串（> 4kb）
    std::vector<std::shared_ptr<arrow::DataType>> types{arrow::int64(),
                                                        is_large_binary ? arrow::large_binary() :arrow::binary()};
    std::shared_ptr<SourceReader> reader = std::make_shared<SourceReader>("table", types, const_batch_size, table_len);
    
    arrow::acero::Declaration data_source{"record_batch_reader_source", arrow::acero::RecordBatchReaderSourceNodeOptions(reader)};

    std::vector<arrow::compute::Aggregate> aggregates;
    aggregates.emplace_back("hash_max" , /*options*/nullptr, arrow::FieldRef("table_1"), /*new field name*/"col1");

    std::vector<arrow::FieldRef> group_by_fields;
    group_by_fields.emplace_back(arrow::FieldRef("table_0"));

    arrow::acero::AggregateNodeOptions agg_options{aggregates, group_by_fields};

    arrow::acero::Declaration agg{"aggregate", {std::move(data_source)}, std::move(agg_options)};

    double start_time = milliseconds_now();
    auto result = arrow::acero::DeclarationToTable(std::move(agg), /*use_threads=*/true);
    double end_time = milliseconds_now();
    std::cout << "agg time: " << (end_time - start_time) / 1000.0 << "s" << std::endl;
    auto final_table = result.ValueOrDie();

    std::shared_ptr<arrow::ChunkedArray> col0 = final_table->GetColumnByName("table_0");
    std::shared_ptr<arrow::ChunkedArray> col1 = final_table->GetColumnByName("col1");

    std::cout << "after agg rows size: " << col1->length() << std::endl;
    int64_t bytes_size = 0;

    if(check_result){
        for(int i = 0 ; i < table_len; ++i){
            std::string id_str = col0->GetScalar(i).ValueOrDie()->ToString();
            std::string str1 = col1->GetScalar(i).ValueOrDie()->ToString();
            // if(str1 != reader->loooooong_prefix + id_str){
            //     std::cout << "i = " << i << std::endl;
            //     std::cout << str1 << std::endl;
            //     std::cout << reader->loooooong_prefix + id_str << std::endl;
            // }
            ASSERT_TRUE(str1 == reader->loooooong_prefix + id_str);

        }

        std::cout << "total date in Byte: " << bytes_size << std::endl;
    }
}

void run_agg_without_join_key_as_two_int(int table_len, bool is_large_binary) {
    auto cup_e = arrow::internal::GetCpuThreadPool();

    // 三列，第一列是id，后两列为4kb字符串（> 4kb）
    std::vector<std::shared_ptr<arrow::DataType>> types{arrow::int64(),
                                                        arrow::int32(),  
                                                        is_large_binary ? arrow::large_binary() :arrow::binary()};
    std::shared_ptr<SourceReader> reader = std::make_shared<SourceReader>("table", types, const_batch_size, table_len);
    
    arrow::acero::Declaration data_source{"record_batch_reader_source", arrow::acero::RecordBatchReaderSourceNodeOptions(reader)};

    std::vector<arrow::compute::Aggregate> aggregates;
    // aggregates.emplace_back("hash_max" , /*options*/nullptr, arrow::FieldRef("table_1"), /*new field name*/"col1");
    aggregates.emplace_back("hash_max" , /*options*/nullptr, arrow::FieldRef("table_2"), /*new field name*/"col2");

    std::vector<arrow::FieldRef> group_by_fields;
    group_by_fields.emplace_back(arrow::FieldRef("table_0"));
    group_by_fields.emplace_back(arrow::FieldRef("table_1"));

    arrow::acero::AggregateNodeOptions agg_options{aggregates, group_by_fields};

    arrow::acero::Declaration agg{"aggregate", {std::move(data_source)}, std::move(agg_options)};
    auto result = arrow::acero::DeclarationToTable(std::move(agg), /*use_threads=*/true);
    auto final_table = result.ValueOrDie();

    std::shared_ptr<arrow::ChunkedArray> col0 = final_table->GetColumnByName("table_0");
    std::shared_ptr<arrow::ChunkedArray> col1 = final_table->GetColumnByName("table_1");
    std::shared_ptr<arrow::ChunkedArray> col2 = final_table->GetColumnByName("col2");

    std::cout << "after agg rows size: " << col1->length() << std::endl;
    int64_t bytes_size = 0;

    bool flag = true;
    for(int i = 0 ; i < table_len; ++i){
        std::string id_str = col0->GetScalar(i).ValueOrDie()->ToString();
        std::string str2 = col2->GetScalar(i).ValueOrDie()->ToString();
        // if(str1 != reader->loooooong_prefix + id_str){
        //     std::cout << "i = " << i << std::endl;
        //     std::cout << str1 << std::endl;
        //     std::cout << reader->loooooong_prefix + id_str << std::endl;
        // }
        ASSERT_TRUE(str2 == reader->loooooong_prefix + id_str);

    }
}

void run_agg_without_join_key_as_binary_and_int(int table_len, bool is_large_binary) {
    auto cup_e = arrow::internal::GetCpuThreadPool();

    // 三列，第一列是id，后两列为4kb字符串（> 4kb）
    std::vector<std::shared_ptr<arrow::DataType>> types{is_large_binary ? arrow::large_binary() :arrow::binary(),
                                                        arrow::int32(),  
                                                        is_large_binary ? arrow::large_binary() :arrow::binary()};
    std::shared_ptr<SourceReader> reader = std::make_shared<SourceReader>("table", types, const_batch_size, table_len);
    
    arrow::acero::Declaration data_source{"record_batch_reader_source", arrow::acero::RecordBatchReaderSourceNodeOptions(reader)};

    std::vector<arrow::compute::Aggregate> aggregates;
    // aggregates.emplace_back("hash_max" , /*options*/nullptr, arrow::FieldRef("table_1"), /*new field name*/"col1");
    aggregates.emplace_back("hash_max" , /*options*/nullptr, arrow::FieldRef("table_2"), /*new field name*/"col2");

    std::vector<arrow::FieldRef> group_by_fields;
    group_by_fields.emplace_back(arrow::FieldRef("table_0"));
    group_by_fields.emplace_back(arrow::FieldRef("table_1"));

    arrow::acero::AggregateNodeOptions agg_options{aggregates, group_by_fields};

    arrow::acero::Declaration agg{"aggregate", {std::move(data_source)}, std::move(agg_options)};
    auto result = arrow::acero::DeclarationToTable(std::move(agg), /*use_threads=*/true);
    auto final_table = result.ValueOrDie();

    std::shared_ptr<arrow::ChunkedArray> col0 = final_table->GetColumnByName("table_0");
    std::shared_ptr<arrow::ChunkedArray> col1 = final_table->GetColumnByName("table_1");
    std::shared_ptr<arrow::ChunkedArray> col2 = final_table->GetColumnByName("col2");

    std::cout << "after agg rows size: " << col1->length() << std::endl;
    int64_t bytes_size = 0;

    bool flag = true;
    for(int i = 0 ; i < table_len; ++i){
        std::string id_str = col1->GetScalar(i).ValueOrDie()->ToString();
        std::string str2 = col2->GetScalar(i).ValueOrDie()->ToString();
        // if(str1 != reader->loooooong_prefix + id_str){
        //     std::cout << "i = " << i << std::endl;
        //     std::cout << str1 << std::endl;
        //     std::cout << reader->loooooong_prefix + id_str << std::endl;
        // }
        ASSERT_TRUE(str2 == reader->loooooong_prefix + id_str);

    }
}

//TEST(test_arrow_vector_execute, join_key_as_string) {
//    std::cout << "==========    1    ========================" << std::endl;
//    join_key_as_string_test(basic_test_len, basic_test_len, /*is_large_binary*/true, /*left_with_null*/ true, /*right_with_null*/true);
//    std::cout << "==========    2    ========================" << std::endl;
//    join_key_as_string_test(basic_test_len, basic_test_len, /*is_large_binary*/false, /*left_with_null*/ true, /*right_with_null*/true);
//    std::cout << "==========    3    ========================" << std::endl;
//    join_key_as_string_test(basic_test_len, basic_test_len, /*is_large_binary*/true, /*left_with_null*/ false, /*right_with_null*/true);
//    std::cout << "==========    4    ========================" << std::endl;
//    join_key_as_string_test(basic_test_len, basic_test_len, /*is_large_binary*/true, /*left_with_null*/ true, /*right_with_null*/false);
//    std::cout << "==========    5    ========================" << std::endl;
//    join_key_as_string_test(basic_test_len, basic_test_len, /*is_large_binary*/true, /*left_with_null*/ false, /*right_with_null*/false);
//    // Test Null
//    // run_test_arrow_run_acero_async(1024 * 1400, 1024 * 1400);
//    // run_agg_without_join_larger_than_4G_async();
//}
//
//// 基本测试，是否包含null
//TEST(test_arrow_vector_execute, TestBasicJoin){
//    // Test Basic
//    basic_join_test(basic_test_len, basic_test_len, true);
//    basic_join_test(basic_test_len, basic_test_len, false);
//}
//
//TEST(test_arrow_vector_execute, join_key_as_two_int){
//
//    join_key_as_two_int_test(basic_test_len, basic_test_len, true);
//    join_key_as_two_int_test(basic_test_len, basic_test_len, false);
//}
//
//TEST(test_arrow_vector_execute, join_key_as_int_and_binary){
//    // Test Basic
//    join_key_as_int_and_binary_test(basic_test_len, basic_test_len, true);
//    join_key_as_int_and_binary_test(basic_test_len, basic_test_len, false);
//}
//
//TEST(test_arrow_vector_execute, group_basic_key_as_binary){
//    run_agg_without_join_key_as_binary(basic_test_len, true);
//    run_agg_without_join_key_as_binary(basic_test_len, false);
//}
//
//TEST(test_arrow_vector_execute, group_basic_key_as_int){
//    // run_agg_without_join_key_as_int(basic_test_len, true);
//    run_agg_without_join_key_as_int(basic_test_len, false);
//}
//
//TEST(test_arrow_vector_execute, agg_without_join_key_as_two_int){
//    run_agg_without_join_key_as_two_int(basic_test_len, true);
//    run_agg_without_join_key_as_two_int(basic_test_len, false);
//}
//
//TEST(test_arrow_vector_execute, agg_without_join_key_as_binary_and_int){
//    run_agg_without_join_key_as_binary_and_int(basic_test_len, true);
//    run_agg_without_join_key_as_binary_and_int(basic_test_len, false);
//}

TEST(test_arrow_vector_execute, test_wide_table_join) {
  wide_join_test(20000000, 20000000);
}

// TEST(test_arrow_vector_execute, join_performence_test){
//     std::cout << "pid = " << getpid() << std::endl;
//     int _4_mb = 1024;
//     for(int i = 8 ; i < 16 ; ++i){
//         int rows = 100 * i * _4_mb;
//         std::cout << "total size: 2 * " << 400 * i <<"MB" << std::endl;
//         std::cout << "type 'goon' thing to continue" << std::endl;
//         std::string str;
//         while(str != "goon"){
//             std::cin >> str;
//         }
//         join_key_as_string_test(rows, rows, /*is_large_binary*/true, false, false);
//     }
// }

// TEST(test_arrow_vector_execute, agg_performence_test){
//     std::cout << "pid = " << getpid() << std::endl;
//     int _4_mb = 1024;
//     int i ; 
//     std::cout << "please input i: " << std::endl;
//     std::cin >> i;
//     int rows = 100 * i * _4_mb;
//     std::cout << "total size: 2 * " << 400 * i <<"MB" << std::endl;
//     std::cout << "type 'goon' thing to continue" << std::endl;
//     std::string str;
//     while(str != "goon"){
//         std::cin >> str;
//     }
//     run_agg_without_join_key_as_int(rows, /*is_large_binary*/true, true);
// }

// TEST(test_arrow_vector_execute, free_test){
//     join_key_as_string_test(4, 4, /*is_large_binary*/true, false, false);
// }

}  // namespace baikal