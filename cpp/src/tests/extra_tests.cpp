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
#include <cassert>

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

namespace arrow_test {

void build_arrow_table2(const std::string& prefix, std::shared_ptr<arrow::Schema>* schema ,std::shared_ptr<arrow::Table>* table, int64_t rows, int skip_gap) {
    // 10000万表，10列，全部都是string
    std::vector<std::shared_ptr<arrow::Array>> array_list;
    std::vector<std::shared_ptr<arrow::Field>> fields;
    for (auto i = 0; i < 1; ++i) {
        fields.emplace_back(arrow::field(prefix + "_" + std::to_string(i), arrow::large_binary()));
    }
    int i = 0;
    for (auto f : fields) {
        std::unique_ptr<arrow::ArrayBuilder> builder = nullptr;
        std::string prefix = "this_is_row_";
        arrow::MakeBuilder(arrow::default_memory_pool(), f->type(), &builder);
        int row_count = 0;
        for (auto i = 0; i < rows; ++i) {
            if(i % skip_gap == skip_gap - 1) {continue;}
            std::string v = prefix + std::to_string(i);
            dynamic_cast<arrow::LargeBinaryBuilder*>((builder).get())->Append(v);
            row_count ++;
        }
        std::cout << "total " << row_count << " rows" << std::endl;
        std::shared_ptr<arrow::Array> a;
        builder->Finish(&a);
        array_list.emplace_back(a);
    }
    *schema = std::make_shared<arrow::Schema>(fields);
    *table = arrow::Table::Make(*schema, array_list, rows % skip_gap + rows / skip_gap * (skip_gap  - 1));
}

int LCM(int left, int right){
    int lcm;
    for(int i = 1; ++i ; i <= right){
        if (left * i % right == 0) {
            return left * i;
        }
    }
    return -1;
}

int match_rows(int total, int skip_l, int skip_r) {
    int lcm = LCM(skip_l, skip_r);

    return total - total / skip_l - total / skip_r + total / lcm;
}

int64_t run_test_arrow_run_acero_async2(int rows, int skip_left, int skip_right, arrow::acero::JoinType join_type) {
    std::cout << "==========================" << std::endl;
    std::cout << "test for " << arrow::acero::ToString(join_type) << std::endl;
    std::shared_ptr<arrow::Table> left_table, right_table;
    std::shared_ptr<arrow::Schema> left_schema, right_schema;
    build_arrow_table2("left", &left_schema, &left_table, rows, skip_left);
    build_arrow_table2("right", &right_schema, &right_table, rows, skip_right);
    arrow::acero::Declaration left{"table_source", arrow::acero::TableSourceNodeOptions(left_table, rows % skip_left + rows / skip_left * (skip_left  - 1))};
    arrow::acero::Declaration right{"table_source", arrow::acero::TableSourceNodeOptions(right_table, rows % skip_right + rows / skip_right * (skip_right  - 1))};
    arrow::acero::HashJoinNodeOptions join_options{join_type,
                                    {arrow::FieldRef("left_0")},
                                    {arrow::FieldRef("right_0")},
                                    arrow::compute::literal(true)};
    arrow::acero::Declaration join{"hashjoin", {std::move(left), std::move(right)}, join_options};
    auto result = arrow::acero::DeclarationToTable(std::move(join), /*use_threads=*/true);
    std::shared_ptr<arrow::Table> final_table = result.ValueOrDie();
    
    int expected_rows;
    // semi join下只有一边输出
    bool checkout_left = false, checkout_right = false;
    switch(join_type){
        case arrow::acero::JoinType::LEFT_SEMI: {
            expected_rows = match_rows(rows, skip_left, skip_right);
            checkout_left = true;
            break;
        }
        case arrow::acero::JoinType::LEFT_ANTI: {
            expected_rows = rows - rows / skip_left - match_rows(rows, skip_left, skip_right);
            checkout_left = true;
            break;
        }
        case arrow::acero::JoinType::LEFT_OUTER: {
            expected_rows = rows - rows / skip_left;
            checkout_left = true;
            checkout_right = true;
            break;
        }
        case arrow::acero::JoinType::INNER: {
            expected_rows = match_rows(rows, skip_left, skip_right);
            checkout_left = true;
            checkout_right = true;
            break;
        }
        case arrow::acero::JoinType::RIGHT_OUTER: {
            expected_rows = rows - rows / skip_right;
            checkout_left = true;
            checkout_right = true;
            break;
        }
        case arrow::acero::JoinType::FULL_OUTER: {
            expected_rows = rows - rows / LCM(skip_left, skip_right);
            checkout_left = true;
            checkout_right = true;
            break;
        }
        case arrow::acero::JoinType::RIGHT_SEMI: {
            expected_rows = match_rows(rows, skip_left, skip_right);
            checkout_right = true;
            break;
        }
        case arrow::acero::JoinType::RIGHT_ANTI: {
            expected_rows = rows - rows / skip_right - match_rows(rows, skip_left, skip_right);
            checkout_right = true;
            break;
        }
    }
    assert(final_table->num_rows() == expected_rows);
    std::cout << "after join large binary, table row: "  << final_table->num_rows() << std::endl;
    if(checkout_left) {
        std::cout << "left: ------------------------>" << std::endl;
        std::shared_ptr<arrow::ChunkedArray> col_left = final_table->GetColumnByName("left_0");
        for(int i = 0 ; i < expected_rows; ++i){
            std::cout << i << " row : " << (col_left->GetScalar(i).ValueOrDie()->is_valid ? col_left->GetScalar(i).ValueOrDie()->ToString(): "null") << std::endl;
        }
    }
    if(checkout_right) {
        std::cout << "right: ------------------------>" << std::endl;
        std::shared_ptr<arrow::ChunkedArray> col_right = final_table->GetColumnByName("right_0");
        for(int i = 0 ; i < expected_rows; ++i){
            std::cout << i << " row : " << (col_right->GetScalar(i).ValueOrDie()->is_valid ? col_right->GetScalar(i).ValueOrDie()->ToString(): "null") << std::endl;
        }
    }

    //std::cout << "Results : " << final_table->ToString() << std::endl;
    return 0;
}

TEST(test_arrow_vector_execute, VariousJoin) {
    run_test_arrow_run_acero_async2(11, 3, 5, arrow::acero::JoinType::LEFT_SEMI);
    run_test_arrow_run_acero_async2(11, 3, 5, arrow::acero::JoinType::LEFT_ANTI);
    run_test_arrow_run_acero_async2(11, 3, 5, arrow::acero::JoinType::LEFT_OUTER);
    run_test_arrow_run_acero_async2(11, 3, 5, arrow::acero::JoinType::INNER);
    run_test_arrow_run_acero_async2(11, 3, 5, arrow::acero::JoinType::RIGHT_OUTER);
    run_test_arrow_run_acero_async2(11, 3, 5, arrow::acero::JoinType::FULL_OUTER);
    run_test_arrow_run_acero_async2(11, 3, 5, arrow::acero::JoinType::RIGHT_SEMI);
    run_test_arrow_run_acero_async2(11, 3, 5, arrow::acero::JoinType::RIGHT_ANTI);
}

}