
#include <gtest/gtest.h>

#include <algorithm>
#include <deque>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include "arrow/api.h"
#include "arrow/scalar.h"
#include "arrow/result.h"
#include "arrow/util/type_fwd.h"
#include "parquet/arrow/writer.h"

#include "arrow/io/api.h"
#include "arrow/io/file.h"
#include "parquet/arrow/reader.h"
#include "parquet/file_reader.h"
#include "parquet/metadata.h"
#include "parquet/page_index.h"
#include "parquet/schema.h"
#include "parquet/thrift_internal.h"

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

namespace csvhelper {
enum Type {
  INT,
  FLOAT,
  DOUBLE,
  STRING,
  BOOL,
};

using RangePair = std::pair<int64_t, int64_t>;

// 按照逗号分割字符串
std::vector<std::string> split(const std::string& str, char delimiter) {
  std::vector<std::string> tokens;
  std::stringstream ss(str);
  std::string token;

  while (std::getline(ss, token, delimiter)) {
    tokens.push_back(token);
  }

  return tokens;
}

template <typename TYPE>
TYPE ParseAs(const std::string& str_val);

template <>
int64_t ParseAs(const std::string& str_val) {
  return std::stoi(str_val);
}

template <>
float ParseAs(const std::string& str_val) {
  return std::stof(str_val);
}

template <>
double ParseAs(const std::string& str_val) {
  return std::stod(str_val);
}

template <>
bool ParseAs(const std::string& str_val) {
  return str_val == "True";
}

class ColumnBase {
 public:
  virtual std::string to_string(int row) const = 0;
  virtual void append_null() = 0;
  virtual bool operator==(const ColumnBase& rhs) const = 0;
  virtual bool empty() const = 0;
};

template <typename T>
class Column : public ColumnBase {
 public:
  std::vector<T>& getValues() { return values_; }

  bool empty() const override {return values_.empty();}

  void append(const T& t) {
    values_.emplace_back(t);
    is_null_.emplace_back(false);
  }

  void append_null() override {
    is_null_.emplace_back(true);
    values_.emplace_back(T());
  }

  [[nodiscard]] std::string to_string(int row) const override {
    std::ostringstream oss;
    if (is_null_[row]) {
      oss << "null";
    } else {
      oss << values_[row];
    };
    return oss.str();
  }

  bool operator==(const ColumnBase& rhs) const override {
    try {
      const auto& rhs_ = dynamic_cast<const Column<T>&>(rhs);
      if (values_.size() != rhs_.values_.size()) {
        return false;
      }
      for (int i = 0; i < values_.size(); ++i) {
        if (is_null_[i] != rhs_.is_null_[i]) {
          return false;
        }
        if (!is_null_[i] && !rhs_.is_null_[i]) {
          if (values_[i] != rhs_.values_[i]) {
            return false;
          }
        } 
      }
      return true;
    } catch (const std::bad_cast& e) {
      return false;
    }
}

private:
  std::vector<T> values_;
  std::vector<bool> is_null_;
};

template <>
bool Column<double>::operator==(const ColumnBase& rhs) const {
  try {
      const auto& rhs_ = dynamic_cast<const Column<double>&>(rhs);
      if (values_.size() != rhs_.values_.size()) {
        return false;
      }
      for (int i = 0; i < values_.size(); ++i) {
        if (is_null_[i] != rhs_.is_null_[i]) {
          return false;
        }
        if (!is_null_[i] && !rhs_.is_null_[i]) {
          if (values_[i] - rhs_.values_[i] > 1e-5
                || values_[i] - rhs_.values_[i] < -1e-5) {
            return false;
          }
        } 
      }
      return true;
    } catch (const std::bad_cast& e) {
      return false;
    }
}

template <>
bool Column<float>::operator==(const ColumnBase& rhs) const {
  try {
      const auto& rhs_ = dynamic_cast<const Column<float>&>(rhs);
      if (values_.size() != rhs_.values_.size()) {
        return false;
      }
      for (int i = 0; i < values_.size(); ++i) {
        if (is_null_[i] != rhs_.is_null_[i]) {
          return false;
        }
        if (!is_null_[i] && !rhs_.is_null_[i]) {
          if (values_[i] - rhs_.values_[i] > 1e-5
                || values_[i] - rhs_.values_[i] < -1e-5) {
            return false;
          }
        } 
      }
      return true;
    } catch (const std::bad_cast& e) {
      return false;
    }
}

class CSVReader {
 public:
  CSVReader(const std::string& file_name, const std::vector<Type>& schema)
      : file_name_(file_name),
        schema_(schema),
        delimiter_(','),
        rows_num_(0),
        use_row_range_(false),
        select_cols_(false) {
    for (const auto& it : schema) {
      switch (it) {
        case INT:
          cols_.emplace_back(std::move(std::make_shared<Column<int64_t>>()));
          break;
        case FLOAT:
          cols_.emplace_back(std::move(std::make_shared<Column<float>>()));
          break;
        case DOUBLE:
          cols_.emplace_back(std::move(std::make_shared<Column<double>>()));
          break;
        case STRING:
          cols_.emplace_back(std::move(std::make_shared<Column<std::string>>()));
          break;
        case BOOL:
          cols_.emplace_back(std::move(std::make_shared<Column<bool>>()));
          break;
      }
    }
  }

  void set_row_range(const std::vector<RangePair>& row_range) {
    row_range_.clear();
    std::transform(row_range.begin(), row_range.end(), std::back_inserter(row_range_),
                   [](const auto& pair) { return pair; });
    use_row_range_ = true;
  }
  void read_lines() {
    std::string line;
    int64_t row_id = 0;
    while (std::getline(file_, line)) {
      if (use_row_range_) {
        if (row_range_.empty()) {
          break;
        }
        if (row_id != row_range_.front().first) {
          ++row_id;
          continue;
        } else {
          if (row_range_.front().first == row_range_.front().second) {
            row_range_.pop_front();
          } else {
            ++row_range_.front().first;
          }
          ++row_id;
        }
      }
      auto values_as_str = split(line, delimiter_);

      for (int i = 0; i < (select_cols_ ? col_ids_.size() : schema_.size()); ++i) {
        int col_id = select_cols_ ? col_ids_[i] : i;
        if (values_as_str[col_id] == "null") {
          cols_[col_id]->append_null();
          continue;
        }
        switch (schema_[col_id]) {
          case INT:
            dynamic_cast<Column<int64_t>*>(cols_[col_id].get())
                ->append(ParseAs<int64_t>(values_as_str[col_id]));
            break;
          case FLOAT:
            dynamic_cast<Column<float>*>(cols_[col_id].get())
                ->append(ParseAs<float>(values_as_str[col_id]));
            break;
          case DOUBLE:
            dynamic_cast<Column<double>*>(cols_[col_id].get())
                ->append(ParseAs<double>(values_as_str[col_id]));
            break;
          case STRING:
            dynamic_cast<Column<std::string>*>(cols_[col_id].get())
                ->append(values_as_str[col_id]);
            break;
          case BOOL:
            dynamic_cast<Column<bool>*>(cols_[col_id].get())
                ->append(ParseAs<bool>(values_as_str[col_id]));
            break;
        }
      }
      ++rows_num_;
    }
  }

  int read() {
    read_header();
    read_lines();
    return 0;
  }

  int read(const std::vector<RangePair>& row_range, bool need_read_header) {
    set_row_range(row_range);
    if (need_read_header) {
      read_header();
    }
    read_lines();
    return 0;
  }

  void set_select_cols(const std::vector<int>& col_ids) {
    col_ids_ = col_ids;
    select_cols_ = true;
  }

  void reset_ptr() {
    file_.clear();  // 清除流的状态
    file_.seekg(0, std::ios::beg);  // 将文件指针移动到文件开头
  }

  void read_header() {
    std::string line;
    std::getline(file_, line);
    col_names_ = split(line, delimiter_);
  }

  int open_file() {
    file_.open(file_name_);
    if (!file_) {
      std::cerr << "Error opening file for reading" << std::endl;
      return -1;
    }
    return 0;
  }

  int close() {
    file_.close();
    return 0;
  }

  void show_header() {
    for (int i = 0; i < (select_cols_ ? col_ids_.size() : schema_.size()) - 1; ++i) {
      std::cout << col_names_[select_cols_ ? col_ids_[i] : i] << ",";
    }
    std::cout << col_names_[select_cols_ ? col_ids_.back() : (col_names_.size() - 1)]
              << std::endl;
  }

  void show_row(int row) {
    size_t loop_times = select_cols_ ? col_ids_.size() : schema_.size();
    for (int i = 0; i < loop_times - 1; ++i) {
      int col_id = select_cols_ ? col_ids_[i] : i;
      std::cout << cols_[col_id]->to_string(row) << ",";
    }
    std::cout << cols_[select_cols_ ? col_ids_.back() : (cols_.size() - 1)]->to_string(
                     row)
              << std::endl;
  }

  void show() {
    show_header();
    for (int i = 0; i < rows_num_; ++i) {
      show_row(i);
    }
  }

  void reset() {
    select_cols_ = false;
    use_row_range_ = false;
    row_range_.clear();
    col_ids_.clear();
    file_.clear();
    file_.seekg(0, std::ios::beg);
  }

  std::shared_ptr<ColumnBase> column(int i) {
    return cols_[select_cols_ ? col_ids_[i] : i];
  }

 private:
  std::vector<Type> schema_;
  std::string file_name_;
  std::vector<std::shared_ptr<ColumnBase>> cols_;
  std::vector<std::string> col_names_;
  std::ifstream file_;
  std::deque<RangePair> row_range_;
  std::vector<int> col_ids_;
  bool select_cols_;
  bool use_row_range_;
  char delimiter_;
  int rows_num_;
};
}  // namespace csvhelper

::arrow::Status ArrayToColumn(std::shared_ptr<arrow::Array> array,
                            std::shared_ptr<csvhelper::ColumnBase>* column_base, bool need_print = false) {
  switch (array->type_id()) {
    case arrow::Type::INT8: {
      *column_base = std::make_shared<csvhelper::Column<int64_t>>();
      auto int64col = dynamic_cast<csvhelper::Column<int64_t>*>(column_base->get());
      for (int row = 0; row < array->length(); ++row) {
        if (!array->IsValid(row)) {
          (*column_base)->append_null();
          continue;
        }
        std::shared_ptr<arrow::Scalar> scalar;
        scalar = array->GetScalar(row).ValueOrDie();
        int64_t value = std::static_pointer_cast<arrow::Int8Scalar>(scalar)->value;
        if (need_print) {
          std::cout << value << std::endl;
        }
        int64col->append(value);
      }
      break;
    }
    case arrow::Type::INT16: {
      *column_base = std::make_shared<csvhelper::Column<int64_t>>();
      auto int64col = dynamic_cast<csvhelper::Column<int64_t>*>(column_base->get());
      for (int row = 0; row < array->length(); ++row) {
        if (!array->IsValid(row)) {
          (*column_base)->append_null();
          continue;
        }
        std::shared_ptr<arrow::Scalar> scalar;
        scalar = array->GetScalar(row).ValueOrDie();
        int64_t value = std::static_pointer_cast<arrow::Int16Scalar>(scalar)->value;
        if (need_print) {
          std::cout << value << std::endl;
        }
        int64col->append(value);
      }
      break;
    }
    case arrow::Type::INT32: {
      *column_base = std::make_shared<csvhelper::Column<int64_t>>();
      auto int64col = dynamic_cast<csvhelper::Column<int64_t>*>(column_base->get());
      for (int row = 0; row < array->length(); ++row) {
        if (!array->IsValid(row)) {
          (*column_base)->append_null();
          continue;
        }
        std::shared_ptr<arrow::Scalar> scalar;
        scalar = array->GetScalar(row).ValueOrDie();
        int64_t value = std::static_pointer_cast<arrow::Int32Scalar>(scalar)->value;
        if (need_print) {
          std::cout << value << std::endl;
        }
        int64col->append(value);
      }
      break;
    }
    case arrow::Type::INT64: {
      *column_base = std::make_shared<csvhelper::Column<int64_t>>();
      auto int64col = dynamic_cast<csvhelper::Column<int64_t>*>(column_base->get());
      for (int row = 0; row < array->length(); ++row) {
        if (!array->IsValid(row)) {
          (*column_base)->append_null();
          continue;
        }
        std::shared_ptr<arrow::Scalar> scalar;
        scalar = array->GetScalar(row).ValueOrDie();
        int64_t value = std::static_pointer_cast<arrow::Int64Scalar>(scalar)->value;
        if (need_print) {
          std::cout << value << std::endl;
        }
        int64col->append(value);
      }
      break;
    }
    case arrow::Type::FLOAT: {
      *column_base = std::make_shared<csvhelper::Column<float>>();
      auto float_col = dynamic_cast<csvhelper::Column<float>*>(column_base->get());
      for (int row = 0; row < array->length(); ++row) {
        if (!array->IsValid(row)) {
          (*column_base)->append_null();
          continue;
        }
        std::shared_ptr<arrow::Scalar> scalar;
        scalar = array->GetScalar(row).ValueOrDie();
        auto value = std::static_pointer_cast<arrow::FloatScalar>(scalar)->value;
        if (need_print) {
          std::cout << value << std::endl;
        }
        float_col->append(value);
      }
      break;
    }
    case arrow::Type::DOUBLE: {
      *column_base = std::make_shared<csvhelper::Column<double>>();
      auto double_col = dynamic_cast<csvhelper::Column<double>*>(column_base->get());
      for (int row = 0; row < array->length(); ++row) {
        if (!array->IsValid(row)) {
          (*column_base)->append_null();
          continue;
        }
        std::shared_ptr<arrow::Scalar> scalar;
        scalar = array->GetScalar(row).ValueOrDie();
        auto value = std::static_pointer_cast<arrow::DoubleScalar>(scalar)->value;
        if (need_print) {
          std::cout << value << std::endl;
        }
        double_col->append(value);
      }
      break;
    }
    case arrow::Type::BINARY: {
      *column_base = std::make_shared<csvhelper::Column<std::string>>();
      auto str_col = dynamic_cast<csvhelper::Column<std::string>*>(column_base->get());
      for (int row = 0; row < array->length(); ++row) {
        if (!array->IsValid(row)) {
          (*column_base)->append_null();
          continue;
        }
        std::shared_ptr<arrow::Scalar> scalar;
        scalar = array->GetScalar(row).ValueOrDie();
        auto value = std::static_pointer_cast<arrow::BinaryScalar>(scalar)->ToString();
        if (need_print) {
          std::cout << value << std::endl;
        }
        str_col->append(value);
      }
      break;
    }
    case arrow::Type::LARGE_BINARY: {
      *column_base = std::make_shared<csvhelper::Column<std::string>>();
      auto str_col = dynamic_cast<csvhelper::Column<std::string>*>(column_base->get());
      for (int row = 0; row < array->length(); ++row) {
        if (!array->IsValid(row)) {
          (*column_base)->append_null();
          continue;
        }
        std::shared_ptr<arrow::Scalar> scalar;
        scalar = array->GetScalar(row).ValueOrDie();
        auto value = std::static_pointer_cast<arrow::LargeBinaryScalar>(scalar)->ToString();
        if (need_print) {
          std::cout << value << std::endl;
        }
        str_col->append(value);
      }
      break;
    }
    case arrow::Type::STRING: {
      *column_base = std::make_shared<csvhelper::Column<std::string>>();
      auto str_col = dynamic_cast<csvhelper::Column<std::string>*>(column_base->get());
      for (int row = 0; row < array->length(); ++row) {
        if (!array->IsValid(row)) {
          (*column_base)->append_null();
          continue;
        }
        std::shared_ptr<arrow::Scalar> scalar;
        scalar = array->GetScalar(row).ValueOrDie();
        auto value = std::static_pointer_cast<arrow::StringScalar>(scalar)->ToString();
        if (need_print) {
          std::cout << value << std::endl;
        }
        str_col->append(value);
      }
      break;
    }
    case arrow::Type::LARGE_STRING: {
      *column_base = std::make_shared<csvhelper::Column<std::string>>();
      auto str_col = dynamic_cast<csvhelper::Column<std::string>*>(column_base->get());
      for (int row = 0; row < array->length(); ++row) {
        if (!array->IsValid(row)) {
          (*column_base)->append_null();
          continue;
        }
        std::shared_ptr<arrow::Scalar> scalar;
        scalar = array->GetScalar(row).ValueOrDie();
        auto value = std::static_pointer_cast<arrow::LargeStringScalar>(scalar)->ToString();
        str_col->append(value);
      }
      break;
    }
    case arrow::Type::BOOL: {
      *column_base = std::make_shared<csvhelper::Column<bool>>();
      auto bool_col = dynamic_cast<csvhelper::Column<bool>*>(column_base->get());
      for (int row = 0; row < array->length(); ++row) {
        if (!array->IsValid(row)) {
          (*column_base)->append_null();
          continue;
        }
        std::shared_ptr<arrow::Scalar> scalar;
        scalar = array->GetScalar(row).ValueOrDie();
        auto value = std::static_pointer_cast<arrow::BooleanScalar>(scalar)->value;
        if (need_print) {
          std::cout << value << std::endl;
        }
        bool_col->append(value);
      }
      break;
    }
  }

  return ::arrow::Status::OK();
}

arrow::Status ReadTest(std::vector<int> cols, std::vector<std::vector<parquet::RowRangeItem>> row_ranges) {
  std::string parquet_file_name = "cpp/src/tests/data/alltypes_tiny_pages.parquet";
  std::string csv_file_name = "cpp/src/tests/data/alltypes_tiny_pages.csv";
  // #include "arrow/io/api.h"
  // #include "arrow/parquet/arrow/reader.h"
  std::vector<csvhelper::Type> schema = {
      csvhelper::INT, csvhelper::BOOL, csvhelper::INT, csvhelper::INT,
      csvhelper::INT, csvhelper::INT, csvhelper::FLOAT, csvhelper::DOUBLE,
      csvhelper::STRING, csvhelper::STRING, csvhelper::STRING,
      csvhelper::INT, csvhelper::INT
  };
  csvhelper::CSVReader csv_reader(csv_file_name, schema);
  if (csv_reader.open_file() != 0) {
    std::cerr << "open csv file failed" << std::endl;
    return arrow::Status::Invalid("");
  }
  csv_reader.set_select_cols(cols);
  for (const auto& row_range: row_ranges) {
    csv_reader.read(row_range, true);
    csv_reader.reset_ptr();
  }
  
  csv_reader.close();

  arrow::MemoryPool* pool = arrow::default_memory_pool();

  // Configure general Parquet reader settings
  auto reader_properties = parquet::ReaderProperties(pool);
  reader_properties.set_buffer_size(4096 * 4);
  reader_properties.enable_buffered_stream();

  // Configure Arrow-specific Parquet reader settings
  auto arrow_reader_props = parquet::ArrowReaderProperties();
  arrow_reader_props.set_batch_size(128 * 1024);  // default 64 * 1024

  parquet::arrow::FileReaderBuilder reader_builder;
  ARROW_RETURN_NOT_OK(
      reader_builder.OpenFile(parquet_file_name, /*memory_map=*/false, reader_properties));
  reader_builder.memory_pool(pool);
  reader_builder.properties(arrow_reader_props);

  std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
  ARROW_ASSIGN_OR_RAISE(arrow_reader, reader_builder.Build());

  std::shared_ptr<::arrow::RecordBatchReader> rb_reader;
  std::vector<int> row_groups(row_ranges.size(), 0);
  ARROW_RETURN_NOT_OK(arrow_reader->GetRecordBatchReader(row_groups, cols, row_ranges, &rb_reader));
  // ARROW_RETURN_NOT_OK(arrow_reader->GetRecordBatchReader({0}, {7, 8} ,&rb_reader));

  bool first_batch = true;
  std::vector<std::shared_ptr<csvhelper::ColumnBase>> output(cols.size());
  for (arrow::Result<std::shared_ptr<arrow::RecordBatch>> maybe_batch : *rb_reader) {
    // Operate on each batch...
    std::shared_ptr<arrow::RecordBatch> batch = maybe_batch.ValueOrDie();
    for (int i = 0 ; i < cols.size(); ++i) {
      ArrayToColumn(batch->column(i), &output[i]);
    }
  }

  if (row_ranges[0].empty()) {
    for (int i = 0; i < cols.size(); ++i) {
      if (output[i] != nullptr) {
        return arrow::Status::Invalid("");
      }
    }
    return arrow::Status::OK();
  }
  for (int i = 0; i < cols.size(); ++i) {
    if (!(*csv_reader.column(i) == *output[i])) {
      return arrow::Status::Invalid("");
    }
  }
  return arrow::Status::OK();
}

// 没有测试多row group的场景
// 从csv里读取数据和从parquet里读取数据作比较
// 读取不要超过一个batch的大小 64 * 1024
TEST(ReadParquetTest, ReadInRowGroup) {
  // 单列单行
  EXPECT_TRUE(ReadTest({0}, {{{1, 1}}}).ok());
  // 单列多行
  EXPECT_TRUE(ReadTest({0}, {{{2, 10}}}).ok());
  // 跨page
  EXPECT_TRUE(ReadTest({0}, {{{2, 40}}}).ok());
  // 多列单行
  EXPECT_TRUE(ReadTest({0, 1, 3}, {{{1, 1}}}).ok());
  // 多列多行
  EXPECT_TRUE(ReadTest({0, 1, 3}, {{{1, 3}}}).ok());
  // 多page
  // 第11列是时间戳，不好测试，跳过
  EXPECT_TRUE(ReadTest({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12}, {{{3, 10}, {41, 43}, {69, 200}, {3000, 4000}}}).ok());
}

TEST(ReadParquetTest, ReadNothing) {
  // 单列单行
  EXPECT_TRUE(ReadTest({0}, {{}}).ok());
}

TEST(ReadParquetTest, ReadExceedRange) {
  // 超出row group最大行数，最大7300
  EXPECT_TRUE(ReadTest({0}, {{{7297, 7299}}}).ok());
  EXPECT_TRUE(!ReadTest({0}, {{{7297, 7301}}}).ok());
}

// 只有一个row group 反复读这个row group
TEST(ReadParquetTest, RepeatReadRowGroup) {
  // 超出row group最大行数，最大7300
  EXPECT_TRUE(ReadTest({0}, {{{2, 10}}, {{4,7}, {9,100}}}).ok());
}
