#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "gaivoronskiy_m_average_vector_sum/common/include/common.hpp"
#include "gaivoronskiy_m_average_vector_sum/mpi/include/ops_mpi.hpp"
#include "gaivoronskiy_m_average_vector_sum/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace gaivoronskiy_m_average_vector_sum {

namespace {

TestType MakeCase(std::vector<double> values, std::string name) {
  return TestType{std::move(values), std::move(name)};
}

TestType MakeArithmeticCase(std::size_t size, double start, double step, std::string name) {
  InType values(size);
  for (std::size_t i = 0; i < size; ++i) {
    values[i] = start + step * static_cast<double>(i);
  }
  return TestType{std::move(values), std::move(name)};
}

}  // namespace

class AverageVectorSumFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return test_param.name;
  }

 protected:
  void SetUp() override {
    const auto &params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = params.values;
    expected_average_ = CalculateAverage(input_data_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const double kEps = 1e-9;
    return std::fabs(output_data - expected_average_) <= kEps;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  static double CalculateAverage(const InType &values) {
    if (values.empty()) {
      return 0.0;
    }
    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    return sum / static_cast<double>(values.size());
  }

  InType input_data_;
  OutType expected_average_ = 0.0;
};

namespace {

const std::array<TestType, 5> kTestCases = {
    MakeCase({1.0, 2.0, 3.0, 4.0}, "small_positive"), MakeCase({-5.0, 0.0, 5.0, 10.0, -10.0}, "mixed_values"),
    MakeCase({42.5}, "single_element"), MakeArithmeticCase(128, -32.0, 0.25, "arithmetic_progression"),
    MakeArithmeticCase(1003, 1.0, 1.0, "long_progression")};

TEST_P(AverageVectorSumFuncTests, ComputesAverageCorrectly) {
  ExecuteTest(GetParam());
}

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<GaivoronskiyMAverageVecSumMPI, InType>(
                                               kTestCases, PPC_SETTINGS_gaivoronskiy_m_average_vector_sum),
                                           ppc::util::AddFuncTask<GaivoronskiyMAverageVecSumSEQ, InType>(
                                               kTestCases, PPC_SETTINGS_gaivoronskiy_m_average_vector_sum));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = AverageVectorSumFuncTests::PrintFuncTestName<AverageVectorSumFuncTests>;

INSTANTIATE_TEST_SUITE_P(AverageCases, AverageVectorSumFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace gaivoronskiy_m_average_vector_sum
