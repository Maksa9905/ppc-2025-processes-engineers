#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <tuple>
#include <vector>

#include "gaivoronskiy_m_gauss_jordan/common/include/common.hpp"
#include "gaivoronskiy_m_gauss_jordan/mpi/include/ops_mpi.hpp"
#include "gaivoronskiy_m_gauss_jordan/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace gaivoronskiy_m_gauss_jordan {

class GaivoronskiyMRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<0>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<1>(params);
    res_ = std::get<2>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (res_.size() != output_data.size()) {
      return false;
    }
    for (size_t i = 0; i < res_.size(); i++) {
      if (std::abs(res_[i] - output_data[i]) > 1e-6) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType res_;
};

namespace {

TEST_P(GaivoronskiyMRunFuncTestsProcesses, GaussJordan) {
  ExecuteTest(GetParam());
}

// Тестовые системы уравнений
// Пример 1: 2x + y - z = 8, -3x - y + 2z = -11, -2x + y + 2z = -3
// Решение: x = 2, y = 3, z = -1
const std::array<TestType, 3> kTestParam = {
    std::make_tuple("test1", std::vector<std::vector<double>>{{2, 1, -1, 8}, {-3, -1, 2, -11}, {-2, 1, 2, -3}},
                    std::vector<double>{2.0, 3.0, -1.0}),
    // Пример 2: x + y = 5, 2x - y = 1
    // Решение: x = 2, y = 3
    std::make_tuple("test2", std::vector<std::vector<double>>{{1, 1, 5}, {2, -1, 1}}, std::vector<double>{2.0, 3.0}),
    // Пример 3: x + 2y = 7, 3x - y = 1
    // Решение: x = 9/7, y = 20/7
    std::make_tuple("test3", std::vector<std::vector<double>>{{1, 2, 7}, {3, -1, 1}},
                    std::vector<double>{9.0 / 7.0, 20.0 / 7.0})};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<GaivoronskiyMGaussJordanMPI, InType>(kTestParam, PPC_SETTINGS_gaivoronskiy_m_gauss_jordan),
    ppc::util::AddFuncTask<GaivoronskiyMGaussJordanSEQ, InType>(kTestParam, PPC_SETTINGS_gaivoronskiy_m_gauss_jordan));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = GaivoronskiyMRunFuncTestsProcesses::PrintFuncTestName<GaivoronskiyMRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(GaussJordanTests, GaivoronskiyMRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace gaivoronskiy_m_gauss_jordan
