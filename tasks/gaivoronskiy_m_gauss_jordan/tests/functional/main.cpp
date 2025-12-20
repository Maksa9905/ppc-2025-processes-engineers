#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "gaivoronskiy_m_gauss_jordan/common/include/common.hpp"
#include "gaivoronskiy_m_gauss_jordan/mpi/include/ops_mpi.hpp"
#include "gaivoronskiy_m_gauss_jordan/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

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
const std::array<TestType, 17> kTestParam = {
    std::make_tuple("test1", std::vector<std::vector<double>>{{2, 1, -1, 8}, {-3, -1, 2, -11}, {-2, 1, 2, -3}},
                    std::vector<double>{2.0, 3.0, -1.0}),
    // Пример 2: x + y = 5, 2x - y = 1
    // Решение: x = 2, y = 3
    std::make_tuple("test2", std::vector<std::vector<double>>{{1, 1, 5}, {2, -1, 1}}, std::vector<double>{2.0, 3.0}),
    // Пример 3: x + 2y = 7, 3x - y = 1
    // Решение: x = 9/7, y = 20/7
    std::make_tuple("test3", std::vector<std::vector<double>>{{1, 2, 7}, {3, -1, 1}},
                    std::vector<double>{9.0 / 7.0, 20.0 / 7.0}),
    // Пример 4: Единичная матрица, x = 1, y = 2, z = 3
    std::make_tuple("test_identity", std::vector<std::vector<double>>{{1, 0, 0, 1}, {0, 1, 0, 2}, {0, 0, 1, 3}},
                    std::vector<double>{1.0, 2.0, 3.0}),
    // Пример 5: Диагональная матрица, 2x = 4, 3y = 9, 4z = 8
    // Решение: x = 2, y = 3, z = 2
    std::make_tuple("test_diagonal", std::vector<std::vector<double>>{{2, 0, 0, 4}, {0, 3, 0, 9}, {0, 0, 4, 8}},
                    std::vector<double>{2.0, 3.0, 2.0}),
    // Пример 6: Система с перестановкой строк (нуль на первом месте)
    // 0x + 2y = 4, x + y = 3
    // Решение: x = 1, y = 2
    std::make_tuple("test_row_swap", std::vector<std::vector<double>>{{0, 2, 4}, {1, 1, 3}},
                    std::vector<double>{1.0, 2.0}),
    // Пример 7: Большая система 4x4 (простая диагональная)
    // x = 1, y = 2, z = 3, w = 4
    std::make_tuple(
        "test_4x4",
        std::vector<std::vector<double>>{{1, 0, 0, 0, 1}, {0, 1, 0, 0, 2}, {0, 0, 1, 0, 3}, {0, 0, 0, 1, 4}},
        std::vector<double>{1.0, 2.0, 3.0, 4.0}),
    // Пример 8: Простая система 2x2 с отрицательными коэффициентами
    // -x + y = 1, x + y = 3
    // Решение: x = 1, y = 2
    std::make_tuple("test_negative_coeffs", std::vector<std::vector<double>>{{-1, 1, 1}, {1, 1, 3}},
                    std::vector<double>{1.0, 2.0}),
    // Пример 9: Система с дробными коэффициентами
    // 0.5x + 0.25y = 1.5, 0.75x - 0.5y = 0.5
    // Решение: x = 2, y = 2
    std::make_tuple("test_fractional", std::vector<std::vector<double>>{{0.5, 0.25, 1.5}, {0.75, -0.5, 0.5}},
                    std::vector<double>{2.0, 2.0}),
    // Пример 10: Система с большими числами
    // 100x + 50y = 350, 200x - 100y = 100
    // Решение: x = 2, y = 3
    std::make_tuple("test_large_numbers", std::vector<std::vector<double>>{{100, 50, 350}, {200, -100, 100}},
                    std::vector<double>{2.0, 3.0}),
    // Пример 11: Система 3x3 с нулями
    // x + 0y + z = 4, 0x + y + z = 5, x + y + 0z = 3
    // Решение: x = 1, y = 2, z = 3
    std::make_tuple("test_with_zeros", std::vector<std::vector<double>>{{1, 0, 1, 4}, {0, 1, 1, 5}, {1, 1, 0, 3}},
                    std::vector<double>{1.0, 2.0, 3.0}),
    // Пример 12: Симметричная система
    // 2x + y = 5, x + 2y = 4
    // Решение: x = 2, y = 1
    std::make_tuple("test_symmetric", std::vector<std::vector<double>>{{2, 1, 5}, {1, 2, 4}},
                    std::vector<double>{2.0, 1.0}),
    // Пример 13: Система с одним уравнением
    // 2x = 6
    // Решение: x = 3
    std::make_tuple("test_single_equation", std::vector<std::vector<double>>{{2, 6}}, std::vector<double>{3.0}),
    // Пример 14: Система 5x5
    // Решение: x1 = 1, x2 = 2, x3 = 3, x4 = 4, x5 = 5
    std::make_tuple(
        "test_5x5",
        std::vector<std::vector<double>>{
            {1, 0, 0, 0, 0, 1}, {0, 1, 0, 0, 0, 2}, {0, 0, 1, 0, 0, 3}, {0, 0, 0, 1, 0, 4}, {0, 0, 0, 0, 1, 5}},
        std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0}),
    // Пример 15: Система с маленькими коэффициентами
    // 0.001x + 0.002y = 0.005, 0.003x - 0.001y = 0.001
    // Решение: x = 1, y = 2
    std::make_tuple("test_small_coeffs",
                    std::vector<std::vector<double>>{{0.001, 0.002, 0.005}, {0.003, -0.001, 0.001}},
                    std::vector<double>{1.0, 2.0}),
    // Пример 16: Система с перестановкой (все нули в первом столбце кроме последней строки)
    // 0x + y + 0z = 2, 0x + 0y + z = 3, x + 0y + 0z = 1
    // Решение: x = 1, y = 2, z = 3
    std::make_tuple("test_column_zeros", std::vector<std::vector<double>>{{0, 1, 0, 2}, {0, 0, 1, 3}, {1, 0, 0, 1}},
                    std::vector<double>{1.0, 2.0, 3.0}),
    // Пример 17: Треугольная система (верхнетреугольная)
    // x + y + z = 6, y + z = 5, z = 3
    // Решение: x = 1, y = 2, z = 3
    std::make_tuple("test_triangular", std::vector<std::vector<double>>{{1, 1, 1, 6}, {0, 1, 1, 5}, {0, 0, 1, 3}},
                    std::vector<double>{1.0, 2.0, 3.0})};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<GaivoronskiyMGaussJordanMPI, InType>(kTestParam, PPC_SETTINGS_gaivoronskiy_m_gauss_jordan),
    ppc::util::AddFuncTask<GaivoronskiyMGaussJordanSEQ, InType>(kTestParam, PPC_SETTINGS_gaivoronskiy_m_gauss_jordan));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = GaivoronskiyMRunFuncTestsProcesses::PrintFuncTestName<GaivoronskiyMRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(GaussJordanTests, GaivoronskiyMRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace gaivoronskiy_m_gauss_jordan
