#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <tuple>
#include <vector>

#include "gaivoronskiy_m_grachem_method/common/include/common.hpp"
#include "gaivoronskiy_m_grachem_method/mpi/include/ops_mpi.hpp"
#include "gaivoronskiy_m_grachem_method/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace gaivoronskiy_m_grachem_method {

class GaivoronskiyMGrahamScanRunFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param).size()) + "_points_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
    expected_hull_size_ = std::stoi(std::get<1>(params));
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != static_cast<size_t>(expected_hull_size_)) {
      return false;
    }

    for (size_t i = 0; i < output_data.size(); i++) {
      for (size_t j = i + 1; j < output_data.size(); j++) {
        if (output_data[i] == output_data[j]) {
          return false;
        }
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  int expected_hull_size_;
};

namespace {

// Тест 1: Простой треугольник
std::vector<Point> GetTrianglePoints() {
  return {Point(0.0, 0.0), Point(1.0, 0.0), Point(0.5, 1.0)};
}

// Тест 2: Квадрат
std::vector<Point> GetSquarePoints() {
  return {Point(0.0, 0.0), Point(1.0, 0.0), Point(1.0, 1.0), Point(0.0, 1.0)};
}

// Тест 3: Квадрат с внутренними точками
std::vector<Point> GetSquareWithInnerPoints() {
  return {Point(0.0, 0.0), Point(1.0, 0.0), Point(1.0, 1.0), Point(0.0, 1.0),
          Point(0.5, 0.5), Point(0.3, 0.3), Point(0.7, 0.7)};
}

// Тест 4: Окружность (8 точек)
std::vector<Point> GetCirclePoints() {
  std::vector<Point> points;
  const int n = 8;
  const double radius = 1.0;
  for (int i = 0; i < n; i++) {
    double angle = 2.0 * M_PI * i / n;
    points.emplace_back(radius * std::cos(angle), radius * std::sin(angle));
  }
  return points;
}

// Тест 5: Множество точек с несколькими на выпуклой оболочке
std::vector<Point> GetComplexPoints() {
  return {Point(0.0, 0.0), Point(3.0, 0.0),  Point(3.0, 3.0), Point(0.0, 3.0), Point(1.0, 1.0), Point(2.0, 1.0),
          Point(1.0, 2.0), Point(2.0, 2.0),  Point(1.5, 1.5), Point(0.5, 0.5), Point(2.5, 0.5), Point(2.5, 2.5),
          Point(0.5, 2.5), Point(1.5, 0.5),  Point(1.5, 2.5), Point(0.5, 1.5), Point(2.5, 1.5), Point(-1.0, 1.5),
          Point(4.0, 1.5), Point(1.5, -1.0), Point(1.5, 4.0)};
}

// Тест 6: Большой набор случайных точек в круге
std::vector<Point> GetRandomPointsInCircle() {
  std::vector<Point> points;
  const int n = 50;
  const double radius = 10.0;

  for (int i = 0; i < 10; i++) {
    double angle = 2.0 * M_PI * i / 10;
    points.emplace_back(radius * std::cos(angle), radius * std::sin(angle));
  }

  for (int i = 0; i < 40; i++) {
    double angle = 2.0 * M_PI * i / 40;
    double r = radius * 0.7 * (1 + (i % 3) * 0.1);
    points.emplace_back(r * std::cos(angle), r * std::sin(angle));
  }

  return points;
}

TEST_P(GaivoronskiyMGrahamScanRunFuncTests, GrahamScanTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 6> kTestParam = {
    std::make_tuple(GetTrianglePoints(), "3"),        std::make_tuple(GetSquarePoints(), "4"),
    std::make_tuple(GetSquareWithInnerPoints(), "4"), std::make_tuple(GetCirclePoints(), "8"),
    std::make_tuple(GetComplexPoints(), "8"),         std::make_tuple(GetRandomPointsInCircle(), "10")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<GaivoronskiyMGrahamScanMPI, InType>(kTestParam, PPC_SETTINGS_gaivoronskiy_m_grachem_method),
    ppc::util::AddFuncTask<GaivoronskiyMGrahamScanSEQ, InType>(kTestParam, PPC_SETTINGS_gaivoronskiy_m_grachem_method));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = GaivoronskiyMGrahamScanRunFuncTests::PrintFuncTestName<GaivoronskiyMGrahamScanRunFuncTests>;

INSTANTIATE_TEST_SUITE_P(GrahamScanTests, GaivoronskiyMGrahamScanRunFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace gaivoronskiy_m_grachem_method
