#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <random>

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

#include "gaivoronskiy_m_grachem_method/common/include/common.hpp"
#include "gaivoronskiy_m_grachem_method/mpi/include/ops_mpi.hpp"
#include "gaivoronskiy_m_grachem_method/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace gaivoronskiy_m_grachem_method {

class GaivoronskiyMGrahamScanRunPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kNumPoints_ = 10000;
  InType input_data_;

  void SetUp() override {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1000.0, 1000.0);

    input_data_.clear();
    input_data_.reserve(kNumPoints_);

    for (int i = 0; i < kNumPoints_; i++) {
      input_data_.emplace_back(dis(gen), dis(gen));
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() < 3) {
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
};

TEST_P(GaivoronskiyMGrahamScanRunPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, GaivoronskiyMGrahamScanMPI, GaivoronskiyMGrahamScanSEQ>(
    PPC_SETTINGS_gaivoronskiy_m_grachem_method);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = GaivoronskiyMGrahamScanRunPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, GaivoronskiyMGrahamScanRunPerfTests, kGtestValues, kPerfTestName);

}  // namespace gaivoronskiy_m_grachem_method
