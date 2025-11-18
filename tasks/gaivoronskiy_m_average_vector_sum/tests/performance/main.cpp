#include <gtest/gtest.h>

#include "gaivoronskiy_m_average_vector_sum/common/include/common.hpp"
#include "gaivoronskiy_m_average_vector_sum/mpi/include/ops_mpi.hpp"
#include "gaivoronskiy_m_average_vector_sum/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace gaivoronskiy_m_average_vector_sum {

class ExampleRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 100;
  InType input_data_{};

  void SetUp() override {
    input_data_ = kCount_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return input_data_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(ExampleRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, GaivoronskiyMAverageVecSumMPI, GaivoronskiyMAverageVecSumSEQ>(PPC_SETTINGS_gaivoronskiy_m_average_vector_sum);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ExampleRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ExampleRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace gaivoronskiy_m_average_vector_sum
