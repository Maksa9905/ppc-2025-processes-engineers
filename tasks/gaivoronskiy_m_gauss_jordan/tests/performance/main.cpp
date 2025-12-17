#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "gaivoronskiy_m_gauss_jordan/common/include/common.hpp"
#include "gaivoronskiy_m_gauss_jordan/mpi/include/ops_mpi.hpp"
#include "gaivoronskiy_m_gauss_jordan/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace gaivoronskiy_m_gauss_jordan {

class GaivoronskiyMRunPerfTestsProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kMatrixSize_ = 50;
  InType input_data_;

  void SetUp() override {
    int n = kMatrixSize_;
    int m = kMatrixSize_;
    input_data_ = InType(static_cast<size_t>(n), std::vector<double>(static_cast<size_t>(m + 1)));

    for (int i = 0; i < n; i++) {
      double sum = 0;
      for (int j = 0; j < m; j++) {
        if (i == j) {
          input_data_[i][j] = 10.0 + ((i % 100) / 10.0);
        } else {
          input_data_[i][j] = (i + j) % 10 / 10.0;
        }
        sum += std::abs(input_data_[i][j]);
      }
      input_data_[i][m] = sum;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return !output_data.empty();
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(GaivoronskiyMRunPerfTestsProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, GaivoronskiyMGaussJordanMPI, GaivoronskiyMGaussJordanSEQ>(
        PPC_SETTINGS_gaivoronskiy_m_gauss_jordan);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = GaivoronskiyMRunPerfTestsProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, GaivoronskiyMRunPerfTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace gaivoronskiy_m_gauss_jordan
