#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <numeric>
#include <vector>

#include "gaivoronskiy_m_average_vector_sum/common/include/common.hpp"
#include "gaivoronskiy_m_average_vector_sum/mpi/include/ops_mpi.hpp"
#include "gaivoronskiy_m_average_vector_sum/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace gaivoronskiy_m_average_vector_sum {

class GaivoronskiyRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    const std::size_t data_size = ResolveInputSize();
    input_data_.resize(data_size);
    for (std::size_t i = 0; i < data_size; ++i) {
      input_data_[i] = static_cast<double>((i % 101) - 50);
    }
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
  static std::size_t ResolveInputSize() {
    constexpr std::size_t kDefaultSize = 1'000'000;
    const char *env_value = std::getenv("PPC_AVG_VEC_SIZE");
    if (env_value == nullptr) {
      return kDefaultSize;
    }
    char *end_ptr = nullptr;
    const unsigned long long parsed = std::strtoull(env_value, &end_ptr, 10);
    if (end_ptr == env_value || parsed == 0) {
      return kDefaultSize;
    }
    const auto size = static_cast<std::size_t>(parsed);
    return size == 0 ? kDefaultSize : size;
  }

  static double CalculateAverage(const InType &values) {
    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    return sum / static_cast<double>(values.size());
  }

  InType input_data_;
  OutType expected_average_ = 0.0;
};

TEST_P(GaivoronskiyRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, GaivoronskiyMAverageVecSumMPI, GaivoronskiyMAverageVecSumSEQ>(
        PPC_SETTINGS_gaivoronskiy_m_average_vector_sum);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = GaivoronskiyRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, GaivoronskiyRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace gaivoronskiy_m_average_vector_sum
