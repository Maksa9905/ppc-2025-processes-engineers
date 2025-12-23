#pragma once

#include <vector>

#include "gaivoronskiy_m_grachem_method/common/include/common.hpp"
#include "task/include/task.hpp"

namespace gaivoronskiy_m_grachem_method {

class GaivoronskiyMGrahamScanSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit GaivoronskiyMGrahamScanSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<Point> points_;
  std::vector<Point> hull_;
};

}  // namespace gaivoronskiy_m_grachem_method
