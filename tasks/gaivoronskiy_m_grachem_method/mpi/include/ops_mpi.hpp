#pragma once

#include "gaivoronskiy_m_grachem_method/common/include/common.hpp"
#include "task/include/task.hpp"

namespace gaivoronskiy_m_grachem_method {

class GaivoronskiyMGrahamScanMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit GaivoronskiyMGrahamScanMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<Point> points_;
  std::vector<Point> local_points_;
  std::vector<Point> hull_;

  static std::vector<Point> grahamScan(const std::vector<Point> &points);
  static std::vector<Point> mergeHulls(const std::vector<Point> &hull1, const std::vector<Point> &hull2);
};

}  // namespace gaivoronskiy_m_grachem_method
