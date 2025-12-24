#pragma once

#include <cstddef>
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

  static size_t FindLowestPoint(const std::vector<Point> &pts);
  static size_t RemoveCollinearPoints(std::vector<Point> &pts, const Point &p0);
  static std::vector<Point> BuildConvexHull(const std::vector<Point> &pts, size_t num_points);
};

}  // namespace gaivoronskiy_m_grachem_method
