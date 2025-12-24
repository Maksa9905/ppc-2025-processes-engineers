#pragma once

#include <cstddef>
#include <vector>

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

  static std::vector<Point> GrahamScan(const std::vector<Point> &points);
  static std::vector<Point> MergeHulls(const std::vector<Point> &hull1, const std::vector<Point> &hull2);
  static std::vector<double> PointsToFlat(const std::vector<Point> &points);
  static std::vector<Point> FlatToPoints(const std::vector<double> &flat_data, int num_points);
  static size_t FindLowestPoint(const std::vector<Point> &pts);
  static size_t RemoveCollinearPoints(std::vector<Point> &pts, const Point &p0);
  static std::vector<Point> BuildConvexHull(const std::vector<Point> &pts, size_t num_points);
  void GatherAndMergeHulls(const std::vector<Point> &local_hull, int rank, int size);
  void BroadcastResult(int rank);
};

}  // namespace gaivoronskiy_m_grachem_method
