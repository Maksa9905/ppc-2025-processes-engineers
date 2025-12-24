#include "gaivoronskiy_m_grachem_method/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stack>
#include <vector>

#include "gaivoronskiy_m_grachem_method/common/include/common.hpp"

namespace gaivoronskiy_m_grachem_method {

namespace {
int Orientation(const Point &p, const Point &q, const Point &r) {
  double val = ((q.y - p.y) * (r.x - q.x)) - ((q.x - p.x) * (r.y - q.y));
  constexpr double kEps = 1e-9;
  if (std::abs(val) < kEps) {
    return 0;
  }
  return (val > 0) ? 1 : 2;
}

double DistSquare(const Point &p1, const Point &p2) {
  return ((p1.x - p2.x) * (p1.x - p2.x)) + ((p1.y - p2.y) * (p1.y - p2.y));
}

bool Compare(const Point &p1, const Point &p2, const Point &p0) {
  int o = Orientation(p0, p1, p2);
  if (o == 0) {
    return DistSquare(p0, p1) < DistSquare(p0, p2);
  }
  return (o == 2);
}
}  // namespace

GaivoronskiyMGrahamScanSEQ::GaivoronskiyMGrahamScanSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool GaivoronskiyMGrahamScanSEQ::ValidationImpl() {
  return GetInput().size() >= 3;
}

bool GaivoronskiyMGrahamScanSEQ::PreProcessingImpl() {
  points_ = GetInput();
  hull_.clear();
  return !points_.empty();
}

size_t GaivoronskiyMGrahamScanSEQ::FindLowestPoint(const std::vector<Point> &pts) {
  size_t min_idx = 0;
  for (size_t i = 1; i < pts.size(); i++) {
    if (pts[i].y < pts[min_idx].y || (pts[i].y == pts[min_idx].y && pts[i].x < pts[min_idx].x)) {
      min_idx = i;
    }
  }
  return min_idx;
}

size_t GaivoronskiyMGrahamScanSEQ::RemoveCollinearPoints(std::vector<Point> &pts, const Point &p0) {
  size_t m = 1;
  for (size_t i = 1; i < pts.size(); i++) {
    while (i < pts.size() - 1 && Orientation(p0, pts[i], pts[i + 1]) == 0) {
      i++;
    }
    pts[m] = pts[i];
    m++;
  }
  return m;
}

std::vector<Point> GaivoronskiyMGrahamScanSEQ::BuildConvexHull(const std::vector<Point> &pts, size_t num_points) {
  std::stack<Point> s;
  s.push(pts[0]);
  s.push(pts[1]);
  s.push(pts[2]);

  for (size_t i = 3; i < num_points; i++) {
    Point top = s.top();
    s.pop();
    while (!s.empty() && Orientation(s.top(), top, pts[i]) != 2) {
      top = s.top();
      s.pop();
    }
    s.push(top);
    s.push(pts[i]);
  }

  std::vector<Point> result;
  while (!s.empty()) {
    result.push_back(s.top());
    s.pop();
  }

  std::ranges::reverse(result);
  return result;
}

bool GaivoronskiyMGrahamScanSEQ::RunImpl() {
  if (points_.size() < 3) {
    return false;
  }

  size_t min_idx = FindLowestPoint(points_);
  std::swap(points_[0], points_[min_idx]);
  const Point p0 = points_[0];

  std::sort(points_.begin() + 1, points_.end(),
            [&p0](const Point &p1, const Point &p2) { return Compare(p1, p2, p0); });

  size_t m = RemoveCollinearPoints(points_, p0);

  if (m < 3) {
    return false;
  }

  hull_ = BuildConvexHull(points_, m);
  GetOutput() = hull_;

  return true;
}

bool GaivoronskiyMGrahamScanSEQ::PostProcessingImpl() {
  return GetOutput().size() >= 3;
}

}  // namespace gaivoronskiy_m_grachem_method
