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

bool GaivoronskiyMGrahamScanSEQ::RunImpl() {
  if (points_.size() < 3) {
    return false;
  }

  size_t min_idx = 0;
  for (size_t i = 1; i < points_.size(); i++) {
    if (points_[i].y < points_[min_idx].y ||
        (points_[i].y == points_[min_idx].y && points_[i].x < points_[min_idx].x)) {
      min_idx = i;
    }
  }

  std::swap(points_[0], points_[min_idx]);
  const Point p0 = points_[0];

  std::sort(points_.begin() + 1, points_.end(),
            [&p0](const Point &p1, const Point &p2) { return Compare(p1, p2, p0); });

  size_t m = 1;
  for (size_t i = 1; i < points_.size(); i++) {
    while (i < points_.size() - 1 && Orientation(p0, points_[i], points_[i + 1]) == 0) {
      i++;
    }
    points_[m] = points_[i];
    m++;
  }

  if (m < 3) {
    return false;
  }

  std::stack<Point> s;
  s.push(points_[0]);
  s.push(points_[1]);
  s.push(points_[2]);

  for (size_t i = 3; i < m; i++) {
    Point top = s.top();
    s.pop();
    while (!s.empty() && Orientation(s.top(), top, points_[i]) != 2) {
      top = s.top();
      s.pop();
    }
    s.push(top);
    s.push(points_[i]);
  }

  hull_.clear();
  while (!s.empty()) {
    hull_.push_back(s.top());
    s.pop();
  }

  std::ranges::reverse(hull_);
  GetOutput() = hull_;

  return true;
}

bool GaivoronskiyMGrahamScanSEQ::PostProcessingImpl() {
  return GetOutput().size() >= 3;
}

}  // namespace gaivoronskiy_m_grachem_method
