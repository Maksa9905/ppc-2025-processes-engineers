#include "gaivoronskiy_m_grachem_method/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <stack>
#include <vector>

#include "gaivoronskiy_m_grachem_method/common/include/common.hpp"

namespace gaivoronskiy_m_grachem_method {

namespace {
int orientation(const Point &p, const Point &q, const Point &r) {
  double val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
  constexpr double eps = 1e-9;
  if (std::abs(val) < eps) {
    return 0;
  }
  return (val > 0) ? 1 : 2;
}

double distSquare(const Point &p1, const Point &p2) {
  return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y);
}

Point p0;

bool compare(const Point &p1, const Point &p2) {
  int o = orientation(p0, p1, p2);
  if (o == 0) {
    return distSquare(p0, p1) < distSquare(p0, p2);
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
  p0 = points_[0];

  std::sort(points_.begin() + 1, points_.end(), compare);

  size_t m = 1;
  for (size_t i = 1; i < points_.size(); i++) {
    while (i < points_.size() - 1 && orientation(p0, points_[i], points_[i + 1]) == 0) {
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
    while (!s.empty() && orientation(s.top(), top, points_[i]) != 2) {
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

  std::reverse(hull_.begin(), hull_.end());
  GetOutput() = hull_;

  return true;
}

bool GaivoronskiyMGrahamScanSEQ::PostProcessingImpl() {
  return GetOutput().size() >= 3;
}

}  // namespace gaivoronskiy_m_grachem_method
