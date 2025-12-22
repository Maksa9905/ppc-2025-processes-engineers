#include "gaivoronskiy_m_grachem_method/mpi/include/ops_mpi.hpp"

#include <mpi.h>

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

Point p0_local;

bool compare(const Point &p1, const Point &p2) {
  int o = orientation(p0_local, p1, p2);
  if (o == 0) {
    return distSquare(p0_local, p1) < distSquare(p0_local, p2);
  }
  return (o == 2);
}
}  // namespace

GaivoronskiyMGrahamScanMPI::GaivoronskiyMGrahamScanMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool GaivoronskiyMGrahamScanMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    return GetInput().size() >= 3;
  }
  return true;
}

bool GaivoronskiyMGrahamScanMPI::PreProcessingImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n_points = 0;
  if (rank == 0) {
    points_ = GetInput();
    n_points = static_cast<int>(points_.size());
  }

  MPI_Bcast(&n_points, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int base_size = n_points / size;
  int remainder = n_points % size;

  std::vector<int> send_counts(size);
  std::vector<int> displs(size);

  for (int i = 0; i < size; i++) {
    send_counts[i] = (base_size + (i < remainder ? 1 : 0)) * 2;
    displs[i] = (i * base_size + std::min(i, remainder)) * 2;
  }

  int local_size = send_counts[rank] / 2;
  local_points_.resize(local_size);

  std::vector<double> flat_points;
  if (rank == 0) {
    flat_points.resize(n_points * 2);
    for (int i = 0; i < n_points; i++) {
      flat_points[i * 2] = points_[i].x;
      flat_points[i * 2 + 1] = points_[i].y;
    }
  }

  std::vector<double> local_flat(local_size * 2);
  MPI_Scatterv(flat_points.data(), send_counts.data(), displs.data(), MPI_DOUBLE, local_flat.data(), send_counts[rank],
               MPI_DOUBLE, 0, MPI_COMM_WORLD);

  for (int i = 0; i < local_size; i++) {
    local_points_[i].x = local_flat[i * 2];
    local_points_[i].y = local_flat[i * 2 + 1];
  }

  return true;
}

bool GaivoronskiyMGrahamScanMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::vector<Point> local_hull;
  if (!local_points_.empty()) {
    local_hull = grahamScan(local_points_);
  }

  int local_hull_size = static_cast<int>(local_hull.size());
  std::vector<int> hull_sizes(size);

  MPI_Gather(&local_hull_size, 1, MPI_INT, hull_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> displs(size);
  std::vector<int> recv_counts(size);
  int total_hull_points = 0;

  if (rank == 0) {
    for (int i = 0; i < size; i++) {
      recv_counts[i] = hull_sizes[i] * 2;
      displs[i] = total_hull_points * 2;
      total_hull_points += hull_sizes[i];
    }
  }

  std::vector<double> local_hull_flat(local_hull_size * 2);
  for (int i = 0; i < local_hull_size; i++) {
    local_hull_flat[i * 2] = local_hull[i].x;
    local_hull_flat[i * 2 + 1] = local_hull[i].y;
  }

  std::vector<double> all_hulls_flat;
  if (rank == 0) {
    all_hulls_flat.resize(total_hull_points * 2);
  }

  MPI_Gatherv(local_hull_flat.data(), local_hull_size * 2, MPI_DOUBLE, all_hulls_flat.data(), recv_counts.data(),
              displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    std::vector<Point> combined_points(total_hull_points);
    for (int i = 0; i < total_hull_points; i++) {
      combined_points[i].x = all_hulls_flat[i * 2];
      combined_points[i].y = all_hulls_flat[i * 2 + 1];
    }

    hull_ = grahamScan(combined_points);
    GetOutput() = hull_;
  }

  return true;
}

bool GaivoronskiyMGrahamScanMPI::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    return GetOutput().size() >= 3;
  }
  return true;
}

std::vector<Point> GaivoronskiyMGrahamScanMPI::grahamScan(const std::vector<Point> &points) {
  if (points.empty()) {
    return std::vector<Point>();
  }
  if (points.size() < 3) {
    return points;
  }

  std::vector<Point> pts = points;

  size_t min_idx = 0;
  for (size_t i = 1; i < pts.size(); i++) {
    if (pts[i].y < pts[min_idx].y || (pts[i].y == pts[min_idx].y && pts[i].x < pts[min_idx].x)) {
      min_idx = i;
    }
  }

  std::swap(pts[0], pts[min_idx]);
  p0_local = pts[0];

  std::sort(pts.begin() + 1, pts.end(), compare);

  size_t m = 1;
  for (size_t i = 1; i < pts.size(); i++) {
    while (i < pts.size() - 1 && orientation(p0_local, pts[i], pts[i + 1]) == 0) {
      i++;
    }
    pts[m] = pts[i];
    m++;
  }

  if (m < 3) {
    return pts;
  }

  std::stack<Point> s;
  s.push(pts[0]);
  s.push(pts[1]);
  s.push(pts[2]);

  for (size_t i = 3; i < m; i++) {
    Point top = s.top();
    s.pop();
    while (!s.empty() && orientation(s.top(), top, pts[i]) != 2) {
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

  std::reverse(result.begin(), result.end());
  return result;
}

std::vector<Point> GaivoronskiyMGrahamScanMPI::mergeHulls(const std::vector<Point> &hull1,
                                                          const std::vector<Point> &hull2) {
  std::vector<Point> combined = hull1;
  combined.insert(combined.end(), hull2.begin(), hull2.end());
  return grahamScan(combined);
}

}  // namespace gaivoronskiy_m_grachem_method
