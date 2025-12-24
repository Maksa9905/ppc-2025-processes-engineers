#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace gaivoronskiy_m_grachem_method {

struct Point {
  double x;
  double y;

  Point() : x(0.0), y(0.0) {}
  Point(double x_val, double y_val) : x(x_val), y(y_val) {}

  bool operator==(const Point &other) const {
    return x == other.x && y == other.y;
  }
};

using InType = std::vector<Point>;
using OutType = std::vector<Point>;
using TestType = std::tuple<std::vector<Point>, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace gaivoronskiy_m_grachem_method
