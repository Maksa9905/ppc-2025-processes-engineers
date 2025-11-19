#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace gaivoronskiy_m_average_vector_sum {

using InType = std::vector<double>;
using OutType = double;

struct TestCase {
  std::vector<double> values;
  std::string name;
};

using TestType = TestCase;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace gaivoronskiy_m_average_vector_sum
