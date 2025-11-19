#pragma once

#include <ostream>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace gaivoronskiy_m_average_vector_sum {

using InType = std::vector<double>;
using OutType = double;

struct TestCase {
  std::string file_name;
  double expected_average;
};

// Provide printing support for Google Test
inline void PrintTo(const TestCase& test_case, std::ostream* os) {
  *os << "TestCase{file_name=\"" << test_case.file_name << "\", expected_average=" << test_case.expected_average << "}";
}

using TestType = TestCase;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace gaivoronskiy_m_average_vector_sum
