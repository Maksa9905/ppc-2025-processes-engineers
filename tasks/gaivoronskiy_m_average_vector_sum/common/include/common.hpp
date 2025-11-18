#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace gaivoronskiy_m_average_vector_sum {

using InType = int;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace gaivoronskiy_m_average_vector_sum
