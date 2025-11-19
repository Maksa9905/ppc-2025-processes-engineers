#include "gaivoronskiy_m_average_vector_sum/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <cstdint>
#include <numeric>

#include "gaivoronskiy_m_average_vector_sum/common/include/common.hpp"

namespace gaivoronskiy_m_average_vector_sum {

GaivoronskiyMAverageVecSumMPI::GaivoronskiyMAverageVecSumMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool GaivoronskiyMAverageVecSumMPI::ValidationImpl() {
  return !GetInput().empty();
}

bool GaivoronskiyMAverageVecSumMPI::PreProcessingImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

  total_size_ = 0;
  if (world_rank_ == 0) {
    distributed_values_ = GetInput();
    total_size_ = distributed_values_.size();
  } else {
    distributed_values_.clear();
  }

  std::uint64_t size_to_share = static_cast<std::uint64_t>(total_size_);
  MPI_Bcast(&size_to_share, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
  total_size_ = static_cast<std::size_t>(size_to_share);

  local_sum_ = 0.0;
  global_sum_ = 0.0;
  return total_size_ > 0;
}

bool GaivoronskiyMAverageVecSumMPI::RunImpl() {
  if (world_size_ <= 0 || total_size_ == 0) {
    return false;
  }

  std::vector<int> send_counts(world_size_, 0);
  std::vector<int> displs(world_size_, 0);

  const std::size_t base_chunk = total_size_ / static_cast<std::size_t>(world_size_);
  const std::size_t remainder = total_size_ % static_cast<std::size_t>(world_size_);

  for (int rank = 0; rank < world_size_; rank++) {
    std::size_t chunk = base_chunk + (static_cast<std::size_t>(rank) < remainder ? 1 : 0);
    send_counts[rank] = static_cast<int>(chunk);
    if (rank > 0) {
      displs[rank] = displs[rank - 1] + send_counts[rank - 1];
    }
  }

  const int recv_count = send_counts[world_rank_];
  const auto buffer_size = recv_count > 0 ? recv_count : 0;
  local_buffer_.assign(static_cast<std::size_t>(buffer_size), 0.0);

  const double *send_buffer = distributed_values_.empty() ? nullptr : distributed_values_.data();
  double *recv_buffer = local_buffer_.empty() ? nullptr : local_buffer_.data();
  MPI_Scatterv(send_buffer, send_counts.data(), displs.data(), MPI_DOUBLE, recv_buffer, recv_count, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);

  local_sum_ = std::accumulate(local_buffer_.begin(), local_buffer_.end(), 0.0);

  MPI_Allreduce(&local_sum_, &global_sum_, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = global_sum_ / static_cast<double>(total_size_);
  return true;
}

bool GaivoronskiyMAverageVecSumMPI::PostProcessingImpl() {
  return std::isfinite(GetOutput());
}

}  // namespace gaivoronskiy_m_average_vector_sum
