#include "gaivoronskiy_m_gauss_jordan/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "gaivoronskiy_m_gauss_jordan/common/include/common.hpp"

namespace gaivoronskiy_m_gauss_jordan {

namespace {

bool is_zero(double value) {
  return std::fabs(value) < 1e-10;
}

void perform_gauss_jordan_elimination(std::vector<std::vector<double>> &matrix, int n, int m) {
  auto find_pivot = [&](int col, int start_row) -> int {
    int global_pivot_row = -1;
    double max_val = 0.0;

    for (int i = start_row; i < n; i++) {
      double abs_val = std::fabs(matrix[i][col]);
      if (abs_val > max_val) {
        max_val = abs_val;
        global_pivot_row = i;
      }
    }

    return (max_val > 1e-10) ? global_pivot_row : -1;
  };

  auto swap_rows = [&](int row1, int row2) {
    if (row1 != row2) {
      std::swap(matrix[row1], matrix[row2]);
    }
  };

  auto normalize_row = [&](int global_row, int pivot_col) {
    double pivot = matrix[global_row][pivot_col];
    if (!is_zero(pivot)) {
      for (int j = 0; j < m; j++) {
        matrix[global_row][j] /= pivot;
      }
    }
  };

  auto eliminate_column = [&](int pivot_row, int pivot_col) {
    for (int i = 0; i < n; i++) {
      if (i == pivot_row) {
        continue;
      }

      double coeff = matrix[i][pivot_col];
      if (!is_zero(coeff)) {
        for (int j = 0; j < m; j++) {
          matrix[i][j] -= coeff * matrix[pivot_row][j];
        }
      }
    }
  };

  int row = 0;
  int col = 0;

  while (row < n && col < m - 1) {
    int pivot_row = find_pivot(col, row);

    if (pivot_row == -1) {
      col++;
      continue;
    }

    if (pivot_row != row) {
      swap_rows(row, pivot_row);
    }

    normalize_row(row, col);
    MPI_Bcast(matrix[row].data(), m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    eliminate_column(row, col);

    for (int i = 0; i < n; i++) {
      MPI_Bcast(matrix[i].data(), m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    row++;
    col++;
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

struct MatrixAnalysisResult {
  bool inconsistent;
  int rank;
  std::vector<double> solution;
};

MatrixAnalysisResult analyze_matrix_and_extract_solution(const std::vector<std::vector<double>> &matrix, int n, int m) {
  MatrixAnalysisResult result;
  result.inconsistent = false;
  result.rank = 0;
  result.solution.resize(m - 1, 0.0);

  for (int i = 0; i < n; i++) {
    bool all_zero = true;
    bool has_non_zero = false;

    for (int j = 0; j < m - 1; j++) {
      if (!is_zero(matrix[i][j])) {
        all_zero = false;
        has_non_zero = true;
        break;
      }
    }

    if (all_zero && !is_zero(matrix[i][m - 1])) {
      result.inconsistent = true;
    }

    if (has_non_zero) {
      result.rank++;
    }

    if (i < m - 1) {
      for (int j = 0; j < m - 1; j++) {
        if (!is_zero(matrix[i][j])) {
          result.solution[j] = matrix[i][m - 1];
          break;
        }
      }
    }
  }

  return result;
}

}  // namespace

GaivoronskiyMGaussJordanMPI::GaivoronskiyMGaussJordanMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  InType tmp(in);
  GetInput().swap(tmp);
}

bool GaivoronskiyMGaussJordanMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    if (GetInput().empty()) {
      return true;
    }
    size_t cols = GetInput()[0].size();
    std::size_t total = 0;
    for (const auto &row : GetInput()) {
      total += row.size();
    }
    bool valid = GetOutput().empty() && (cols != 0) && ((cols * GetInput().size()) == total);
    int valid_int = valid ? 1 : 0;
    MPI_Bcast(&valid_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return valid;
  } else {
    int valid_int = 0;
    MPI_Bcast(&valid_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return valid_int != 0;
  }
}

bool GaivoronskiyMGaussJordanMPI::PreProcessingImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  GetOutput().clear();

  int n = 0;
  int m = 0;

  if (rank == 0) {
    n = static_cast<int>(GetInput().size());
    m = (n > 0) ? static_cast<int>(GetInput()[0].size()) : 0;
  }

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (n == 0 || m == 0) {
    return true;
  }

  if (rank == 0) {
    std::vector<double> flat_matrix(static_cast<size_t>(n * m));
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        flat_matrix[static_cast<size_t>(i) * static_cast<size_t>(m) + static_cast<size_t>(j)] = GetInput()[i][j];
      }
    }
    MPI_Bcast(flat_matrix.data(), n * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  } else {
    std::vector<double> flat_matrix(static_cast<size_t>(n * m));
    MPI_Bcast(flat_matrix.data(), n * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    GetInput() = InType(static_cast<size_t>(n), std::vector<double>(static_cast<size_t>(m)));
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        GetInput()[i][j] = flat_matrix[static_cast<size_t>(i) * static_cast<size_t>(m) + static_cast<size_t>(j)];
      }
    }
  }

  return true;
}

bool GaivoronskiyMGaussJordanMPI::RunImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (GetInput().empty()) {
    return true;
  }

  std::vector<std::vector<double>> matrix = GetInput();
  int n = static_cast<int>(matrix.size());
  int m = (n > 0) ? static_cast<int>(matrix[0].size()) : 0;

  if (m == 0) {
    return false;
  }

  perform_gauss_jordan_elimination(matrix, n, m);

  MatrixAnalysisResult local_result = analyze_matrix_and_extract_solution(matrix, n, m);

  bool inconsistent_global = false;
  int rank_global = 0;
  MPI_Reduce(&local_result.inconsistent, &inconsistent_global, 1, MPI_C_BOOL, MPI_LOR, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_result.rank, &rank_global, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Bcast(&rank_global, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int solution_type = 0;
  if (rank == 0) {
    if (inconsistent_global) {
      solution_type = 0;
      GetOutput() = std::vector<double>();
    } else if (rank_global < m - 1 && rank_global < n) {
      solution_type = -1;
      GetOutput() = std::vector<double>();
    } else {
      solution_type = 1;
      GetOutput() = local_result.solution;
    }
  }

  MPI_Bcast(&solution_type, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (solution_type != 1) {
    if (rank != 0) {
      GetOutput() = std::vector<double>();
    }
    return false;
  }

  int solution_size = static_cast<int>(GetOutput().size());
  MPI_Bcast(&solution_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    GetOutput().resize(static_cast<size_t>(solution_size));
  }

  if (solution_size > 0) {
    MPI_Bcast(GetOutput().data(), solution_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  return true;
}

bool GaivoronskiyMGaussJordanMPI::PostProcessingImpl() {
  return true;
}

}  // namespace gaivoronskiy_m_gauss_jordan
