#include "gaivoronskiy_m_gauss_jordan/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "gaivoronskiy_m_gauss_jordan/common/include/common.hpp"

namespace gaivoronskiy_m_gauss_jordan {

GaivoronskiyMGaussJordanSEQ::GaivoronskiyMGaussJordanSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  InType tmp(in);
  GetInput().swap(tmp);
}

bool GaivoronskiyMGaussJordanSEQ::ValidationImpl() {
  if (GetInput().empty()) {
    return true;
  }
  size_t cols = GetInput()[0].size();
  std::size_t total = 0;
  for (const auto &row : GetInput()) {
    total += row.size();
  }
  return GetOutput().empty() && (cols != 0) && ((cols * GetInput().size()) == total);
}

bool GaivoronskiyMGaussJordanSEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool GaivoronskiyMGaussJordanSEQ::RunImpl() {
  if (GetInput().empty()) {
    return true;
  }

  std::vector<std::vector<double>> matrix = GetInput();
  int n = static_cast<int>(matrix.size());
  int m = (n > 0) ? static_cast<int>(matrix[0].size()) : 0;

  if (m == 0) {
    return false;
  }

  auto is_zero = [](double value) { return std::fabs(value) < 1e-10; };

  auto find_pivot = [&](int row, int col) -> int {
    for (int i = row; i < n; i++) {
      if (!is_zero(matrix[i][col])) {
        return i;
      }
    }
    return -1;
  };

  auto swap_rows = [&](int row1, int row2) {
    for (int j = 0; j < m; j++) {
      std::swap(matrix[row1][j], matrix[row2][j]);
    }
  };

  auto divide_row = [&](int row, double divisor) {
    for (int j = 0; j < m; j++) {
      matrix[row][j] /= divisor;
    }
  };

  auto subtract_rows = [&](int target_row, int source_row, double coefficient) {
    for (int j = 0; j < m; j++) {
      matrix[target_row][j] -= coefficient * matrix[source_row][j];
    }
  };

  int row = 0;
  int col = 0;

  while (row < n && col < m - 1) {
    int pivot_row = find_pivot(row, col);

    if (pivot_row == -1) {
      col++;
      continue;
    }

    if (pivot_row != row) {
      swap_rows(row, pivot_row);
    }

    double pivot_value = matrix[row][col];
    if (!is_zero(pivot_value)) {
      divide_row(row, pivot_value);
    }

    for (int i = 0; i < n; i++) {
      if (i != row && !is_zero(matrix[i][col])) {
        double coeff = matrix[i][col];
        subtract_rows(i, row, coeff);
      }
    }

    row++;
    col++;
  }

  std::vector<double> solution(m - 1, 0.0);

  for (int i = 0; i < n; i++) {
    bool all_zero = true;
    for (int j = 0; j < m - 1; j++) {
      if (!is_zero(matrix[i][j])) {
        all_zero = false;
        break;
      }
    }
    if (all_zero && !is_zero(matrix[i][m - 1])) {
      GetOutput() = std::vector<double>();
      return false;
    }
  }

  int rank = 0;
  for (int i = 0; i < n; i++) {
    bool has_non_zero = false;
    for (int j = 0; j < m - 1; j++) {
      if (!is_zero(matrix[i][j])) {
        has_non_zero = true;
        break;
      }
    }
    if (has_non_zero) {
      rank++;
    }
  }

  if (rank < m - 1 && rank < n) {
    GetOutput() = std::vector<double>();
    return false;
  }

  for (int i = 0; i < std::min(n, m - 1); i++) {
    bool found = false;
    for (int j = 0; j < m - 1; j++) {
      if (!is_zero(matrix[i][j])) {
        solution[j] = matrix[i][m - 1];
        found = true;
        break;
      }
    }
    if (!found && i < m - 1) {
      solution[i] = 0.0;
    }
  }

  GetOutput() = solution;
  return true;
}

bool GaivoronskiyMGaussJordanSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace gaivoronskiy_m_gauss_jordan
