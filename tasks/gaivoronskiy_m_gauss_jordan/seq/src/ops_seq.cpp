#include "gaivoronskiy_m_gauss_jordan/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "gaivoronskiy_m_gauss_jordan/common/include/common.hpp"

namespace gaivoronskiy_m_gauss_jordan {

namespace {

bool IsZero(double value) {
  return std::fabs(value) < 1e-10;
}

int FindPivot(const std::vector<std::vector<double>> &matrix, int row, int col, int n) {
  for (int i = row; i < n; i++) {
    if (!IsZero(matrix[i][col])) {
      return i;
    }
  }
  return -1;
}

void SwapRows(std::vector<std::vector<double>> &matrix, int row1, int row2, int m) {
  for (int j = 0; j < m; j++) {
    std::swap(matrix[row1][j], matrix[row2][j]);
  }
}

void DivideRow(std::vector<std::vector<double>> &matrix, int row, double divisor, int m) {
  for (int j = 0; j < m; j++) {
    matrix[row][j] /= divisor;
  }
}

void SubtractRows(std::vector<std::vector<double>> &matrix, int target_row, int source_row, double coefficient, int m) {
  for (int j = 0; j < m; j++) {
    matrix[target_row][j] -= coefficient * matrix[source_row][j];
  }
}

void PerformGaussJordanElimination(std::vector<std::vector<double>> &matrix, int n, int m) {
  int row = 0;
  int col = 0;

  while (row < n && col < m - 1) {
    int pivot_row = FindPivot(matrix, row, col, n);

    if (pivot_row == -1) {
      col++;
      continue;
    }

    if (pivot_row != row) {
      SwapRows(matrix, row, pivot_row, m);
    }

    double pivot_value = matrix[row][col];
    if (!IsZero(pivot_value)) {
      DivideRow(matrix, row, pivot_value, m);
    }

    for (int i = 0; i < n; i++) {
      if (i != row && !IsZero(matrix[i][col])) {
        double coeff = matrix[i][col];
        SubtractRows(matrix, i, row, coeff, m);
      }
    }

    row++;
    col++;
  }
}

bool IsRowAllZeros(const std::vector<double> &row, int cols) {
  for (int j = 0; j < cols; j++) {
    if (!IsZero(row[j])) {
      return false;
    }
  }
  return true;
}

bool CheckInconsistency(const std::vector<std::vector<double>> &matrix, int n, int m) {
  for (int i = 0; i < n; i++) {
    if (IsRowAllZeros(matrix[i], m - 1) && !IsZero(matrix[i][m - 1])) {
      return true;
    }
  }
  return false;
}

int CalculateRank(const std::vector<std::vector<double>> &matrix, int n, int m) {
  int rank = 0;
  for (int i = 0; i < n; i++) {
    if (!IsRowAllZeros(matrix[i], m - 1)) {
      rank++;
    }
  }
  return rank;
}

std::vector<double> ExtractSolution(const std::vector<std::vector<double>> &matrix, int n, int m) {
  std::vector<double> solution(m - 1, 0.0);
  for (int i = 0; i < std::min(n, m - 1); i++) {
    bool found = false;
    for (int j = 0; j < m - 1; j++) {
      if (!IsZero(matrix[i][j])) {
        solution[j] = matrix[i][m - 1];
        found = true;
        break;
      }
    }
    if (!found && i < m - 1) {
      solution[i] = 0.0;
    }
  }
  return solution;
}

}  // namespace

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

  PerformGaussJordanElimination(matrix, n, m);

  if (CheckInconsistency(matrix, n, m)) {
    GetOutput() = std::vector<double>();
    return false;
  }

  int rank = CalculateRank(matrix, n, m);
  if (rank < m - 1 && rank < n) {
    GetOutput() = std::vector<double>();
    return false;
  }

  GetOutput() = ExtractSolution(matrix, n, m);
  return true;
}

bool GaivoronskiyMGaussJordanSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace gaivoronskiy_m_gauss_jordan
