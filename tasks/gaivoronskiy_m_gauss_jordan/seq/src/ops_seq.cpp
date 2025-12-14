#include "gaivoronskiy_m_gauss_jordan/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
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

  auto isZero = [](double value) { return std::fabs(value) < 1e-10; };

  auto findPivot = [&](int row, int col) -> int {
    for (int i = row; i < n; i++) {
      if (!isZero(matrix[i][col])) {
        return i;
      }
    }
    return -1;
  };

  auto swapRows = [&](int row1, int row2) {
    for (int j = 0; j < m; j++) {
      std::swap(matrix[row1][j], matrix[row2][j]);
    }
  };

  auto divideRow = [&](int row, double divisor) {
    for (int j = 0; j < m; j++) {
      matrix[row][j] /= divisor;
    }
  };

  auto subtractRows = [&](int targetRow, int sourceRow, double coefficient) {
    for (int j = 0; j < m; j++) {
      matrix[targetRow][j] -= coefficient * matrix[sourceRow][j];
    }
  };

  int row = 0;
  int col = 0;

  while (row < n && col < m - 1) {
    int pivotRow = findPivot(row, col);

    if (pivotRow == -1) {
      col++;
      continue;
    }

    if (pivotRow != row) {
      swapRows(row, pivotRow);
    }

    double pivotValue = matrix[row][col];
    if (!isZero(pivotValue)) {
      divideRow(row, pivotValue);
    }

    for (int i = 0; i < n; i++) {
      if (i != row && !isZero(matrix[i][col])) {
        double coeff = matrix[i][col];
        subtractRows(i, row, coeff);
      }
    }

    row++;
    col++;
  }

  std::vector<double> solution(m - 1, 0.0);

  for (int i = 0; i < n; i++) {
    bool allZero = true;
    for (int j = 0; j < m - 1; j++) {
      if (!isZero(matrix[i][j])) {
        allZero = false;
        break;
      }
    }
    if (allZero && !isZero(matrix[i][m - 1])) {
      GetOutput() = std::vector<double>();
      return false;
    }
  }

  int rank = 0;
  for (int i = 0; i < n; i++) {
    bool hasNonZero = false;
    for (int j = 0; j < m - 1; j++) {
      if (!isZero(matrix[i][j])) {
        hasNonZero = true;
        break;
      }
    }
    if (hasNonZero) {
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
      if (!isZero(matrix[i][j])) {
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
