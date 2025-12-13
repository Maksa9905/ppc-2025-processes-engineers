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

  // Копируем матрицу для работы
  std::vector<std::vector<double>> matrix = GetInput();
  int n = static_cast<int>(matrix.size());
  int m = (n > 0) ? static_cast<int>(matrix[0].size()) : 0;

  if (m == 0) {
    return false;
  }

  // Проверка на нулевой элемент
  auto isZero = [](double value) { return std::fabs(value) < 1e-10; };

  // Поиск ненулевого элемента в столбце
  auto findPivot = [&](int row, int col) -> int {
    for (int i = row; i < n; i++) {
      if (!isZero(matrix[i][col])) {
        return i;
      }
    }
    return -1;
  };

  // Перестановка строк
  auto swapRows = [&](int row1, int row2) {
    for (int j = 0; j < m; j++) {
      std::swap(matrix[row1][j], matrix[row2][j]);
    }
  };

  // Деление строки на число
  auto divideRow = [&](int row, double divisor) {
    for (int j = 0; j < m; j++) {
      matrix[row][j] /= divisor;
    }
  };

  // Вычитание строки, умноженной на коэффициент
  auto subtractRows = [&](int targetRow, int sourceRow, double coefficient) {
    for (int j = 0; j < m; j++) {
      matrix[targetRow][j] -= coefficient * matrix[sourceRow][j];
    }
  };

  // Прямой ход - приведение к диагональному виду
  int row = 0;
  int col = 0;

  while (row < n && col < m - 1) {
    // Ищем опорный элемент
    int pivotRow = findPivot(row, col);

    if (pivotRow == -1) {
      // Все элементы в столбце нулевые, переходим к следующему столбцу
      col++;
      continue;
    }

    // Переставляем строку с опорным элементом на текущую позицию
    if (pivotRow != row) {
      swapRows(row, pivotRow);
    }

    // Нормализуем текущую строку (делаем опорный элемент равным 1)
    double pivotValue = matrix[row][col];
    if (!isZero(pivotValue)) {
      divideRow(row, pivotValue);
    }

    // Обнуляем текущий столбец во всех других строках
    for (int i = 0; i < n; i++) {
      if (i != row && !isZero(matrix[i][col])) {
        double coeff = matrix[i][col];
        subtractRows(i, row, coeff);
      }
    }

    row++;
    col++;
  }

  // Извлечение решения
  std::vector<double> solution(m - 1, 0.0);

  // Проверяем на противоречивость
  for (int i = 0; i < n; i++) {
    bool allZero = true;
    for (int j = 0; j < m - 1; j++) {
      if (!isZero(matrix[i][j])) {
        allZero = false;
        break;
      }
    }
    if (allZero && !isZero(matrix[i][m - 1])) {
      // Нет решений
      GetOutput() = std::vector<double>();
      return false;
    }
  }

  // Проверяем на бесконечное количество решений
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

  // Если rank < m - 1 и rank < n, то бесконечно много решений
  if (rank < m - 1 && rank < n) {
    // Бесконечно много решений
    GetOutput() = std::vector<double>();
    return false;
  }

  // Извлекаем единственное решение
  // После метода Гаусса-Жордана каждая строка должна иметь единицу на диагонали
  for (int i = 0; i < std::min(n, m - 1); i++) {
    bool found = false;
    for (int j = 0; j < m - 1; j++) {
      if (!isZero(matrix[i][j])) {
        // Находим переменную, соответствующую этой строке
        // После нормализации это должен быть единичный элемент
        solution[j] = matrix[i][m - 1];
        found = true;
        break;
      }
    }
    if (!found && i < m - 1) {
      solution[i] = 0.0;  // Свободная переменная
    }
  }

  GetOutput() = solution;
  return true;
}

bool GaivoronskiyMGaussJordanSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace gaivoronskiy_m_gauss_jordan
