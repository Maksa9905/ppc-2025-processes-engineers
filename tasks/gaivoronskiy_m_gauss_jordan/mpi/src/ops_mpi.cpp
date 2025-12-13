#include "gaivoronskiy_m_gauss_jordan/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "gaivoronskiy_m_gauss_jordan/common/include/common.hpp"

namespace gaivoronskiy_m_gauss_jordan {

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

  // Распределяем матрицу по процессам
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

  // Рассылаем матрицу всем процессам
  if (rank == 0) {
    std::vector<double> flat_matrix(static_cast<size_t>(n * m));
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        flat_matrix[static_cast<size_t>(i * m + j)] = GetInput()[i][j];
      }
    }
    MPI_Bcast(flat_matrix.data(), n * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  } else {
    std::vector<double> flat_matrix(static_cast<size_t>(n * m));
    MPI_Bcast(flat_matrix.data(), n * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    GetInput() = InType(static_cast<size_t>(n), std::vector<double>(static_cast<size_t>(m)));
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        GetInput()[i][j] = flat_matrix[static_cast<size_t>(i * m + j)];
      }
    }
  }

  return true;
}

bool GaivoronskiyMGaussJordanMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

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

  // Все процессы имеют полную копию матрицы (уже распределена в PreProcessingImpl)

  // Проверка на нулевой элемент
  auto isZero = [](double value) { return std::fabs(value) < 1e-10; };

  // Получить глобальную строку (все процессы имеют полную копию матрицы)
  auto getGlobalRow = [&](int global_row, std::vector<double> &buffer) {
    buffer = matrix[global_row];
  };

  // Установить глобальную строку (все процессы выполняют одинаковые операции)
  auto setGlobalRow = [&](int global_row, const std::vector<double> &row) {
    matrix[global_row] = row;
  };

  // Найти опорный элемент в столбце
  auto findPivot = [&](int col, int start_row) -> int {
    // Все процессы имеют полную копию матрицы, ищем локально
    int global_pivot_row = -1;
    double max_val = 0.0;

    for (int i = start_row; i < n; i++) {
      double abs_val = std::fabs(matrix[i][col]);
      if (abs_val > max_val) {
        max_val = abs_val;
        global_pivot_row = i;
      }
    }

    // Все процессы находят одинаковый результат, так как имеют одинаковую матрицу
    if (max_val > 1e-10) {
      return global_pivot_row;
    }

    return -1;  // Все элементы нулевые
  };

  // Поменять строки местами
  auto swapRows = [&](int global_row1, int global_row2) {
    if (global_row1 == global_row2) return;

    std::vector<double> row1(m), row2(m);
    getGlobalRow(global_row1, row1);
    getGlobalRow(global_row2, row2);
    setGlobalRow(global_row1, row2);
    setGlobalRow(global_row2, row1);
  };

  // Нормализовать строку
  auto normalizeRow = [&](int global_row, int pivot_col) {
    std::vector<double> row(m);
    getGlobalRow(global_row, row);

    double pivot = row[pivot_col];
    if (!isZero(pivot)) {
      for (int j = 0; j < m; j++) {
        row[j] /= pivot;
      }
    }

    setGlobalRow(global_row, row);
  };

  // Обнулить столбец в других строках
  auto eliminateColumn = [&](int pivot_row, int pivot_col) {
    std::vector<double> pivot_row_data(m);
    getGlobalRow(pivot_row, pivot_row_data);

    double pivot_value = pivot_row_data[pivot_col];

    if (isZero(pivot_value)) return;

    // Все процессы обрабатывают все строки (у всех полная копия)
    for (int i = 0; i < n; i++) {
      if (i == pivot_row) continue;

      double coeff = matrix[i][pivot_col];

      if (!isZero(coeff)) {
        for (int j = 0; j < m; j++) {
          matrix[i][j] -= coeff * pivot_row_data[j] / pivot_value;
        }
      }
    }
  };

  // Основной цикл метода Гаусса-Жордана
  int row = 0;
  int col = 0;

  while (row < n && col < m - 1) {
    // Находим опорный элемент
    int pivot_row = findPivot(col, row);

    if (pivot_row == -1) {
      col++;
      continue;
    }

    // Переставляем строки
    if (pivot_row != row) {
      swapRows(row, pivot_row);
    }

    // Нормализуем опорную строку
    normalizeRow(row, col);

    // Обнуляем столбец во всех строках
    eliminateColumn(row, col);

    row++;
    col++;

    // Синхронизация после каждой итерации
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // Собираем результаты (все процессы имеют полную матрицу)
  std::vector<double> solution(m - 1, 0.0);

  // Проверяем все строки
  bool inconsistent_local = false;
  bool infinite_solutions_local = false;

  for (int i = 0; i < n; i++) {
    bool all_zero = true;
    bool has_non_zero = false;

    for (int j = 0; j < m - 1; j++) {
      if (!isZero(matrix[i][j])) {
        all_zero = false;
        has_non_zero = true;
      }
    }

    // Проверка на противоречивость
    if (all_zero && !isZero(matrix[i][m - 1])) {
      inconsistent_local = true;
    }

    // Проверка на линейную зависимость
    if (!has_non_zero && i < m - 1) {
      infinite_solutions_local = true;
    }

    // Извлекаем решение из диагональных элементов
    if (i < m - 1) {
      for (int j = 0; j < m - 1; j++) {
        if (!isZero(matrix[i][j])) {
          solution[j] = matrix[i][m - 1];
          break;
        }
      }
    }
  }

  // Собираем флаги со всех процессов
  bool inconsistent_global, infinite_solutions_global;
  MPI_Reduce(&inconsistent_local, &inconsistent_global, 1, MPI_C_BOOL, MPI_LOR, 0, MPI_COMM_WORLD);
  MPI_Reduce(&infinite_solutions_local, &infinite_solutions_global, 1, MPI_C_BOOL, MPI_LOR, 0,
             MPI_COMM_WORLD);

  // Определяем тип решения (все процессы имеют одинаковые данные)
  if (inconsistent_global) {
    GetOutput() = std::vector<double>();  // Нет решений
    return false;
  } else if (infinite_solutions_global) {
    GetOutput() = std::vector<double>();  // Бесконечно много решений
    return false;
  } else {
    GetOutput() = solution;  // Единственное решение
  }

  return !GetOutput().empty();
}

bool GaivoronskiyMGaussJordanMPI::PostProcessingImpl() {
  return true;
}

}  // namespace gaivoronskiy_m_gauss_jordan
