#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <mpi.h>

using namespace std;

class ParallelGaussJordan {
private:
    vector<vector<double>> local_matrix; // Локальная часть матрицы
    vector<double> solution; // Решение
    int n; // Общее количество уравнений
    int m; // Общее количество столбцов (переменные + 1)
    int rank; // Ранг процесса
    int size; // Количество процессов
    int rows_per_process; // Количество строк на процесс
    
    // Проверка на нулевой элемент
    bool isZero(double value) {
        return fabs(value) < 1e-10;
    }
    
    // Получить глобальную строку (для вывода)
    void getGlobalRow(int global_row, vector<double>& buffer) {
        int owner = global_row / rows_per_process;
        
        if (rank == owner) {
            int local_row = global_row % rows_per_process;
            buffer = local_matrix[local_row];
        }
        
        // Рассылаем строку всем процессам
        MPI_Bcast(buffer.data(), m, MPI_DOUBLE, owner, MPI_COMM_WORLD);
    }
    
    // Установить глобальную строку
    void setGlobalRow(int global_row, const vector<double>& row) {
        int owner = global_row / rows_per_process;
        
        if (rank == owner) {
            int local_row = global_row % rows_per_process;
            local_matrix[local_row] = row;
        }
        
        // Рассылаем обновленную строку процессу-владельцу
        if (rank == 0) {
            MPI_Send(row.data(), m, MPI_DOUBLE, owner, 0, MPI_COMM_WORLD);
        }
        if (rank == owner && rank != 0) {
            MPI_Recv(local_matrix[global_row % rows_per_process].data(), m, 
                    MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    
    // Найти опорный элемент в столбце
    int findPivot(int col, int start_row) {
        int global_pivot_row = -1;
        double max_val = 0.0;
        
        // Каждый процесс ищет максимальный элемент в своем сегменте
        for (int local_row = 0; local_row < rows_per_process; local_row++) {
            int global_row = rank * rows_per_process + local_row;
            if (global_row >= start_row && global_row < n) {
                double abs_val = fabs(local_matrix[local_row][col]);
                if (abs_val > max_val) {
                    max_val = abs_val;
                    global_pivot_row = global_row;
                }
            }
        }
        
        // Находим глобальный максимум среди всех процессов
        struct {
            double value;
            int rank;
        } local_max, global_max;
        
        local_max.value = max_val;
        local_max.rank = rank;
        
        MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
        
        // Если нашли ненулевой элемент
        if (global_max.value > 1e-10) {
            // Процесс с максимальным значением сообщает номер строки
            if (rank == global_max.rank) {
                MPI_Bcast(&global_pivot_row, 1, MPI_INT, global_max.rank, MPI_COMM_WORLD);
            } else {
                MPI_Bcast(&global_pivot_row, 1, MPI_INT, global_max.rank, MPI_COMM_WORLD);
            }
            return global_pivot_row;
        }
        
        return -1; // Все элементы нулевые
    }
    
    // Поменять строки местами
    void swapRows(int global_row1, int global_row2) {
        if (global_row1 == global_row2) return;
        
        vector<double> row1(m), row2(m);
        
        // Получаем обе строки
        getGlobalRow(global_row1, row1);
        getGlobalRow(global_row2, row2);
        
        // Меняем местами
        setGlobalRow(global_row1, row2);
        setGlobalRow(global_row2, row1);
    }
    
    // Нормализовать строку (разделить на опорный элемент)
    void normalizeRow(int global_row, int pivot_col) {
        vector<double> row(m);
        getGlobalRow(global_row, row);
        
        double pivot = row[pivot_col];
        if (!isZero(pivot)) {
            for (int j = 0; j < m; j++) {
                row[j] /= pivot;
            }
        }
        
        setGlobalRow(global_row, row);
    }
    
    // Обнулить столбец в других строках
    void eliminateColumn(int pivot_row, int pivot_col) {
        vector<double> pivot_row_data(m);
        getGlobalRow(pivot_row, pivot_row_data);
        
        double pivot_value = pivot_row_data[pivot_col];
        
        if (isZero(pivot_value)) return;
        
        // Каждый процесс обрабатывает свои строки
        for (int local_row = 0; local_row < rows_per_process; local_row++) {
            int global_row = rank * rows_per_process + local_row;
            
            if (global_row >= n) continue;
            if (global_row == pivot_row) continue;
            
            double coeff = local_matrix[local_row][pivot_col];
            
            if (!isZero(coeff)) {
                for (int j = 0; j < m; j++) {
                    local_matrix[local_row][j] -= coeff * pivot_row_data[j] / pivot_value;
                }
            }
        }
    }
    
public:
    ParallelGaussJordan(const vector<vector<double>>& matrix, int argc, char** argv) {
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        
        n = matrix.size();
        m = (n > 0) ? matrix[0].size() : 0;
        
        // Распределяем строки по процессам
        rows_per_process = (n + size - 1) / size; // Округление вверх
        
        // Выделяем память для локальной матрицы
        local_matrix.resize(rows_per_process, vector<double>(m, 0.0));
        
        // Распределяем данные по процессам
        distributeData(matrix);
    }
    
    ~ParallelGaussJordan() {
        MPI_Finalize();
    }
    
    // Распределение исходной матрицы по процессам
    void distributeData(const vector<vector<double>>& matrix) {
        if (rank == 0) {
            // Процесс 0 распределяет данные
            for (int i = 0; i < n; i++) {
                int dest_process = i / rows_per_process;
                if (dest_process == 0) {
                    local_matrix[i % rows_per_process] = matrix[i];
                } else {
                    MPI_Send(matrix[i].data(), m, MPI_DOUBLE, 
                            dest_process, 0, MPI_COMM_WORLD);
                }
            }
        } else {
            // Остальные процессы получают данные
            for (int i = rank * rows_per_process; 
                 i < min((rank + 1) * rows_per_process, n); i++) {
                int local_idx = i % rows_per_process;
                MPI_Recv(local_matrix[local_idx].data(), m, MPI_DOUBLE, 
                        0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        
        // Синхронизация
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    // Основной метод решения
    pair<vector<double>, int> solve() {
        int row = 0;
        int col = 0;
        
        // Основной цикл метода Гаусса-Жордана
        while (row < n && col < m - 1) {
            if (rank == 0 && col % 10 == 0) {
                cout << "Обработка столбца " << col + 1 << " из " << m - 1 << endl;
            }
            
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
        
        // Собираем результаты
        return gatherSolution();
    }
    
    // Сбор решения со всех процессов
    pair<vector<double>, int> gatherSolution() {
        solution.resize(m - 1, 0.0);
        
        // Каждый процесс проверяет свои строки на противоречивость
        bool inconsistent_local = false;
        bool infinite_solutions_local = false;
        
        for (int local_row = 0; local_row < rows_per_process; local_row++) {
            int global_row = rank * rows_per_process + local_row;
            if (global_row >= n) break;
            
            bool all_zero = true;
            bool has_non_zero = false;
            
            for (int j = 0; j < m - 1; j++) {
                if (!isZero(local_matrix[local_row][j])) {
                    all_zero = false;
                    has_non_zero = true;
                }
            }
            
            // Проверка на противоречивость
            if (all_zero && !isZero(local_matrix[local_row][m - 1])) {
                inconsistent_local = true;
            }
            
            // Проверка на линейную зависимость
            if (!has_non_zero && global_row < m - 1) {
                infinite_solutions_local = true;
            }
            
            // Извлекаем решение из диагональных элементов
            if (global_row < m - 1) {
                for (int j = 0; j < m - 1; j++) {
                    if (!isZero(local_matrix[local_row][j])) {
                        // Отправляем значение в процесс 0
                        double value = local_matrix[local_row][m - 1];
                        if (rank == 0) {
                            solution[j] = value;
                        } else {
                            // Отправляем процессу 0
                            MPI_Send(&value, 1, MPI_DOUBLE, 0, j, MPI_COMM_WORLD);
                        }
                        break;
                    }
                }
            }
        }
        
        // Собираем флаги со всех процессов
        bool inconsistent_global, infinite_solutions_global;
        MPI_Reduce(&inconsistent_local, &inconsistent_global, 1, 
                  MPI_C_BOOL, MPI_LOR, 0, MPI_COMM_WORLD);
        MPI_Reduce(&infinite_solutions_local, &infinite_solutions_global, 1,
                  MPI_C_BOOL, MPI_LOR, 0, MPI_COMM_WORLD);
        
        // Процесс 0 собирает все части решения
        if (rank == 0) {
            // Принимаем значения от других процессов
            for (int j = 0; j < m - 1; j++) {
                if (isZero(solution[j])) {
                    double value;
                    MPI_Status status;
                    // Проверяем, есть ли сообщение
                    int flag;
                    MPI_Iprobe(MPI_ANY_SOURCE, j, MPI_COMM_WORLD, &flag, &status);
                    if (flag) {
                        MPI_Recv(&value, 1, MPI_DOUBLE, status.MPI_SOURCE, 
                                j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        solution[j] = value;
                    }
                }
            }
            
            // Определяем тип решения
            if (inconsistent_global) {
                return {solution, 0}; // Нет решений
            } else if (infinite_solutions_global) {
                return {solution, -1}; // Бесконечно много решений
            } else {
                return {solution, 1}; // Единственное решение
            }
        } else {
            // Для других процессов возвращаем пустой результат
            return {vector<double>(), 0};
        }
    }
    
    // Вывод матрицы (только процессом 0)
    void printMatrix(const string& title = "Матрица:") {
        if (rank != 0) return;
        
        cout << title << endl;
        vector<double> row_buffer(m);
        
        for (int i = 0; i < n; i++) {
            getGlobalRow(i, row_buffer);
            
            for (int j = 0; j < m; j++) {
                if (j == m - 1) {
                    cout << " | ";
                }
                cout << setw(10) << fixed << setprecision(4) << row_buffer[j];
            }
            cout << endl;
        }
        cout << endl;
    }
    
    // Вывод решения
    static void printSolution(const vector<double>& solution, int solutionType, int rank) {
        if (rank != 0) return;
        
        cout << "\nРезультат:" << endl;
        switch (solutionType) {
            case 0:
                cout << "Система не имеет решений (противоречива)" << endl;
                break;
            case -1:
                cout << "Система имеет бесконечно много решений" << endl;
                break;
            case 1:
                cout << "Система имеет единственное решение:" << endl;
                for (size_t i = 0; i < solution.size(); i++) {
                    cout << "x" << i + 1 << " = " << fixed << setprecision(6) << solution[i] << endl;
                }
                break;
        }
    }
    
    // Получить ранг процесса
    int getRank() const { return rank; }
    
    // Получить количество процессов
    int getSize() const { return size; }
};

// Функция для создания тестовой матрицы
vector<vector<double>> createTestMatrix(int n, int m) {
    vector<vector<double>> matrix(n, vector<double>(m + 1));
    
    // Создаем диагонально доминирующую матрицу для устойчивости
    for (int i = 0; i < n; i++) {
        double sum = 0;
        for (int j = 0; j < m; j++) {
            if (i == j) {
                matrix[i][j] = 10.0 + (rand() % 100) / 10.0;
            } else {
                matrix[i][j] = (rand() % 10) / 10.0;
            }
            sum += fabs(matrix[i][j]);
        }
        // Правая часть - сумма элементов строки
        matrix[i][m] = sum;
    }
    
    return matrix;
}

int main(int argc, char** argv) {
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if (rank == 0) {
        cout << "ПАРАЛЛЕЛЬНЫЙ МЕТОД ГАУССА-ЖОРДАНА С ИСПОЛЬЗОВАНИЕМ MPI" << endl;
        cout << "=====================================================" << endl;
    }
    
    int n = 100; // Размер системы
    int m = 100; // Количество переменных
    
    vector<vector<double>> matrix;
    
    if (rank == 0) {
        cout << "\nСоздание тестовой матрицы " << n << "x" << m << "..." << endl;
        matrix = createTestMatrix(n, m);
        
        // Выводим информацию о параллельном выполнении
        int size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        cout << "Количество процессов: " << size << endl;
        cout << "Распределение: примерно " << (n + size - 1) / size 
             << " строк на процесс" << endl;
    }
    
    // Широковещательная рассылка размеров матрицы
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Создаем решатель
    ParallelGaussJordan solver(matrix, argc, argv);
    
    double start_time, end_time;
    
    if (rank == 0) {
        start_time = MPI_Wtime();
        cout << "\nНачало вычислений..." << endl;
    }
    
    // Решаем систему
    auto [solution, solutionType] = solver.solve();
    
    if (rank == 0) {
        end_time = MPI_Wtime();
        cout << "Вычисления завершены за " << fixed << setprecision(4) 
             << (end_time - start_time) << " секунд" << endl;
    }
    
    // Выводим результат
    ParallelGaussJordan::printSolution(solution, solutionType, rank);
    
    // Статистика производительности
    if (rank == 0) {
        cout << "\nСтатистика производительности:" << endl;
        cout << "Размер системы: " << n << " уравнений, " << m << " переменных" << endl;
        cout << "Время выполнения: " << fixed << setprecision(4) 
             << (end_time - start_time) << " секунд" << endl;
    }
    
    MPI_Finalize();
    return 0;
}