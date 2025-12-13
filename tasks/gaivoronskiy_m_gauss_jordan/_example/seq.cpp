#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>

using namespace std;

class GaussJordanSolver {
private:
    vector<vector<double>> matrix;
    int n; // количество уравнений (строк)
    int m; // количество переменных + 1 (столбцов)

    // Проверка на нулевой элемент для избежания деления на ноль
    bool isZero(double value) {
        return fabs(value) < 1e-10;
    }

    // Поиск ненулевого элемента в столбце ниже текущей строки
    int findPivot(int row, int col) {
        for (int i = row; i < n; i++) {
            if (!isZero(matrix[i][col])) {
                return i;
            }
        }
        return -1; // Все элементы в столбце нулевые
    }

    // Перестановка строк
    void swapRows(int row1, int row2) {
        for (int j = 0; j < m; j++) {
            swap(matrix[row1][j], matrix[row2][j]);
        }
    }

    // Деление строки на число
    void divideRow(int row, double divisor) {
        for (int j = 0; j < m; j++) {
            matrix[row][j] /= divisor;
        }
    }

    // Вычитание строки, умноженной на коэффициент, из другой строки
    void subtractRows(int targetRow, int sourceRow, double coefficient) {
        for (int j = 0; j < m; j++) {
            matrix[targetRow][j] -= coefficient * matrix[sourceRow][j];
        }
    }

public:
    // Конструктор
    GaussJordanSolver(const vector<vector<double>>& inputMatrix) {
        n = inputMatrix.size();
        m = (n > 0) ? inputMatrix[0].size() : 0;
        matrix = inputMatrix;
    }

    // Основной метод решения
    pair<vector<double>, int> solve() {
        int row = 0;
        int col = 0;
        
        // Прямой ход - приведение к диагональному виду
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
        
        // Проверка решений
        return extractSolution();
    }

    // Извлечение решения из преобразованной матрицы
    pair<vector<double>, int> extractSolution() {
        vector<double> solution(m - 1, 0.0);
        
        // Проверяем на противоречивость (строка вида 0 0 0 ... | b, где b != 0)
        for (int i = 0; i < n; i++) {
            bool allZero = true;
            for (int j = 0; j < m - 1; j++) {
                if (!isZero(matrix[i][j])) {
                    allZero = false;
                    break;
                }
            }
            if (allZero && !isZero(matrix[i][m - 1])) {
                return {solution, 0}; // Нет решений
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
            if (hasNonZero) rank++;
        }
        
        if (rank < m - 1 && rank < n) {
            return {solution, -1}; // Бесконечно много решений
        }
        
        // Извлекаем единственное решение
        for (int i = 0; i < min(n, m - 1); i++) {
            bool found = false;
            for (int j = 0; j < m - 1; j++) {
                if (!isZero(matrix[i][j])) {
                    solution[j] = matrix[i][m - 1];
                    found = true;
                    break;
                }
            }
            if (!found && i < m - 1) {
                solution[i] = 0; // Свободная переменная
            }
        }
        
        return {solution, 1}; // Единственное решение
    }

    // Вывод матрицы
    void printMatrix(const string& title = "Матрица:") {
        cout << title << endl;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (j == m - 1) {
                    cout << " | ";
                }
                cout << setw(10) << fixed << setprecision(4) << matrix[i][j];
            }
            cout << endl;
        }
        cout << endl;
    }

    // Вывод решения
    static void printSolution(const vector<double>& solution, int solutionType) {
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
};

// Функция для ввода матрицы с клавиатуры
vector<vector<double>> inputMatrix() {
    int n, m;
    cout << "Введите количество уравнений: ";
    cin >> n;
    cout << "Введите количество переменных: ";
    cin >> m;
    
    // Матрица коэффициентов + столбец свободных членов
    vector<vector<double>> matrix(n, vector<double>(m + 1));
    
    cout << "\nВведите коэффициенты матрицы (по строкам):" << endl;
    cout << "Формат: a11 a12 ... a1m b1" << endl;
    
    for (int i = 0; i < n; i++) {
        cout << "Строка " << i + 1 << ": ";
        for (int j = 0; j <= m; j++) {
            cin >> matrix[i][j];
        }
    }
    
    return matrix;
}

// Примеры тестовых систем
vector<vector<double>> getExample1() {
    // Система с единственным решением:
    // 2x + y - z = 8
    // -3x - y + 2z = -11
    // -2x + y + 2z = -3
    return {
        {2, 1, -1, 8},
        {-3, -1, 2, -11},
        {-2, 1, 2, -3}
    };
}

vector<vector<double>> getExample2() {
    // Система без решений:
    // x + y = 1
    // x + y = 2
    return {
        {1, 1, 1},
        {1, 1, 2}
    };
}

vector<vector<double>> getExample3() {
    // Система с бесконечным числом решений:
    // x + y + z = 6
    // 2x + y + 3z = 14
    // x + 2y + 2z = 11
    return {
        {1, 1, 1, 6},
        {2, 1, 3, 14},
        {1, 2, 2, 11}
    };
}

int main() {
    setlocale(LC_ALL, "Russian");
    
    cout << "МЕТОД ГАУССА-ЖОРДАНА ДЛЯ РЕШЕНИЯ СИСТЕМ ЛИНЕЙНЫХ УРАВНЕНИЙ" << endl;
    cout << "==========================================================" << endl;
    
    int choice;
    cout << "\nВыберите способ ввода:" << endl;
    cout << "1. Ввести матрицу с клавиатуры" << endl;
    cout << "2. Использовать пример 1 (единственное решение)" << endl;
    cout << "3. Использовать пример 2 (нет решений)" << endl;
    cout << "4. Использовать пример 3 (бесконечно много решений)" << endl;
    cout << "Ваш выбор: ";
    cin >> choice;
    
    vector<vector<double>> matrix;
    
    switch (choice) {
        case 1:
            matrix = inputMatrix();
            break;
        case 2:
            matrix = getExample1();
            cout << "\nПример 1:" << endl;
            cout << "2x + y - z = 8" << endl;
            cout << "-3x - y + 2z = -11" << endl;
            cout << "-2x + y + 2z = -3" << endl;
            break;
        case 3:
            matrix = getExample2();
            cout << "\nПример 2:" << endl;
            cout << "x + y = 1" << endl;
            cout << "x + y = 2" << endl;
            break;
        case 4:
            matrix = getExample3();
            cout << "\nПример 3:" << endl;
            cout << "x + y + z = 6" << endl;
            cout << "2x + y + 3z = 14" << endl;
            cout << "x + 2y + 2z = 11" << endl;
            break;
        default:
            cout << "Неверный выбор. Используем пример 1." << endl;
            matrix = getExample1();
    }
    
    // Создаем решатель
    GaussJordanSolver solver(matrix);
    
    // Выводим исходную матрицу
    solver.printMatrix("Исходная матрица:");
    
    // Решаем систему
    auto [solution, solutionType] = solver.solve();
    
    // Выводим преобразованную матрицу
    solver.printMatrix("Приведенная матрица:");
    
    // Выводим результат
    GaussJordanSolver::printSolution(solution, solutionType);
    
    return 0;
}