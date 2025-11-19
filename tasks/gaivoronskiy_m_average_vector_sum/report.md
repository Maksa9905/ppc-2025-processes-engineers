# Average Vector Mean (Gaivoronskiy M.)

- Student: Gaivoronskiy Maksim, group 3823Б1ПР1
- Technology: SEQ, MPI
- Variant: 1

## 1. Introduction
Цель работы — реализовать задачу вычисления среднего значения элементов вектора двумя способами: последовательно и с использованием MPI. В отчёте описаны алгоритмы, схема распараллеливания и измерения производительности, достаточные для воспроизведения результатов.

## 2. Problem Statement
На вход поступает вектор вещественных чисел фиксированной длины *N*. Требуется вернуть среднее арифметическое:  
`avg = (1 / N) * Σ_i v[i]`. Ограничения: `N > 0`, значения хранятся в `std::vector<double>`, результат — `double`.

## 3. Baseline Algorithm (Sequential)
Базовый алгоритм выполняет один проход по вектору, аккумулирует сумму в `double` (через `std::accumulate`) и делит её на `N`. Важные детали:
- валидация проверяет непустой вход;
- данные копируются на этапе `PreProcessing` для единообразия с MPI‑версией;
- на этапе `PostProcessing` рассчитывается окончательное среднее и проверяется `std::isfinite`.

## 4. Parallelization Scheme
MPI‑версия использует одноразовый Scatter/Reduce‑паттерн:
- **Распределение данных**: корневой процесс (ранг 0) делит исходный вектор на почти равные блоки (разница не превышает 1 элемент) и рассылает их через `MPI_Scatterv`.
- **Локальные вычисления**: каждый ранг суммирует свой блок.
- **Глобальное объединение**: `MPI_Allreduce` собирает локальные суммы и возвращает итоговую сумму на все ранги.
- **Финализация**: корень (и все ранги) делят глобальную сумму на `N` и записывают результат.
Топология — полносвязная через `MPI_COMM_WORLD`, дополнительных синхронизаций нет.

## 5. Implementation Details
- Код расположен в `seq/src/ops_seq.cpp` и `mpi/src/ops_mpi.cpp`, заголовки — в `seq/include` и `mpi/include`.
- Типы объявлены в `common/include/common.hpp`: вход — `std::vector<double>`, выход — `double`.
- MPI‑класс хранит: исходные данные на ранге 0, локальный буфер, размеры, суммы, а также кэш ранга/размера мира.
- Обработка краевых случаев: пустой вектор (валидатор отклоняет), нечисловые значения (постобработка отвергает `inf/nan`).
- Для perf‑тестов размер входа задаётся переменной `PPC_AVG_VEC_SIZE` (по умолчанию 1 000 000).

## 6. Experimental Setup
- **Hardware / OS**: Apple M1 (8 ядер, 4P+4E), 16 GB LPDDR4X, macOS 14.5 (Darwin 23.5.0).
- **Toolchain**: CMake + Apple Clang 15.0.0, `CMAKE_BUILD_TYPE=Release`.
- **Environment variables**:
  - SEQ тесты: стандартные (`PPC_AVG_VEC_SIZE` по необходимости).
  - MPI тесты: запуск `mpirun -np 4` (4 процесса), переменная `PPC_AVG_VEC_SIZE` передаётся через `-x`.
- **Данные**: синтетический детерминированный вектор `v[i] = (i mod 101) - 50`, формируется в perf‑тесте `tests/performance/main.cpp`.
- **Команды для воспроизведения**:
  ```bash
  cmake --build build -j4
  ./build/bin/ppc_func_tests '--gtest_filter=*gaivoronskiy_m_average_vector_sum*'
  mpirun -np 4 ./build/bin/ppc_func_tests '--gtest_filter=*gaivoronskiy_m_average_vector_sum_mpi*'
  # Perf, базовый размер
  ./build/bin/ppc_perf_tests '--gtest_filter=*gaivoronskiy_m_average_vector_sum_seq*'
  mpirun -np 4 ./build/bin/ppc_perf_tests '--gtest_filter=*gaivoronskiy_m_average_vector_sum_mpi*'
  # Perf, увеличенный размер
  PPC_AVG_VEC_SIZE=5000000 ./build/bin/ppc_perf_tests '--gtest_filter=*seq*'
  mpirun -np 4 -x PPC_AVG_VEC_SIZE=5000000 ./build/bin/ppc_perf_tests '--gtest_filter=*mpi*'
  ```

## 7. Results and Discussion

### 7.1 Correctness
- Функциональные тесты (`ppc_func_tests`) покрывают пять наборов значений различной длины и проверяют совпадение среднего с эталонным значением в диапазоне `1e-9`.
- MPI‑вариант прогоняется под `mpirun -np 4`, все тесты завершаются без расхождений.

### 7.2 Performance
Все значения времени — усреднённые выводы `ppc_perf_tests` (режим `pipeline`). Speedup рассчитывается относительно последовательного `pipeline`‑времени на том же размере данных; эффективность = `speedup / proc_count`.

**Размер = 1 000 000 элементов (по умолчанию)**

| Mode | Processes | Time, s | Speedup | Efficiency |
|------|-----------|---------|---------|------------|
| seq (pipeline) | 1 | 0.00155 | 1.00 | N/A |
| mpi (pipeline) | 4 | 0.00215 | 0.72 | 18% |

Дополнительно: `task_run` режим даёт 0.00106 с (SEQ) и 0.00189 с (MPI). Потери MPI обусловлены стоимостью Scatter/Allreduce при малом объёме работы на процесс.

**Размер = 5 000 000 элементов (`PPC_AVG_VEC_SIZE=5000000`)**

| Mode | Processes | Time, s | Speedup | Efficiency |
|------|-----------|---------|---------|------------|
| seq (pipeline) | 1 | 0.00727 | 1.00 | N/A |
| mpi (pipeline) | 4 | 0.00898 | 0.81 | 20% |

На крупном наборе доля коммуникаций уменьшается, однако ускорение всё ещё < 1. Для дальнейшего масштабирования необходимы большие объёмы данных и/или более производительная сеть.

**Размер = 20 000 000 элементов (`PPC_AVG_VEC_SIZE=20000000`)**

| Mode | Processes | Time, s | Speedup | Efficiency |
|------|-----------|---------|---------|------------|
| seq (pipeline) | 1 | 0.03022 | 1.00 | N/A |
| mpi (pipeline) | 4 | 0.02982 | 1.01 | 25% |

На 20 млн элементов MPI и SEQ достигают схожего времени: вычислительная нагрузка доминирует над передачами, но эффективность остаётся невысокой из‑за малого числа процессов и использования общего узла.

**Размер = 100 000 000 элементов (`PPC_AVG_VEC_SIZE=100000000`)**

| Mode | Processes | Time, s | Speedup | Efficiency |
|------|-----------|---------|---------|------------|
| seq (pipeline) | 1 | 0.16902 | 1.00 | N/A |
| mpi (pipeline) | 4 | 0.16057 | 1.05 | 26% |
| mpi (pipeline) | 8 | 0.32813 | 0.52 | 6% |

Только на 100 млн элементов MPI впервые обгоняет последовательную версию (~5 % при 4 процессах), однако при увеличении числа процессов до 8 вычисления становятся медленнее последовательных (speedup < 1) из‑за конкуренции за память и необходимости обменивать больше блоков между процессами одного узла.

## 8. Conclusions
- Реализованы и проверены SEQ и MPI версии среднего значения вектора.
- Корректность подтверждена функциональными тестами под `gtest`.
- MPI‑вариант даёт ограниченное ускорение из-за доминирующих коммуникационных накладных расходов на малых размерах; при увеличении размера разница сокращается, но преимущества не достигаются на 4 рангах и одном узле Apple M1.
- Улучшения: использование `MPI_Reduce` на корне вместо `Allreduce`, передача данных в `double` без копий, анализ блокировок на больших рангах.

## 9. References
1. MPI Forum, *MPI: A Message-Passing Interface Standard Version 4.0*, https://www.mpi-forum.org
2. ISO/IEC 14882:2020, *Programming Languages — C++*.

## Appendix (Optional)
```cpp
// Отрывок вычисления локальной суммы в MPI-варианте
local_sum_ = std::accumulate(local_buffer_.begin(), local_buffer_.end(), 0.0);
MPI_Allreduce(&local_sum_, &global_sum_, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
GetOutput() = global_sum_ / static_cast<double>(total_size_);
```

