import os
import time
import random
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt

SIZES = [200, 400, 800, 1200, 1600, 2000]
THREADS = [1, 2, 4, 8]

def worker_multiply_row(args):
    """Функция для одного вычислительного потока: умножает одну строку матрицы A на матрицу B"""
    row_A, B, N = args
    result_row = [0.0] * N
    for j in range(N):
        sub_sum = 0.0
        for k in range(N):
            sub_sum += row_A[k] * B[k][j]
        result_row[j] = sub_sum
    return result_row

def parallel_matrix_multiply(A, B, N, num_threads):
    """Параллельное умножение матриц (аналог OpenMP)"""
    pool_args = [(A[i], B, N) for i in range(N)]
    with Pool(processes=num_threads) as pool:
        C = pool.map(worker_multiply_row, pool_args)
    return C

def generate_matrices(n):
    """Генерация случайных матриц в памяти"""
    A = [[random.uniform(-10, 10) for _ in range(n)] for _ in range(n)]
    B = [[random.uniform(-10, 10) for _ in range(n)] for _ in range(n)]
    return A, B

if __name__ == '__main__':
    raw_data = {size: {} for size in SIZES}

    print("=== НАЧАЛО АВТОМАТИЧЕСКОГО ПАРАЛЛЕЛЬНОГО ТЕСТИРОВАНИЯ ===")
    print(f"Обнаружено ядер на вашем процессоре: {cpu_count()}")

    for n in SIZES:
        print(f"\n[Матрица {n}x{n}] Генерируем данные...")
        A, B = generate_matrices(n)
        
        for t in THREADS:
            print(f"  -> Расчет: {t} поток(ов)... ", end="", flush=True)
            
            t0 = time.perf_counter()
            C = parallel_matrix_multiply(A, B, n, t)
            dt_ms = (time.perf_counter() - t0) * 1000
            
            raw_data[n][t] = dt_ms
            print(f"Успешно за {dt_ms:.1f} мс")

    print("\n=== ТЕСТЫ ЗАВЕРШЕНЫ. ТАБЛИЦА РЕЗУЛЬТАТОВ ДЛЯ ОТЧЕТА ===")
    print(f"{'Размер':<8} | {'Потоки':<7} | {'Время (мс)':<12} | {'Ускорение':<10} | {'Эффективность':<13}")
    print("-" * 60)

    for n in SIZES:
        t1_time = raw_data[n].get(1)
        for t in THREADS:
            t_time = raw_data[n].get(t)
            speedup = t1_time / t_time
            efficiency = (speedup / t) * 100
            print(f"{n:<8} | {t:<7} | {t_time:<12.2f} | {speedup:<10.2f} | {efficiency:.1f}%")
        print("-" * 60)

    print("\nСохраняем графические зависимости для отчета...")

    plt.figure(figsize=(10, 5))
    for t in THREADS:
        times = [raw_data[n][t] for n in SIZES]
        plt.plot(SIZES, times, marker='o', label=f"Потоки: {t}")
    plt.title("Зависимость времени вычислений от размера матрицы")
    plt.xlabel("Размер матрицы (N)")
    plt.ylabel("Время выполнения (мс)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.savefig("1_time_vs_size.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(THREADS, THREADS, 'k--', alpha=0.7, label="Идеальное ускорение")
    for n in SIZES:
        t1 = raw_data[n].get(1)
        speedups = [t1 / raw_data[n][t] for t in THREADS]
        plt.plot(THREADS, speedups, marker='s', label=f"Размер: {n}x{n}")
    plt.title("Ускорение работы программы (Speedup)")
    plt.xlabel("Количество потоков (p)")
    plt.ylabel("Ускорение (S_p)")
    plt.xticks(THREADS)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.savefig("2_speedup_graphs.png", dpi=300)
    plt.close()

    print("Готово! Графики сохранены под именами '1_time_vs_size.png' и '2_speedup_graphs.png'.")
