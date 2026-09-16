import time
import random
import matplotlib.pyplot as plt

SIZES = [10, 50, 100, 150, 200, 250, 300]
gflops_list = []

print("Начинаем тестирование производительности процессора...")

for n in SIZES:
    print(f"Считаем размер {n}x{n}...", end="", flush=True)
    
    A = [[random.random() for _ in range(n)] for _ in range(n)]
    B = [[random.random() for _ in range(n)] for _ in range(n)]
    C = [[0.0 for _ in range(n)] for _ in range(n)]
    
    t0 = time.perf_counter()
    
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
                
    dt = time.perf_counter() - t0
    
    gflops = (2 * (n ** 3)) / (dt * 1e9) if dt > 0 else 0
    gflops_list.append(gflops)
    print(f" Готово! Скорость: {gflops:.4f} GFLOPS")

print("\nОтрисовка графика...")
plt.figure(figsize=(9, 5))
plt.plot(SIZES, gflops_list, 'r-o', linewidth=2, label="Обычное умножение")
plt.title("Производительность процессора при умножении матриц")
plt.xlabel("Размер матрицы N")
plt.ylabel("Скорость вычислений (GFLOPS)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

plt.savefig("performance.png", dpi=300)
print("Файл 'performance.png' успешно сохранен в вашу папку лабы!")
plt.show()