import matplotlib.pyplot as plt
import numpy as np

procs = np.array([1, 2, 4, 8])

data_time = {
    200:  np.array([0.010906, 0.006294, 0.002842, 0.001645]),
    400:  np.array([0.126024, 0.062416, 0.031869, 0.016438]),
    800:  np.array([0.850862, 0.562955, 0.312944, 0.218947]),
    1200: np.array([3.011620, 1.798687, 1.319525, 0.774712]),
    1600: np.array([11.780742, 5.909463, 3.346051, 2.450282]),
    2000: np.array([24.757467, 12.382097, 5.410097, 2.926085])
}

markers = {200: 'o', 400: 's', 800: '^', 1200: 'd', 1600: 'x', 2000: 'v'}

# === ГРАФИК 1: ВРЕМЯ ВЫПОЛНЕНИЯ ===
plt.figure(figsize=(9, 5))
for N, times in data_time.items():
    plt.plot(procs, times, marker=markers[N], label=f'N={N}')
plt.title('Время выполнения программы')
plt.xlabel('Число процессов')
plt.ylabel('Время (сек, лог. шкала)')
plt.yscale('log')
plt.xticks(procs)
plt.grid(True, which="both", linestyle=':', alpha=0.5)
plt.legend(title="Размер N")
plt.tight_layout()
plt.savefig('1_mpi_time.png', dpi=300)
plt.show()

# === ГРАФИК 2: УСКОРЕНИЕ (SPEEDUP) ===
plt.figure(figsize=(9, 5))
plt.plot(procs, procs, color='gray', linestyle='--', linewidth=2, label='Идеальное (S=p)')

for N, times in data_time.items():
    speedup = times[0] / times  # T1 / Tp
    plt.plot(procs, speedup, marker=markers[N], label=f'N={N}')

plt.title('Ускорение параллельной программы (Speedup)')
plt.xlabel('Число процессов')
plt.ylabel('Коэффициент ускорения S')
plt.xticks(procs)
plt.yticks(range(0, 10))
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend(title="Размер N")
plt.tight_layout()
plt.savefig('2_mpi_speedup.png', dpi=300)
plt.show()

# === ГРАФИК 3: ЭФФЕКТИВНОСТЬ (EFFICIENCY) ===
plt.figure(figsize=(9, 5))
plt.axhline(y=100, color='gray', linestyle='--', linewidth=2, label='Идеальная (E=100%)')

for N, times in data_time.items():
    speedup = times[0] / times
    efficiency = (speedup / procs) * 100  # (S / p) * 100%
    plt.plot(procs, efficiency, marker=markers[N], label=f'N={N}')

plt.title('Эффективность использования ядер (Efficiency)')
plt.xlabel('Число процессов')
plt.ylabel('Эффективность E (%)')
plt.xticks(procs)
plt.ylim(0, 110)
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend(title="Размер N")
plt.tight_layout()
plt.savefig('3_mpi_efficiency.png', dpi=300)
plt.show()
