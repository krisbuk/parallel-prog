import numpy as np
import sys

if len(sys.argv) < 4:
    print("Usage: python check.py <N> <A.txt> <B.txt>")
    sys.exit(1)

n = int(sys.argv[1])
A = np.loadtxt(sys.argv[2])
B = np.loadtxt(sys.argv[3])
C = np.loadtxt("result.txt")

expected = np.dot(A, B)

expected_rounded = np.round(expected, 4)

diff = np.linalg.norm(C - expected_rounded)

print(f"Проверка для N={n} завершена.")
if diff < 1e-2:
    print("РЕЗУЛЬТАТ: Успешно (матрицы совпадают)")
else:
    print(f"РЕЗУЛЬТАТ: Ошибка! Норма разности: {diff}")