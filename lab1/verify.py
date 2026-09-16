import numpy as np

nA, *elementsA = open("matrixA.txt").read().split()
nB, *elementsB = open("matrixB.txt").read().split()

N = int(nA)
A = np.array(elementsA, dtype=float).reshape(N, N)
B = np.array(elementsB, dtype=float).reshape(N, N)

with open("result.txt") as f:
    lines = f.read().splitlines()
result_elements = " ".join(lines[-N:]).split()
cpp_result = np.array(result_elements, dtype=float).reshape(N, N)

python_result = np.dot(A, B)

print("\nVerification of results:")
if np.allclose(cpp_result, python_result, atol=0.01):
    print("SUCCESS! The matrices match.")
else:
    print("ERROR! Matrices do NOT match.")