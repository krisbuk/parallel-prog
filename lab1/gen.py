import numpy as np

N = 500  

def save_matrix(filename, size):
    matrix = np.random.uniform(1.0, 100.0, size=(size, size))

    with open(filename, "w") as f:
        f.write(f"{size}\n")
        for row in matrix:
            f.write(" ".join(f"{num:.2f}" for num in row) + "\n")


print(f"Generating matrix {N}x{N}...")
save_matrix("matrixA.txt", N)
save_matrix("matrixB.txt", N)
print("Files matrixA.txt and matrixB.txt were successfully created!")