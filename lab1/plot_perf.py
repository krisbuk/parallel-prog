import matplotlib.pyplot as plt

sizes = ['10', '100', '200', '300', '400', '500']
gflops = [0.00, 2.00, 4.00, 0.75, 3.05, 2.32]

plt.figure(figsize=(7, 4.5))

plt.bar(sizes, gflops, color="purple", width=0.5, alpha=0.85)

plt.title("Matrix Multiplication Performance")
plt.xlabel("Matrix Size (n x n)")
plt.ylabel("Performance (GFLOPS)")
plt.grid(axis='y', linestyle="--", alpha=0.5)

for i, y in enumerate(gflops):
    plt.text(i, y + 0.08, f"{y:.2f}", ha="center", fontsize=9, fontweight="bold")

plt.tight_layout()

plt.savefig("performance.png", dpi=150)
plt.show()