import os, subprocess, time, random
import matplotlib.pyplot as plt

EXE = "./lab1.exe" if os.name == "nt" else "./lab1"
subprocess.run(["g++", "-O3", "lab1.cpp", "-o", EXE])

SIZES = [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
gflops_list = []

for n in SIZES:
    for name in ["matrixA.txt", "matrixB.txt"]:
        with open(name, "w") as f:
            f.write(f"{n}\n" + "\n".join(" ".join(f"{random.random():.2f}" for _ in range(n)) for _ in range(n)))
    t0 = time.perf_counter()
    subprocess.run([EXE], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    dt = time.perf_counter() - t0
    
    gflops = (2 * (n ** 3)) / (dt * 1e9) if dt > 0 else 0
    gflops_list.append(gflops)
    print(f"N = {n}: {gflops:.2f} GFLOPS")

plt.plot(SIZES, gflops_list, 'r-o', linewidth=2)
plt.title("Производительность процессора на lab1.cpp")
plt.xlabel("Размер матрицы N")
plt.ylabel("Производительность (GFLOPS)")
plt.grid(True)
plt.savefig("performance.png", dpi=300)
plt.show()
