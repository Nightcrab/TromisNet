import torch
import time
import matplotlib.pyplot as plt

cpu_time = []
gpu_time = []

matrix = [[0.7385] * 50 ] * 50

def stack(a,b):
    n = []
    for i in range(b):
        n.append(a.copy())
    return n

for x in range(10):

    cpu_time.append(0)
    gpu_time.append(0)

    for y in range(2 ** (10-x)):
        A = torch.tensor(stack(matrix, 2 ** x))
        t = time.perf_counter()
        A = torch.matmul(A,A)
        cpu_time[len(cpu_time)-1] += time.perf_counter() - t

    for y in range(2 ** (10-x)):
        A = torch.tensor(stack(matrix, 2 ** x))
        t = time.perf_counter()
        A.cuda()
        A = torch.matmul(A,A)
        gpu_time[len(gpu_time)-1] += time.perf_counter() - t


plt.plot(cpu_time, label="CPU")
plt.plot(gpu_time, label="GPU")
plt.ylabel("time (s)")
plt.xlabel("log_2 batch size")
plt.legend()
plt.show()