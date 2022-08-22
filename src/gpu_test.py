import torch
import time

dim = [9513, 22120]

startTime = time.time()

a = torch.Tensor(dim[0], dim[1])
b = torch.Tensor(dim[1], dim[0])

c = a.mm(b)

endTime = time.time()
print("CPU Runtime: %.5f [sec]"%(endTime-startTime))

startTime = time.time()

ac = a.cuda()
bc = b.cuda()
cc = ac.mm(bc)

endTime = time.time()
print("GPU Runtime: %.5f [sec]"%(endTime-startTime))