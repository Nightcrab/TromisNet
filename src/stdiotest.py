from subprocess import Popen, PIPE, STDOUT
import time

p = Popen(["echo.exe"], stdout=PIPE, stdin=PIPE, stderr=PIPE)

out = p.stdout.readline().decode()
print(out)

tic = time.perf_counter()

for x in range(0,1000):
    p.stdin.write(b'9\n')
    p.stdin.flush()
    out = p.stdout.readline().decode()

toc = time.perf_counter()

p.stdin.write(b'0\n')
p.stdin.flush()
out = p.stdout.readline().decode()
print(out)

print(str(1000/(toc-tic)) + " exchanges per second")