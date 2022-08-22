import matplotlib.pyplot as plt


points = []

with open("plot1.txt", 'r') as file:
    points = file.read().split()

points = [float(x) for x in points]

plt.plot(points)
plt.ylabel("loss")
plt.xlabel("epoch")
plt.show()