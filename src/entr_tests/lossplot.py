import matplotlib.pyplot as plt

def loadxy (path):
    points = []
    x = []
    y = []
    with open(path, 'r') as file:
        points = file.readlines()

    for i in range(len(points)-1):
        x_,y_ = [float(k) for k in points[i+1].split()]
        x.append(x_)
        y.append(y_)

    return x, y

points = loadxy("./rewards0.txt")

plt.plot(points[0], points[1], label="0.001")
points = loadxy("./rewards1.txt")

plt.plot(points[0], points[1], label="0.005")
points = loadxy("./rewards2.txt")

plt.plot(points[0], points[1], label="0.01")
points = loadxy("./rewards3.txt")

plt.plot(points[0], points[1], label="no entropy")
plt.ylabel("win rate")
plt.xlabel("updates")
plt.legend()
plt.title("entropy comparison, batch_size=512, lr=0.000008")
plt.show()