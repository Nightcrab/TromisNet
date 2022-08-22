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

plt.plot(points[0], points[1], label="10^-6")
points = loadxy("./rewards3.txt")

plt.plot(points[0], points[1], label="30^-6")
points = loadxy("./rewards1.txt")

plt.plot(points[0], points[1], label="50^-6")
points = loadxy("./rewards4.txt")

plt.plot(points[0], points[1], label="80^-6")
points = loadxy("./rewards2.txt")

plt.plot(points[0], points[1], label="10^-5")
plt.ylabel("win rate")
plt.xlabel("updates")
plt.legend()
plt.title("LR comparison, batch_size=512")
plt.show()