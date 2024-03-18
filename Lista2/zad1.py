import numpy as np
import matplotlib.pyplot as plt

num_points = 6

yfunc = lambda x : x**3 - 4*x**2 + 3*x + 1
xdata = np.linspace(0.0, 3.0, num=num_points)
ydata = [yfunc(x) for x in xdata]

realx = np.linspace(0.0, 3.0, num=num_points * 100)
realy = [yfunc(x) for x in realx]

def linear_interpolation(x, y):
    interpolatedx = []
    interpolatedy = []
    for iter in range(1, len(x)):
        x0 = x[iter - 1]
        x1 = x[iter]
        y0 = y[iter - 1]
        y1 = y[iter]
        for new in range(100):
            currentx = x0 + ((x1 - x0) * new / 100)
            currenty = y0 + ((y1 - y0) * (currentx - x0) / (x1 - x0))
            interpolatedx.append(currentx)
            interpolatedy.append(currenty)
    return interpolatedx, interpolatedy

interpolatedx, interpolatedy = linear_interpolation(xdata, ydata)

plt.scatter(realx, realy, color="red", alpha=0.5, s=2, label="Funkcja")
plt.scatter(interpolatedx, interpolatedy, color="tab:blue", alpha=0.5, s=2, label="Interpolacja")
plt.scatter(xdata, ydata, color="black", label="Próbki")
plt.legend()
plt.show()