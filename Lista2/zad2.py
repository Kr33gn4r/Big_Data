import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate

num_points = 4

yfunc = lambda x : x**3 - 4*x**2 + 3*x + 1
xdata = np.linspace(0.0, 3.0, num=num_points)
ydata = [yfunc(x) for x in xdata]

realx = np.linspace(0.0, 3.0, num=num_points * 100)
realy = [yfunc(x) for x in realx]

lagrange_interpolation = scipy.interpolate.lagrange(xdata, ydata)

interpolatedx = np.linspace(0.0, 3.0, num=num_points * 100)
interpolatedy = lagrange_interpolation(interpolatedx)

plt.scatter(realx, realy, color="red", alpha=0.5, s=2, label="Funkcja")
plt.scatter(interpolatedx, interpolatedy, color="tab:blue", alpha=0.5, s=2, label="Interpolacja")
plt.scatter(xdata, ydata, color="black", label="Próbki")
plt.legend()
plt.show()