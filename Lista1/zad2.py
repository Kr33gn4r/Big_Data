import numpy as np
from matplotlib import pyplot as plt
from random import uniform
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

size = 500
func = lambda k: k ** 2 - 3 * k + 7

x = [uniform(-5, 10) for i in range(size)]
y = [func(x[i]) + np.random.normal(0, 1, 1) * 5 for i in range(size)]

x_train = x[:-(int(0.2 * size))]
x_test = x[-(int(0.2 * size)):]

y_train = y[:-(int(0.2 * size))]
y_test = y[-(int(0.2 * size)):]

model = Pipeline([('poly', PolynomialFeatures(degree=2)),
                  ('linear', linear_model.LinearRegression(fit_intercept=False))])

model = model.fit(np.array(x_train).reshape(-1, 1), np.array(y_train).reshape(-1, 1))
y_pred = model.predict(np.sort(np.array(x_test).reshape(-1, 1), axis=0))

q = min(y_pred)[0]
p = np.sort(np.array(x_test).reshape(-1, 1), axis=0)[np.argmin(y_pred)][0]
newy = y_pred[0][0]
newx = np.sort(np.array(x_test).reshape(-1, 1), axis=0)[0][0]
a = (newy - q) / ((newx - p) ** 2)

print(f"Model ma parametry y = {a}(x - {p})^2 + {q}")
plt.scatter(x_test, y_test, s=10, color="tab:blue")
plt.plot(np.sort(np.array(x_test).reshape(-1, 1), axis=0), y_pred, color="tab:orange")
plt.show()
