import numpy as np
from matplotlib import pyplot as plt
from random import uniform
from sklearn import linear_model

size = 500
func = lambda k: k + 10

x = [uniform(0, 100) for i in range(size)]
y = [func(x[i]) + np.random.normal(0, 1, 1) * 10 for i in range(size)]

x_train = x[:-(int(0.2 * size))]
x_test = x[-(int(0.2 * size)):]

y_train = y[:-(int(0.2 * size))]
y_test = y[-(int(0.2 * size)):]

regr = linear_model.ElasticNet()
regr.fit(np.array(x_train).reshape(-1, 1), np.array(y_train).reshape(-1, 1))
y_pred = regr.predict(np.array(x_test).reshape(-1, 1))

print(f"Model ma parametry y = {(max(y_pred) - min(y_pred)) / 100} * x + {min(y_pred)}")
plt.scatter(x_test, y_test, s=10, color="tab:blue")
plt.plot(x_test, y_pred, color="tab:orange")
plt.show()