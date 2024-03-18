import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from random import uniform, randint
from datetime import datetime, timedelta
from sklearn import preprocessing

def generate_timeseries(amount):
    timeseries = [[datetime.today(), 10, 20]]
    for i in range(1, amount):
        temp, temp2 = -1, -1
        while (temp >= 20 or temp <= -10) or (temp2 >= 60 or temp2 <= 20):
            temp = timeseries[i-1][1] + uniform(-1, 1)
            temp2 = timeseries[i-1][2] + uniform(-2, 2)
        timeseries.append([timeseries[i-1][0] + timedelta(minutes=randint(1, 15)), temp, temp2])
    return np.array(timeseries)

ts = generate_timeseries(1000)

normalizer = preprocessing.Normalizer()
temp_list = [[i[1], i[2]] for i in ts]
temp_norm = normalizer.transform(temp_list)
norm_ts = [[i[0]] for i in ts]
for n in range(len(norm_ts)):
    norm_ts[n].append(temp_norm[n][0])
    norm_ts[n].append(temp_norm[n][1])

standarizer = preprocessing.StandardScaler()
temp_stan = standarizer.fit_transform(temp_list)
stan_ts = [[i[0]] for i in ts]
for n in range(len(stan_ts)):
    stan_ts[n].append(temp_stan[n][0])
    stan_ts[n].append(temp_stan[n][1])

scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
temp_scal = scaler.fit_transform(temp_list)
scal_ts = [[i[0]] for i in ts]
for n in range(len(scal_ts)):
    scal_ts[n].append(temp_scal[n][0])
    scal_ts[n].append(temp_scal[n][1])

style.use('ggplot')
fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, figsize=(16,7), sharex=True)
fig.autofmt_xdate(rotation=45)
ax1.plot([i[0] for i in ts], [i[1] for i in ts], color='tab:orange')
ax1.plot([i[0] for i in ts], [i[2] for i in ts], color='tab:blue')
ax1.set_title("Szereg czasowy")
ax2.plot([i[0] for i in norm_ts], [i[1] for i in norm_ts], color='tab:orange')
ax2.plot([i[0] for i in norm_ts], [i[2] for i in norm_ts], color='tab:blue')
ax2.set_title("Znormalizowany szereg czasowy")
ax3.plot([i[0] for i in stan_ts], [i[1] for i in stan_ts], color='tab:orange')
ax3.plot([i[0] for i in stan_ts], [i[2] for i in stan_ts], color='tab:blue')
ax3.set_title("Standaryzowany szereg czasowy")
ax4.plot([i[0] for i in scal_ts], [i[1] for i in scal_ts], color='tab:orange')
ax4.plot([i[0] for i in scal_ts], [i[2] for i in scal_ts], color='tab:blue')
ax4.set_title("Skalowany szereg czasowy")

plt.show()