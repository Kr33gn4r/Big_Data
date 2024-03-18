import matplotlib.pyplot as plt
import warnings
import numpy as np
import statistics
import nolds
import pandas as pd
from scipy import stats
from matplotlib import style
from random import uniform, randint
from datetime import datetime, timedelta
from hurst import compute_Hc
from sklearn import preprocessing

warnings.filterwarnings("ignore")
style.use('ggplot')

def generate_timeseries(amount):
    ts = pd.DataFrame({'dt':[datetime.today()], 'd1':[10], 'd2':[30]})
    for i in range(1, amount):
        temp, temp2 = -1, -1
        while (temp >= 20 or temp <= 0) or (temp2 >= 60 or temp2 <= 20):
            temp = round(ts['d1'][i-1] + uniform(-1, 1), 1)
            temp2 = round(ts['d2'][i-1] + uniform(-2, 2), 3)
        ts = pd.concat([ts, pd.DataFrame({'dt':[ts['dt'][i-1] + timedelta(minutes=randint(1, 15))], 'd1':[temp], 'd2':[temp2]})], ignore_index=True)
    return ts

def generate_parameters(tslist, col, strpoint=0):
    data = {}
    data['col'] = [col]
    data['strpoint'] = [int(strpoint)]
    data['avg'] = [statistics.fmean(tslist)]
    data['stdev'] = [statistics.stdev(tslist)]
    data['vmin'] = [min(tslist)]
    data['lmin'] = [strpoint + tslist.index(data['vmin'][0])]
    data['vmax'] = [max(tslist)]
    data['lmax'] = [strpoint + tslist.index(data['vmax'][0])]
    data['median'] = [statistics.median(tslist)]
    data['kurt'] = [stats.kurtosis(tslist)]
    data['skew'] = [stats.skew(tslist)]

    val, count = np.unique(tslist, return_counts=True)
    pk = count / len(tslist)
    data['entropy'] = [stats.entropy(pk)]
    data['hurst'] = [compute_Hc(tslist)[0]]
    data['fract'] = [nolds.corr_dim(tslist, 2)]
    data['lyap'] = [nolds.lyap_r(tslist, 2)]
    return data

def parameter_dataframe(ts, windows=1):
    ps = pd.DataFrame({
        'col':[],
        'strpoint':[],
        'avg':[],
        'stdev':[],
        'vmin':[],
        'lmin':[],
        'vmax':[],
        'lmax':[],
        'median':[],
        'kurt':[],
        'skew':[],
        'entropy':[],
        'hurst':[],
        'fract':[],
        'lyap':[]
    })
    for col in ts:
        if ts[col].dtypes == 'float64':
            for inc in range(windows):
                temp = generate_parameters(ts[col].to_list()[int(inc * len(ts.index)/windows) : int((inc+1) * len(ts.index)/windows)], col, int(inc * len(ts.index)/windows))
                ps = pd.concat([ps, pd.DataFrame.from_dict(temp)], ignore_index=True)
    return ps

def normalize_dataframe(ts, windows=1):
    newts = pd.DataFrame({'dt':ts['dt'].to_list()})
    d1list = []
    d2list = []
    for inc in range(windows):
        tempd1 = ts['d1'].to_list()[int(inc * len(ts.index)/windows) : int((inc+1) * len(ts.index)/windows)]
        tempd2 = ts['d2'].to_list()[int(inc * len(ts.index)/windows) : int((inc+1) * len(ts.index)/windows)]
        normalizing = [[tempd1[i], tempd2[i]] for i in range(int(len(ts.index)/windows))]
        normalized = preprocessing.Normalizer().fit_transform(normalizing)
        d1list.extend([i[0] for i in normalized])
        d2list.extend([i[1] for i in normalized])
    newts['d1'] = d1list
    newts['d2'] = d2list
    return newts

def standarize_dataframe(ts, windows=1):
    newts = pd.DataFrame({'dt': ts['dt'].to_list()})
    d1list = []
    d2list = []
    for inc in range(windows):
        tempd1 = ts['d1'].to_list()[int(inc * len(ts.index) / windows): int((inc + 1) * len(ts.index) / windows)]
        tempd2 = ts['d2'].to_list()[int(inc * len(ts.index) / windows): int((inc + 1) * len(ts.index) / windows)]
        standardizing = [[tempd1[i], tempd2[i]] for i in range(int(len(ts.index) / windows))]
        standardized = preprocessing.StandardScaler().fit_transform(standardizing)
        d1list.extend([i[0] for i in standardized])
        d2list.extend([i[1] for i in standardized])
    newts['d1'] = d1list
    newts['d2'] = d2list
    return newts

ts = generate_timeseries(1000)
nts = normalize_dataframe(ts, 10)
sts = standarize_dataframe(ts, 10)
param_ts_nw = parameter_dataframe(ts)
param_ts_w = parameter_dataframe(ts, 10)
param_nts_w = parameter_dataframe(nts, 10)
param_sts_w = parameter_dataframe(sts, 10)

#print(param_nts_w.loc[param_nts_w['col'] == 'd1'])
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.autofmt_xdate(rotation=45)
ax1.plot(ts['dt'], ts['d1'], color='tab:orange', alpha=0.8)
ax1.plot(ts['dt'], ts['d2'], color='tab:blue', alpha=0.8)
ax1.scatter(ts['dt'].iloc[int(param_ts_nw['lmin'].iloc[0])], param_ts_nw['vmin'].iloc[0], color='tab:orange', marker='v', edgecolors='tab:blue')
ax1.scatter(ts['dt'].iloc[int(param_ts_nw['lmax'].iloc[0])], param_ts_nw['vmax'].iloc[0], color='tab:orange', marker='^', edgecolors='tab:blue')
ax1.scatter(ts['dt'].iloc[int(param_ts_nw['lmin'].iloc[1])], param_ts_nw['vmin'].iloc[1], color='tab:blue', marker='v', edgecolors='tab:orange')
ax1.scatter(ts['dt'].iloc[int(param_ts_nw['lmax'].iloc[1])], param_ts_nw['vmax'].iloc[1], color='tab:blue', marker='^', edgecolors='tab:orange')
ax1.axhline(param_ts_nw['avg'].iloc[0], color='tab:orange', linestyle='dashed')
ax1.axhline(param_ts_nw['avg'].iloc[1], color='tab:blue', linestyle='dashed')

ax2.plot(ts['dt'], ts['d1'], color='tab:orange', alpha=0.8)
ax2.plot(ts['dt'], ts['d2'], color='tab:blue', alpha=0.8)
for i in range(10):
    ax2.scatter(ts['dt'].iloc[int(param_ts_w['lmin'].iloc[i])], param_ts_w['vmin'].iloc[i], color='tab:orange',
                marker='v', edgecolors='tab:blue')
    ax2.scatter(ts['dt'].iloc[int(param_ts_w['lmax'].iloc[i])], param_ts_w['vmax'].iloc[i], color='tab:orange',
                marker='^', edgecolors='tab:blue')
    ax2.scatter(ts['dt'].iloc[int(param_ts_w['lmin'].iloc[i+10])], param_ts_w['vmin'].iloc[i+10], color='tab:blue',
                marker='v', edgecolors='tab:orange')
    ax2.scatter(ts['dt'].iloc[int(param_ts_w['lmax'].iloc[i+10])], param_ts_w['vmax'].iloc[i+10], color='tab:blue',
                marker='^', edgecolors='tab:orange')
    ax2.axvline(ts['dt'].iloc[int(param_ts_w['strpoint'].iloc[i])], color='black', linestyle='dashed', alpha=0.2)
ax2.plot(ts['dt'].iloc[[0,100,200,300,400,500,600,700,800,900]], param_ts_w['avg'].iloc[0:10], color='orange', linestyle='dotted')
ax2.plot(ts['dt'].iloc[[0,100,200,300,400,500,600,700,800,900]], param_ts_w['avg'].iloc[10:20], color='blue', linestyle='dotted')

ax3.plot(nts['dt'], nts['d1'], color='tab:orange', alpha=0.8)
ax3.plot(nts['dt'], nts['d2'], color='tab:blue', alpha=0.8)
for i in range(10):
    ax3.scatter(nts['dt'].iloc[int(param_nts_w['lmin'].iloc[i])], param_nts_w['vmin'].iloc[i], color='tab:orange',
                marker='v', edgecolors='tab:blue')
    ax3.scatter(nts['dt'].iloc[int(param_nts_w['lmax'].iloc[i])], param_nts_w['vmax'].iloc[i], color='tab:orange',
                marker='^', edgecolors='tab:blue')
    ax3.scatter(nts['dt'].iloc[int(param_nts_w['lmin'].iloc[i+10])], param_nts_w['vmin'].iloc[i+10], color='tab:blue',
                marker='v', edgecolors='tab:orange')
    ax3.scatter(nts['dt'].iloc[int(param_nts_w['lmax'].iloc[i+10])], param_nts_w['vmax'].iloc[i+10], color='tab:blue',
                marker='^', edgecolors='tab:orange')
    ax3.axvline(nts['dt'].iloc[int(param_nts_w['strpoint'].iloc[i])], color='black', linestyle='dashed', alpha=0.2)
ax3.plot(nts['dt'].iloc[[0,100,200,300,400,500,600,700,800,900]], param_nts_w['avg'].iloc[0:10], color='orange', linestyle='dotted')
ax3.plot(nts['dt'].iloc[[0,100,200,300,400,500,600,700,800,900]], param_nts_w['avg'].iloc[10:20], color='blue', linestyle='dotted')

ax4.plot(sts['dt'], sts['d1'], color='tab:orange', alpha=0.8)
ax4.plot(sts['dt'], sts['d2'], color='tab:blue', alpha=0.8)
for i in range(10):
    ax4.scatter(sts['dt'].iloc[int(param_sts_w['lmin'].iloc[i])], param_sts_w['vmin'].iloc[i], color='tab:orange',
                marker='v', edgecolors='tab:blue')
    ax4.scatter(sts['dt'].iloc[int(param_sts_w['lmax'].iloc[i])], param_sts_w['vmax'].iloc[i], color='tab:orange',
                marker='^', edgecolors='tab:blue')
    ax4.scatter(sts['dt'].iloc[int(param_sts_w['lmin'].iloc[i+10])], param_sts_w['vmin'].iloc[i+10], color='tab:blue',
                marker='v', edgecolors='tab:orange')
    ax4.scatter(sts['dt'].iloc[int(param_sts_w['lmax'].iloc[i+10])], param_sts_w['vmax'].iloc[i+10], color='tab:blue',
                marker='^', edgecolors='tab:orange')
    ax4.axvline(sts['dt'].iloc[int(param_sts_w['strpoint'].iloc[i])], color='black', linestyle='dashed', alpha=0.2)
ax4.plot(sts['dt'].iloc[[0,100,200,300,400,500,600,700,800,900]], param_sts_w['avg'].iloc[0:10], color='orange', linestyle='dotted')
ax4.plot(sts['dt'].iloc[[0,100,200,300,400,500,600,700,800,900]], param_sts_w['avg'].iloc[10:20], color='blue', linestyle='dotted')


plt.show()