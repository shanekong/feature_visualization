# -*- coding: utf-8 -*-
# @date 2020/09/22
# @author shanekong
# @description: 直方图+ kde.
# 直方图：对数据做分箱处理，然后统计每个箱内观察值的数量，这就是真正的直方图所要做的工作。
# KDE（Kernel density estimation）是核密度估计的意思，它用来估计随机变量的概率密度函数，可以将数据变得更平缓。

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def show_hist():
    """ show hist"""

    means = 10, 20
    stdevs = 4, 2
    dist = pd.DataFrame(np.random.normal(loc=means, scale=stdevs, size=(1000, 2)), columns=['a', 'b'])
    dist.agg(['min', 'max', 'mean', 'std']).round(decimals=2)

    fig, ax = plt.subplots()
    dist.plot.kde(ax=ax, legend=False, title='Histogram: A vs. B')
    dist.plot.hist(density=True, ax=ax, rwidth=0.9)
    ax.set_ylabel('Probability')
    ax.grid(axis='y')
    ax.set_facecolor('#d8dcd6')
    plt.show()


def show_data():
    mean1 = 10
    stdevs1 = 4
    x1_samples = np.random.normal(loc=mean1, scale=stdevs1, size=100)

    mean2 = 20
    stdevs2 = 2
    x2_samples = np.random.normal(loc=mean2, scale=stdevs2, size=100)
    print(x1_samples.shape)
    print(x1_samples)
    print(x2_samples.shape)
    print(x2_samples)

    dist = pd.DataFrame(zip(x1_samples, x2_samples), columns=['a', 'b'])
    dist.agg(['min', 'max', 'mean', 'std']).round(decimals=2)
    print(dist)

    fig, ax = plt.subplots()
    dist.plot.kde(ax=ax, legend=False, title='Histogram: A vs. B')
    dist.plot.hist(density=True, ax=ax, rwidth=0.9)
    ax.set_ylabel('Probability')
    ax.grid(axis='y')
    # ax.set_facecolor('#d8dcd6')
    plt.show()


if __name__ == "__main__":
    # show_hist()
    show_data()
