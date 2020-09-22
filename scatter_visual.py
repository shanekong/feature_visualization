# -*- coding: utf-8 -*-
# @date 2020/09/22
# @author shanekong
# @description: scatter 散点图.


import numpy as np
import matplotlib.pyplot as plt
import random


def sample1():
    """example1"""
    x = np.arange(1, 10)
    y = x
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('Scatter Plot')
    plt.xlabel('X')
    plt.ylabel('Y')

    ax1.scatter(x, y, c='r', marker='o')
    plt.legend("x1", loc='upper right')  # 示意图标
    plt.show()


def sample2():
    """example2
       one figure; multi data;
    """
    from matplotlib import pyplot as plt
    import numpy as np

    # Generating a Gaussion dataset:
    # creating random vectors from the multivariate normal distribution
    # given mean and covariance
    mu_vec1 = np.array([0, 0])
    cov_mat1 = np.array([[2, 0], [0, 2]])

    x1_samples = np.random.multivariate_normal(mu_vec1, cov_mat1, 100)
    x2_samples = np.random.multivariate_normal(mu_vec1 + 0.2, cov_mat1 + 0.2, 100)
    x3_samples = np.random.multivariate_normal(mu_vec1 + 0.4, cov_mat1 + 0.4, 100)

    # x1_samples.shape -> (100, 2), 100 rows, 2 columns
    print(x1_samples.shape)
    print(x1_samples)
    print(x1_samples[:, 0])  # 取第0列数据;
    print(x1_samples[:, 1])  # 取第1列数据;

    plt.figure(figsize=(8, 6))

    plt.scatter(x1_samples[:, 0], x1_samples[:, 1], marker='x',
                color='blue', alpha=0.7, label='x1 samples')
    plt.scatter(x2_samples[:, 0], x1_samples[:, 1], marker='o',
                color='green', alpha=0.7, label='x2 samples')
    plt.scatter(x3_samples[:, 0], x1_samples[:, 1], marker='^',
                color='red', alpha=0.7, label='x3 samples')
    plt.title('Basic scatter plot')
    plt.ylabel('variable X')
    plt.xlabel('Variable Y')
    plt.legend(loc='upper right')

    plt.show()


if __name__ == "__main__":
    # sample1()
    sample2()