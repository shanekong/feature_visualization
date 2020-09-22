# -*- coding: utf-8 -*-
# @date 2020/09/22
# @author shanekong
# @description: t-distributed Stochastic Neighbor Embedding(t-SNE)
# t-SNE可降样本点间的相似度关系转化为概率：在原空间（高维空间）中转化为基于高斯分布的概率,在嵌入空间（二维空间）中转化为基于t分布的概率。
# ref:https://blog.csdn.net/hustqb/article/details/80628721


import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets


def show_data():
    """show data miniset"""
    digits = datasets.load_digits(n_class=6)
    X, y = digits.data, digits.target
    n_samples, n_features = X.shape
    print("n_samples:{}, n_features:{}".format(str(n_samples), str(n_features)))

    '''显示原始数据'''
    n = 20  # 每行20个数字，每列20个数字
    img = np.zeros((10 * n, 10 * n))
    for i in range(n):
        ix = 10 * i + 1
        for j in range(n):
            iy = 10 * j + 1
            img[ix:ix + 8, iy:iy + 8] = X[i * n + j].reshape((8, 8))
    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap=plt.cm.binary)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def show_tsne():
    """show tsne """
    digits = datasets.load_digits(n_class=6)
    X, y = digits.data, digits.target
    n_samples, n_features = X.shape
    print("n_samples:{}, n_features:{}".format(str(n_samples), str(n_features)))

    '''t-SNE'''
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)

    print("Org data dimension is {}.Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

    '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    plt.figure(figsize=(8, 8))
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.show()


if __name__ == "__main__":
    show_data()
    show_tsne()