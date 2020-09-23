# -*- coding: utf-8 -*-
# @date 2020/09/23
# @author shanekong
# @description: sequence bar, 序列条状图.


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import pandas as pd


def show_seq():
    """ one figure, one seq with several items for demo"""

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.axis([0, 15, 0, 4])
    ax1.add_patch(
        patches.Rectangle(
            (0.1, 1.0),  # (x,y)
            0.3,  # width
            0.3,  # height
        )
    )
    ax1.add_patch(
        patches.Rectangle(
            (0.4, 1.0),  # (x,y)
            0.8,  # width
            0.3,  # height
            facecolor='#00FFFF'
        )
    )
    ax1.add_patch(
        patches.Rectangle(
            (1.2, 1.0),  # (x,y)
            0.5,  # width
            0.3,  # height
            facecolor='g'
        )
    )

    ax1.add_patch(
        patches.Rectangle(
            (1.7, 1.0),  # (x,y)
            1.5,  # width
            0.3,  # height
            facecolor='r'
        )
    )

    fig1.savefig('./pngs/seq_rect.png', dpi=90, bbox_inches='tight')
    fig1.show()


if __name__ == "__main__":
    show_seq()


