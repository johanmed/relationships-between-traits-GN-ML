#!/usr/bin/env python3

"""
Library to manage rendering of graph/network plots

Adapted from https://github.com/nelsocs/dzcnapyd/blob/master/dzcnapy_plotlib.py written by Dimitry 
"""

import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use("grayscale")

attrs = {
    "edge_color" : "gray",
    "edgecolors" : "black", # For Networkx 2.0
    "linewidths" : 1,       # For Networkx 2.0
    "font_family" : "Liberation Sans Narrow",
    "font_size" : 14,
    "node_color" : "pink",
    "node_size" : 700,
    "width" : 2,
}



def set_extent(positions, axes, title=None):
    """
    Given node coordinates pos and the subplot,
    calculate and set its extent.
    """
    axes.tick_params(labelbottom="off")
    axes.tick_params(labelleft="off")
    if title:
        axes.set_title(title)

    x_values, y_values = zip(*positions.values())
    x_max = max(x_values)
    y_max = max(y_values)
    x_min = min(x_values)
    y_min = min(y_values)
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
    try:
        axes.set_xlim(x_min - x_margin, x_max + x_margin)
        axes.set_ylim(y_min - y_margin, y_max + y_margin)
    except AttributeError:
        axes.xlim(x_min - x_margin, x_max + x_margin)
        axes.ylim(y_min - y_margin, y_max + y_margin)

def plot(fname, save = False):
    plt.tight_layout()
    if save:
        plt.savefig("../../output/{}.png".format(fname), dpi=500)
    plt.show()
