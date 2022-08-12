# k_means_sklearn.py


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import AutoMinorLocator
import sys
import os

K_CLUSTERS = 3


def plot(ax):
    ax.set_title(f"k-Means Clustering (k={K_CLUSTERS})")
    ax.set_xlim(-5, 45)
    ax.set_ylim(-5, 45)
    # ax.set_aspect('equal')

    file_name = os.path.dirname(sys.argv[0]) + '/cluster_samples.csv'
    points = np.genfromtxt(file_name, delimiter=',')
    points = points[:-1, :]

    np.random.seed(2016)
    kmeans = KMeans(K_CLUSTERS)
    kmeans.fit(points)
    y_kmeans = kmeans.predict(points)
    centers = kmeans.cluster_centers_

    print(f"\ncenters = [x,y]\n {centers}\n")

    ax.scatter(points[:, 0], points[:, 1], c=y_kmeans)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.scatter(centers[:, 0], centers[:, 1], c='black', marker='s')
    ymin, ymax = ax.get_ylim()
    ax.vlines(x=[centers], ymin=ymin, ymax=ymax,
              color='green', linestyle='dotted')
    xmin, xmax = ax.get_xlim()
    ax.hlines(y=[centers], xmin=xmin, xmax=xmax,
              color='green', linestyle='dotted')


def main():
    fig = plt.figure()
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0, 0])
    plot(ax)
    plt.show()


if __name__ == "__main__":
    main()
