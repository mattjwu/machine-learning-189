import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

def graph(part, m1, m2, x_min, x_max, y_min, y_max, delta=.025):
    x = np.arange(x_min, x_max, delta)
    y = np.arange(y_min, y_max, delta)
    X, Y = np.meshgrid(x, y)
    mux, muy, sigmax, sigmay, sigmaxy = m1
    Z1 = mlab.bivariate_normal(X, Y, sigmax, sigmay, mux, muy, sigmaxy)
    if m2:
        mux, muy, sigmax, sigmay, sigmaxy = m2
        Z2 = mlab.bivariate_normal(X, Y, sigmax, sigmay, mux, muy, sigmaxy)
        Z = Z1 - Z2
    else:
        Z = Z1
    plt.figure()
    CS = plt.contour(X, Y, Z)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('Isocontour for part (' + part + ')')

def part_a():
    m1 = [1, 1, 1, 2**.5, 0]
    m2 = None
    x_min, x_max = -4, 4
    y_min, y_max = -4, 4
    graph('a', m1, m2, x_min, x_max, y_min, y_max)

def part_b():
    m1 = [-1, 2, 2**.5, 3**.5, 1]
    m2 = None
    x_min, x_max = -4, 4
    y_min, y_max = -4, 4
    graph('b', m1, m2, x_min, x_max, y_min, y_max)

def part_c():
    m1 = [0, 2, 2**.5, 1, 1]
    m2 = [2, 0, 2**.5, 1, 1]
    x_min, x_max = -4, 4
    y_min, y_max = -4, 4
    graph('c', m1, m2, x_min, x_max, y_min, y_max)

def part_d():
    m1 = [0, 2, 2**.5, 1, 1]
    m2 = [2, 0, 2**.5, 3**.5, 1]
    x_min, x_max = -4, 4
    y_min, y_max = -4, 4
    graph('d', m1, m2, x_min, x_max, y_min, y_max)

def part_e():
    m1 = [1, 1, 2**.5, 1, 0]
    m2 = [-1, -1, 2**.5, 2**.5, 1]
    x_min, x_max = -4, 4
    y_min, y_max = -4, 4
    graph('e', m1, m2, x_min, x_max, y_min, y_max)

part_a()
part_b()
part_c()
part_d()
part_e()

plt.show()