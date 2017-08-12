import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

X1 = np.random.normal(3, 3, 100)
X2 = np.random.normal(.5*X1 + 4, 2)

x1_mean, x2_mean = np.mean(X1), np.mean(X2)

Sigma = np.cov(X1, X2)

eig_vals, eig_vectors = np.linalg.eig(Sigma)

v1, v2 = eig_vectors[:,0], eig_vectors[:,1]

v1, v2 = v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)

v1_x = v1[0] * eig_vals[0]
v1_y = v1[1] * eig_vals[0]

v2_x = v2[0] * eig_vals[1]
v2_y = v2[1] * eig_vals[1]

string = "Randomly generated (X1, X2) coordinate pairs:\n"
for i in range(100):
    x1, x2 = X1[i], X2[i]
    s = '(' + str(x1) + ', ' + str(x2) + ')\n'
    string += s

string += '\n\n' + str('Mean of the sample: (' + str(x1_mean) + ', ' + str(x2_mean) + ')\n\n')
string += '2x2 Covariance Matrix:\n'
string += '[ ' + str(Sigma[0][0]) + '  ' + str(Sigma[0][1]) + ' ]\n'
string += '[ ' + str(Sigma[1][0]) + '  ' + str(Sigma[1][1]) + ' ]\n\n'
string += 'Eigenvalues and Eigenvectors of the Covariance Matrix:\n'
string += 'v1: ' + str(v1) + '  λ1: ' + str(eig_vals[0]) + '\n'
string += 'v2: ' + str(v2) + '  λ2: ' + str(eig_vals[1]) + '\n'

f = open('question_3.txt', 'w', encoding='utf-8')
f.write(string)
f.close()

plt.figure(figsize=(7.5, 7.5))
plt.xlabel("X1")
plt.ylabel("X2")
plt.axis((-15, 15, -15, 15))
plt.plot(X1, X2, 'ro')
plt.arrow(x1_mean, x2_mean, v1_x, v1_y, length_includes_head=True, head_width=.5, color='blue')
plt.arrow(x1_mean, x2_mean, v2_x, v2_y, length_includes_head=True, head_width=.5, color='blue')

ut = np.array([v1, v2])
x_rotated = np.dot(ut, np.array([X1 - x1_mean, X2 - x2_mean]))
plt.figure(figsize=(7.5, 7.5))
plt.axis((-15, 15, -15, 15))
plt.plot(x_rotated[0], x_rotated[1], 'mo')
plt.arrow(0, 0, eig_vals[0], 0, length_includes_head=True, head_width=.5, color='cyan')
plt.arrow(0, 0, 0, eig_vals[1], length_includes_head=True, head_width=.5, color='cyan')

plt.show()