import numpy as np

X = np.array(
    [[0, 3, 1],
    [1, 3, 1],
    [0, 1, 1],
    [1, 1, 1]]
    )
y = np.array([[1], [1], [0], [0]])

lam = 0.07

w = np.array([[-2], [1], [0]])

def sigmoid(x):
    return 1/(1+np.exp(-x))

for i in range(2):
    s = []
    for j in range(4):
        s.append(sigmoid(np.dot(X[j], w)))
    s = np.array(s).reshape(4, 1)
    print("s" + str(i) + " = " + str(np.transpose(s)))
    omega = np.array(
        [[s[0][0]*(1-s[0][0]), 0, 0, 0],
        [0, s[1][0]*(1-s[1][0]), 0, 0],
        [0, 0, s[2][0]*(1-s[2][0]), 0],
        [0, 0, 0, s[3][0]*(1-s[3][0])]])
    A = np.linalg.inv(np.add(2*lam*np.eye(3), np.dot(np.transpose(X), np.dot(omega, X))))
    B = np.subtract(2*lam*w, np.dot(np.transpose(X), y-s))
    w = w - np.dot(A, B)
    print("w" + str(i+1) + " = " + str(np.transpose(w)))