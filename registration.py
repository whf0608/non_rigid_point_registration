# author: 龚潇颖(Xiaoying Gong)
# date： 2019/9/18 20:59
# IDE：PyCharm
# des:
# input(s)：
# output(s)：
import numpy as np
import matplotlib.pyplot as plt
import constants

def gaussian_kernel(Y, beta=2):
    (M, D) = Y.shape
    XX = np.reshape(Y, (1, M, D))
    YY = np.reshape(Y, (M, 1, D))
    XX = np.tile(XX, (M, 1, 1))
    YY = np.tile(YY, (1, M, 1))
    diff = XX-YY
    diff = np.multiply(diff, diff)
    diff = np.sum(diff, 2)
    return np.exp(-diff / (2 * beta))

def Q_energy_fun(X, Y, K, param):
    Q_1 = np.square(np.linalg.norm(X - Y - np.dot(K, param)))
    Q_2 = (constants.LAMBDA / 2) * (np.dot(np.transpose(param), K, param))
    Q = Q_1 + Q_2
    return Q

def solve_param(X, Y):
    constants.LAMBDA = 1
    K = gaussian_kernel(Y, beta=2)
    K_T = np.transpose(K)
    W_1 = 2 * np.dot(K_T, X) - 2 * np.dot(K_T, Y)
    W_2 = np.linalg.inv(2 * np.dot(K_T, K) + (constants.LAMBDA / 2)*(K + K_T))
    W = np.dot(W_2, W_1)
    return W


def solve_param_2(X, Y, C):
    K = gaussian_kernel(Y, beta=2)
    K_T = np.transpose(K)
    C_T = np.transpose(C)
    W_1 = 2 * np.dot(np.dot(K_T, C_T), np.dot(C, X)) - 2 * np.dot(np.dot(K_T, C_T), np.dot(C, Y))
    W_2 = np.linalg.inv( 2 * np.dot(np.dot(K_T, C_T), np.dot(C, K)) + (constants.LAMBDA / 2) * (K + K_T ))
    W = np.dot(W_2, W_1)
    return W

X = np.array([[1, 2], [2, 1], [2, 3], [3, 2], [4, 3], [5, 1], [15, 10]])
Y = np.array([[0, 2], [2, 2], [2, 4], [2, 3], [3, 3], [4, 10], [-10, -10]])
# X = np.array([[2, 2], [3, 1], [3, 3]])
# Y = np.array([[12, 12], [13, 11], [13, 13]])
# C = np.diag(np.array([1, 1, 0]))

plt.figure(num="OKKKK")
plt.ion()
for i in range(50):
   # if i == 30:
   #     C = np.diag(np.array([1, 1, 1, 1, 0, 1]))
   W = solve_param(X, Y)
   Y = Y + np.dot(gaussian_kernel(Y), W)
   # Y = Y + np.dot(gaussian_kernel(Y), W, C)
   plt.scatter(np.transpose(X)[0, :], np.transpose(X)[1, :])
   plt.scatter(np.transpose(Y)[0, :], np.transpose(Y)[1, :], c='r')
   plt.pause(0.1)
   if i != 49:
        plt.clf()
print("OVER")
plt.ioff()
plt.show()



