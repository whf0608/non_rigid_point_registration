# author: 龚潇颖(Xiaoying Gong)
# date： 2019/9/28 21:11  
# IDE：PyCharm 
# des:
# input(s)：
# output(s)：
import numpy as np
import matplotlib.pyplot as plt
import constants
from knn.knn import K_NearestNeighbors

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

def solve_param(X, Y):
    constants.LAMBDA = 1
    K = gaussian_kernel(Y, beta=2)
    K_T = np.transpose(K)
    X_gauss_weight, Y_gauss_weight = cal_gauss_weight(X, Y)
    inlier_diag = np.diag(np.ones(len(X))* (-5))
    W_1 = X_gauss_weight - inlier_diag
    W_2 = Y_gauss_weight - inlier_diag
    U = np.dot(W_1, X) - np.dot(W_2, Y)
    V = W_2
    W_P1 = 2 * np.dot(K_T, X) - 2 * np.dot(K_T, Y) + constants.ITA * np.dot(K, np.dot(V.T, U))
    W_P2 = np.linalg.inv(2 * np.dot(K_T, K) + (constants.LAMBDA / 2)*(K + K_T) + constants.ITA*np.dot(
        np.dot(K, V.T), np.dot(V, K)
    ))
    W = np.dot(W_P2, W_P1)
    return W

def cal_gauss_weight(X, Y):
    knn_1 = K_NearestNeighbors(X)
    knn_2 = K_NearestNeighbors(Y)
    result_1 = np.zeros([len(X), len(Y)])
    result_2 = np.zeros([len(X), len(Y)])
    dist_1, index_1 = knn_1.get_k_neighbors_exclude_0(X, 5)
    dist_2, index_2 = knn_2.get_k_neighbors_exclude_0(Y, 5)
    var_1 = np.var(dist_1)
    var_2 = np.var(dist_2)
    for i in range(len(X)):
        result_1[i, index_1[i]] = np.exp(-1 * (dist_1[i]**2) / (var_1))
        result_2[i, index_2[i]] = np.exp(-1 * (dist_2[i]**2) / (var_2))
    return result_1, result_2

def Q_energy(X, Y, W):
    K = gaussian_kernel(Y, beta=2)
    X_gauss_weight, Y_gauss_weight = cal_gauss_weight(X, Y)
    inlier_diag = np.diag(np.ones(len(X))* (-5))
    W_1 = X_gauss_weight - inlier_diag
    W_2 = Y_gauss_weight - inlier_diag
    U = np.dot(W_1, X) - np.dot(W_2, Y)
    V = W_2
    Q_1 = np.square(np.linalg.norm(X - Y - np.dot(K, W)))
    Q_2 = (constants.LAMBDA / 2) * np.trace((np.dot(np.transpose(W), K, W)))
    Q_3 = (constants.ITA / 2) * (np.trace(np.dot(U, U.T)) - 2 * np.trace(np.dot(np.dot(W.T, K), np.dot(V.T, U))) +
                                 np.trace(np.dot(np.dot(W.T, K, V.T), np.dot(K, W))))
    Q = Q_1 + Q_2 + Q_3
    return Q


def Q_energy_fun(X, Y, K, param):
    Q_1 = np.square(np.linalg.norm(X - Y - np.dot(K, param)))
    Q_2 = (constants.LAMBDA / 2) * (np.dot(np.transpose(param), K, param))
    Q = Q_1 + Q_2
    return Q

X = np.array([[1, 2], [2, 1], [2, 3], [3, 2], [4, 3], [5, 1], [15, 10], [-1, -1]])
Y = np.array([[0, 2], [2, 2], [2, 4], [2, 3], [3, 3], [4, 10], [-1, -1], [-100, -100]])
# X = np.array([[2, 2], [3, 1], [3, 3]])
# Y = np.array([[12, 12], [13, 11], [13, 13]])
# C = np.diag(np.array([1, 1, 0]))

plt.figure(num="OKKKK")
plt.ion()
for i in range(50):
   W = solve_param(X, Y)
   Y = Y + np.dot(gaussian_kernel(Y), W)

   Q = Q_energy(X, Y, W)
   print(Q)

   plt.scatter(np.transpose(X)[0, :], np.transpose(X)[1, :])
   plt.scatter(np.transpose(Y)[0, :], np.transpose(Y)[1, :], c='r')
   plt.pause(0.1)
   if i != 49:
        plt.clf()
print("OVER")
plt.ioff()
plt.show()