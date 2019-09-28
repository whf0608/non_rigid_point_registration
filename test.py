# author: 龚潇颖(Xiaoying Gong)
# date： 2019/9/22 16:33  
# IDE：PyCharm 
# des:
# input(s)：
# output(s)：
import numpy as np
import scipy.io as io

from knn.knn import K_NearestNeighbors

X = np.array([[1, 1], [1, 2], [1, 3], [2, 1], [3, 1], [10, 10], [11, 11]], dtype=float)
Y = np.array([[2, 2], [2, 3], [3, 4], [3, 2], [4, 2], [11, 11], [12, 12]], dtype=float)


def cal_gausWeight(points_1, points_2):
    knn_1 = K_NearestNeighbors(points_1)
    knn_2 = K_NearestNeighbors(points_2)
    result_1 = np.zeros([len(points_1), len(points_2)])
    result_2 = np.zeros([len(points_1), len(points_2)])
    dist_1, index_1 = knn_1.get_k_neighbors_exclude_0(X, 5)
    dist_2, index_2 = knn_2.get_k_neighbors_exclude_0(Y, 5)
    var_1 = np.var(dist_1)
    var_2 = np.var(dist_2)
    for i in range(len(points_1)):
        result_1[i, index_1[i]] = np.exp(-1 * (dist_1[i]**2) / (var_1))
        result_2[i, index_2[i]] = np.exp(-1 * (dist_2[i]**2) / (var_2))
    return result_1, result_2


# result_1, result_2 = cal_gausWeight(X, Y)
# io.savemat("./data/savemat.mat", {"result_1": result_1, "result_2": result_2})

inlier_index = np.array([1,1,1,1])
inlier_diag = np.diag(np.ones(len(inlier_index))* (-5))
print(inlier_diag)