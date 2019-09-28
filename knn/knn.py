'''
author:龚潇颖
des:找到相领的内点
'''

import numpy as np
from sklearn.neighbors import NearestNeighbors

class K_NearestNeighbors:
    def __init__(self, train_data):
        self.train_data = train_data
        self.nn = NearestNeighbors(algorithm='kd_tree').fit(self.train_data)

    # 寻找的点不会在kd树当中
    def get_k_neighbors_v0(self, aim_point, k):
        k_nearest_neighbors_dist, k_nearest_neighbors_index = self.nn.kneighbors(X=aim_point, n_neighbors=k)
        return k_nearest_neighbors_dist, k_nearest_neighbors_index

    # 寻找的点在kd树种
    def get_k_neighbors_v1(self, aim_point, k):
        aim_point = np.array([aim_point])
        # 没找到足量的点 就一直循环
        k_new = k
        while True:
            k_nearest_neighbors_dist, k_nearest_neighbors_index = self.nn.kneighbors(X=aim_point, n_neighbors=k_new)
            #去除掉自己
            zero_index = np.array(np.where(k_nearest_neighbors_dist[0] == 0.))
            k_nearest_neighbors_dist = np.delete(k_nearest_neighbors_dist, zero_index)
            k_nearest_neighbors_index = np.delete(k_nearest_neighbors_index, zero_index)
            if len(k_nearest_neighbors_dist) != k or len(k_nearest_neighbors_index) != k:
                k_new = k_new + 1
            else:
                break
        return k_nearest_neighbors_dist, k_nearest_neighbors_index

    def get_k_neighbors(self, aim_point, k):
        # aim_point = np.array([aim_point])
        k_nearest_neighbors_dist, k_nearest_neighbors_index = self.nn.kneighbors(X=aim_point, n_neighbors=k)
        return k_nearest_neighbors_dist, k_nearest_neighbors_index

    def get_k_neighbors_boardcast(self, aim_point, k):
        k_nearest_neighbors_dist, k_nearest_neighbors_index = self.nn.kneighbors(X=aim_point, n_neighbors=k)
        return k_nearest_neighbors_dist, k_nearest_neighbors_index


    def get_k_neighbors_exclude_0(self, aim_point, k):
        '''
        一个为了效率而委屈求全的办法，以排除掉dist==0的点
        为了鲁棒性，可以把初始搜索范围设定为全部的点
        为了效率，可以只设置部分点（log的搜索时间也是时间啊！！！！！）
        '''
        search_range = 6
        result_matrix_mask = (np.ones([len(aim_point), search_range]) == 1)
        k_nearest_neighbors_dist, k_nearest_neighbors_index = self.nn.kneighbors(X=aim_point, n_neighbors=search_range)
        zero_pos = np.argwhere(k_nearest_neighbors_dist == 0.)
        result_matrix_mask[zero_pos[:, 0], zero_pos[:, 1]] = False
        k_nearest_neighbors_dist = self.choose_k_elements_with_mask_2d(k_nearest_neighbors_dist, result_matrix_mask, k)
        k_nearest_neighbors_index = self.choose_k_elements_with_mask_2d(k_nearest_neighbors_index, result_matrix_mask, k, dtype=int)
        return k_nearest_neighbors_dist, k_nearest_neighbors_index


    def choose_k_elements_with_mask_2d(self, data, mask, k, dtype=float):
        row_counts = len(data)
        result = np.zeros([row_counts, k], dtype)
        for i in range(row_counts):
            temp = data[i, mask[i]]
            result[i] = temp[:k]
        return result
