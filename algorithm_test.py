# author: 龚潇颖(Xiaoying Gong)
# date： 2019/9/28 11:38  
# IDE：PyCharm 
# des:
# input(s)：
# output(s)：
import numpy as np

bool_mask = np.array([[False, True, True, True, True, True, True],
                      [False, False, True, True, True, True, True]])
index = np.array([[0, 1, 2, 3, 4, 5, 6],
                  [0, 1, 2, 3, 4, 5, 6]])


def choose_k_elements_with_mask_2d(data, mask, k):
    row_counts = len(data)
    result = np.zeros([row_counts, k])
    for i in range(row_counts):
        temp = data[i, mask[i]]
        result[i] = temp[:k]
    return result


# print(choose_k_elements_with_mask_2d(index, bool_mask, 5))
a = np.array([[5, 5], [5, 5]])
print(np.var(a))
print(np.sum(a))