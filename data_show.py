# author: 龚潇颖(Xiaoying Gong)
# date： 2019/9/22 11:36  
# IDE：PyCharm 
# des:
# input(s)：
# output(s)：
import os
import numpy as np

files = os.listdir("./label_result")
i = 0
for file in files:
    i += 1
    file_content = np.load("./label_result/" + file)
    a = file_content["correspondence_label"][4, :]
    print(i, ":", np.argwhere(a != 1))

