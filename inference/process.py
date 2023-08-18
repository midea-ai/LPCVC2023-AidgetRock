import numpy as np

def process0(image_array):
    
    # 统计得到的各类像素点数目最小值
    thresholds = {
        0: 512,
        1: 3795,
        2: 32,
        3: 8,
        4: 168,
        5: 107,
        6: 51,
        7: 101,
        8: 45927,
        9: 36,
        10: 3,
        11: 649,
        12: 36,
        13: 12,
    }

    # 统计得到的互斥类，互斥类
    mutual_exclusion = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0], 
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 
                        [0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0], 
                        [0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], 
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 
                        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0], 
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    
    #统计像素点
    unique_class = np.unique(image_array)
    pixel_class_num = {}
    pixel_class_num[0]=0
    for cls in unique_class:
        num = np.sum(image_array == cls)
        pixel_class_num[cls] = num
    modified_image_array = np.copy(image_array)
    
    # 基于最少像素点的后处理
    for cls in unique_class:
        if pixel_class_num[cls] < thresholds[cls]/3:
            modified_image_array[modified_image_array == cls] = 0
            pixel_class_num[0] = pixel_class_num[0]+pixel_class_num[cls]
            del pixel_class_num[cls]
    
    #统计像素点有剩哪几类
    pixel_class_num = sorted(pixel_class_num.items(), key=lambda x:x[1], reverse=True) #对字典从大到小排序
    unique_class = []
    for k,v in pixel_class_num:
        unique_class.append(k)
    n = len(unique_class)
    
    #基于互斥类的后处理
    flag = 0
    for i in range(n):
        cls1 = unique_class[i]
        for j in range(i+1,n):
            cls2 = unique_class[j] #cls2的像素点数一定少于cls1
            if(mutual_exclusion[cls1][cls2]==1):
                modified_image_array[modified_image_array == cls2] = 0
                flag = 1 #找到一组互斥类，不再处理其他互斥类
                break
        if flag==1:
            break
    
    return modified_image_array


def process1(prior_mat,image):
    result = np.empty((512, 512), dtype=int)
    for i in range(512):
        for j in range(512):
            r, g, b = image[i][j]
            result[i][j] = prior_mat[r // 32][g // 32][b // 32]
    return result