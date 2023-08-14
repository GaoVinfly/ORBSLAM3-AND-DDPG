#!/usr/bin/python

import numpy as np

def rigid_transform_3D(A, B):
    # find mean column wise
    centroid_A = np.mean(A, axis=1)   # 求质心   axis=0按列求均值 #axis=1按行求均值
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B
    An = np.sum(Am*Am, axis=0)    # “×”表示矩阵对应元素相乘  axis=0按列求和
    Bn = np.sum(Bm*Bm, axis=0)
    H = Am @ np.transpose(Bm)

    #计算尺度的平均值
    lam = (An/Bn)
    lam1 = np.mean(lam, axis=0)
    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T    #“@"矩阵相乘

    # t = -R @ centroid_A + centroid_B
    la = np.sqrt(lam1)
    R = R/(la)
    t = -R/(la)@centroid_A + centroid_B
    return R, t
