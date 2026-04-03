import numpy as np
import sklearn.neighbors as skn
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import warnings
import matplotlib.pyplot as plt
import cv2
from scipy.spatial import KDTree
import os
import time
from functools import wraps

def timer(name=None, repeat=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            times = []
            result = None
            
            for _ in range(repeat):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                end = time.perf_counter()
                times.append(end - start)
            
            avg_time = sum(times) / repeat
            print(f"[TIMER] {name or func.__name__}: {avg_time:.4f}s (avg over {repeat})")
            return result
        
        return wrapper
    return decorator

def knn_brute(features, k):
    nbrs = skn.NearestNeighbors(n_neighbors=k+1, algorithm='brute').fit(features)
    dist, idx = nbrs.kneighbors(features)
    return dist[:,1:], idx[:,1:]

def knn_kd_tree(features, k):
    tree = skn.NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree').fit(features)
    dist, idx = tree.kneighbors(features)
    return dist[:,1:], idx[:,1:]

def knn_flann(features, k):
    features32 = features.astype(np.float32)
    
    index_params = dict(algorithm=1, trees=5)  # KDTree
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(features32, features32, k=k+1)
    
    distances = []
    indices = []
    
    for m in matches:
        distances.append([match.distance for match in m[1:]])
        indices.append([match.trainIdx for match in m[1:]])
    
    return np.array(distances), np.array(indices)

def knn_matting(image, trimap, task_id, knn_k, my_lambda=100):
    [h, w, c] = image.shape
    image, trimap = image / 255.0, trimap / 255.0
    # print(f"Unique values in trimap: {np.unique(trimap)}")
    foreground = (trimap > 0.9).astype(int)
    background = (trimap < 0.1).astype(int)
    # mask = foreground + background
    # cv2.imwrite(f'./result/{task_id}_mask.png', (mask * 255).astype(np.uint8))
    ####################################################
    # find KNN for the given image
    ####################################################
    N = h * w
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    spatial_scale = np.linalg.norm([h, w])
    
    # RGB, x, y
    # features = np.zeros((N, c+2))
    
    # features[:, :c] = image.reshape(-1, c)
    # features[:, c] = x.reshape(-1) / spatial_scale
    # features[:, c+1] = y.reshape(-1) / spatial_scale
    
    # LAB, x, y
    features = np.zeros((N, 5))
    features[:, :c] = image.reshape(-1, c)
    features[:, c] = x.reshape(-1) / spatial_scale
    features[:, c+1] = y.reshape(-1) / spatial_scale
    
    tree = KDTree(features)
    distances, indices = tree.query(features, k=knn_k+1)
    # discard self neighbor
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    ####################################################
    #       compute the affinity matrix A
    #       and all other stuff needed
    ####################################################
    sigma = np.mean(distances[:, -1])   # distance to k-th neighbor
    print(f"Estimated sigma: {sigma}")
    if sigma < 1e-6:
        print("Warning: sigma is too small, using default value")
        sigma = 1.0
    # sigma = 0.05
    weights = np.exp(-distances**2 / (sigma**2))
    row_indices = np.repeat(np.arange(N), knn_k)
    col_indices = indices.flatten()
    data = weights.flatten()

    A = sp.csr_matrix((data, (row_indices, col_indices)), shape=(N, N))
    A = A.maximum(A.T)  # make it symmetric
    D = sp.diags(A.sum(axis=1).A1)
    # L = D - A
    eps = 1e-6
    L = D - A + eps * sp.eye(N)  # add small value to diagonal for stability
    foreground_flat = (trimap.reshape(-1) > 0.9)
    background_flat = (trimap.reshape(-1) < 0.1)
    
    constraint_mask = foreground_flat | background_flat
    # Ensure constraint_mask is numeric
    constraint_mask = constraint_mask.astype(float)
    constraint_values = foreground_flat.astype(float)
    # background already 0
    
    # Build diagonal matrix for constraints
    C = sp.diags(constraint_mask)
    
    # Final system: (L + lambda * C) * alpha = lambda * constraint_values
    M = L + my_lambda * C
    b = my_lambda * constraint_values
    
    ####################################################
    #       solve for the linear system,
    #       note that you may encounter en error
    #       if no exact solution exists
    ####################################################
    warnings.filterwarnings('error')
    
    try:
        alpha = spla.spsolve(M, b)
    except Warning as e:
        print("Error solving linear system:", e)
        alpha, _ = spla.cg(M, b, rtol=1e-5, maxiter=100)
    
    alpha = np.clip(alpha, 0, 1).reshape(h, w)
    return alpha

if __name__ == '__main__':
    abs_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(abs_path)
    os.makedirs('./result', exist_ok=True)
    new_background = cv2.imread(f'./background/garden.png')
    # tasks = ["bear", "white_cloth", "woman"]
    tasks = ["white_cloth"]
    for task_id in range(len(tasks)):
        print(f"Processing task: {tasks[task_id]}...")
        image = cv2.imread(f'./image/{tasks[task_id]}.png')
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        trimap = cv2.imread(f'./trimap/{tasks[task_id]}.png', cv2.IMREAD_GRAYSCALE)
        # thresholds = [0, 32, 233, 255]
        trimap[trimap < 32] = 0
        trimap[(trimap >= 32) & (trimap < 233)] = 127
        trimap[trimap >= 233] = 255
        foreground = (trimap >= 233)
        background = (trimap < 32)
        # print(np.unique(trimap))
        print(f"Image shape: {image.shape}, Trimap shape: {trimap.shape}")
        # scale = 0.75
        # image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
        # trimap = cv2.resize(trimap, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        ####################################################
        # TODO: pick up your own background image, 
        #       and merge it with the foreground
        ####################################################
        # K = [3, 4, 5, 6, 7, 8, 9, 10]
        K = [20]
        for k in K:
            task_id_str = f"{tasks[task_id]}_k_{k}"
            alpha = knn_matting(lab_image, trimap, task_id=task_id_str, knn_k=k, my_lambda=1000)
            alpha = alpha[:, :, np.newaxis]
            alpha = (alpha * 255).astype(np.uint8)
            print(f"Saving matting result for K={k}...")
            cv2.imwrite(f'./result/{task_id_str}.png', alpha)
            print(f"Saving foreground for K={k}...")
            foreground_composite = (alpha / 255.0) * image
            cv2.imwrite(f'./result/{task_id_str}_foreground.png', foreground_composite)
            print(f"Saving composing result for K={k}")
            # composed = (alpha / 255.0) * image + (1 - alpha / 255.0) * new_background
            # composed = cv2.cvtColor(composed.astype(np.uint8), cv2.COLOR_LAB2BGR)
            # cv2.imwrite(f'./result/{task_id_str}_composed.png', composed)