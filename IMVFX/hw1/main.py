import numpy as np
import sklearn.neighbors
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import warnings
import matplotlib.pyplot as plt
import cv2
from scipy.spatial import KDTree
import os


def knn_matting(image, trimap, knn_k, my_lambda=100):
    [h, w, c] = image.shape
    # hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image, trimap = image / 255.0, trimap / 255.0
    foreground = (trimap == 1.0).astype(int)
    background = (trimap == 0.0).astype(int)

    ####################################################
    # TODO: find KNN for the given image
    ####################################################
    N = h * w
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    spatial_scale = np.linalg.norm([h, w])
    
    features = np.zeros((N, 5))
    
    features[:, :3] = image.reshape(-1, 3)
    features[:, 3] = x.reshape(-1) / spatial_scale
    features[:, 4] = y.reshape(-1) / spatial_scale
    # features = np.zeros((N, 6))
    # # cos(h), sin(h), s, v, x, y
    # features[:, 0] = np.cos(hsv_image[:, :, 0].reshape(-1) * np.pi / 180)  # cos(hue)
    # features[:, 1] = np.sin(hsv_image[:, :, 0].reshape(-1) * np.pi / 180)  # sin(hue)
    # features[:, 2:4] = hsv_image[:, :, 1:3].reshape(-1, 2)  # saturation and value
    # # Add spatial coordinates
    # features[:, 4] = x.reshape(-1) / spatial_scale
    # features[:, 5] = y.reshape(-1) / spatial_scale
    
    tree = KDTree(features)
    distances, indices = tree.query(features, k=knn_k+1)
    # discard self neighbor
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    ####################################################
    # TODO: compute the affinity matrix A
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
    L = D - A
    # eps = 1e-6
    # L = D - A + eps * sp.eye(N)  # add small value to diagonal for stability
    
    constraint_mask = np.zeros(N)
    constraint_mask[foreground] = 1.0
    constraint_mask[background] = 1.0
    constraint_values = np.zeros(N)
    constraint_values[foreground] = 1.0   # alpha=1 for FG
    # background already 0
    
    # Build diagonal matrix for constraints
    C = sp.diags(constraint_mask)
    
    # Final system: (L + lambda * C) * alpha = lambda * constraint_values
    M = L + my_lambda * C
    b = my_lambda * constraint_values
    
    ####################################################
    # TODO: solve for the linear system,
    #       note that you may encounter en error
    #       if no exact solution exists
    ####################################################
    warnings.filterwarnings('error')
    # alpha = []
    
    # Preconditioner: Jacobi (diagonal)
    # M_diag = M.diagonal()
    # def precond(x):
    #     return x / M_diag
    # M = spla.spilu(M.tocsc(), drop_tol=1e-4)
    try:
        # alpha = spla.spsolve(M, b)
        alpha, info = spla.cg(M, b, rtol=1e-5, maxiter=1000)
    except Warning as e:
        print("Error solving linear system:", e)
        alpha, info = spla.cg(M, b, rtol=1e-5, maxiter=1000)
        if info != 0:
            print("Conjugate Gradient did not converge:", info)
            # try least squares solution
            alpha = spla.lsqr(M, b)[0]
    
    alpha = np.clip(alpha, 0, 1).reshape(h, w)
    return alpha

if __name__ == '__main__':
    abs_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(abs_path)
    os.makedirs('./result', exist_ok=True)
    tasks = ["bear", "white_cloth", "woman"]
    task_id = 0
    print(f"Processing task: {tasks[task_id]}...")
    image = cv2.imread(f'./image/{tasks[task_id]}.png')
    trimap = cv2.imread(f'./trimap/{tasks[task_id]}.png', cv2.IMREAD_GRAYSCALE)
    print(f"Image shape: {image.shape}, Trimap shape: {trimap.shape}")
    # scale = 0.75
    # image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    # trimap = cv2.resize(trimap, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    ####################################################
    # TODO: pick up your own background image, 
    #       and merge it with the foreground
    ####################################################
    K = [3, 4, 5, 6, 7, 8, 9, 10]
    for k in K:
        # print(f"Processing K={k}...")
        alpha = knn_matting(image, trimap, knn_k=k, my_lambda=100)
        alpha = alpha[:, :, np.newaxis]
        alpha = (alpha * 255).astype(np.uint8)
        print(f"Saving result for K={k}...")
        cv2.imwrite(f'./result/{tasks[task_id]}_{k}.png', alpha)
