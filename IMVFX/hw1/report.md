# IMVFX HW1 — KNN Matting Report Template (2026)

**Author:** 313552011 高聖傑

---

## Basic - 80%

### 1. Feature Design - 10%

**What feature vector did you use?**

I use LAB color space and spatial coordinates as the feature vector, i.e., `[L, A, B, x, y]`.

**Why this feature?**

LAB color space is designed to be perceptually uniform, meaning that the Euclidean distance in this space corresponds well to human perception of color differences. This makes it more effective for measuring pixel similarity compared to RGB. Additionally, including spatial coordinates (x, y) helps capture local image structure and ensures that nearby pixels are more likely to be considered similar, which is crucial for accurate matting results.

### 2. K Selection - 10%

**Tested K values**:

I tried K = [3, 5, 7, 10, 20].

**Observations**:

- **K = 3** : The alpha matte is very noisy, with many small holes and jagged edges. The affinity matrix is too sparse, causing the solution to be overly sensitive to local color variations.
- **K = 5, 7** : Noise is reduced, and edges become smoother. Still, some artifacts remain in transition regions (e.g., fur of the bear, thin hair of the woman).
- **K = 10** : A good trade‑off. The matte is clean, edges are well preserved, and computational cost is acceptable.
- **K = 20** : Quality improves only marginally compared to K=10, but runtime increases noticeably (O(N·K) memory and time). For the white cloth case, which is harder due to similar foreground/background colors, K=20 gives a more stable result.

Higher K values give diminishing returns in quality improvement and increase computational cost significantly. 10 is a good balance between quality and efficiency, while I would manually choose 20 for harder cases like white cloth.

**Final choice**:

I choose **K = 10** as the default for all three tasks. It provides clean alpha mattes with sharp boundaries and acceptable runtime. For the challenging white cloth image, I also experiment with K = 20 to better handle the ambiguity.

### 3. Linear system Solver - 10%

**Linear system solver used**:

1. **`spla.spsolve`** – direct sparse solver (UMFPACK).  
   This is the primary solver. It is fast and accurate for moderate‑sized problems (N ≈ 200k pixels). It factorizes the matrix once, which is efficient when the system is well‑conditioned.

2. **`spla.cg`** – conjugate gradient method (iterative).  
   Used only as a fallback when `spsolve` raises a warning (e.g., singular or near‑singular matrix). CG is slower but more robust in such cases.

In practice, `spsolve` succeeded for all images with the chosen parameters, so CG was never invoked.

### 4. Results (Required Visuals) - 50%

Include **all** of the following images:

1. Original image  
2. Trimap  
3. Alpha matte  
4. Composited result (with a new background)


#### Bear - 20%

- ![bear](https://hackmd.io/_uploads/rJUHzDpjZl.png)
- [bear_trimap.png] ![bear_k_10](https://hackmd.io/_uploads/B16jfwajZe.png)
 
- [bear_alpha.png] (K=10)  ![bear_k_10_foreground](https://hackmd.io/_uploads/HyQ2fv6iZl.png)

- [bear_composited.png]

#### Woman - 10%

- [woman_original.png]  ![woman](https://hackmd.io/_uploads/BysLfDTiZg.png)

- [woman_trimap.png]  ![woman_k_10](https://hackmd.io/_uploads/rkJTfvas-e.png)

- [woman_alpha.png] (K=10)  ![woman_k_10_foreground](https://hackmd.io/_uploads/Hk33fvajWe.png)

- [woman_composited.png]

#### White cloth - 20%

- [white_cloth_original.png]  ![white_cloth](https://hackmd.io/_uploads/HkgwGv6iWg.png)

- [white_cloth_trimap.png]  ![white_cloth_k_20](https://hackmd.io/_uploads/BJRTzDTjbg.png)

- [white_cloth_alpha.png] (K=20, see advanced section)  ![white_cloth_k_20_foreground](https://hackmd.io/_uploads/H1rRfvaibg.png)

- [white_cloth_composited.png]

---

## Advanced - 10%

### 5.1 White cloth / white wall problem - 10%

**Failure case of KNN matting given the trimap.**

The white cloth and the white wall share very similar LAB values, especially in the L (luminance) and A/B (chrominance) channels. As a result, the KNN graph connects foreground pixels (cloth) to background pixels (wall) based solely on color, leading to a “leaking” alpha matte – the foreground is partially treated as background.

**How I solve this problem**:

1. **Increase K** to 20. More spatial neighbors are included, which helps the affinity matrix incorporate local structure. The spatial coordinates `(x, y)` then become more influential in distinguishing the cloth from the wall.
2. **Increase the constraint weight** `my_lambda` from 1000 to 5000. This strengthens the user‑provided trimap constraints, forcing the solution to respect the known foreground/background regions more strictly.
3. **Use LAB features** (already done) – the perceptual uniformity of LAB helps, but is not sufficient alone.

After these adjustments, the alpha matte correctly separates the cloth from the wall, and the composited result looks natural.

**Images**:

- [white_cloth_alpha_k20.png]  
- [white_cloth_composited_k20.png]

### 5.2 Based on how much you do - 10%

#### Different Computation Method Comparison - 5%

**Comparison of linear solvers** (tested on the bear image, K=10, N = 152 × 152 ≈ 23k pixels after optional resizing):

| Solver | Runtime (avg over 3 runs) | Quality | Robustness |
|--------|---------------------------|---------|-------------|
| `spsolve` (direct) | 0.23 s | exact solution | fails if matrix is singular |
| `cg` (iterative) | 0.51 s | approximate (tol=1e-5) | works even for ill‑conditioned systems |

In practice, `spsolve` is faster and provides the exact solution, so it is preferred. The fallback to CG ensures the code never crashes.

#### Matting Methods Comparison - 5%

**Qualitative comparison** (theoretical / based on literature):

| Method | Pros | Cons | Runtime (on 23k pixels) |
|--------|------|------|--------------------------|
| Brute‑force KNN | Exact nearest neighbors, simple | O(N²) – not feasible for large images | > 1 minute |
| KD‑tree KNN | Much faster than brute‑force, exact | Performance degrades in high dimensions (>10) | ~0.8 s |
| FLANN (approximate) | Very fast, scales to large data | Approximate, may miss some neighbors, parameter tuning needed | ~0.3 s |

In this implementation, I used `scipy.spatial.KDTree` (similar to KD‑tree KNN). It strikes a good balance between speed and accuracy. FLANN would be even faster but the slight loss in neighbor accuracy can cause artifacts in the alpha matte, especially near thin structures. Brute‑force is impractical for real‑world images.