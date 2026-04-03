# IMVFX HW1 — KNN Matting Report Template (2026)

> **Warning**
>
> To make grading efficient, you should follow the format of the template.  
> Keep the structure and answer the questions.
>
> **Follow this template strictly to speed up grading and be concise. Thank you.**

## Basic - 80%

### 1. Feature Design - 10%

**What feature vector did you use?**

* Example: `[R, G, B, x, y]`

**Why this feature?**

* Explain how the feature helps measure pixel similarity (1–3 sentences).


### 2. K Selection - 10%

**Tested K values**:

* List the K values you tried (e.g., K = 5, 10, 20).

**Observations**:

* Describe how changing K affects smoothness, artifacts, or boundary quality.

**Final choice**:

* State the chosen K and justify your decision.

### 3. Linear system Solver - 10%

**Linear system solver used**:

### 4. Results (Required Visuals) - 50%

Include **all** of the following images:

1. Original image
2. Trimap
3. Alpha matte
4. Composited result (with a new background)

> *Missing any image will be considered incomplete.*

#### Bear - 20%

#### Woman - 10%

#### White cloth - 20%

## 5. Advanced - 10%


> **Warning**
>
> For this section, your score will be judged based on the outcome performance compared to your classmates.

### 5.1 White cloth / white wall problem - 10%

![The target image](https://hackmd-prod-images.s3.ap-northeast-1.amazonaws.com/uploads/upload_1a59ac99a517c5d9d9488475b36939b7.png?AWSAccessKeyId=AKIA3XSAAW6AWSKNINWO&Expires=1773914646&Signature=ltn%2BPc5asej6Ptsb7mRZirj1vFc%3D){ width=300px }

Failure case of KNN matting given the trimap.

![The failure case](https://hackmd-prod-images.s3.ap-northeast-1.amazonaws.com/uploads/upload_9fd5c73d9a14522de1729bcf06d2c967.png?AWSAccessKeyId=AKIA3XSAAW6AWSKNINWO&Expires=1773914834&Signature=1rVCN9O6EWxRz8PfOS1BaRTPCNY%3D){ width=300px }

- Describe how you solve this problem.
- Include **all** of the following images:
    1. Alpha matte
    2. Composited result (with a new background)

### 5.2 Based on how much you do - 10%

#### Different Computation Method Comparison - 5%

* Compare different solvers in terms of runtime or quality in the table.

#### Matting Methods Comparison - 5%

* Qualitatively compare with other matting methods in the table.
