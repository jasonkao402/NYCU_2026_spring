---
hackmd:
  url: https://hackmd.io/v5Xhi5JWTKGhygyhmO8hEw
  title: IMVFX HW2-1 — DDPM Report Template (2026)
  lastSync: 2026-04-30T06:35:14.591Z
---
# IMVFX HW2-1 — DDPM Report Template (2026)

> ID: 313552011, Name: 高聖傑
>warning 
To make the grading efficient, you should follow the format of the template. That is keep the structure and answer the questions. 
**Follow this template strictly to speed up grading and be concise, please.
Thank you.**


## Basic - 70%

### MNIST - 30%

**Training Loss Curve** - 15%

* Insert **loss curve figure**
    * x-axis: training step / epoch
    * y-axis: loss

**Generated Samples (5×5)** - 15%

* Insert **one 5×5 image grid**
* Each image: 28×28

### Anime Face - 40%

If you generate only grayscale images, you will get only - 10% 
If you generate color images, you will get - 20%

**Training Loss Curve** - 10%

* Insert **loss curve figure**

**Generated Samples (5×5)** - 10%

* Insert **one 5×5 image grid**
* Each image: 64×64
* Indicate whether the result is **grayscale or color**

## Advanced - 30%

### Noise Scheduler Comparison - 15%

* Insert **comparison figures** with description.
* Explain your observation
* Example:
  * Linear vs Cosine
  * Forward noising visualization
  * Sampling quality comparison

### Benchmark on *Anime Face* - 15% 

Explain the modification that make performance better.

Fill in the table below:

| Method / Setting | **FID** ↓ | **Sampling Steps** ↓ | **Params** ↓ |
| ---------------- | ------------- | ------------------------ | --------------------- |
| DDPM    |               |                          |                       |


**Notes**:

* **FID**: lower is better
* **Sampling steps**: total reverse steps used during inference
* **Model size**: number of parameters or model size
* **Scoring based on Ranking which is relative to classmates**
