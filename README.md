## CALM
Code for the article **'CALM: A Copula-Augmented Lightweight Deep Learning Model for Robust Pulmonary Function Assessment from CT Imaging'**
## Introduction
Chronic respiratory diseases (CRDs) pose a significant threat to public health and life expectancy. Pulmonary function testing (PFT) is crucial for the evaluation and diagnosis of CRDs, yet its widespread adoption and patient compliance remain limited. Recent studies have demonstrated strong correlations between chest CT imaging features and PFT metrics, suggesting a viable alternative approach for assessing pulmonary function.

This paper proposes a copula-augmented lightweight deep learning model (CALM) for robust pulmonary function assessment based on CT imaging. Specifically, our study focuses on three key pulmonary function indices: forced vital capacity (FVC), forced expiratory volume in 1 second (FEV1), and total lung capacity (TLC). The proposed CALM framework integrates a multi-view guided attention mechanism and adopts a copula-based loss function specifically designed to model the conditional dependencies among pulmonary function index parameters. Experiments on a real-world hospital dataset demonstrate that CALM achieves higher prediction accuracy compared to baseline methods while requiring fewer trainable parameters.
## Model
Figure 1 shows the flowchart of the proposed CALM.
![image](https://github.com/xy015/CALM/blob/main/CALM.png?raw=true)
The input consists of a series of CT images from a specific patient, fused through three views.The network consists of two stages. In the first stage, training is performed using a conventional empirical loss function to estimate the parameters required for the Copula Loss. This stage also serves as a baseline for evaluating the performance improvement achieved in the second stage. In the second stage, the model is trained using the Copula Loss, ultimately predicting three pulmonary function indices.
