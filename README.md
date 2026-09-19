# ADHD Diagnostics Pro

<p align="center">
  <strong>Multimodal Deep Learning for ADHD Research</strong><br/>
  EEG Spectrograms + 3D MRI + Transformer-Based Fusion + Explainable AI
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/Streamlit-Application-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
</p>

<p align="center">
  <img src="https://img.shields.io/github/license/RITESH2127/ADHD-DETECTION-MINI-PROJECT-?style=flat-square" alt="License"/>
  <img src="https://img.shields.io/github/last-commit/RITESH2127/ADHD-DETECTION-MINI-PROJECT-?style=flat-square" alt="Last commit"/>
  <img src="https://img.shields.io/github/repo-size/RITESH2127/ADHD-DETECTION-MINI-PROJECT-?style=flat-square" alt="Repository size"/>
  <img src="https://img.shields.io/github/languages/top/RITESH2127/ADHD-DETECTION-MINI-PROJECT-?style=flat-square" alt="Top language"/>
</p>

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="#research-pipeline">Research Pipeline</a> ·
  <a href="#explainability">Explainability</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#limitations">Limitations</a>
</p>

---

## Overview

**ADHD Diagnostics Pro** is a research-oriented multimodal machine learning project exploring ADHD-versus-control classification using complementary neural information from EEG and structural MRI.

The project combines:

- EEG time-frequency representations
- 3D structural MRI volumes
- Transformer-based modality encoders
- Bidirectional cross-modal attention
- Deep-learning classification
- Explainable AI
- A Streamlit research interface

The objective is to build an end-to-end experimental framework for multimodal representation learning while keeping model behavior inspectable.

> **Research status:** This is an academic/research prototype. It is not a clinically validated diagnostic device and must not be used to diagnose, treat, or make medical decisions about an individual.

---

# Architecture

## End-to-End System

~~~mermaid
flowchart TD
    A["Raw EEG"] --> B["Band-Pass Filtering"]
    B --> C["Windowing"]
    C --> D["Time-Frequency Spectrograms"]

    E["Structural MRI"] --> F["NaN Handling"]
    F --> G["Intensity Normalization"]
    G --> H["3D Resampling"]

    D --> I["EEG Spectrogram Transformer"]
    H --> J["3D MRI Transformer"]

    I --> K["Cross-Modal Attention"]
    J --> K
    K --> L["Feature Fusion"]
    L --> M["Classification Head"]
    M --> N["ADHD / Control"]
    N --> O["Explainability"]
~~~

## Core Model

### EEG branch

EEG recordings are filtered, divided into overlapping windows, converted into channel-wise spectrograms, logarithmically transformed, patch-projected, and processed by a Transformer encoder.

Current research configuration:

| Parameter | Value |
|---|---:|
| EEG channels | 19 |
| Sampling rate | 128 Hz |
| Band-pass | 0.5–50 Hz |
| Window size | 256 samples |
| Step size | 128 samples |
| Embedding dimension | 128 |
| Attention heads | 4 |
| Transformer layers | 2 |

### MRI branch

Structural MRI volumes are loaded as NIfTI data, sanitized, normalized, resampled to 96 × 96 × 96, converted into 3D patches, and processed through a Transformer-based encoder.

### Cross-modal fusion

Instead of simply concatenating independent features, the model uses bidirectional multi-head cross-attention so that learned EEG and MRI representations can interact before classification.

~~~text
EEG embedding ──────────────┐
                            │
                            ▼
                    Cross-Modal Attention
                            ▲
                            │
MRI embedding ──────────────┘
                            │
                            ▼
                    Fused representation
                            │
                            ▼
                     Classification
                            │
                     ┌──────┴──────┐
                     ▼             ▼
                   ADHD         Control
~~~

---

# Research Pipeline

## EEG Processing

1. Load subject-level EEG data.
2. Remove missing observations.
3. Select the EEG channels.
4. Apply a fourth-order Butterworth band-pass filter.
5. Generate overlapping windows.
6. Compute spectrograms.
7. Apply logarithmic transformation.
8. Preserve subject identifiers for grouped evaluation.

~~~text
Raw EEG
   ↓
Band-pass filter
   ↓
Sliding windows
   ↓
Channel-wise spectrograms
   ↓
Log transformation
   ↓
Patch representation
   ↓
EEG Transformer
~~~

## MRI Processing

1. Load NIfTI volumes with NiBabel.
2. Replace invalid numerical values safely.
3. Normalize intensity values.
4. Resize volumes to 96 × 96 × 96.
5. Convert volumes to tensors.
6. Extract 3D patch representations.
7. Encode patches using a Transformer.

## Subject-Aware Validation

The research notebook uses subject identifiers with GroupShuffleSplit to reduce the risk of placing windows from the same participant into both training and validation partitions.

This distinction is important because overlapping windows from a single participant are not independent biological observations.

---

# Explainability

The project includes an explainability layer to make model behavior easier to inspect.

## Grad-CAM

Grad-CAM produces activation maps associated with model predictions.

~~~text
Input
  ↓
Feature extraction
  ↓
Target-class gradients
  ↓
Weighted activation map
  ↓
Heatmap
  ↓
Visual interpretation
~~~

## LIME

LIME provides local feature attribution by creating perturbed samples, observing model responses, and fitting an interpretable local approximation.

~~~text
Input image
    ↓
Superpixel segmentation
    ↓
Perturbed samples
    ↓
Model predictions
    ↓
Local surrogate model
    ↓
Feature attribution
~~~

### Important interpretation principle

XAI visualizations describe aspects of model behavior. They do not establish causal biomarkers, biological mechanisms, or clinical validity.

---

# Technology Stack

| Layer | Technology |
|---|---|
| Language | Python |
| Deep Learning Research | PyTorch |
| Application / Inference | TensorFlow / Keras |
| Web Interface | Streamlit |
| EEG Processing | NumPy, SciPy, Pandas |
| MRI Processing | NiBabel |
| Computer Vision | OpenCV |
| Image Processing | Pillow, scikit-image |
| Explainable AI | LIME, Grad-CAM-oriented visualization |
| Visualization | Matplotlib |
| Data Splitting | scikit-learn |
| Experiment Environment | Jupyter / Kaggle-compatible workflow |

---

# Project Structure

~~~text
ADHD-DETECTION-MINI-PROJECT-/
│
├── app.py
│   └── Streamlit application layer
│
├── ADHD_MINI_PROJECT.ipynb
│   └── Main research notebook
│
├── Copy_of_ADHD_MINI_PROJECT.ipynb
│   └── Additional notebook version
│
├── notebook18c25b8789 (1).ipynb
│   └── Extended experimentation / training notebook
│
├── requirements.txt
│   └── Application dependencies
│
├── run.txt
│   └── Launch command
│
├── run_app.bat
│   └── Windows launcher
│
├── .python-version
│   └── Python version specification
│
├── .gitignore
├── LICENSE
└── README.md
~~~

---

# Quick Start

## 1. Clone

~~~bash
git clone https://github.com/RITESH2127/ADHD-DETECTION-MINI-PROJECT-.git
cd ADHD-DETECTION-MINI-PROJECT-
~~~

## 2. Create an environment

### Windows

~~~powershell
python -m venv venv
venv\Scripts\activate
~~~

### macOS / Linux

~~~bash
python3 -m venv venv
source venv/bin/activate
~~~

## 3. Install dependencies

~~~bash
python -m pip install --upgrade pip
pip install -r requirements.txt
~~~

## 4. Launch

~~~bash
streamlit run app.py
~~~

The application normally opens at:

~~~text
http://localhost:8501
~~~

Windows users can also use the included run_app.bat launcher.

---

# Dataset Workflow

The research notebooks reference public ADHD-related EEG and ADHD-200 MRI datasets hosted through Kaggle-compatible environments.

The original notebook paths are environment-specific. When running the notebooks locally, replace those paths with the local dataset locations.

### EEG

The training workflow expects subject-level EEG data containing identifiers, class labels, and EEG channels.

### MRI

The MRI pipeline expects preprocessed NIfTI anatomical volumes.

> Always review dataset licensing, terms of use, participant privacy requirements, and redistribution restrictions before using or sharing research data.

---

# Training Workflow

~~~mermaid
flowchart LR
    A["EEG CSV"] --> B["EEG preprocessing"]
    B --> C["Spectrogram dataset"]

    D["MRI NIfTI"] --> E["MRI preprocessing"]
    E --> F["3D tensor dataset"]

    C --> G["Subject-aware split"]
    F --> G

    G --> H["EEG Transformer"]
    G --> I["MRI Transformer"]

    H --> J["Cross-modal attention"]
    I --> J

    J --> K["Classifier"]
    K --> L["Loss"]
    L --> M["Backpropagation"]
    M --> H
    M --> I
~~~

The current notebook uses Adam optimization, cross-entropy loss, GPU acceleration when available, and automatic mixed precision when CUDA is available.

---

# Evaluation Framework

A serious neuroimaging ML evaluation should go beyond a single accuracy value.

| Metric / Analysis | Purpose |
|---|---|
| Accuracy | Overall classification correctness |
| Precision | Reliability of positive predictions |
| Recall / Sensitivity | Detection of positive cases |
| Specificity | Identification of controls |
| F1-score | Precision-recall balance |
| ROC-AUC | Threshold-independent ranking performance |
| Confusion matrix | Error distribution |
| Subject-level validation | Participant-level generalization |
| External validation | Robustness on independent data |
| Calibration | Reliability of predicted probabilities |
| Ablation studies | Contribution of each modality/component |

A strong internal metric does not by itself establish clinical validity or generalization.

---

# Limitations

This repository should be understood as a research prototype.

### Dataset limitations

Results can be affected by sample composition, acquisition protocols, hardware, preprocessing decisions, demographic characteristics, and dataset-specific artifacts.

### Generalization

Performance on a development dataset does not establish robustness across unseen participants, institutions, devices, populations, or independent datasets.

### Multimodal alignment

The current research implementation uses dataset indexing and length matching when pairing EEG-derived samples with MRI files. A production-grade multimodal research system should explicitly align modalities using validated participant identifiers and acquisition metadata.

### Explainability

Grad-CAM and LIME explain model behavior; they do not prove that highlighted patterns are causal neurological biomarkers.

### Clinical validity

The system has not been clinically validated and should not be used for diagnosis, treatment, medication decisions, triage, or other medical decision-making.

---

# Responsible Use

### Intended for

- Academic research
- Machine learning experimentation
- Neuroimaging education
- Multimodal representation-learning research
- Explainable AI experimentation
- Research-oriented application demonstrations

### Not intended for

- Self-diagnosis
- Clinical diagnosis
- Treatment selection
- Medication decisions
- Medical triage
- Educational or employment decisions based solely on model output

Any future clinical application would require substantially more evidence, independent validation, appropriate governance, regulatory assessment, and qualified professional oversight.

---

# Future Research

## Modeling

- Subject-aware multimodal alignment
- Stronger cross-modal Transformer architectures
- Temporal Transformer modeling for raw EEG
- Self-supervised pretraining
- Contrastive EEG-MRI representation learning
- Uncertainty-aware inference
- Modality ablation experiments

## Evaluation

- Nested subject-level cross-validation
- Independent external validation
- Dataset-shift analysis
- Probability calibration
- Confidence intervals
- Statistical significance testing
- Robustness benchmarking

## Explainability

- Integrated Gradients
- SHAP
- Attention rollout
- Modality-specific attribution
- Counterfactual explanations
- Explanation stability analysis

## Engineering

- Configuration management
- Automated tests
- GitHub Actions CI
- Experiment tracking
- Model versioning
- Containerized deployment
- Hardware-aware inference optimization

---

# Reproducibility Checklist

For meaningful research comparisons:

- Fix dataset versions.
- Record preprocessing parameters.
- Preserve subject-level partitions.
- Record random seeds.
- Version model configurations.
- Save training metrics and checkpoints.
- Separate model artifacts from source code.
- Evaluate on unseen participants.
- Perform independent external validation.
- Report failure cases rather than only successful predictions.

For large model artifacts, Git LFS or dedicated model storage is preferable to committing large binaries directly to the repository.

---

# Research Roadmap

~~~mermaid
timeline
    title ADHD Diagnostics Pro
    2026 : Multimodal EEG + MRI prototype
         : Transformer modality encoders
         : Cross-modal attention
         : Streamlit research interface
    Next : Subject-level multimodal alignment
         : Stronger validation
         : External dataset evaluation
         : Calibration and uncertainty
         : Quantitative XAI evaluation
    Future : Robust multimodal representation learning
           : Reproducible experiment tracking
           : Large-scale independent validation
~~~

---

# Citation

If you use this repository in an academic project, report, presentation, or derivative research work:

~~~bibtex
@software{ritesh_kumar_adhd_diagnostics_pro,
  author  = {Ritesh Kumar},
  title   = {ADHD Diagnostics Pro},
  year    = {2026},
  url     = {https://github.com/RITESH2127/ADHD-DETECTION-MINI-PROJECT-},
  license = {MIT}
}
~~~

---

# License

Released under the **MIT License**.

See [LICENSE](LICENSE) for the complete license text.

---

# Author

<p align="center">
  <strong>Ritesh Kumar</strong><br/>
  Computer Science Engineering
</p>

<p align="center">
  <a href="https://github.com/RITESH2127">GitHub Profile</a>
  ·
  <a href="https://github.com/RITESH2127/ADHD-DETECTION-MINI-PROJECT-">Repository</a>
</p>

---

<p align="center">
  <sub>Research, engineering, and responsible exploration of multimodal AI for neuroimaging.</sub>
</p>
