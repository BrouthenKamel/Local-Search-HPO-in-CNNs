# Stochastic Best Improvement with Progressive Halving for CNN Hyperparameter Optimization

## Overview

This project explores **lightweight hyperparameter optimization (HPO)** for Convolutional Neural Networks (CNNs).

The approach combines:

* **Stochastic Best Improvement (SBI)** – a local search method that samples from neighboring hyperparameter configurations.
* **Progressive Halving (PH)** – an adaptive resource allocation scheme inspired by Hyperband, training promising candidates longer while pruning poor ones.

The search is applied to **MobileNetV3-Small**, with **block-wise transfer learning**: early layers are frozen (using pretrained ImageNet weights), while later layers are optimized.
Although a surrogate modeling module was explored, it was ultimately **abandoned due to computational constraints**.


## Repository Structure

```
├── notebooks/             # notebook for experiments in cloud environments (such as Google Colab)
├── scripts/               # Training and experimentation scripts
├── src/
│   ├── optim/             # SBI + Progressive Halving implementation
│   ├── neighborhood/      # Neighbor sampling strategies
│   ├── loading/           # Model and data loaders
│   ├── schema/            # Structural representations of model blocks and parameters
│   ├── training/          # Training utilities
│   ├── surrogate_modeling/ # (Deprecated) surrogate-based accuracy prediction
│   └── utils/             # Helper functions
├── dataset/               # CIFAR-10
├── docs/                  # Approach and environment setup documentation
├── environment.yaml        # Conda environment specification
└── tests/                  # Unit tests
```

## Installation

Follow these [Guidelines](./docs/guides/conda.md)

## Usage

To reproduce the experiments:

```bash
# Pretrained MobileNetV3 fine-tuning
python scripts/pretrained_training.py

# Run Stochastic Best Improvement with Progressive Halving
python src/optim/sa_optimization/main.py
```

## Results

The proposed **Stochastic Best Improvement with Progressive Halving (SBI-PH)** method was tested on **MobileNetV3-Small** using the **CIFAR-10** dataset.
Each optimization run is compared to the baseline pretrained model.

| Run                 | Accuracy    | Change vs Baseline | Parameters | Reduction vs Baseline |
| ------------------- | ----------- | ------------------ | ---------- | --------------------- |
| **Baseline**        | 82.45 %     | –                  | 1.528 M    | –                     |
| **Initial HPO Run** | 82.90 %     | +0.45 %            | 1.528 M    | 0 %                   |
| **Second HPO Run**  | **86.80 %** | **+4.35 %**        | 1.019 M    | **−33 %**             |
| **Third HPO Run**   | 86.48 %     | +4.03 %            | 0.574 M    | **−62 %**             |

The best configuration achieved **86.8 % accuracy**, a gain of about **4 %** over the baseline,
while reducing the model size by roughly **one-third**.
A more compact variant (0.57 M parameters, about **60 % smaller**) maintained nearly the same accuracy.

## Read More

If you find this interesting, you can read the following:
- [Literature Review: Convolutional Neural Network (CNN) Architecture and Hyper-Parameter Optimization (HPO) for Image Classification](https://www.researchgate.net/publication/396448471_Literature_Review_Convolutional_Neural_Network_CNN_Architecture_and_Hyper-Parameter_Optimization_HPO_for_Image_Classification)
- [Stochastic Best Improvement with Progressive Halving Hyperparameter Optimization for Image Classification](TO BE PUBLISHED)

## License

This project is done within the Semestrial Project for "Intelligent Systems & Data" Option at [Ecole nationale Supérieure d’Informatique (ESI)](https://www.esi.dz/presentationesi/)

### Contributors LinkedIn (ordered alphabetically)
- [ABOUD Ibrahim](https://www.linkedin.com/in/ibrahim-aboud04/)
- [AKEB Abdelaziz](https://www.linkedin.com/in/abdelaziz-akeb-4064a9282/)
- [BAGHDADI Mouadh](https://www.linkedin.com/in/mouad-baghdadi-2111871b9/)
- [BROUTHEN Kamel](https://www.linkedin.com/in/brouthen-kamel/)
- [MOKRANE Imed Eddine](https://www.linkedin.com/in/imed-eddine-mokrane-3a1a17232/)
- [SEGHAIRI Abderraouf](https://www.linkedin.com/in/abderraouf-seghairi-1970b8275/)

### Supervisors Google Scholar / Research Gate (ordered alphabetically)
- [ARIES Abdelkrime](https://scholar.google.com/citations?user=FYJlQL4AAAAJ&hl=en)
- [BENATCHBA Karima](https://scholar.google.com/citations?user=7iGY4M0AAAAJ&hl=en&oi=ao)
- [BENBOUZID SI-TAYEB Fatima](https://scholar.google.com/citations?user=ri2J--kAAAAJ&hl=en&oi=ao)
- [BESSEDIK Malika](https://scholar.google.com/citations?user=FTzfUeEAAAAJ&hl=en&oi=ao)
- [FAISAL Touka](https://www.researchgate.net/profile/Touka-Faisal-2)

This project is released under the **MIT License**.