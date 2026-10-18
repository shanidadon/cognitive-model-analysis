# Beyond Explicit Ratings: Reaction Time as an Implicit Measure of Category Structure

**Computational cognitive science · Deep learning · Human behavior**

**Shani Dadon** · Research with **Itay Inbar** and **Tal Golan**  
**Brains & Machines Lab**, School of Brain Sciences and Cognition, Ben-Gurion University of the Negev

## Overview

What makes an image a *typical* example of a category? Can computational models of visual representations and category structure predict how quickly people classify that image?

Typicality is often studied using explicit ratings, which may be influenced by the way a question is asked. This project investigates **human classification reaction time (RT) as an implicit behavioral correlate of category typicality**. The central prediction is that images with higher computational typicality scores will tend to elicit faster human classifications.

Using **CIFAR-10H** behavioral responses, this project connects three lines of analysis:

1. **Visual representation validation:** Extract features from pretrained deep neural networks and assess their usefulness for classifying low-resolution CIFAR-10 images.
2. **Behavioral validation:** Test whether human reaction times increase with classification uncertainty, as predicted by **Hyman–Hick's law**.
3. **Model–behavior comparison:** Relate human reaction times to previously computed typicality scores from **10 neural network representations × 3 cognitive categorization models**.

> **Interpretation:** RT is treated as a potential *implicit proxy* for typicality, not as a direct or exclusive measurement of typicality. A relationship between entropy and RT supports its sensitivity to decision uncertainty but does not, by itself, establish that RT measures typicality.

## Key findings

| Analysis | Finding |
| --- | --- |
| Representation validation | Selected pretrained visual representations supported approximately **74–78%** CIFAR-10 classification accuracy using Ridge classifiers, versus a **10%** chance baseline (reported in the research poster). |
| Uncertainty and reaction time | The research poster reports a positive association in the higher-entropy subset (**Spearman ρ = 0.689**, **p < 0.0001**), consistent with Hyman–Hick's law. |
| Typicality and reaction time | **29 of 30** representation × categorization combinations showed negative Spearman correlations in `results/correlations/summary.csv`. |
| Strongest typicality association | **SigLIP2 + Decision-Bound**, **Spearman ρ = −0.466** (`n = 10,000` images). |

![Spearman correlations across computational model combinations](assets/typicality_rt_correlations.png)

*Figure: Spearman correlation between model-derived typicality and human mean RT. Negative values indicate that images assigned higher typicality tend to be classified faster. Generated from the committed correlation summary.*

## Research questions

- Which visual representational spaces best capture behaviorally relevant category structure?
- How do **prototype**, **exemplar**, and **decision-bound** accounts of categorization compare when evaluated against implicit human behavior?
- Does human classification time reflect uncertainty in a manner consistent with Hyman–Hick's law?

## Data

The analyses use **CIFAR-10H**, which extends CIFAR-10 with human image-classification judgments (approximately 51 judgments per image), and associated behavioral measurements. CIFAR-10 images are originally **32 × 32 pixels**.

Key local files:

- `data/cifar10h-counts.npy` and `data/cifar10h-probs.npy`: distributions of human category labels.
- `data/image_reaction_times.csv`: image-level human reaction-time summaries.
- `data/cifar10_individual_typicality_scores (1).csv`: model-based typicality scores from earlier lab research, used here for behavioral comparisons.

The typicality analysis matches images by `category` and `image_index` and uses the **CIFAR-10 test split**.

## Methods

### 1. Frozen deep neural network representations

A **PyTorch-based feature extraction pipeline** loads pretrained visual encoders through the lab's `bml_vision` toolkit, runs images in batches without gradient updates, and converts extracted features into NumPy arrays for downstream analysis. CIFAR-10 images are upsampled (bicubic resizing) and normalized for the pretrained models.

The study considers 10 representations across three families:

| Family | Representations |
| --- | --- |
| Vision–language | CLIP ViT-L; SLIP ViT-L (CLIP); SLIP ViT-L (SLIP); SigLIP2 |
| Supervised image classification | Google ViT-L; ConvNeXt-L; Inception-V3; CORnet-RT |
| Self-supervised | DINOv2 ViT-L; SLIP ViT-L (SimCLR) |

Feature quality is assessed with **scikit-learn `RidgeClassifierCV`**, using **5-fold cross-validation** for regularization selection. The experiment pipeline calculates accuracy, precision, recall, and F1. The reported accuracy range above reflects successful research runs, **not** the separate GPU-unavailable failures recorded in `results/all_models_results.csv`.

### 2. Classification uncertainty and Hyman–Hick's law

Human classification uncertainty is estimated from each image's label distribution using **Shannon entropy** with the **Miller–Madow bias correction**:

\[
H(X)=-\sum_i p_i\log_2 p_i, \qquad \widehat H_{MM}=\widehat H+\frac{k-1}{2N\ln 2}.
\]

The analysis relates entropy to mean log reaction time, examines near-unanimous versus higher-disagreement images, and computes correlation and group-comparison statistics. The expected pattern is **slower responses for images with greater classification uncertainty**.

Implementation: `src/experiment/hicks_law_analysis.py`.

### 3. Cognitive category models

Typicality predictions were computed in earlier lab work using three accounts of category structure:

- **Prototype:** category structure summarized by central tendencies.
- **Exemplar:** category judgments informed by similarity to individual examples.
- **Decision-Bound:** category membership characterized by boundaries in representational space.

Combining these three accounts with ten neural representations yields **30 sets of typicality predictions**. This repository's behavioral analysis **uses those existing scores**; it does not claim to have originally computed them.

### 4. Typicality–RT correlation analysis

The project joins per-image typicality predictions with human mean reaction times, then computes **Spearman's ρ** and **Kendall's τ** for each neural-network × categorization-model combination. Spearman correlation is appropriate here because the main prediction concerns the **rank-order relationship** between typicality and response speed.

Implementation: `src/experiment/typicality_rt_correlation.py`; results: `results/correlations/summary.csv`.

## My contributions

My work in this project focused on extending prior model-based typicality research to human implicit behavioral measures:

- **Implemented and ran a PyTorch feature-extraction and evaluation workflow** for pretrained visual models using the lab's model-loading infrastructure.
- **Validated neural representations** with Ridge classification and model-performance metrics.
- **Analyzed human reaction times and classification uncertainty**, including entropy estimation, Hyman–Hick's law tests, and statistical comparisons.
- **Integrated existing model-based typicality scores with human behavioral data** and systematically evaluated correlations across 30 model combinations.
- **Produced and interpreted statistical summaries and visualizations** comparing computational predictions with human behavior.

**Attribution:** The pretrained architectures and `bml_vision` infrastructure were not developed from scratch in this project. The typicality scores were calculated in earlier research by Itay Inbar; this project contributes the feature-validation and reaction-time analyses and their integration with those scores.

## Repository structure

```text
.
├── configs/
│   ├── cifar10.yaml                  # CIFAR-10 feature-extraction experiment
│   └── local.yaml                    # Additional/template configuration
├── data/                             # Human labels, RTs, typicality scores
├── src/
│   ├── cli.py                        # Experiment command-line interface
│   └── experiment/
│       ├── models.py                 # Pretrained model wrappers
│       ├── pipeline.py               # Feature extraction and Ridge evaluation
│       ├── hicks_law_analysis.py     # Entropy and reaction-time analysis
│       └── typicality_rt_correlation.py
├── scripts/
│   └── plot_correlations.py          # Correlation visualization utility
├── results/
│   ├── correlations/                 # Correlation summaries and merged data
│   ├── hicks_law/                    # Behavioral analysis outputs
│   └── model_accuracy.csv            # Recorded model validation results
├── notebooks/                        # Exploratory analyses
├── assets/                           # Figures displayed in this README
├── requirements.txt
└── pyproject.toml
```

## Installation and usage

Use **Python 3.10+** in a virtual environment. Run commands from the repository root.

```bash
git clone https://github.com/shanidadon/cognitive-model-analysis.git
cd cognitive-model-analysis
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Additional environment requirements:** The feature-extraction workflow imports the lab's **`bml_vision`** toolkit, which is not declared in `requirements.txt` and may require separate lab-specific installation and model weights. Some pretrained models may require an available CUDA GPU. The standalone statistical analyses also import **SciPy**, which is not explicitly listed in the current requirements file. Install it if it is not already present (`pip install scipy`).

### Inspect the experiment configuration

```bash
python -m src.cli validate configs/cifar10.yaml
```

### Run the representation experiment

```bash
python -m src.cli run configs/cifar10.yaml --ridge
```

This command depends on the pretrained model toolkit and compatible hardware; it has **not been verified end-to-end in a clean environment**.

### Run the Hyman–Hick analysis

```bash
python -m src.experiment.hicks_law_analysis
```

This analysis may download the CIFAR-10H raw response archive if it is absent. Outputs are written to `results/hicks_law/`.

### Recompute all typicality–RT correlations

```bash
python -m src.experiment.typicality_rt_correlation --all
```

**Known code issue:** The uploaded version of `typicality_rt_correlation.py` contains an extraneous character (`ן`) immediately before the `summary = pd.DataFrame(...)` line in `compute_all_combos()`. Remove that character before running this command. The existing `results/correlations/summary.csv` can still be inspected without rerunning the script.

### Explore existing results

- `results/correlations/summary.csv`: all 30 Spearman and Kendall correlations.
- `results/hicks_law/hicks_law_results.csv`: per-image uncertainty and RT summaries.
- `results/model_accuracy.csv`: recorded representation-validation accuracies.

## Limitations and next steps

- Reaction time is affected by factors beyond typicality, including visual difficulty and response uncertainty.
- Correlations describe associations and should not be interpreted as evidence of causation.
- Comparing correlation magnitudes across model combinations warrants further statistical testing.
- Future analyses can compare **implicit RT-based findings** directly against **explicit human typicality ratings** and evaluate robustness across additional datasets.

## Research team and acknowledgments

**Shani Dadon**, **Itay Inbar**, and **Tal Golan**  
Brains & Machines Lab, School of Brain Sciences and Cognition, Ben-Gurion University of the Negev.

This project builds on prior typicality-modeling work and lab infrastructure. CIFAR-10H provides the human classification data underlying the behavioral analyses.
## Research Poster

Explore the research questions, methodology, and key findings in the full research poster.

📄 [View Research Poster (PDF)](Implicit_Typicality_Poster_new.pdf)
