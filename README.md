# plant-geopriors

**Late-fusion of visual and geographic-temporal context for fine-grained plant species classification.**

Neha Sajja, Ceci Zhang, Manini Banerjee — DS26 Final Project (May 2026)

---

## Overview

Fine-grained plant species classification from photographs is hard: many species are visually near-identical yet ecologically distinct. This project asks whether **geographic and seasonal priors** — latitude, longitude, and day-of-year — can improve classification when fused with a CNN trained on images.

Working with **44 species** across three California families (Asteraceae, Poaceae, Apiaceae) sourced from GBIF / iNaturalist 2025, we train two single-modality baselines and two late-fusion models, then compare their accuracy at the overall, family, and per-species level.

This work is directly inspired by Mac Aodha et al. (2019), *Presence-Only Geographical Priors for Fine-Grained Image Classification* (ICCV).

---

## Research Question

> **Do geographic and seasonal priors improve plant species classification accuracy, and for which species do they help most?**

---

## Headline Result

| Model | Balanced Accuracy (Test) |
| --- | --- |
| Geo Baseline (MLP) | 0.108 |
| ResNet50 (Image Only) | 0.448 |
| Weighted Addition (α·image + β·context) | 0.489 |
| **Fusion MLP (trainable head over logits)** | **0.548** |

The Fusion MLP achieves the highest balanced accuracy and brings every species off zero — the geo-only baseline produced 0% accuracy on 21 of 44 species, and the image-only baseline on 1. Gains concentrate in **range-restricted species with clean phenology** (e.g., Sanicula arctopoides: 87%, Encelia californica / farinosa: 80%) while widespread weedy species (e.g., Centaurea solstitialis) can regress because the geographic prior is uninformative for them.

---

## Target Families

| Family | Visual Challenge |
| --- | --- |
| **Asteraceae** | Many small yellow composites, hard to separate even for trained botanists |
| **Poaceae** | Grasses — notoriously hard to distinguish from photographs |
| **Apiaceae** | Umbel-bearing species with near-identical inflorescences |

---

## Data

**Source:** California-restricted GBIF / iNaturalist 2025 export for Asteraceae, Poaceae, and Apiaceae. Filtered to species with ≥ 300 observations and sufficient seasonal and geographic spread → **44 species**.

**Two aligned datasets** with the same 44-class label space:

| Dataset | Rows | Train / Val / Test | Use |
| --- | --- | --- | --- |
| Image-backed | 4,400 | 3,080 / 660 / 660 | ResNet50 + fusion head training & evaluation |
| Expanded (geo-only) | 103,314 | 72,319 / 15,497 / 15,498 | Geo MLP baseline (preserves image-backed splits verbatim — no leakage) |

**Feature engineering — sinusoidal encoding** (avoids the day-365 / day-1 wraparound problem):

```
lat_sin/cos = sin/cos(π · lat / 90)
lon_sin/cos = sin/cos(π · lon / 180)
doy_sin/cos = sin/cos(π · (doy − 183) / 183)
```

**Species selection:** Three strategies compared in [04_species_comparison.ipynb](notebooks/04_species_comparison.ipynb): raw count (`top_n`), seasonal coverage (`time_diverse`), and joint temporal-spatial diversity via k-means on (lat, lon, doy) (`time_geo_diverse`). The third was adopted because it spreads training mass across each species' range rather than concentrating on hotspot counties.

---

## Models

### 1. Image-Only Baseline — ResNet50
ImageNet-pretrained, frozen except for a replaced final FC layer (2048 → 44). Standard ImageNet augmentations, weighted cross-entropy on inverse capped frequencies, Adam @ lr = 1e-3.

### 2. Geo Baseline — Small MLP
Four-layer MLP (three hidden blocks of 256, ReLU, dropout 0.3) mapping the 6 sinusoidal features → 44 logits. Trained on the expanded 103k-row split.

### 3. Weighted-Addition Fusion (zero trainable parameters)
Both baselines loaded frozen. Compute `final_logits = α · image_logits + β · context_logits`, sweep β ∈ [0, 1] on a 21-point grid with α = 1 − β on validation; evaluate the chosen (α, β) once on test. This is the Bayesian factorization `P(y | I, x) ∝ P(y | I) · P(y | x)` from Mac Aodha et al. (2019).

### 4. Trainable Fusion MLP
Small head over concatenated baseline logits: R⁸⁸ → Linear → R²⁵⁶ → ReLU + Dropout → R⁴⁴. Backbones frozen; only the ~25k-parameter head trains. Batch size 256, Adam @ lr = 1e-3, per-epoch per-species capping at 500, early stopping on validation balanced accuracy.

---

## Key Findings

- **Per-family:** All three families end up at similar fusion-MLP balanced accuracy (Apiaceae 0.53, Asteraceae 0.49, Poaceae 0.55), but the **gain from fusion** differs sharply. **Apiaceae** benefits most (ResNet 0.41 → Fusion 0.53, **+0.12**) — the umbellifers are visually ambiguous, so the geographic prior has the most room to help. **Asteraceae** sees a modest lift (0.43 → 0.49, **+0.06**). **Poaceae** is already the strongest family under image alone (0.54) and gains almost nothing from fusion (0.55, **+0.01**) — most California grasses have wide overlapping ranges, so the geo prior carries little discriminative signal. The geo baseline alone is uniformly weak across families (0.06–0.13).
- **Per-species (Fusion vs ResNet):** 10 species regress slightly; the species that improve do so substantially (up to +0.60, e.g., Daucus carota: 0.00 → 0.60). The trainable head is more aggressive than weighted addition in both directions — it amplifies geo signal where it helps, but can over-correct where it doesn't.
- **Regression case study:** Centaurea solstitialis drops from 67% (ResNet) to 33% (Fusion). Its geographically widespread distribution overlaps with its top confusers, so the geo prior adds noise rather than signal.
- **Deployment tradeoff:** weighted addition is *safer* (fewer regressions); Fusion MLP wins on *overall balanced accuracy* but takes bigger swings per species.
- **Caveat:** the image-backed test set has only 15 observations per species, so a single flip = 0.067 accuracy swing. Per-species comparisons should be read with that in mind.

---

## Project Structure

```
plant-geopriors/
├── notebooks/
│   ├── 01_eda.ipynb                                     # California EDA
│   ├── 02_results.ipynb                                 # Aggregated results & figures
│   ├── 03_us_species_selection.ipynb                    # US-wide selection (exploratory)
│   ├── 04_species_comparison.ipynb                      # top_n vs time_diverse vs time_geo_diverse
│   ├── 05_making_datasets_BaselineModels.ipynb          # Dataset construction
│   ├── 06_EDA_Baseline_Modeling_and_Pipeline_Development.ipynb
│   ├── 07_weighted_addition_model.ipynb                 # α·image + β·context fusion
│   └── 08_Expanded_Baselines_MLP_combination.ipynb      # Trainable Fusion MLP head
├── src/
│   ├── data.py           # Loading, preprocessing, sinusoidal encoding
│   ├── models.py         # ResNet50, Geo MLP, Fusion MLP definitions
│   ├── train.py          # Training loop
│   └── eval.py           # Evaluation metrics (balanced accuracy, per-family, per-species)
├── scripts/
│   ├── prepare_data.py
│   ├── build_expanded_splits.py    # Builds the 103k-row split preserving image-backed assignments
│   ├── run_train.py
│   └── run_eval.py
├── configs/
│   ├── exp_image.yaml
│   ├── exp_context.yaml
│   └── exp_multimodal.yaml
├── artifacts/
│   ├── master_observations.csv                  # 4,400-row image-backed table
│   ├── master_observations_geo_expanded.csv     # 103,314-row geo-expanded table
│   ├── split_iid.csv                            # Image-backed splits
│   ├── split_geo_expanded.csv                   # Expanded splits (preserves image-backed assignments)
│   ├── label_encoder.pkl
│   ├── label_mapping.json
│   └── selections_ca/                           # EDA caches and selected-species artifacts
├── checkpoints/
│   ├── resnet50_finetune.pth
│   ├── geo_baseline.pth
│   ├── geo_baseline_expanded.pth
│   ├── fusion_mlp.pth
│   └── weighted_addition_logits.pth
├── results/                                     # Confusion matrices, per-species plots, deltas
└── reports/
    └── final_report_v1.tex
```

---

## Setup

```bash
pip install -r requirements.txt
jupyter notebook
```

The notebooks are numbered to reflect the recommended reading order. Trained checkpoints in [checkpoints/](checkpoints/) and result artifacts in [results/](results/) let you reproduce all figures in the report without re-training.

---

## Future Work

1. **Richer geo features.** Replace or augment the 6 sinusoidal features with bioclim variables, elevation, or species-distribution-model priors — Mac Aodha et al. (2019) suggest this as the most natural next step.
2. **End-to-end fine-tuning.** Both fusion models freeze the backbones. Unfreezing the ResNet50 alongside concatenated context features would let the visual representation adapt to the prior.
3. **Per-species fusion weights.** The weighted-addition uses a single global (α, β). Learning a per-species or per-family weight could capture the within-family heterogeneity our analysis surfaces — e.g., geo lifts most Apiaceae species but several widespread Poaceae grasses (*Arundo donax*, *Distichlis spicata*, *Bromus hordeaceus*) regress under fusion because the geographic prior is uninformative for them.

---

## Broader Impact

Plant identification tools shape how citizen scientists, land managers, and educators perceive biodiversity. False positives on widespread weedy species risk inflating reports of invasive presence; false negatives on rare natives risk underestimating conservation value. Fusing geographic priors helps the easy cases (range-restricted natives) but can mask the hard cases (cosmopolitan invasives) by collapsing them onto more common neighbors. The citizen-science data underlying this project also inherits well-documented spatial biases toward accessible, populated areas — models trained on it may systematically underrepresent remote or under-observed taxa.

---

## References

1. GBIF.org. *Global Biodiversity Information Facility Occurrence Download* (California, Asteraceae / Poaceae / Apiaceae).
2. Mac Aodha, O., Cole, E., Perona, P. *Presence-Only Geographical Priors for Fine-Grained Image Classification.* ICCV 2019. [arXiv:1906.05272](https://arxiv.org/abs/1906.05272).
3. Claude Code and Codex were used as coding support, primarily for debugging.

---

## Team

Neha Sajja · Ceci Zhang · Manini Banerjee — DS26 Final Project, May 2026
