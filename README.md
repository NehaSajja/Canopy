# plant-geopriors

Can geographic and seasonal context improve fine-grained plant species classification for visually ambiguous species? This project investigates whether auxiliary metadata (GPS coordinates, day-of-year) can improve accuracy and confidence calibration in a multimodal deep learning classifier trained on iNaturalist observations.

---

## Research Question

> **Can geographic and seasonal priors improve plant species classification accuracy and calibration, particularly for visually similar species within the same family?**

Sub-questions:

- Under what conditions do geo/seasonal priors help most?
- Do they hurt under distribution shift (e.g., testing on unseen regions)?
- How does model confidence calibration change when priors are added?

---

## Target Families

Three plant families chosen specifically because species within each family are visually challenging to distinguish:

| Family         | Challenge                                        | CA Species (raw) | Selected (US) |
| -------------- | ------------------------------------------------ | ---------------- | ------------- |
| **Asteraceae** | Daisy/aster family — many lookalikes             | 303              | ~45           |
| **Poaceae**    | Tall grasses — highly similar morphology         | 72               | ~15           |
| **Apiaceae**   | Carrot/parsley family — nearly identical flowers | 45               | ~15           |

---

## Data

**Source:** iNaturalist research-grade observations via GBIF (Global Biodiversity Information Facility), supplemented with the iNaturalist 2021 Competition dataset.

**Raw data:** `data/CA_3Species_Raw.csv` (~210 MB) — California observations for the three families.

**Features used:**

- `decimalLatitude`, `decimalLongitude` — geographic location
- `eventDate` → day-of-year, month — seasonal signal
- Specimen images downloaded from GBIF API

**Filters applied:**

- Research-grade observations only
- MediaType = StillImage (image-backed observations only)
- Removed records with missing coordinates or fuzzy taxonomic matches
- Minimum 300 observations per species; minimum 2 months of seasonal coverage

---

## What Has Been Done

### Notebook 01 — California EDA (`notebooks/01_eda.ipynb`)

**Key analyses:**

- Cleaned and filtered raw GBIF data (dropped sparse columns, missing coords, non-image records)
- Analyzed species coverage per family; identified species with >100 observations as viable candidates
- **Geographic baseline:** Trained a logistic regression on lat/lon only
  - Asteraceae: **6% accuracy** (6× above random 1/303 baseline)
  - Poaceae: **16% accuracy** (16× above random 1/72 baseline)
  - Apiaceae: **16% accuracy** (16× above random 1/45 baseline)
  - **Takeaway:** Strong spatial structure exists — geography is a meaningful discriminative signal even without any image features
- **Temporal analysis:** Examined day-of-year distributions per species
  - Most observations cluster in late spring/summer (DOY ~100–250) due to observer behavior bias
  - Within-family, species do peak at somewhat different times — potential discriminative signal, but weaker than geography
- **Image inspection:** Downloaded 2–3 sample images per species from GBIF API; confirmed fine-grained challenge (species within families look very similar)
- **Species selection strategies compared:**
  1. Top-N by count (high data volume, low diversity)
  2. Temporally diverse (good seasonal spread, lower counts)
  3. **Time + geo diverse** ← recommended — best balance of observation count (~1200+ per species) and diversity across both temporal and spatial dimensions

### Notebook 03 — US-Wide Species Selection (`notebooks/03_us_species_selection.ipynb`) [Skeleton: not used or tested yet]

- Generalized the selection pipeline from California to the full United States GBIF dataset [not yet started]
- Same three families remain viable with higher observation counts and broader geographic spread
- Score each species on three dimensions: count, temporal entropy (month distribution), geographic range (lat/lon spread)
- Applied composite `score_time_geo` to select final species lists (~45 Asteraceae, ~15 Poaceae, ~15 Apiaceae)
- Outputs: per-species summary stats and eligible species dataframes, ready for dataset construction

### Notebook 02 — Results (`notebooks/02_results.ipynb`)

- Placeholder only — reserved for model training results, accuracy metrics, and calibration curves

---

## What Still Needs to Happen

- [ ] **Finalize species lists** — confirm final per-family species from US-wide selection
- [ ] **Download full image dataset** — pull high-res images for all selected species from GBIF API
- [ ] **Build train/val/test splits** — design splits that avoid geographic and temporal leakage:
  - Random stratified split (45% / 15% / 40%)
  - Geographic split (held-out regions)
  - Temporal split (held-out seasons)
- [ ] **Engineer input features** — finalize encoding for lat, lon, day-of-year, month
- [ ] **Train three model variants:**
  - Image-only CNN baseline (ResNet/EfficientNet)
  - Context-only baseline (geo + seasonal features only)
  - Multimodal model (image + geo + seasonal)
- [ ] **Evaluate and compare** — accuracy, top-k accuracy, confidence calibration (ECE, reliability diagrams)
- [ ] **Out-of-distribution evaluation** — test all models on held-out geographic/temporal splits
- [ ] **Write up results** — populate `notebooks/02_results.ipynb` with findings

---

## Project Structure

```
plant-geopriors/
├── notebooks/
│   ├── 01_eda.ipynb                  # California EDA — COMPLETE
│   ├── 02_results.ipynb              # Model results — PLACEHOLDER
│   └── 03_us_species_selection.ipynb # US-wide species selection — COMPLETE
├── src/
│   ├── data.py                       # Data loading and preprocessing
│   ├── models.py                     # CNN / multimodal model definitions
│   ├── train.py                      # Training loop
│   └── eval.py                       # Evaluation metrics
├── scripts/
│   ├── prepare_data.py               # Data preparation pipeline
│   ├── run_train.py                  # Training runner
│   └── run_eval.py                   # Evaluation runner
├── configs/
│   ├── exp_image.yaml                # Image-only model config
│   ├── exp_context.yaml              # Context-only model config
│   └── exp_multimodal.yaml           # Multimodal model config
├── data/
│   ├── CA_3Species_Raw.csv           # Raw GBIF data for California (not tracked)
│   ├── iNaturalist_2021_DataSet/     # Full 2021 competition data (not tracked)
│   └── selections_ca/               # Selected species metadata + sample images
├── checkpoints/                      # Saved model weights (not tracked)
└── results/                          # Evaluation outputs (not tracked)
```

---

## Setup

```bash
pip install -r requirements.txt
jupyter notebook
```

---

## Team

DS26 Final Project
