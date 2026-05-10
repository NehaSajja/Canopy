# CS 1090B MS4 Submission - Group 49

Project: plant-geopriors

Group members: Ceci Zhang; Manini Banerjee; Neha Sajja

Main notebook:

- `notebooks/cs1090b_ms4_main_group49.ipynb`

This submission folder contains the main notebook, supporting notebooks, helper script, processed artifacts, checkpoints, and result figures needed to reproduce the final checkpoint-based evaluation.

The main notebook is designed to run from a fresh kernel using relative paths from this folder. It loads processed artifacts and checkpoints, evaluates the baseline models, evaluates weighted-addition fusion, and reports the final Fusion MLP results. It uses cached logits by default, so raw images are not required for the quick run.

Important submission note:

This folder is larger than the 50 MB Canvas zip limit because it includes model checkpoints and result artifacts. Submit a link to a specific public Git commit containing this folder.

Raw images:

The full `data/images/` directory is not included. If image logits need to be recomputed, place images under `data/images/<Species_name>/<gbifID>.jpg`. The current main notebook does not require that directory when cached logits are present.

AI use disclosure:

OpenAI Codex (GPT-5) was used on 9 May 2026 to help organize and draft the new main notebook, create the submission-folder structure, add checkpoint-first evaluation code, and generate the cached Fusion MLP logits file from the team's existing checkpoints. The project data collection, modeling approach, checkpoints, supporting notebooks, interpretation, and final model choice are the group's work.
