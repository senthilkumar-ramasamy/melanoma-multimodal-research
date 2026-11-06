# Part 2 - Dataset audit and baseline models

Leakage-aware audit of HAM10000, a lesion-grouped train/validation/test partition, and two reference baselines evaluated under one protocol:

- image-only: ImageNet-pretrained EfficientNet-B0
- metadata-only: logistic regression, random forest, and gradient boosting on age, sex, and anatomical site

Task: binary classification, melanoma (positive) vs. the six other diagnostic categories.

## Folders
- `code/` - the five pipeline scripts and their README
- `notebooks/` - the Google Colab notebook that runs the pipeline end to end
- `results/` - result tables, metrics, saved predictions, partitions, and figures from the verified run

## Data
This repository contains code only. Download HAM10000 from Harvard Dataverse (https://doi.org/10.7910/DVN/DBW86T, licence CC BY-NC 4.0). Do not redistribute the images.
