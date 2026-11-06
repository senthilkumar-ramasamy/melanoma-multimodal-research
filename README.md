# Melanoma Detection from Dermoscopic Images and Clinical Metadata

Research code for a multimodal melanoma detection programme using dermoscopic images, patient demographics, and clinical metadata with transfer learning. The work is decision support research and is not a clinical diagnostic tool.

| Part | Folder | Status |
|---|---|---|
| 1. Literature review | `part1_literature_review/` | Written |
| 2. Dataset audit and baselines | `part2_dataset_baselines/` | Complete; baseline results reported |
| 3. Multimodal model | `part3_multimodal_model/` | Planned |
| 4. Final evaluation | `part4_final_evaluation/` | Planned |

## Getting started
1. Download HAM10000 from Harvard Dataverse: https://doi.org/10.7910/DVN/DBW86T (CC BY-NC 4.0).
2. Follow `part2_dataset_baselines/code/README.md` or run `part2_dataset_baselines/notebooks/baselines_colab.ipynb` on Google Colab.

## Requirements
Python 3, PyTorch and torchvision, scikit-learn, pandas, numpy, scipy, matplotlib, and Pillow (see `requirements.txt`). The runs were made on Google Colab with a T4 GPU.

## Licence
Code: MIT (see `LICENSE`). The HAM10000 data keep their own CC BY-NC 4.0 licence.
