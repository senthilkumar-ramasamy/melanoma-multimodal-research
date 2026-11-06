# Dataset audit and baseline code (HAM10000, melanoma vs. other lesions)

Run these in order. Times are for Google Colab with a free T4 GPU. Do not start a step until the previous one has printed its final line.

| Step | Script | Typical time | Final line |
|---|---|---|---|
| 1 | `01_audit.py` | 5-15 min | `Audit done` |
| 2 | `02_split.py` | under 1 min | `OK: zero shared lesion groups` |
| 3 | `03_metadata.py` | 15-40 min | last line for seed 46 hgb |
| 4 | `04_image.py` | about 20-120 min on GPU | `done. best val PR-AUC` |
| 5 | `05_evaluate.py` | 3-10 min | `Results and figures written` |
| 6 | `06_sensitivity.py` | under 1 min | `Sensitivity analysis written` |

The simplest way to run everything is `../notebooks/baselines_colab.ipynb` on Google Colab.

Notes
- Perceptual-hash pairs within Hamming distance 2 are flagged and listed; only identical hashes (`--merge_dist 0`) are merged into one group, because chaining at larger distances joins unrelated look-alike images into very large groups.
- `make_synthetic.py` builds a tiny fake dataset for smoke-testing the code. Never use it for reported results.
- `--no_pretrained` and `--max_images` in `04_image.py` are for smoke tests only.
- The test partition is used once, in step 5. Thresholds and model selection use validation data only.
