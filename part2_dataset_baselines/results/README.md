Results of the verified run (HAM10000, melanoma vs. other lesions).

- `results_table_V.csv`, `results_main.json` - test metrics with 95% lesion-clustered bootstrap intervals
- `results_repeated_runs.csv` - metadata-only models over five grouped partitions
- `false_positive_rates.csv` - false-positive rate per negative diagnostic category
- `near_duplicate_sensitivity.csv` - results after excluding test images similar to training images
- `preds/` - saved predictions and run settings; `splits/` - partitions; `audit/` - audit report and exclusion log; `figures/` - result figures

Model checkpoints, cached image arrays, and dataset images are not included.
