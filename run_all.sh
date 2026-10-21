#!/usr/bin/env bash
# Reproduce everything on one GPU (about 2 h for KdV and 3 h for turbulence on a laptop RTX 5060). Resumable.
set -e
python experiments/make_data.py                       # reference solutions -> data/ (~430 MB)
for s in kdv ns; do
  python experiments/zoo.py --system $s               # train + evaluate the model zoo
  python experiments/summarize.py --system $s         # population statistics
  python experiments/monitor_eval.py --system $s      # label-free drift monitor vs twin-model baseline
  python experiments/export.py --system $s            # rollout galleries + demo data
done
python experiments/spectra.py                         # turbulence energy spectra, incl. the enstrophy-projection variant
python experiments/figures.py
python -m pytest tests -q
