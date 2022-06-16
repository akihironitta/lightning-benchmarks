# Benchmarks on PyTorch Lightning

- [akihironitta.com/lightning-benchmarks](https://www.akihironitta.com/lightning-benchmarks): See benchmark results.
- [asv documentation](https://asv.readthedocs.io/en/stable/)

## What's this repository?

This repository measures memory usage and speed of simple benchmark scripts using PyTorch Lightning.
It originally started as my private repository, but I made it pubilc hoping that someone in the community would benefit from it.

So far, I've used this repo to locate the following regressions:

- [tqdm progress bar in v1.6 is slower than v1.5 #13179](https://github.com/Lightning-AI/pytorch-lightning/issues/13179)
- [v1.6 is slower than v1.5 #12713](https://github.com/Lightning-AI/pytorch-lightning/issues/12713)

## Writing benchmarks

- Benchmarking scripts have to be compatible to 1.5 and 1.6!
- See examples:
  - [NumPy benchmarks](https://github.com/numpy/numpy/tree/main/benchmarks/benchmarks)
  - [SciPy benchmarks](https://github.com/scipy/scipy/tree/main/benchmarks/benchmarks)

## Running benchmarks

```
# run all benchmarks on HEAD commit of each branch configured in asv.conf.json
asv run

# run all benchmarks on one hash/tag/branch
asv run master^!

# run all benchmarks across commits between hashes/tags/branches
asv run 1.5.0..master

asv run \
  --python=/path/to/python \  # specify Python interpreter
  --bench "Boring*" \  # run benchmarks that matches a regexp.
  --show-stderr \
  -v

# create report to be previewed
asv publish

# and access http://localhost:8080
asv preview
```

### TODO / open questions

- add smaller benchmarks (non end-to-end)
- add more benchmarks
