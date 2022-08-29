# Benchmarks on PyTorch Lightning

- See results at [akihironitta.com/lightning-benchmarks](https://www.akihironitta.com/lightning-benchmarks).
- See [asv documentation](https://asv.readthedocs.io/en/stable/) for writing benchmarks.

## What's this repository?

This repository was created to measure memory usage and speed of simple benchmark scripts using PyTorch Lightning.
It originally started as my private repository, but I made it pubilc hoping that someone in the community would benefit from it.

So far, I've used this repo to locate the following regressions:

- [Lightning-AI/lightning#13179 tqdm progress bar in v1.6 is slower than v1.5](https://github.com/Lightning-AI/lightning/issues/13179)
- [Lightning-AI/lightning#12713 v1.6 is slower than v1.5](https://github.com/Lightning-AI/pytorch-lightning/issues/12713)

## Writing benchmarks

- Benchmarking scripts have to be compatible to 1.5 and 1.6.
- See examples:
  - [NumPy benchmarks](https://github.com/numpy/numpy/tree/main/benchmarks/benchmarks)
  - [SciPy benchmarks](https://github.com/scipy/scipy/tree/main/benchmarks/benchmarks)
- Notes on method names:
  - `track_` suffix records a return value.
  - `time_` suffix records time.
  - `timeraw_` suffix runs the benchmark runs in a seperate process and records time. Useful for measuring import time.
  - `peakmem_` suffix records peak memory usage. It also captures memory usage during `setup`. To avoid this, use `setup_cache`.
  - `mem_` suffix records the size of a returned Python object.
  - `setup_cache` runs only once per benchmark class while `setup` runs before each benchmark case within the benchmark class.

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

- add more benchmarks
- add smaller benchmarks
