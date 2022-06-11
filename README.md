## Note to self

Benchmarking scripts have to be compatible to 1.5 and 1.6!

### comands

```
asv run  # run all benchmarks on HEAD commit of each branch configured in asv.conf.json
asv run 1.5.0..master  # run all benchmarks across commits between hashes/tags/branches
asv run \
  --python=/path/to/python \  # specify Python interpreter
  --bench "Boring*" \  # run benchmarks that matches a regexp.
  --show-stderr \
  -v
```

```
asv publish  # create report to be previewed
asv preview  # and access http://localhost:8080
```

### docs

- https://asv.readthedocs.io/en/stable/
- https://github.com/numpy/numpy/tree/main/benchmarks/benchmarks
- https://github.com/scipy/scipy/tree/main/benchmarks/benchmarks

### TODO/questions

- add smaller benchmarks (non end-to-end)
- add more benchmarks?
- should each test case be hardware-agnostic?
  - yes, maybe, to visualise multiple stats in the same graph. If the stats of running on gpus goes down, we'll know that it's something wrong with gpu-related code.
