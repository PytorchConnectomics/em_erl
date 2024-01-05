# erl
Evaluation script for expected run length (ERL)


- Installation
```
# create a new environment
conda create -n erl-eval python==3.9.0
source activate erl-eval

pip install --editable .
```

Acknowledgement
---
- [Funkelab](https://github.com/funkelab): The main functionality is adapted from [this file](https://github.com/funkelab/funlib.evaluate/blob/master/funlib/evaluate/run_length.py)
- [jasonkena](https://jasonkena.github.io/): Reduce the memory usage by replacing the networkx object with a self-defined networkx-lite object.
