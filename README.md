# Learning Multi-level Dependencies for Robust Word Recognition
Pytorch implementation of MUDE.  Some parts of the code are adapted from the implementation of [scRNN](https://github.com/keisks/robsut-wrod-reocginiton).

## Usage
Our repository is arranged as follows:
```
    data/ # contains the original datasets
    experiment.py #run experiment to produce Table 2 and 3
    model.py # comtains the MUDE
    utils.py # utility functions
    generalization_pipeline.py # produce Table 4 and 5
```
When data is ready, the code can directly run with PyTorch  1.0.0.

