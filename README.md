# CART-ELC

CART-ELC is an oblique decision tree induction algorithm. Our approach has a comparatively high computational cost, but the resulting models often have higher classification accuracies and better interpretability. Given the computational cost of our algorithm, it is best applied to small datasets ($n < 5000$). 

Our paper is currently under review at TMLR, but is accessible on [arXiv](https://doi.org/10.48550/arXiv.2505.05402). 

This repository contains the following:

### Implementations

| Model       | Language        | Notes                                                  |
|-------------|-----------------|--------------------------------------------------------|
| CART-ELC    | C++ / Python    | Includes C++ tests                                     |
| HHCART(A)   | Python          | See [HHCART](https://doi.org/10.48550/arXiv.1504.03415)|
| HHCART(D)   | Python          | See [HHCART](https://doi.org/10.48550/arXiv.1504.03415)|

### Results

- Experiment scripts for HHCART(A), HHCART(D), and CART-ELC
- Experimental results for CART-ELC

### Documentation

- Usage instructions / implementation details
- Source files for our paper
- Source files for our presentation

# Usage

## Dependencies

- Python 3.8+
- NumPy
- scikit-learn (for loading datasets)
- pandas
- A C++ compiler (e.g., 'g++')
- Make
- Graphviz (for rendering dot output)
- Setuptools
- pybind11
- python3-dev

If you encounter issues on Linux, please feel free to open an issue.

## Run Experiments

**Note:** Running the full experiment suite may take a significant amount of time. Additionally, you will need to create a 'datasets' directory in the root of the repository and populate it with the 'housing.csv', 'diabetes.csv', 'cancer.csv', 'dim.csv', and 'bright.csv' files corresponding to the datasets described in our paper.

To run the CART-ELC experiments from our paper, from the root of the repository, run:

```
make experiments
```

This will compile the C++ code, move the compiled binary to the directory with the python code, and run all of our python experiments in series.

## Fit and Predict

To use our implementation of CART-ELC you will first need to compile the C++ code for use with our python bindings. To do this, from the root of the repository, run 

```
make so
```

This compiles the shared object and places it in the 'examples/cart-elc-experiments' directory. You can then cd to this directory and you should see a new file named 'decision_tree.so'. You should also see other .py files. Each of these files has been configured to read in a dataset, call the fit method, and call the predict method. These .py file should be sufficient to understand how CART-ELC can be used. Additionally, these files are the files we used for the experiments in our paper and can thus be ran individually to validate our findings.

## Python Interfaces

Shown below are the interfaces for the 'ELCClassifier' object:

```python
class ELCClassifier:
    def __init__(self, depth: int, combinations: int, threadCount: int, objectiveFunction: Optional[str] = None) -> None: ...
    def fit(self, X: np.ndarray, samples: int, y: np.ndarray, features: int) -> None: ...
    def predict(self, X: np.ndarray, samples: int, features: int) -> np.ndarray: ...
    def getDot(self) -> str: ...
    def getSplits(self) -> int: ...
```

The primary distinction between sklearn's implementations of decision trees and ours is that we require the consumer to pass in the sizes for the arrays being passed to the classifier. This was done for ease of development as under the hood we are primarily using raw c-style arrays that don't have a size attribute.

We also note getDot differs slightly from sklearn's plot_tree implementation. When called, the object returns a dot language string. This string can then be rendered using Graphviz.


# Acknowledgments

This project uses the Eigen library, a third-party C++ library for linear algebra. We are not affiliated with or endorsed by the developers or maintainers of the Eigen library. The use of Eigen in this project is for research and educational purposes only.
