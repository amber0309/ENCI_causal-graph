# Kernel Embedding-based Nonstationary Causal Model Inference

Python code of causal inferene algorithm for **causal graphs** proposed in paper [A Kernel Embedding–Based Approach for Nonstationary Causal Model Inference](https://www.mitpressjournals.org/doi/abs/10.1162/neco_a_01064) (ENCI).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
- NumPy
- SciPy
- lingam

We test the code using **Anaconda3 5.0.1 64-bit for python 3.6.3** on windows. Any later version should still work perfectly. The download page of Anaconda is [here](https://www.anaconda.com/download/).

## Apply **ENCI** on your data

### Usage

Import **ENCI** using

```python
from ENCI import ENCI
```

Apply **ENCI** on your data

```python
md = ENCI(X, B_gt)
B_est, prc, rcl = md.fit()
```

### Description

Input of function **ENCI()**

| Argument  | Description  |
|---|---|
|X | List of *numpy arrays* or *list of numpy arrays*. Each element in X represents a domain.<br/>If elements are *numpy arrays*, rows and columns correspond to i.i.d. samples and variables, respectively.<br/>If elements are *list of numpy arrays*, each array should be a column vector consists of all samples of a variable.|
|B_gt (optional) | The true adjacency matrix|

Output of function **ENCI.fit()**

| Argument  | Description  |
|---|---|
|B_est | The estimated adjacency matrix|
|prc | Precision of estimated result|
|rcl | Recall of estimated result|

## Authors

* **Shoubo Hu** - shoubo [dot] sub [at] gmail [dot] com

See also the list of [contributors](https://github.com/amber0309/ENCI_graph/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
