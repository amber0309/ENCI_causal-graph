# Kernel Embedding-based Nonstationary Causal Model Inference

MATLAB code of causal discovery algorithm for **causal graphs** proposed in paper *A Kernel Embedding–Based Approach for Nonstationary Causal Model Inference* (ENCI).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

We test the code using **MATLAB R2017b** on windows. Any later version should still work perfectly.

## Running the tests

In MATLAB, change your *current folder* to "ENCI_graph/code" and run the file *example.m* to see whether it could run normally.

The test does the following:
1. it generate 1000 groups of a causal graph of 4 variables and put all groups in a MATLAB *cell array*.
(Each group is a L by 4 matrix where L is the number of points ranging from 40 to 50. The synthetic causal graph generated is shown below.)
2. ENCI is applied on the generated data set to infer the causal structure.

<img src="https://user-images.githubusercontent.com/9404561/38377065-9f2e7626-392c-11e8-9599-ef406e56b4e7.PNG" width="100" height="200">

## Apply on your data

### Usage

Change your current folder to "ENCI_graph/code" and use the following commands

```
[B, prc, rcl] = ENCI_graph(X, B_true)
[B, prc, rcl] = ENCI_graph(X)
```

### Notes

Input of function **ENCI_graph()**

| Argument  | Description  |
|---|---|
|XY | Cell array of matrix. Rows of each matrix represent i.i.d. samples, each column corresponds to a variable in the causal graph.|
|B_true (optional) |  A matrix representing the ground truth of the causal structure. It is used for computing the precision and recall with respect to edge estimation. |

Output of function **ENCI_graph()**

| Argument  | Description  |
|---|---|
|B | A matrix representing the estimated causal structure|
|prc |Precision with respect to edge estimation. -1 if B_true not given |
|rcl |Recall with respect to edge estimation. -1 if B_true not given |

## Authors

* **Shoubo Hu** - shoubo.sub@gmail.com

See also the list of [contributors](https://github.com/amber0309/ENCI_graph/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to [Shohei Shimizu](https://sites.google.com/site/sshimizu06/) for his elegant [LiNGAM](https://www.cs.helsinki.fi/u/ahyvarin/papers/JMLR06.pdf) code
