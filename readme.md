## Usage

Our codes are very readable, and most function names and variable names are consistent with the paper. If you want to change the parameters for specific methods, you can change the code in the corresponding functions.

### SVD experiment (section 5.2 in the paper)

If you want to run the SVD experiment, you can simply run:

```python
python3 svd_tester.py
```

This tester runs three methods: k-adaptive rPCA, randQB_EI and svds.

### Recommend experiment (section 5.3 in the paper)

If you want to run the recommend experiment, you first need to compile the C++ code:

```
g++ -std=c++11 -O3 calc.cpp -fPIC -shared -o calc.so
```

The C++ code is used to predict ratings efficiently. It is automaticaly called by the Python tester.

Then run `recommender_tester.py` to run different recommenders:

```
python3 recommender_tester.py
```

This tester runs three methods: recommendation based on k-adaptive rPCA, RMF_model and SVD model. These methods are consistent with the paper.

If you want to run the fSVT method, you need to do the following steps after you run `recommender_tester.py`. Because we used random sample in the test, we save test data in directory input_for_SVT. There're six files in this directory: A_col.txt, A_row.txt, A_val.txt, test_col.txt, test_row.txt, test_val.txt. The first three files have same lines, where line i for each file denotes that triple (row[i], col[i], val[i]) is a value in sparse matrix A (after we removed validation set and test set). The latter three files have similar meaning, which represent the test data from A. You need to download the fSVT code shared in https://github.com/XuFengthucs/fSVT. You need to use Matlab to read matrix A and test set from the six files in input_for_SVT, then call `fastSVT_U` to complete the matrix, and compute the MAE on test set.