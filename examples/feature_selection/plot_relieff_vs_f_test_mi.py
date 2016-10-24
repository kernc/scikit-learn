"""
======================================================================
Comparison of ReliefF, F-test, and mutual information on a XOR problem
======================================================================

This example illustrates the performance of ReliefF feature scoring algorithm
in comparison with univariate F-test and mutual information.

We consider a synthetic classification problem of 20 features, distributed
uniformly over [0, 1], where the class variable depends on the first three
features in the following manner:

    y = ((x_1 > .5) ^ (x_2 > .5)) | (x_3 > .6)

The remaining 17 features are irrelevant.

The code below plots the relative feature relevance in respect to y as
measured by univariate F-test statistics, mutual information, and ReliefF.

Univariate F-test and mutual information are too myopic to notice the
conditional dependence (XOR) between the first two features and thus fail
to mark them of any importance. ReliefF, on the other hand, marks all
three features significantly more relevant than the rest.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import relieff, f_classif, mutual_info_classif

np.random.seed(0)
X = np.random.rand(1000, 20)
y = ((X[:, 0] > .5) ^ (X[:, 1] > .5)) | (X[:, 2] > .6)

f_test, _ = f_classif(X, y)
f_test /= np.max(f_test)

mi = mutual_info_classif(X, y)
mi /= np.max(mi)

rf = relieff(X, y)
rf /= np.max(np.abs(rf))

plt.figure(figsize=(15, 15))
plt.suptitle('Relative feature importances as measured by ...', fontsize=20)
for i, (title, scores) in enumerate((('F-test', f_test),
                                     ('Mutual Information', mi),
                                     ('ReliefF', rf))):
    plt.subplot(3, 1, i + 1)
    plt.bar(np.arange(X.shape[1]), scores)
    plt.ylabel(title, fontsize=16)

plt.show()
