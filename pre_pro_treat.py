import numpy as np
from numpy.core.numeric import NaN

def dropnan(X=None, y=None, missing_feature=NaN):
    '''Drops the instances that have a missing value for the given feature.
    
    The input dataset (X, y) is not modified. A new dataset is created and returned.
    
    Parameters
    ----------
        X : numpy array (matrix).
            A dataset. Each row corresponds to an instance, each column to a feature.
        y : numpy array.
            The vector with the target values for each instance.
        missing_feature : int
            The index of the column of X that corresponds to the feature with missing values.
    
    Returns
    -------
        A tuple.
            A tuple (X_nomissing, y_nomissing) that corresponds to the input dataset (X, y) without the instances
            with missing values.
    '''
    res = list()
    if not(X is None):
        X_nomissing = X.copy()
        X_nomissing = X_nomissing[~np.isnan(X_nomissing).any(axis=1)]
        res.append(X_nomissing)
    if not(y is None):
        y_nomissing = y.copy()
        y_nomissing = np.delete(y_nomissing, np.argwhere(np.isnan(X_nomissing[:, missing_feature])), axis=0)
        res.append(y_nomissing)
    if (len(res) > 1):
        return tuple(res)
    else:
        return res[0]