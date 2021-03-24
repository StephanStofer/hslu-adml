# Code CheatSheet

Contains several code snippets and usful library functions.

## Python

Code|Parameters|Description
---|---|---
`~bool`||flip boolean result

## Pandas

Code|Parameters|Description
---|---|---
`pd.qcut(data['column-name'],q=4,labels=False)`|q=number of Quantiles, labels=binlabelsarray or False|bin numerical data in $q$ buckets
`pd.get_dummies(data, drop_first=True)`|drop_first= removes one of the columns to get $k-1$|convert categorical features to numerical ones. decide for each feature to use *
one hot encoding* or *label encoding*
`df.duplicated()`|keep=first or last remains in set|list all duplicate entries. keep=False shows all duplicated entries. if param omitted one entry remains in the set
`df.drop_duplicates()`|inplace=True removes duplicates directly, otherwise a copy gets returned (default false)|drop all duplicates
`df.columnname.nunique()||returns the number of unique values in total

## Numpy

Code|Parameters|Description
---|---|---
`np.asarray(x)`||returns a 1d array
`np.asarray(x).shape||returns dimension of $n$-d array. (160,3) means 160 data points in 3 dimensions
`np.roll(a,shift,axis=None)|a=array, shift=number of places elements are shifted|shift elements in direction of axis

## Sklearn

Code|Parameters|Description
---|---|---
`sklearn.neighbors.KNeighborsRegressor()`|weights=Gewichtung der $k$-nahesten Datenpunkte, metric=Distancemetric|K-NN Regression
`train_test_split(X,y, test_size=0.2,random_state=42)`|*X & y* must have the same size, *
test_size* represent the proportion of the dataset, *
randsom_state* controls the shuffling applied - with an int output is reproducible|Split arrays or matrices into random train and test subsets. Returns for each array **
two** lists with a train and test set
`DummyRegressor()`|*
strategy* with options {mean, median, quantile, constant}|instantiate a DummyRegressor to get an idea how good our model performs.

