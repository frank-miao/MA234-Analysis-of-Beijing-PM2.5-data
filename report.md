# REPORT on Introduction to Theoretical and Practical Data Science
> 11911819 缪方然 11911915 曹书伟

## Data exploration


### data statistics

### data visualization

## Data preprocessing

### dataset partitioning

### detecting missing values

* **Detect**

We use `np.where` and `np.isnan` to find and locate the NAN data.
Then generate the `nan_index` and `non_nan_index` to help us.
The related code is in `data_preprocess.py->detect_missiing_data`.

* **Fill**

We use several methods to fill the NAN data.
We use 0, average, mode and median to fill the NAN.
Since it will appear large amounts of times, it is, generally, not a proper way.

Then we use KNN to fill.
We find 5 nearest neighbors for a certain row which contains a missing value.
And we take the average of the 5 nearest neighbors' values of pm2.5 to fill the NA.

Last we use interpolation to fill it.
The most natural idea is that the value of pm2.5 is continuous about time.
Thus, we use the interpolation of pm2.5 versus time.

### data conversion

There is only one column need to be converted in the raw data---'cbwd'.
First use the function `set` to find the number of unique values.
As a result, there are 4-- "NW", "NE", "SE", "cv".
Then use `dict` to store a mapping relation, i.e. "NW" maps to 1, "NE" maps to 2, "SE" maps to 3, "cv" maps to 4.
Finally, replace in the original data.

### data normalization

We use `sklearn.preprocessing.StandardScaler` to normalize data.
Note that there are still some features that we don't want to normalize, e.g. "cbwd".
Since this kind of data is just a representation of the direction of the wind, and it does not necessarily follow the
Gauss Distribution.

## Model construction

In machine learning, we use seven different models to train our data.

The machine learning model files are stored in `models.py` file

* **linear regression model**

Implemented by `sklearn.linear_model.LinearRegression  `

* **ridge regression**

Implemented by `sklearn.linear_model.Ridge`

* **LASSO regression model**

Implemented by `sklearn.linear_model.Lasso`

* **random forset regressor model**

Implemented by `sklearn.ensemble.RandomForestRegressor`

* **extra trees regressor**

Implemented by `sklearn.ensemble.ExtraTreesRegressor`

* **gradient boosting regressor**

Implemented by `sklearn.ensemble.GradientBoostingRegressor`

* **support vector regressor**

Implemented by `sklearn.svm`

## Feature selection and Model selection and Feature creation

### Feature selection

In this part, we use several ways to select features.

* **AIC and BIC**

We use forward AIC and BIC to select the features step by step.
Since scikit-learn doesn't provide the related API, we implemented this part by ourselves.

Both methods show that `DEWP`, `TEMP`, `PRES`, `cbwd`, `Iws` are the selected feature.

* **LASSO**

We use decreasing $\alpha$ to select features.
The result shows that `Iws`, `DEWP`, `PRES`, `TEMP`, `Ir`, `cbwd` are selected features.

* **Random Forest**

Random forest can calculate the feature importance.
And we used the built-in API in this package.
The result shows that `DEWP`, `TEMP`, `Iws`, `PRES`, `cbwd` are important.

* **Mutual Information**

We use the mutual information to select the features.
The result shows that `DEWP`, `Iws`, `cbwd`, `PRES`, `TEMP` are the top 5 important.

* **Conclusion**

We can find out that `DEWP`, `Iws`, `cbwd`, `PRES`, `TEMP` are important features.
And `Ir`, `Is` are not so important features.

### Feature creation

### Model selection

## Model evaluation

### R2 score



### RMSE



### cross validation





## Conclusion

