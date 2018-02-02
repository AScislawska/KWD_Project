# 1. Uczenie liniowej hipotezy dla problemu przewidywania cen nieruchomości (Housing Data)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn import linear_model, datasets
from sklearn.metrics import mean_squared_error, r2_score

boston = load_boston()
print(boston.DESCR)
features = boston.feature_names
print(features)
print(len(features))
print('-----------------')

print(len(boston.data))
invalid_rows = dict()
print('-----------------')
for idx, val in enumerate(boston.data):
    if len(val) != len(features):
        invalid_rows[idx] = (len(val))

if invalid_rows:
    print('invalid sets:')
    print(invalid_rows)
    quit(0)
else:
    print("all sets have data")

print('-----------------')
print(len(boston.target))

print('-----------------')
print(boston.keys())

r_squared_list = []

all_features = boston.data
all_features_target = boston.target

all_features_train, all_features_test, all_features_target_train, all_features_target_test = train_test_split(
    all_features, all_features_target, train_size=0.1)

print(len(all_features_target_train))

for idx, name in enumerate(features):
    print(name)

    boston_x_train = all_features_train[:, idx]
    boston_x_train = boston_x_train.reshape(-1, 1)

    boston_x_test = all_features_test[:, idx]
    boston_x_test = boston_x_test.reshape(-1, 1)

    boston_y_train = all_features_target_train
    boston_y_test = all_features_target_test

    print('boston_x_train size ', len(boston_x_train))
    print('boston_x_test size', len(boston_x_test))
    plt.figure(idx)
    plt.subplot('221')
    plt.scatter(boston_x_train, boston_y_train)
    plt.title('Training dataset ' + name)
    plt.xlabel('x' + str(idx))
    plt.ylabel('target')
    plt.grid()

    plt.subplot('222')
    plt.scatter(boston_x_test, boston_y_test)
    plt.title('Test dataset ' + name)
    plt.xlabel('x' + str(idx))
    plt.ylabel('target')
    plt.grid()

    regr = linear_model.LinearRegression()
    regr.fit(boston_x_train, boston_y_train)

    plt.subplot('223')
    plt.scatter(boston_x_train, boston_y_train)
    plt.plot(boston_x_train, regr.predict(boston_x_train), color='green', linewidth=3)
    plt.title('Learned linear regression - train set')
    plt.xlabel('x' + str(idx))
    plt.ylabel('target')
    plt.tight_layout()
    plt.grid()

    plt.subplot('224')
    plt.scatter(boston_x_test, boston_y_test)
    plt.plot(boston_x_test, regr.predict(boston_x_test), color='green', linewidth=3)
    plt.title('Learned linear regression - test set')
    plt.xlabel('x' + str(idx))
    plt.ylabel('target')
    plt.tight_layout()
    plt.grid()

    print('Coefficient: ', regr.coef_, ' intercept: ', regr.intercept_)
    print('Mean squared error: %.4f' % mean_squared_error(boston_y_test, regr.predict(boston_x_test)))
    r_squared = r2_score(boston_y_test, regr.predict(boston_x_test))
    print('r2: %.4f' % r_squared)
    r_squared_list.append(r_squared)

print('r_squared for separate features: ', r_squared_list)
print('-----------------')
from sklearn.metrics import mean_squared_error, r2_score

# regression trained with all features


linear_regression_model = linear_model.LinearRegression()
linear_regression_model.fit(all_features_train, all_features_target_train)

predicts = linear_regression_model.predict(all_features_test)

print('coefficients of all features regression:', linear_regression_model.coef_)

print('Mean error over test data: %.4f' % mean_squared_error(all_features_target_test, predicts))
print('R2  test data: %.4f' % r2_score(all_features_target_test, predicts))
print('-----------------')

# Ridge regression
data = datasets.load_boston()

data_train, data_test, target_train, target_test = train_test_split(data.data, data.target, train_size=0.1)

ridge = linear_model.Ridge(alpha=0.2)
ridge.fit(data_train, target_train)

predicted = ridge.predict(data_test)

print('Ridge')
print('Mean error over test data: %.4f' % mean_squared_error(target_test, predicts))
print('R2  test data: %.4f' % r2_score(target_test, predicted))

from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import cross_val_score

lstat_data=all_features[:, 12];
lstat_data=lstat_data.reshape(-1,1)

regressor_x_train, regressor_x_test, regressor_target_train, regressor_target_test = train_test_split(
    lstat_data, data.target, train_size=0.80)

regressor = SGDRegressor(loss='squared_loss', max_iter=1000, tol=0.001, alpha=0.01)
scores = cross_val_score(regressor, regressor_x_train, regressor_target_train, cv=5)
regressor.fit(regressor_x_train, regressor_target_train)

predicted_lstat=regressor.predict(regressor_x_test)

print('-----------------')
print('SGDRegressor - last feature')
print('Cross validation r-squared scores: ', scores)
print('Average cross validation r-squared score: ', np.mean(scores))
print('Test set r-squared score ', regressor.score(regressor_x_test, regressor_target_test))
print('R2  test data: %.4f' % r2_score(regressor_target_test, predicted_lstat))

plt.show()
