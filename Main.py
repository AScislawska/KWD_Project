# 1. Uczenie liniowej hipotezy dla problemu przewidywania cen nieruchomości (Housing Data)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn import linear_model, datasets

boston = load_boston()

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
print("let's do this")

r_squared_list = []
for idx, name in enumerate(features):
    print(name)
    boston_x = boston.data[:, idx]
    boston_x = boston_x[:, np.newaxis]

    boston_x_train, boston_x_test, boston_y_train, boston_y_test = \
        train_test_split(boston_x, boston.target, test_size=0.3)

    plt.figure(idx)
    plt.subplot('221')
    plt.scatter(boston_x_train, boston_y_train)
    plt.title('Training dataset ' + name)
    plt.xlabel('x'+str(idx))
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
    plt.plot(boston_x_train, regr.predict(boston_x_train), color='green',
             linewidth=3)
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

    from sklearn.metrics import mean_squared_error, r2_score

    print('Coefficients: \n', regr.coef_)
    print("Mean squared error: %.2f" % mean_squared_error(boston_y_test, regr.predict(boston_x_test)))
    r_squared = r2_score(boston_y_test, regr.predict(boston_x_test))
    print('r2: %.2f' % r_squared)
    r_squared_list.append(r_squared)


print('coefficients for separate '+r_squared_list)
print('-----------------')
from sklearn.metrics import mean_squared_error, r2_score

# regression trained with all features
all_features = boston.data
all_features_target = boston.target

all_features_train, all_features_test, all_features_target_train, all_features_target_test = train_test_split(
    all_features, all_features_target, train_size=0.3)

linear_regression_model = linear_model.LinearRegression()
linear_regression_model.fit(all_features_train, all_features_target_train)

predicts = linear_regression_model.predict(all_features_test)

print("coefficients of all features regression:", linear_regression_model.coef_)

print("Mean error over test data: %.2f" % mean_squared_error(all_features_target_test, predicts))
print("R2  test data: %.2f" % r2_score(all_features_target_test, predicts))
print('-----------------')

plt.show()
