# 1. Uczenie liniowej hipotezy dla problemu przewidywania cen nieruchomości (Housing Data)

from sklearn.datasets import load_boston

boston = load_boston()

features=boston.feature_names
print(features)
print(len(features))
print('-----------------')
print(boston.data)

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
print(boston.target)
print(len(boston.target))



print("let's do this")
