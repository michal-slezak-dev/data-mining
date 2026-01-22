import pandas as pd
from numpy import argmax

def print_result(x: pd.DataFrame, result: str, result_name: str) -> None:
    print("*====== Wynik klasyfikacji ======* ")
    for col in x:
        print(f"{str(col)}: {str(x[col][0])}")
    print(f"{result_name} (Klasyfikacja): {result}")
    print("*================================* ")


def naive_bayes(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame):
    y_uniques = y_train.unique()
    prob = [len(y_train[y_train == y]) / len(y_train) for y in y_uniques]

    for col in X_train:
        matching = X_train[col][X_train[col] == X_test[col][0]]
        ys = (y_train[matching.index])

        for i in range(len(prob)):
            prob[i] *= len(ys[ys == y_uniques[i]]) / len(ys)

    result = y_uniques[argmax(prob)]

    return result
