import pandas as pd
from math import sqrt
from numpy import argmax
from typing import Callable

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


def knn(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, k: int, dist_f: Callable):
    def codify_df(df: pd.DataFrame, reference_df: pd.DataFrame = None) -> pd.DataFrame:
        if reference_df is None:
            reference_df = df
        new_df = pd.DataFrame({
            col: [
                list(reference_df[col].unique()).index(df[col][i])
                for i in range(len(df[col]))
            ]
            for col in df
        })
        return new_df

    Xs_train = codify_df(X_train)
    Xs_test = codify_df(X_test, X_train)
    ys_train = pd.Series([
        list(y_train.unique()).index(y)
        for y in y_train
    ])

    x_test = list([Xs_test[col][0] for col in Xs_test.columns])
    distances = [
        dist_f(list([xs[1][col] for col in Xs_train.columns]), x_test)
        for xs in Xs_train.iterrows()
    ]
    Xs_train["Y"] = ys_train
    Xs_train["Distance"] = distances
    Xs_train.sort_values(by=["Distance"], ascending=True, inplace=True)

    k_nearest = []
    for ki in range(k):
        neighbours = Xs_train[Xs_train["Distance"] == min(Xs_train["Distance"])]
        Xs_train = Xs_train[Xs_train["Distance"] != min(Xs_train["Distance"])]
        k_nearest.extend(neighbours.Y)

    return y_train.unique()[argmax(len([x for x in k_nearest if x == y]) for y in y_train.unique())]


def euler_dist(x: list[int], y: list[int]) -> float:
    return sqrt(sum(abs(_x - _y) for _x, _y in zip(x, y)))


def manhattan_dist(x: list[int], y: list[int]) -> float:
    return sum(abs(_x - _y) for _x, _y in zip(x, y))
