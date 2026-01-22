import csv
import pandas as pd

from dataclasses import dataclass


@dataclass
class Quarter:
    quarter: int
    year: int

    def __str__(self) -> str:
        return f'Q{self.quarter} {self.year}'

    def __repr__(self) -> str:
        return f'Q{self.quarter} {self.year}'

    def __eq__(self, other) -> bool:
        return self.quarter == other.quarter and self.year == other.year

    def __ne__(self, other) -> bool:
        return self.quarter != other.quarter or self.year != other.year

    def __lt__(self, other) -> bool:
        if self.year < other.year:
            return True
        if self.year > other.year:
            return False
        return self.quarter < other.quarter

    def __le__(self, other) -> bool:
        if self.year < other.year:
            return True
        if self.year > other.year:
            return False
        return self.quarter <= other.quarter

    def __gt__(self, other) -> bool:
        if self.year > other.year:
            return True
        if self.year < other.year:
            return False
        return self.quarter > other.quarter

    def __ge__(self, other) -> bool:
        if self.year > other.year:
            return True
        if self.year < other.year:
            return False
        return self.quarter >= other.quarter

    @staticmethod
    def from_str(text: str) -> 'Quarter':
        quarter, year = text.split(' ')
        quarter = int(quarter.strip('Q'))
        year = int(year.strip('\'')) + 2000
        return Quarter(quarter, year)

def dane_uzytkownicy() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with open('Lab-1-Zadanie-1-Dane-liczba-użytkowników.csv', 'r') as f:
        reader = csv.reader(f, delimiter=';')
        labels = next(reader)
        dane_uzytkownicy = pd.DataFrame(
            [[int(x[0]), Quarter.from_str(x[1]), int(x[2])]
             for x in reader],
            columns=labels)

    dane_uzytkownicy_train = dane_uzytkownicy[dane_uzytkownicy['QY'] <= Quarter(4, 2017)]
    dane_uzytkownicy_train = dane_uzytkownicy_train[dane_uzytkownicy_train['QY'] >= Quarter(1, 2009)]

    dane_uzytkownicy_test = dane_uzytkownicy[dane_uzytkownicy['QY'] >= Quarter(1, 2018)]

    return dane_uzytkownicy, dane_uzytkownicy_train, dane_uzytkownicy_test

def dane_przychody() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with open('Lab-1-Zadanie-1-Dane-przychody.csv', 'r', encoding="UTF-8") as f:
        reader = csv.reader(f, delimiter=';')
        labels = next(reader)
        dane_przychody = pd.DataFrame([
            [int(x[0]), int(x[1]), int(x[2])]
            for x in reader],
            columns=labels)

    dane_przychody_train = dane_przychody[dane_przychody['Year'] <= 2017]
    dane_przychody_train = dane_przychody_train[dane_przychody_train['Year'] >= 2007]

    dane_przychody_test = dane_przychody[dane_przychody['Year'] >= 2018]

    return dane_przychody, dane_przychody_train, dane_przychody_test



