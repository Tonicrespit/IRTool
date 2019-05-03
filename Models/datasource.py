import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
import sys


class DataSource:
    def __init__(self, data, format='csv'):
        if format == 'csv':
            self.data = pd.read_csv(data)
        elif format in ['xls', 'xlsx']:
            self.data = pd.read_excel(data)
        else:
            raise Exception
        self.data.columns = self.data.columns.str.replace("[^a-zA-Z0-9]", "_")
        self.raw_data = self.data.copy(deep=True)
        self.data_train = self.data.copy(deep=True)
        self.data_test = pd.DataFrame()
        self.target = None
        self.target_value = None
        self.columns = list(self.data.columns)
        self.column_cuts_mapping = {}
        self.column_item_mapping = {}
        for col in self.get_columns():
            if self.is_column_categorical(col):
                self.data_train[col] = self.data_train[col].astype('category')

    def get_data(self, test_data=False) -> pd.DataFrame:
        if self.columns is not None and len(self.columns) > 0:
            if test_data and len(self.data_test.index > 0):
                return self.data_train[self.columns].append(self.data_test[self.columns], ignore_index=True)
            else:
                return self.data_train[self.columns]
        else:
            if test_data and len(self.data_test.index > 0):
                return self.data_train
            else:
                return self.data_train.append(self.data_test, ignore_index=True)

    def get_test_set(self) -> pd.DataFrame:
        if self.data_test is None:
            return pd.DataFrame()
        return self.data_test[self.columns]

    def get_data_as_str(self) -> pd.DataFrame:
        return self.get_data().applymap(str)

    def get_raw(self) -> pd.DataFrame:
        return self.raw_data

    def get_without_target(self, test: bool = False) -> pd.DataFrame:
        return self.get_data(test).drop(columns=self.get_target_column())

    def get_columns(self) -> list:
        if self.columns is not None and len(self.columns) > 0:
            return self.columns
        else:
            return self.data.columns

    def set_columns(self, cols: list):
        self.columns = list(cols)

    def get_col_max(self, col: str) -> int:
        if col in self.columns and not self.is_column_categorical(col):
            return self.data_train[col].max()
        return 0

    def get_col_min(self, col: str) -> int:
        if col in self.columns and not self.is_column_categorical(col):
            return self.data_train[col].min()
        return 0

    def get_col_quantiles(self, col: str, nquant: int) -> list:
        quantiles = []
        if col in self.columns and not self.is_column_categorical(col):
            for q in range(0, nquant):
                nth_q = self.data_train[col].quantile(q/(nquant-1))
                quantiles.append(nth_q)
        return quantiles

    def get_unique(self, col: str) -> list:
        return self.data[col].unique()

    def set_target(self, col: str):
        self.target = col

    def get_target(self, test: bool = False) -> pd.Series:
        return self.get_data(test)[self.get_target_column()]

    def get_target_column(self) -> str:
        return self.target

    def get_target_value(self) -> str:
        return self.target_value

    def set_target_value(self, value: str):
        self.target_value = value

    def get_ncat(self, col: str) -> int:
        return len(self.get_data()[col].unique())

    def is_column_categorical(self, col: str, original: bool = False) -> bool:
        if original:
            return col in self.raw_data.columns and len(self.raw_data[col].unique()) <= 12
        else:
            return col in self.columns and len(self.get_data(test_data=True)[col].unique()) <= 12

    def get_column_by_target_value(self, col: str) -> object:
        series = [self.get_data().loc[self.get_data()[self.target] == self.target_value, col],
                  self.get_data().loc[self.get_data()[self.target] != self.target_value, col]]
        for serie in series:
            serie.dropna(inplace=True)
        return series

    def default_column_categorization(self, col: str, target_x: list = None, target_y: list = None,
                                      other_x: list = None, other_y: list = None):
        if col not in self.columns or col in self.column_cuts_mapping.keys():
            return
        if self.is_column_categorical(col):
            cuts = self.get_data()[col].unique()
        elif target_x is not None:
            s1 = {}
            s2 = {}
            for i in range(len(target_x)):
                s1[target_x[i]] = target_y[i]
                s2[other_x[i]] = other_y[i]

            x_list = sorted(list(target_x + other_x))
            y_list = []
            for i in x_list:
                x1 = min(s1.keys(), key=lambda x: abs(x - i))
                x2 = min(s2.keys(), key=lambda x: abs(x - i))

                if list(s1.keys()).index(x1) == 0 or list(s1.keys()).index(x1) == len(s1.keys()) - 1 or \
                        list(s2.keys()).index(x2) == 0 or list(s2.keys()).index(x2) == len(s2.keys()) - 1:
                    y_list.append(' ')
                elif s1[x1] > s2[x2]:
                    y_list.append('+')
                else:
                    y_list.append('-')

            for i in range(len(y_list)):
                if y_list[i] == ' ':
                    closest = sys.maxsize
                    for j in range(len(y_list)):
                        if y_list[j] != ' ' and abs(i - j) < abs(i - closest):
                            closest = j
                    y_list[i] = y_list[closest]

            smoothed = []
            for i in range(1, len(y_list) - 1):
                if y_list[i - 1:i + 1].count('+') > y_list[i - 1:i + 1].count('-'):
                    smoothed.append('+')
                else:
                    smoothed.append('-')
            last_segment_start = 0
            last_segment_len = 0
            small_categories = True
            while small_categories:
                small_categories = False
                for i in range(1, len(smoothed)):
                    if smoothed[i - 1] != smoothed[i] and last_segment_len > 0:
                        if last_segment_len < len(smoothed) * (5 / 100):
                            small_categories = True
                            for j in range(last_segment_start, i):
                                smoothed[j] = smoothed[i]
                        last_segment_start = i
                        last_segment_len = 0
                    last_segment_len += 1
            cuts = []
            for i in range(len(smoothed) - 1):
                if smoothed[i] != smoothed[i + 1]:
                    cuts.append(x_list[i] + (abs(x_list[i] - x_list[i + 1]))/2)
            cuts = [x_list[0]] + cuts + [x_list[-1]]
        else:
            cuts = self.get_col_quantiles(col, 6)
        self.column_cuts_mapping[col] = cuts

    def default_column_itemization(self, col: str):
        if col in self.column_item_mapping.keys() or col == self.get_target_column():
            return
        counts = {}
        unique_values = self.get_data()[col].dropna().unique()

        for v in unique_values:
            data_by_cat = self.get_data().loc[self.get_data()[col] == v]
            vcount = len(data_by_cat.index)
            vtarget_count = sum(data_by_cat[self.target])
            counts[v] = vtarget_count - (vcount - vtarget_count)
        intervals = sorted(counts, key=counts.get, reverse=False)

        self.column_item_mapping[col] = {}
        order = {}
        i = 1
        for c in intervals:
            cat = c if self.is_column_categorical(col, original=True) else c.left
            self.column_item_mapping[col][i] = cat
            order[cat] = i
            i = i + 1

        # Train data
        column_recoded = self.data_train[col].cat.rename_categories(
            lambda x: order[x if self.is_column_categorical(col, original=True) else x.left])
        cat_type = CategoricalDtype(categories=range(1, len(intervals) + 1), ordered=True)
        column_recoded = column_recoded.astype(cat_type)
        self.data_train[col] = column_recoded

        # Test data
        column_recoded = self.data_test[col].cat.rename_categories(
            lambda x: order[x if self.is_column_categorical(col, original=True) else x.left])
        cat_type = CategoricalDtype(categories=range(1, len(intervals) + 1), ordered=True)
        column_recoded = column_recoded.astype(cat_type)
        self.data_test[col] = column_recoded

    def get_column_categories(self, col: str) -> list:
        if col not in self.columns:
            return []
        if col not in self.column_cuts_mapping.keys():
            self.default_column_categorization(col)
        return self.column_cuts_mapping[col]

    def set_column_categories(self, col: str, cuts: int):
        if col in self.columns and not self.is_column_categorical(col):
            self.column_cuts_mapping[col] = cuts

    def dropna(self):
        self.data_train.dropna(inplace=True)

    @staticmethod
    def calculate_rank(vector, invert: bool = False):
        a = {}
        rank = 1
        for num in sorted(vector, reverse=not invert):
            if num not in a.keys():
                a[num] = rank
                rank += 1
        return [a[i] for i in vector]

    def preview_categorization(self, col: str) -> pd.DataFrame:
        if col in self.column_cuts_mapping.keys():
            cats = self.column_cuts_mapping[col]

            cats_df = pd.DataFrame()
            if self.is_column_categorical(col):
                cats_df['cats'] = self.get_data()[col]
            else:
                cats_df['cats'] = pd.cut(self.get_data()[col], cats, duplicates='drop', include_lowest=True)
            cats_df['target'] = self.get_target() == self.get_target_value()
            items = []
            for cat in cats_df['cats'].value_counts().index:
                target = cats_df.loc[cats_df['cats'] == cat, 'target'].value_counts()
                if len(target.index) > 0 and True not in target.index:
                    items.append(-target[False])
                elif len(target.index) > 0 and False not in target.index:
                    items.append(target[True])
                elif len(target.index) > 0:
                    items.append(target[True] - target[False])
                else:
                    items.append(0)
            result = cats_df['cats'].value_counts()
            items = self.calculate_rank(items, invert=True)

            output = result.rename_axis('category').reset_index(name='counts')
            output['label'] = items
            output = output.iloc[output['category'].cat.codes.argsort()]
            output['category'] = output['category'].astype(str)
            return output
        else:
            return pd.DataFrame([])

    def categorize(self):
        columns = self.columns.copy()
        for col in columns:
            if col not in self.column_cuts_mapping.keys():
                self.default_column_categorization(col)
            if not self.is_column_categorical(col):
                cats = self.column_cuts_mapping[col]
                new_name = col

                self.data_train[new_name] = pd.cut(self.data_train[col], cats, duplicates='drop', include_lowest=True)
                self.data_test[new_name] = pd.cut(self.data_test[col], cats, duplicates='drop', include_lowest=True)
                self.data[new_name] = pd.cut(self.data[col], cats, duplicates='drop', include_lowest=True)

                self.columns.remove(col)
                self.columns.append(new_name)
        self.data_train[self.target] = self.data_train[self.target] == self.target_value
        self.data_train[self.target] = self.data_train[self.target].astype(int)
        self.data_test[self.target] = self.data_test[self.target] == self.target_value
        self.data_test[self.target] = self.data_test[self.target].astype(int)
        self.data[self.target] = self.data[self.target] == self.target_value
        self.data[self.target] = self.data[self.target].astype(int)
        self.target_value = 1

    def test_train_split(self, train_size: float, rnd_state: int, stratify: bool, shuffle: bool):
        from sklearn.model_selection import train_test_split

        x_train, x_test, y_train, y_test = train_test_split(self.get_without_target(True),
                                                            self.get_target(True),
                                                            test_size=1-train_size,
                                                            random_state=rnd_state,
                                                            stratify=self.get_target(True) if stratify else None,
                                                            shuffle=shuffle)
        self.data_train = x_train
        self.data_train[self.get_target_column()] = y_train
        self.data_test = x_test
        self.data_test[self.get_target_column()] = y_test

    def compute_cuts_precision_recall(self, col: str, average: str = 'weighted') -> tuple:
        from sklearn.metrics import precision_score, recall_score

        if col not in self.column_cuts_mapping.keys():
            self.default_column_categorization(col)

        data = self.get_data().dropna()
        cats = self.column_cuts_mapping[col]
        if self.is_column_categorical(col):
            col_serie = data[col]
        else:
            col_serie = pd.cut(data[col], cats, duplicates='drop', include_lowest=True)
        df = pd.DataFrame([col_serie.values, data[self.target].values], columns=[col, self.target])

        counts = {}
        for i, row in df.iterrows():
            if row[col] not in counts.keys():
                counts[row[col]] = {}
            if row[self.target] not in counts[row[col]].keys():
                counts[row[col]][row[self.target]] = 0

            counts[row[col]][row[self.target]] += 1

        y_pred = []
        for i in df[col]:
            y_pred.append(max(counts[i], key=lambda k: counts[i][k]))
        y_true = data[self.target].values

        p = precision_score(y_true, y_pred, average=average)
        r = recall_score(y_true, y_pred, average=average)

        return p, r
