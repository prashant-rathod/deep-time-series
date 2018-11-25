import pandas as pd
import numpy as np


def create_test_train_split(series):
    x = series[:-1]
    y = series[1:]
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    import sklearn.model_selection as sk
    return sk.train_test_split(x, y, test_size=0.30, random_state=42)

def load_M3Year():
    dfy = pd.read_excel(open('data\\M3C.xls', 'rb'), sheet_name='M3Year');
    dfy['Index'] = dfy['Series']
    dfy['Index'] = dfy['Index'].apply(lambda x: x.replace(" ", ""))
    dfy.drop(['Series', 'N', 'NF', 'Starting Year', 'Category', 'Unnamed: 5'], axis=1, inplace=True);
    dfy.set_index(['Index'], inplace=True)

    dfy = dfy.T
    dfy.reindex()
    dfy['Date'] = range(0, dfy.shape[0])

    return dfy

def load_M3Quart():
    df = pd.read_excel(open('data\\M3C.xls', 'rb'), sheet_name='M3Quart')
    df['Index'] = df['Series']
    df['Index'] = df['Index'].apply(lambda x: x.replace(" ", ""))
    df.drop(['Series', 'N', 'NF', 'Starting Year', 'Category', 'Starting Quarter'], axis=1, inplace=True);
    df.set_index(['Index'], inplace=True)

    df = df.T
    df.reindex()
    df['Date'] = range(0, df.shape[0])

    return df


def load_M3Month():
    df = pd.read_excel(open('data\\M3C.xls', 'rb'), sheet_name='M3Month')
    df['Index'] = df['Series']
    df['Index'] = df['Index'].apply(lambda x: x.replace(" ", ""))
    df.drop(['Series', 'N', 'NF', 'Starting Year', 'Category', 'Starting Month'], axis=1, inplace=True);
    df.set_index(['Index'], inplace=True)

    df = df.T
    df.reindex()
    df['Date'] = range(0, df.shape[0])

    return df


def load_M3Other():
    df = pd.read_excel(open('data\\M3C.xls', 'rb'), sheet_name='M3Other')
    df['Index'] = df['Series']
    df['Index'] = df['Index'].apply(lambda x: x.replace(" ", ""))
    df.drop(['Series', 'N', 'NF', 'Category', 'Unnamed: 4', 'Unnamed: 5'], axis=1, inplace=True);
    df.set_index(['Index'], inplace=True)

    df = df.T
    df.reindex()
    df['Date'] = range(0, df.shape[0])

    return df
