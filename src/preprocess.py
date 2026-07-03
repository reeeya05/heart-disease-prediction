import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(filepath):
    """
    Load the dataset.
    """
    df = pd.read_csv(filepath)
    return df


def preprocess_data(df):
    """
    Clean the dataset.
    """
    # Remove unwanted column
    df.drop(columns=['education'], inplace=True)

    # Rename column
    df.rename(columns={'male': 'Sex_male'}, inplace=True)

    # Remove missing values
    df.dropna(inplace=True)

    return df


def prepare_features(df):
    """
    Create X and y.
    """
    X = df[['age',
            'Sex_male',
            'cigsPerDay',
            'totChol',
            'sysBP',
            'glucose']]

    y = df['TenYearCHD']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, scaler