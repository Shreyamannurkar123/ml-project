# preprocess-model branch

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path):
    data = pd.read_csv(path)
    return data

def preprocess_data(data):
    # Example: drop null values
    data = data.dropna()

    # Example: separate features & target
    X = data.drop("target", axis=1)
    y = data["target"]

    # Example: scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)


if __name__ == "__main__":
    data = load_data("data.csv")   # your dataset
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Preprocessing Done")