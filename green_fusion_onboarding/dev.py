import pandas as pd
import torch

from sklearn.preprocessing import OneHotEncoder


def one_hot_encode(train_series, test_series):
    # One-hot encode the labels
    enc = OneHotEncoder(handle_unknown='ignore')
    train_one_hot = enc.fit_transform(train_series.to_frame()).toarray()
    test_one_hot = enc.transform(test_series.to_frame()).toarray()

    # convert it to pytorch tensor
    train_labels = torch.tensor(train_one_hot)
    test_labels = torch.tensor(test_one_hot)

    return train_labels, test_labels


# Example usage:
train = pd.Series(['cat', 'dog', 'bird', 'cat', 'bird'])
test = pd.Series(['cat'])

one_hot_encoded_series = one_hot_encode(train, test)
print(one_hot_encoded_series)
