import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from torch import Tensor


class LabelEncoder:
    def __init__(self):
        self.encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.categories = None

    def fit(self, label_series):
        # Reshape the input series to a 2D array
        label_array = label_series.values.reshape(-1, 1)

        # Fit the OneHotEncoder and store the categories for later transformation
        self.encoder.fit(label_array)
        self.categories = label_series.unique()

    def transform_one_hot(self, color):
        # Transform a single color value to a one-hot-encoded vector
        transformed = self.encoder.transform([[color]])
        return transformed.flatten()

    def inverse_transform_one_hot(self, one_hot_vector):
        # Transform a one-hot-encoded vector back to the original color category
        inverse_transformed = self.encoder.inverse_transform([one_hot_vector])
        return inverse_transformed[0, 0]

    def transform(self, label_series, return_df=False):

        if isinstance(label_series, pd.Series):
            label_values = label_series.values.reshape(-1, 1)
        elif isinstance(label_series, pd.DataFrame):
            label_values = label_series.values
        elif isinstance(label_series, np.ndarray):
            label_values = label_series
        elif isinstance(label_series, Tensor):
            label_values = label_series.numpy()
        else:
            raise ValueError("The input must be a pandas Series or DataFrame")

        # Transform the entire series to a one-hot-encoded DataFrame
        transformed = self.encoder.transform(label_values)
        columns = [f'{category}_encoded' for category in self.categories]
        one_hot_df = pd.DataFrame(transformed, columns=columns)

        if return_df:
            return one_hot_df
        else:
            return one_hot_df.values

    def inverse_transform(self, one_hot_values, return_type='numpy'):
        # Transform a one-hot-encoded DataFrame back to the original series
        if isinstance(one_hot_values, pd.DataFrame):
            values = one_hot_values.values
        elif isinstance(one_hot_values, np.ndarray):
            values = one_hot_values
        elif isinstance(one_hot_values, Tensor):
            values = one_hot_values.numpy()
        else:
            raise ValueError("The input must be a pandas DataFrame")

        inverse_transformed = self.encoder.inverse_transform(values)
        inverse_series = pd.Series(inverse_transformed.flatten())

        if return_type == 'numpy':
            return inverse_series.values
        elif return_type == 'pandas':
            return inverse_series
        elif return_type == 'torch':
            return Tensor(inverse_series.values)
        else:
            raise ValueError("The return_type must be either 'numpy', 'pandas', or 'torch'")


def main():
    # Example usage:
    colors = pd.Series(['Red', 'Green', 'Blue', 'Red', 'Green', 'Blue'])
    encoder = LabelEncoder()
    encoder.fit(colors)

    # Transform a single color value to one-hot-encoded vector
    red_vector = encoder.transform_one_hot('Red')
    print(f"One-hot-encoded vector for 'Red': {red_vector}")

    # Transform a one-hot-encoded vector back to the original color value
    original_color = encoder.inverse_transform_one_hot(red_vector)
    print(f"Original color for the one-hot-encoded vector: {original_color}")

    # Transform the entire series to a one-hot-encoded DataFrame
    one_hot_df = encoder.transform(colors)
    print("\nOne-hot-encoded DataFrame:")
    print(one_hot_df)

    # Transform a one-hot-encoded DataFrame back to the original series
    inverse_series = encoder.inverse_transform(one_hot_df)
    print("\nOriginal series from one-hot-encoded DataFrame:")
    print(inverse_series)


if __name__ == '__main__':
    main()
