import keras
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.layers import Input, Dense, LSTM

# 1. Load the data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.width", None)  # Use full available width
pd.set_option("display.colheader_justify", "center")  # Optional: nicer column headers


image_to_print = x_train[110]

tabulate = pd.DataFrame(image_to_print)
tabulate.columns = [i for i in range(28)]

print(tabulate)

# Convert each value to hexadecimal string
hex_tabulate = tabulate.applymap(lambda x: f"{x:02X}" if x != 0 else "  ")

print(hex_tabulate.to_string(index=False))  # index=False removes row numbers
keras.datasets.