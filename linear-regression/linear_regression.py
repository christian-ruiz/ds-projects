
# upload packages
import pandas as pd
import numpy as np
import os
import requests

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# import df
df = pd.read_csv('linear-regression/USA_Housing.csv')