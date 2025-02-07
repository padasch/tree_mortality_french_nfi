import os
from sklearn.inspection import PartialDependenceDisplay

from pyprojroot import here

# Data wrangling
import pandas as pd
import numpy as np
import random
import rasterstats

# Data visualisation
from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pylab import f

# Machine learning
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.inspection import PartialDependenceDisplay

import formulaic
from scipy import stats
from scipy.stats import pearsonr
import statsmodels.api as sm

# My functions
import sys

sys.path.insert(0, "../../src")
from run_mp import *
from utilities import *
from random_forest_utils import *

# Other
from os import error
import datetime
from io import StringIO
import re
import warnings
import chime
from pyprojroot import here

chime.theme("material")

# # Magic
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

# Function to initialize notebook with magic commands
def init_notebook():
    # Import IPython's get_ipython function to access the current IPython session
    from IPython import get_ipython
    ipython = get_ipython()

    # Magic commands
    if ipython:
        ipython.run_line_magic('matplotlib', 'inline')
        ipython.run_line_magic('load_ext', 'autoreload')
        ipython.run_line_magic('autoreload', '2')