# Standard library
import datetime
import os
import random
import re
import sys
import warnings
from io import StringIO

# Data wrangling
import numpy as np
import pandas as pd
import rasterstats

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport

# Machine learning
from scipy import stats
from scipy.stats import pearsonr
import statsmodels.api as sm
import formulaic
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder

# Custom utilities
sys.path.insert(0, "../../src")
import chime
from random_forest_utils import *
from run_mp import *
from utilities import *

# Other
from pyprojroot import here

# Set chime theme
chime.theme("material")

# Function to initialize notebook with magic commands
def init_notebook():
    """Initializes Jupyter Notebook with useful magic commands."""
    from IPython import get_ipython

    ipython = get_ipython()
    if ipython:
        ipython.run_line_magic("matplotlib", "inline")
        ipython.run_line_magic("load_ext", "autoreload")
        ipython.run_line_magic("autoreload", "2")
