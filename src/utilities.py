
import os
import io
from sklearn.inspection import PartialDependenceDisplay

from pyprojroot import here
from IPython.display import clear_output  # For clearing the output of a cell

# Data wrangling
import glob
from datetime import datetime
from datetime import datetime, date
import pandas as pd
import geopandas as gpd
import numpy as np
import random
import rasterio
from rasterio.transform import Affine

from tqdm import tqdm
import xarray as xr
import random as rnd
from rasterstats import zonal_stats

import itertools


# Data visualisation
from ydata_profiling import ProfileReport

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolor
import matplotlib.gridspec as gridspec

from matplotlib.pylab import f
from matplotlib.ticker import MaxNLocator


# Machine learning
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    KFold,
)
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.impute import KNNImputer
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, roc_auc_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve, average_precision_score

from imblearn.over_sampling import SMOTE

from pymer4.models import Lmer
from pymer4.models import Lm

from scipy import stats
from scipy.stats import mode
import shap
import pickle
import ast

from scipy.stats import pearsonr, f_oneway, chi2_contingency

import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess

import math as math
import pymannkendall as mk

# My functions
import sys

sys.path.insert(0, "../../src")
from run_mp import *
from random_forest_utils import *

# Other
from os import error
import datetime
from io import StringIO
import re
import warnings
import chime
from pyprojroot import here
from IPython.display import Image

chime.theme("material")


# --------------------------------------------------------------------------------
def ___GENERAL_FUNCTIONS___():
    pass

def ring_when_done(stop=False):
    print(f"Finished at {datetime.datetime.now().strftime('%H:%M:%S')}")
    chime.success("Done!")
    if stop:
        raise ValueError("Stopped.")

def countdown(seconds):
    import time
    finish_time = datetime.datetime.now() + datetime.timedelta(seconds=seconds)
    for i in range(seconds, 0, -1):
        print(f"Waiting for {i} seconds... (until: {finish_time})", end="\r")
        time.sleep(1)
    print("Time's up!                   ")  # Clear the line after the countdown

def start_time(verbose=True):
    import time
    if verbose:
        print(f"Started at {datetime.datetime.now().strftime('%H:%M:%S')}")
    return datetime.datetime.now()

def end_time(start_time, folderpath=None, ring=True):
    import time
    end = datetime.datetime.now()
    time_diff = str(end - start_time)
    text = f"Time elapsed: {time_diff} \n   Start:\t {start_time} \n   End:\t\t {end}"
    display("")
    print(f"⌛ {text}")
    if folderpath:
        folderpath = folderpath + "/time_elapsed.txt"
        with open(folderpath, "w") as file:
            file.write(text)
    if ring:
        chime.success()

# -----------------------------------------------------------------------------
def load_hexmap():
    """
    Loads the hexmap from the data folder.
    """

    # Load hexmap
    hexmap = gpd.read_file(here("data/raw/maps/france_geojson/hex.geojson"))
    # Check if fresh hexmap is loaded (has not hex id yet)
    if "hex" in hexmap.columns:
        return hexmap

    # If hex id is not present, add and save hexmap
    hexmap = hexmap.reset_index()[["index", "geometry"]].rename(
        {"index": "hex"}, axis=1
    )
    hexmap.to_file(here("data/raw/maps/france_geojson/hex.geojson"), driver="GeoJSON")
    return hexmap


# -----------------------------------------------------------------------------
def filter_report(filter, df_before, df_after, site_level=False):
    if site_level:
        sites_before = df_before["idp"].nunique()
        sites_after = df_after["idp"].nunique()

        sites_removed = sites_before - sites_after
        sites_removed_percentage = round(100 - (sites_after / sites_before) * 100)

        print(
            f" - Filter: {filter:<30} |\tSites from {sites_before} to {sites_after} (= {sites_removed:>10}, {sites_removed_percentage}%)\t|\t",
            (
                f"❗More than 5% of sites removed❗"
                if sites_removed_percentage > 5
                else ""
            ),
        )

    else:
        trees_before = df_before["tree_id"].nunique()
        sites_before = df_before["idp"].nunique()

        trees_after = df_after["tree_id"].nunique()
        sites_after = df_after["idp"].nunique()

        sites_removed = sites_before - sites_after
        sites_removed_percentage = round(100 - (sites_after / sites_before) * 100)

        trees_removed = trees_before - trees_after
        trees_removed_percentage = round(100 - (trees_after / trees_before) * 100)

        print(
            f" - Filter: {filter:<30} |\tSites from {sites_before} to {sites_after} (= {sites_removed:>10}, {sites_removed_percentage}%)\t|\t",
            f"Trees from {trees_before} to {trees_after} (= {trees_removed:>5}, {trees_removed_percentage}%)",
            (
                f"❗More than 5% of sites removed❗"
                if sites_removed_percentage > 5
                else ""
            ),
            (
                f"❗More than 5% of trees removed❗"
                if trees_removed_percentage > 5
                else ""
            ),
        )


def list_all_cols_in_df(df):
    for col in df.columns:
        print(f" - {col}")
    pass


def ___NFI_WRANGLING_FUNCTIONS___():
    pass


def get_latest_nfi_raw_data():

    print("Loading latest NFI raw data wrangled in R...")

    # files = glob.glob(f"{here('data/tmp')}/*nfi_dataset_for_analysis*.csv")
    files = glob.glob(f"{here('data/tmp/nfi/from-R')}/*_nfi_dataset_raw.csv")

    # Sorting files by date so that the first in list is the latest
    files.sort(reverse=True)
    # Pick latest file
    latest_file = files[0]
    # Get the modification date and time of the latest file
    modification_time = os.path.getctime(latest_file)
    modification_date_time = datetime.datetime.fromtimestamp(
        modification_time
    ).strftime("%A %Y-%m-%d, %H:%M")

    # Calculate the difference between today and the modification date
    today = date.today()
    modification_date = datetime.datetime.strptime(
        modification_date_time, "%A %Y-%m-%d, %H:%M"
    ).date()
    diff = today - modification_date

    print(
        f"👉 Latest file is {latest_file}",
        f"\n👉 Created on {modification_date_time} which is {diff.days} days ago.",
    )

    # Load the file
    nfi_data_raw = pd.read_csv(latest_file, index_col=0)
    return nfi_data_raw


def load_file_and_report_recency(latest_file, verbose=True):

    # Get the modification date and time of the latest file
    modification_time = os.path.getctime(latest_file)
    modification_date_time = datetime.datetime.fromtimestamp(
        modification_time
    ).strftime("%A %Y-%m-%d, %H:%M")

    # Calculate the difference between today and the modification date
    today = date.today()
    modification_date = datetime.datetime.strptime(
        modification_date_time, "%A %Y-%m-%d, %H:%M"
    ).date()
    diff = today - modification_date

    if verbose:
        print(
            f"\n - Latest file is {latest_file}",
            f"\n - Created on {modification_date_time} which is {diff.days} days ago.",
            "\n",
        )

    # Detect ending of file
    if latest_file.endswith(".csv"):
        data = pd.read_csv(latest_file, index_col=0)
    elif latest_file.endswith(".feather"):
        data = pd.read_feather(latest_file)
    else:
        raise ValueError("File type not recognized.")

    return data


def get_feature_database_sheet(sheet=None):
    if sheet is None:
        # Get sheets in the excel file
        sheets = pd.read_excel(
            here("docs/ifna_predictor_database.xlsx"),
            sheet_name=None,
        ).keys()

        print(f"Available sheets in the excel file:")
        for sheet in sheets:
            print(f"  - {sheet}")

        raise ValueError("Please provide a sheet name.")

    nfi_org = pd.read_excel(
        here("docs/ifna_predictor_database.xlsx"),
        sheet_name=sheet,
    )[["var", "type", "level", "remove"]]

    # Add suffixes _1 and _2 to the original variables to distinguish them if sampled from different years
    suffix_1 = nfi_org.copy()
    suffix_2 = nfi_org.copy()

    suffix_1["var"] = suffix_1["var"].astype(str).apply(lambda x: x + "_1")
    suffix_2["var"] = suffix_2["var"].astype(str).apply(lambda x: x + "_2")

    nfi_org_with_suffixes = pd.concat([nfi_org, suffix_1, suffix_2])
    return nfi_org_with_suffixes


# -----------------------------------------------------------------------------
def get_final_nfi_coordinates(noisy_or_corrected=None, geojson_or_csv=None, epsg=None):
    if noisy_or_corrected == "noisy":
        if geojson_or_csv == "csv":
            print("Loading noisy coordinates from csv.")
            return pd.read_csv(
                here("data/final/nfi/coords_of_sites_with_idp.csv"), index_col=None
            )
        elif geojson_or_csv == "geojson":
            if epsg == "4326":
                print("Loading noisy coordinates from geojson EPSG 4326.")
                return gpd.read_file(
                    here("data/final/nfi/sites_with_idp_epsg4326.geojson")
                )
            elif epsg == "2154":
                print("Loading noisy coordinates from geojson EPSG 2154.")
                return gpd.read_file(
                    here("data/final/nfi/sites_with_idp_epsg2154.geojson")
                )
            else:
                raise ValueError("Please specify EPSG 4326 or 2154.")
        else:
            raise ValueError("Please specify if you want to load geojson or csv.")

    elif noisy_or_corrected == "corrected":
        raise ValueError("Getting corrected coordinates implemented yet.")
    else:
        raise ValueError("Please specify if you want noisy or corrected coordinates.")


def get_final_nfi_data_for_analysis(verbose=True):

    if verbose:
        print("\nLoading final NFI data for analysis... (output of python wrangling)")

    # Fixed filename
    latest_file = here("data/final/nfi/nfi_ready_for_analysis.feather")

    # Get the modification date and time of the latest file
    modification_time = os.path.getctime(latest_file)
    modification_date_time = datetime.datetime.fromtimestamp(
        modification_time
    ).strftime("%A %Y-%m-%d, %H:%M")

    # Calculate the difference between today and the modification date
    today = date.today()
    modification_date = datetime.datetime.strptime(
        modification_date_time, "%A %Y-%m-%d, %H:%M"
    ).date()
    diff = today - modification_date

    # Load the file
    df = pd.read_feather(latest_file)
    
    if verbose:
        print(
            f"- Latest file is {latest_file}",
            f"\n- Created on {modification_date_time} which is {diff.days} days ago.",
        )
        print("  Number of trees: ", len(df))
        print("  Number of sites: ", len(df.idp.unique()))

    return df


def calculate_harvest(df_in, grouping_variable="idp"):

    n_alive_xxx = df_in.query("tree_state_1 == 'alive'").shape[0]
    n_alive_alive = df_in.query("tree_state_change == 'alive_alive'").shape[0]
    n_alive_dead = df_in.query("tree_state_change == 'alive_dead'").shape[0]
    n_alive_cut = df_in.query("tree_state_change == 'alive_cut'").shape[0]

    group = df_in[grouping_variable].unique()[0]
    perc_cut = n_alive_cut / n_alive_xxx * 100 / 5 if n_alive_xxx > 0 else np.nan
    perc_dead = n_alive_dead / n_alive_xxx * 100 / 5 if n_alive_xxx > 0 else np.nan
    perc_cutdead = (
        (n_alive_cut + n_alive_dead) / n_alive_xxx * 100 / 5
        if n_alive_xxx > 0
        else np.nan
    )

    df_out = pd.DataFrame(
        {
            grouping_variable: group,
            "n_alive_xxx": n_alive_xxx,
            "n_alive_alive": n_alive_alive,
            "n_alive_dead": n_alive_dead,
            "n_alive_cut": n_alive_cut,
            "perc_cut": perc_cut,
            "perc_dead": perc_dead,
            "perc_cutdead": perc_cutdead,
        },
        index=[0],
    )

    return df_out


def calculate_harvest_loop(df_in, grouping_variable="idp"):

    groups = df_in[grouping_variable].unique()
    df_list = []

    for g in groups:
        df_group = df_in.query(f"{grouping_variable} == @g")
        df_list.append(calculate_harvest(df_group, grouping_variable))

    return pd.concat(df_list)


def calculate_harvest_loop_mp(df_in, grouping_variable="idp"):

    # Split into list
    df_in = split_df_into_list_of_group_or_ns(df_in, 10, grouping_variable)

    # Run mp
    df_mp = run_mp(
        calculate_harvest_loop,
        arg_list=df_in,
        combine_func=pd.concat,
        num_cores=10,
        progress_bar=True,
        grouping_variable=grouping_variable,
    )

    chime.success()

    return df_mp


# -----------------------------------------------------------------------------
def calculate_growth_mortality(
    df_in,
    verbose=False,
    divide_by_nplots=False,
    grouping_variable="idp",
    per_year=True,
    min_trees_per_plot=None,
):
    """
    Function to calculate growth and mortality metrics for a given group.

    Raises:
        ValueError: _description_

    Returns:
        panda dataframe: mortality metrics per group
    """

    raise ValueError(
        "This function is depreciated. Use calculate_growth_mortality_optimized() instead."
    )

    # Check if grouping variable is valid
    if grouping_variable not in df_in.columns:
        raise ValueError(
            f"Grouping variable {grouping_variable} is not in the dataframe."
        )

    # Get grouping variable
    my_group = df_in[grouping_variable].unique()[0]

    if verbose:
        print(
            f"> calculate_growth_mortality():\n  Calculating growth and mortality for {grouping_variable}: {my_group}."
        )

    # Check by how many years metrics of change must be divided
    if per_year:
        timespan = 5  # Default. This gives the change per year.
    else:
        timespan = 1  # This equals the entire change over 5 years.

    # -------------------------------------------------------------------------
    # Stem-based metrics
    # ! Important: To calculate natural stem-based mortality, we exclude trees that have been cut
    n_ini = df_in.query("tree_state_1 == 'aliveand tree_state_2 != 'cut'").shape[0]
    n_ini_withcuts = df_in.query("tree_state_1 == 'alive'").shape[0]
    n_sur = df_in.query("tree_state_change == 'alive_alive'").shape[0]
    n_rec = df_in.query("tree_state_change == 'new_alive'").shape[0]
    n_die = df_in.query("tree_state_change == 'alive_dead'").shape[0]
    n_cut = df_in.query("tree_state_change == 'alive_cut'").shape[0]
    n_fin = n_ini + n_rec

    # Check if metrics should be calculated based on min_trees_per_plot
    if min_trees_per_plot is not None:
        if n_ini < min_trees_per_plot:
            df_out = pd.DataFrame(
                {
                    grouping_variable: my_group,
                    "n_plots": np.nan,
                    # Tree count
                    "n_ini": np.nan,
                    "n_ini_withcuts": np.nan,
                    "n_sur": np.nan,
                    "n_fin": np.nan,
                    "n_rec": np.nan,
                    "n_die": np.nan,
                    "n_cut": np.nan,
                    # Basal Area
                    "ba_at_v1_of_alive_trees": np.nan,
                    "ba_at_v2_of_alive_trees": np.nan,
                    "ba_at_v1_of_survived": np.nan,
                    "ba_at_v2_of_survived": np.nan,
                    "ba_at_v1_of_died": np.nan,
                    "ba_at_v2_of_died": np.nan,
                    "ba_at_v2_of_recruited": np.nan,
                    "ba_at_v2_of_cut": np.nan,
                    # Change
                    # - Mortality
                    "mort_stems_prc_yr_esq": np.nan,
                    "mort_stems_prc_yr_hoshino": np.nan,
                    "mort_ba_prc_yr_hoshino": np.nan,
                    "mort_ba_yr_v1": np.nan,
                    "mort_ba_yr_v2": np.nan,
                    "mort_ba_prc_yr_v1": np.nan,
                    "mort_ba_prc_yr_v2": np.nan,
                    # - Growth
                    "rec_stems_prc_yr_hoshino": np.nan,
                    "tot_growth_ba_prc_yr_hoshino": np.nan,
                    "sur_growth_ba_prc_yr_hoshino": np.nan,
                    "tot_growth_ba_yr": np.nan,
                    "sur_growth_ba_yr": np.nan,
                    "tot_growth_ba_prc_yr": np.nan,
                    "sur_growth_ba_prc_yr": np.nan,
                    # - Cutting
                    "cut_ba_yr_v1": np.nan,
                    "cut_ba_prc_yr_v1": np.nan,
                },
                index=[0],
            )

            return df_out

    # -------------------------------------------------------------------------
    # Basal area-based metrics
    ba_at_v1_of_alive_trees = df_in[df_in["tree_state_1"] == "alive"]["ba_1"].sum()
    ba_at_v2_of_alive_trees = df_in.query(
        "tree_state_change == 'alive_alive| tree_state_change == 'new_alive'"
    )["ba_2"].sum()

    ba_at_v1_of_survived = df_in.query("tree_state_change == 'alive_alive'")[
        "ba_1"
    ].sum()
    ba_at_v2_of_survived = df_in.query("tree_state_change == 'alive_alive'")[
        "ba_2"
    ].sum()

    ba_at_v1_of_died = df_in.query("tree_state_change == 'alive_dead'")["ba_1"].sum()
    ba_at_v2_of_died = df_in.query("tree_state_change == 'alive_dead'")["ba_2"].sum()
    ba_at_v2_of_recruited = df_in.query("tree_state_change == 'new_alive'")[
        "ba_2"
    ].sum()

    ba_at_v1_of_cut = df_in.query("tree_state_change == 'alive_cut'")["ba_1"].sum()

    # Mortality following Esquivel et al.
    if n_ini == 0:
        mort_stems_prc_yr_esq = np.nan
    else:
        mort_stems_prc_yr_esq = (1 - (n_sur / n_ini) ** (1 / timespan)) * 100

    # Mortality and Recruitment following Hoshino et al.
    if n_sur == 0:
        mort_stems_prc_yr_hoshino = np.nan
        rec_stems_prc_yr_hoshino = np.nan
    else:
        mort_stems_prc_yr_hoshino = np.log(n_ini / n_sur) / timespan * 100
        rec_stems_prc_yr_hoshino = np.log(n_fin / n_sur) / timespan * 100

    # Loss, gain, ingrowth following Hoshino et al.
    if ba_at_v1_of_survived == 0:
        mort_ba_prc_yr_hoshino = np.nan
        tot_growth_ba_prc_yr_hoshino = np.nan
        sur_growth_ba_prc_yr_hoshino = np.nan
    else:
        mort_ba_prc_yr_hoshino = (
            np.log(ba_at_v1_of_alive_trees / ba_at_v1_of_survived) / timespan * 100
        )
        tot_growth_ba_prc_yr_hoshino = (
            np.log(ba_at_v2_of_alive_trees / ba_at_v1_of_survived) / timespan * 100
        )
        sur_growth_ba_prc_yr_hoshino = (
            np.log(ba_at_v2_of_survived / ba_at_v1_of_survived) / timespan * 100
        )

    # Absolute changes
    tot_growth_ba_yr = (ba_at_v2_of_alive_trees - ba_at_v1_of_survived) / timespan
    sur_growth_ba_yr = (ba_at_v2_of_survived - ba_at_v1_of_survived) / timespan
    mort_ba_yr_v1 = ba_at_v1_of_died / timespan
    mort_ba_yr_v2 = ba_at_v2_of_died / timespan
    cut_ba_yr_v1 = ba_at_v1_of_cut / timespan

    # Relative changes
    #  - Growth
    if ba_at_v1_of_survived == 0:
        tot_growth_ba_prc_yr = np.nan
        sur_growth_ba_prc_yr = np.nan
    else:
        tot_growth_ba_prc_yr = tot_growth_ba_yr / ba_at_v1_of_survived * 100
        sur_growth_ba_prc_yr = sur_growth_ba_yr / ba_at_v1_of_survived * 100

    # - Mortality
    #   - With respect to total basal area at v1
    if ba_at_v1_of_alive_trees == 0:
        mort_ba_prc_yr_v1 = np.nan
    else:
        mort_ba_prc_yr_v1 = mort_ba_yr_v1 / ba_at_v1_of_alive_trees * 100

    #   - With respect to total basal area at v2 (not very logical...)
    if (ba_at_v2_of_died + ba_at_v2_of_alive_trees) == 0:
        mort_ba_prc_yr_v2 = np.nan
    else:
        mort_ba_prc_yr_v2 = (
            mort_ba_yr_v2 / (ba_at_v2_of_died + ba_at_v2_of_alive_trees) * 100
        )

    # - Cutting
    if ba_at_v1_of_alive_trees == 0:
        cut_ba_prc_yr_v1 = np.nan
    else:
        cut_ba_prc_yr_v1 = cut_ba_yr_v1 / ba_at_v1_of_alive_trees * 100

    # -------------------------------------------------------------------------
    # Divide by number of plots if required
    if divide_by_nplots:
        if verbose:
            print(
                f"\n Dividing all metrics by number of plots per grouping variable: {grouping_variable} so that they are in terms of per plot."
            )

        raise ValueError(
            "This is not implemented yet. Can be simplified by just dividing final df by n_plots, except for grouping variable."
        )

        if (divide_by_nplots) & (grouping_variable == "idp"):
            os.error("Cannot divide by number of plots if grouping variable is idp.")

        # n_plots = df_in["idp"].nunique()
        # # Tree counts
        # n_ini = n_ini / n_plots
        # n_sur = n_sur / n_plots
        # n_fin = n_fin / n_plots
        # n_rec = n_rec / n_plots
        # n_die = n_die / n_plots
        # n_cut = n_cut / n_plots
        # # Basal Area
        # ba_at_v1_of_alive_trees = ba_at_v1_of_alive_trees / n_plots
        # ba_at_v2_of_alive_trees = ba_at_v2_of_alive_trees / n_plots
        # ba_at_v1_of_survived = ba_at_v1_of_survived / n_plots
        # ba_at_v2_of_survived = ba_at_v2_of_survived / n_plots
        # ba_at_v2_of_died = ba_at_v2_of_died / n_plots
        # ba_at_v2_of_recruited = ba_at_v2_of_recruited / n_plots
        # # Changes
        # # - Mortality
        # mort_stems_prc_yr_esq = mort_stems_prc_yr_esq / n_plots
        # mort_stems_prc_yr_hoshino = mort_stems_prc_yr_hoshino / n_plots
        # mort_ba_prc_yr_hoshino = mort_ba_prc_yr_hoshino / n_plots
        # mort_ba_yr_v1 = mort_ba_yr_v1 / n_plots
        # mort_ba_yr_v2 = mort_ba_yr_v2 / n_plots
        # mort_ba_prc_yr_v1 = mort_ba_prc_yr_v1 / n_plots
        # mort_ba_prc_yr_v2 = mort_ba_prc_yr_v2 / n_plots
        # # - Growth
        # rec_stems_prc_yr_hoshino = rec_stems_prc_yr_hoshino / n_plots
        # tot_growth_ba_prc_yr_hoshino = tot_growth_ba_prc_yr_hoshino / n_plots
        # tot_growth_ba_yr = tot_growth_ba_yr / n_plots
        # tot_growth_ba_prc_yr = tot_growth_ba_prc_yr / n_plots
        # sur_growth_ba_prc_yr_hoshino = sur_growth_ba_prc_yr_hoshino / n_plots
        # sur_growth_ba_yr = sur_growth_ba_yr / n_plots
        # sur_growth_ba_prc_yr = sur_growth_ba_prc_yr / n_plots
        # # - Cutting

    else:
        n_plots = df_in["idp"].nunique()

    df_out = pd.DataFrame(
        {
            grouping_variable: my_group,
            "n_plots": n_plots,
            # Tree count
            "n_ini": n_ini,
            "n_ini_withcuts": n_ini_withcuts,
            "n_sur": n_sur,
            "n_fin": n_fin,
            "n_rec": n_rec,
            "n_die": n_die,
            "n_cut": n_cut,
            # Basal Area
            "ba_at_v1_of_alive_trees": ba_at_v1_of_alive_trees,
            "ba_at_v2_of_alive_trees": ba_at_v2_of_alive_trees,
            "ba_at_v1_of_survived": ba_at_v1_of_survived,
            "ba_at_v2_of_survived": ba_at_v2_of_survived,
            "ba_at_v1_of_died": ba_at_v1_of_died,
            "ba_at_v2_of_died": ba_at_v2_of_died,
            "ba_at_v2_of_recruited": ba_at_v2_of_recruited,
            "ba_at_v2_of_cut": ba_at_v1_of_cut,
            # Change
            # - Mortality
            "mort_stems_prc_yr_esq": mort_stems_prc_yr_esq,
            "mort_stems_prc_yr_hoshino": mort_stems_prc_yr_hoshino,
            "mort_ba_prc_yr_hoshino": mort_ba_prc_yr_hoshino,
            "mort_ba_yr_v1": mort_ba_yr_v1,
            "mort_ba_yr_v2": mort_ba_yr_v2,
            "mort_ba_prc_yr_v1": mort_ba_prc_yr_v1,
            "mort_ba_prc_yr_v2": mort_ba_prc_yr_v2,
            # - Growth
            "rec_stems_prc_yr_hoshino": rec_stems_prc_yr_hoshino,
            "tot_growth_ba_prc_yr_hoshino": tot_growth_ba_prc_yr_hoshino,
            "sur_growth_ba_prc_yr_hoshino": sur_growth_ba_prc_yr_hoshino,
            "tot_growth_ba_yr": tot_growth_ba_yr,
            "sur_growth_ba_yr": sur_growth_ba_yr,
            "tot_growth_ba_prc_yr": tot_growth_ba_prc_yr,
            "sur_growth_ba_prc_yr": sur_growth_ba_prc_yr,
            # - Cutting
            "cut_ba_yr_v1": cut_ba_yr_v1,
            "cut_ba_prc_yr_v1": cut_ba_prc_yr_v1,
        },
        index=[0],
    )

    return df_out


# -----------------------------------------------------------------------------


def mp_calculate_growth_mortality(df_in, grouping_variable="idp", per_year=True):

    # Keep only relevant columns
    keep_these = list(
        set(
            [
                grouping_variable,
                "idp",
                "tree_id",
                "tree_state_1",
                "tree_state_2",
                "tree_state_change",
                "ba_1",
                "ba_2",
                "v",
            ]
        )
    )
    df_in = df_in[keep_these]
    
    # display("🚨🚨🚨 I CHANGED mp_calculate_growth_mortality() TO ONLY USE 9 CORES TO CONTINUE WORKING WHILE RUNNING 🚨🚨🚨")
    ncores = 10
    
    # Split into list
    df_in = split_df_into_list_of_group_or_ns(df_in, ncores, grouping_variable)
    # Run mp
    df_mp = run_mp(
        calculate_growth_mortality_optimized_loop,
        arg_list=df_in,
        combine_func=pd.concat,
        num_cores=ncores,
        progress_bar=True,
        grouping_variable=grouping_variable,
        per_year=per_year,
    )

    chime.success()

    return df_mp.reset_index(drop=True)


def calculate_growth_mortality_optimized_loop(
    df_in=None, grouping_variable="idp", per_year=True
):
    groups = df_in[grouping_variable].unique()
    df_list = []

    for g in groups:
        df_group = df_in.query(f"{grouping_variable} == @g")
        df_list.append(
            calculate_growth_mortality_optimized(
                df_in=df_group, grouping_variable=grouping_variable, per_year=per_year
            )
        )

    return pd.concat(df_list)


# Turn of formatting just for this function
# fmt: off
def calculate_growth_mortality_optimized(
    df_in=None,
    verbose=False,
    grouping_variable="idp",
    per_year=True,
    return_metrics_of_change=False,
):
    """
    Optimized function to calculate growth and mortality metrics for a given input df.

    Args:
        df_in (pd.DataFrame): Input dataframe.
        verbose (bool, optional): Verbose output. Defaults to False.
        divide_by_nplots (bool, optional): Divide metrics by number of plots. Defaults to False.
        grouping_variable (str, optional): Grouping variable name (grouping done outside, not within the function!). Defaults to "idp".
        per_year (bool, optional): Calculate metrics per year. Defaults to True.
        min_trees_per_plot (int, optional): Minimum trees per plot. Defaults to 1.

    Raises:
        ValueError: If the grouping variable is not in the dataframe.

    Returns:
        pd.DataFrame: Mortality metrics per group.
    """

    # Return mertics of change
    # ! Make sure to update this if changing the metrics!
    if return_metrics_of_change:
        metrics = sorted([
            "mort_nat_stems_prc_yr_esq",
            "mort_tot_stems_prc_yr_esq",
            "mort_cut_stems_prc_yr_esq",
            "mort_tot_stems_prc_yr",
            "mort_nat_stems_prc_yr",
            "mort_cut_stems_prc_yr",
            "mort_tot_ba_yr",
            "mort_tot_ba_prc_yr",
            "mort_nat_ba_yr",
            "mort_nat_ba_prc_yr",
            "mort_cut_ba_yr",
            "mort_cut_ba_prc_yr",
            "mort_tot_vol_yr",
            "mort_tot_vol_prc_yr",
            "mort_nat_vol_yr",
            "mort_nat_vol_prc_yr",
            "mort_cut_vol_yr",
            "mort_cut_vol_prc_yr",
            "grwt_stems_prc_yr",
            "grwt_tot_ba_yr",
            "grwt_tot_ba_prc_yr",
            "grwt_tot_ba_prc_yr_hos",
            "grwt_sur_ba_yr",
            "grwt_sur_ba_prc_yr",
            "grwt_sur_ba_prc_yr_hos",
            "grwt_rec_ba_yr",
            "grwt_rec_ba_prc_yr",
            "change_tot_ba_yr",
            "change_tot_ba_prc_yr",
            ])
        
        return metrics
    
    # Check if grouping variable is valid
    if grouping_variable not in df_in.columns:
        raise ValueError(f"Grouping variable {grouping_variable} is not in the dataframe.")
    # Get grouping variable
    my_group = df_in[grouping_variable].unique()[0]

    if verbose:
        print(f"> Calculating growth and mortality for {grouping_variable}: {my_group}.")

    # Set timespan
    timespan = 1 / 5 if per_year else 1

    # Remove all trees that were dead at the first visit
    df_in = df_in[df_in['tree_state_1'] != 'dead']
    
    # Subset to relevant columns
    relevant_cols = list(set([grouping_variable, "idp", "tree_id", "tree_state_1", "tree_state_2", "tree_state_change", "ba_1", "ba_2", "v"]))
    df_in = df_in[relevant_cols]

    # Compute conditions once
    alive_1     = df_in['tree_state_1']      == 'alive'
    alive_2     = df_in['tree_state_2']      == 'alive'
    alive_alive = df_in['tree_state_change'] == 'alive_alive'
    alive_dead  = df_in['tree_state_change'] == 'alive_dead'
    alive_cut   = df_in['tree_state_change'] == 'alive_cut'
    new_alive   = df_in['tree_state_change'] == 'new_alive'

    # Stem-based metrics
    # Define stand-metrics dictionary
    stem_metrics = pd.Series({
        "n_a1": df_in[alive_1].shape[0],     # a1 = alive at first visit
        "n_a2": df_in[alive_2].shape[0],     # a2 = alive at second visit                              
        "n_aa": df_in[alive_alive].shape[0], # aa = alive_alive       
        "n_ad": df_in[alive_dead].shape[0],  # ad = alive_dead       
        "n_ac": df_in[alive_cut].shape[0],   # ac = alive_cut   
        "n_na": df_in[new_alive].shape[0],   # na = new_alive   
    })
        
    # ! Notes:
    # ! 1.) I am calculation mortality only with respect to the first visit so that we know how much of the initial stand was lost per year. This means that I can also calculate metrics wrt. volume!
    # ! 2.) The mortality calculations from Hoshino and Esquivel are very similar, so I am only using the Esquivel one.
    # !     - Esquivel: 1 - (survivors / initals) ** (1 / 5) * 100
    # !     - Hoshino: ln(initals / survivors) / 5 * 100
    
    # Size-based metrics
    size_metrics = df_in[["ba_1", "ba_2", "v"]].assign(
        # Basal Area
        ba_ax_v1 = alive_1 * df_in["ba_1"],
        ba_ax_v2 = alive_2 * df_in["ba_2"],
        
        ba_aa_v1 = alive_alive * df_in["ba_1"],
        ba_aa_v2 = alive_alive * df_in["ba_2"],
        
        ba_ad_v1 = alive_dead * df_in["ba_1"],
        ba_ac_v1 = alive_cut * df_in["ba_1"],
        ba_na_v2 = new_alive * df_in["ba_2"],
        
        # Volume
        vol_ax_v1 = alive_1 * df_in["v"],
        vol_aa_v1 = alive_alive * df_in["v"],
        vol_ad_v1 = alive_dead * df_in["v"],
        vol_ac_v1 = alive_cut * df_in["v"],
        
    ).sum().drop(["ba_1", "ba_2", "v"], axis=0)

    # Merge stand metrics and size-based metrics
    sm = pd.concat([stem_metrics, size_metrics])

    # Calculate metrics
    metrics_of_change = pd.Series({
        # Mortality
        # - Stem-Based
        #   - Esquivel Equation: 1 - (survivors / initals) ** (1 / 5) | How many trees die per year with respect to current population?
        #   - To separate natural mortality, we need to add the number of trees that were cut to the survivors because they "survived nature"
        #     And vice versa for the number of trees that were cut.
        "mort_tot_stems_prc_yr_esq": np.nan if sm.n_a1 == 0 else (1 - (sm.n_aa / sm.n_a1) ** timespan) * 100,  # Total (natural and human)
        "mort_nat_stems_prc_yr_esq": np.nan if sm.n_a1 == 0 else (1 - ((sm.n_aa + sm.n_ac) / sm.n_a1) ** timespan) * 100,  # Natural
        "mort_cut_stems_prc_yr_esq": np.nan if sm.n_a1 == 0 else (1 - ((sm.n_aa + sm.n_ad) / sm.n_a1) ** timespan) * 100,  # Human Cutting
        
        #   - Simple: How many trees died per year with respect to initial population?
        #     Here the logic is reversed to the Esquivel equation, because taking the percentage of nd to initial trees
        "mort_tot_stems_prc_yr":     np.nan if sm.n_a1 == 0 else (sm.n_ad + sm.n_ac) / sm.n_a1 * timespan * 100,  # Total (natural and human)
        "mort_nat_stems_prc_yr":     np.nan if sm.n_a1 == 0 else sm.n_ad / sm.n_a1 * timespan * 100,  # Natural
        "mort_cut_stems_prc_yr":     np.nan if sm.n_a1 == 0 else sm.n_ac / sm.n_a1 * timespan * 100,  # Human Cutting
        
        # - Size-Based
        #   Question: Of mortality process types (total, natural, cutting), how much of the initial size alive of subset was lost to that process?
        #   Note that ba_v1_a is the same as summing up ba_aa_v1, ba_ad_v1, ba_ac_v1
        #   - Basal Area
        "mort_tot_ba_yr":                                  (sm.ba_ad_v1 + sm.ba_ac_v1) * timespan,
        "mort_tot_ba_prc_yr": np.nan if sm.ba_ax_v1 == 0 else (sm.ba_ad_v1 + sm.ba_ac_v1) * timespan / sm.ba_ax_v1 * 100,
        
        "mort_nat_ba_yr":                                  sm.ba_ad_v1 * timespan,
        "mort_nat_ba_prc_yr": np.nan if sm.ba_ax_v1 == 0 else sm.ba_ad_v1 * timespan / sm.ba_ax_v1 * 100,
        
        "mort_cut_ba_yr":                                  sm.ba_ac_v1 * timespan,
        "mort_cut_ba_prc_yr": np.nan if sm.ba_ax_v1 == 0 else sm.ba_ac_v1 * timespan / sm.ba_ax_v1 * 100,
        
        #  - Volume
        "mort_tot_vol_yr":                                   (sm.vol_ad_v1 + sm.vol_ac_v1) * timespan,
        "mort_tot_vol_prc_yr": np.nan if sm.vol_ax_v1 == 0 else (sm.vol_ad_v1 + sm.vol_ac_v1) * timespan / sm.vol_ax_v1 * 100,
        
        "mort_nat_vol_yr":                                   sm.vol_ad_v1 * timespan,
        "mort_nat_vol_prc_yr": np.nan if sm.vol_ax_v1 == 0 else sm.vol_ad_v1 * timespan / sm.vol_ax_v1 * 100,
        
        "mort_cut_vol_yr":                                   sm.vol_ac_v1 * timespan,
        "mort_cut_vol_prc_yr": np.nan if sm.vol_ax_v1 == 0 else sm.vol_ac_v1 * timespan / sm.vol_ax_v1 * 100,
        
        # Growth (de-coupled from mortality!)
        # - Stem-Based | Hoshino Equation: ln(finals / survivors) / 5
        "grwt_stems_prc_yr": np.nan if sm.n_aa == 0 else np.log(sm.n_a2 / sm.n_aa) * timespan * 100,

        # - Size-Based (only doable for basal area, because volume is not available for the second visit)
        #   - Total Growth
        "grwt_tot_ba_yr":                                         (sm.ba_aa_v2 - sm.ba_aa_v1 + sm.ba_na_v2) * timespan, # Difference of survivors between v1 and v2 plus all of v2 of new trees
        "grwt_tot_ba_prc_yr":     np.nan if sm.ba_aa_v1 == 0 else (sm.ba_aa_v2 - sm.ba_aa_v1 + sm.ba_na_v2) * timespan / sm.ba_aa_v1 * 100,
        "grwt_tot_ba_prc_yr_hos": np.nan if sm.ba_aa_v1 == 0 else np.log((sm.ba_aa_v2 + sm.ba_na_v2) / sm.ba_aa_v1) * timespan * 100,
        
        "grwt_sur_ba_yr":                                         (sm.ba_aa_v2 - sm.ba_aa_v1) * timespan,
        "grwt_sur_ba_prc_yr":     np.nan if sm.ba_aa_v1 == 0 else (sm.ba_aa_v2 - sm.ba_aa_v1) * timespan / sm.ba_aa_v1 * 100,
        "grwt_sur_ba_prc_yr_hos": np.nan if sm.ba_aa_v1 == 0 else np.log(sm.ba_aa_v2 / sm.ba_aa_v1) * timespan * 100,
        
        "grwt_rec_ba_yr":                                         sm.ba_na_v2 * timespan,
        "grwt_rec_ba_prc_yr":     np.nan if sm.ba_aa_v1 == 0 else sm.ba_na_v2 * timespan / sm.ba_aa_v1 * 100,
        
        # Change of alive biomass (How has the total alive biomass changed over time?)
        "change_tot_ba_yr":                                       (sm.ba_ax_v2 - sm.ba_ax_v1) * timespan,
        "change_tot_ba_prc_yr":   np.nan if sm.ba_ax_v1 == 0 else (sm.ba_ax_v2 - sm.ba_ax_v1) * timespan / sm.ba_ax_v1 * 100,
    })

    # Combine all metrics
    df_out = pd.DataFrame({grouping_variable: my_group, 
                           "n_plots": df_in["idp"].nunique()},index=[0])
    df_out = df_out.assign(**sm, **metrics_of_change)
    df_out.insert(0, "n_plots", df_out.pop("n_plots"))
    df_out.insert(0, grouping_variable, df_out.pop(grouping_variable))

    return df_out


# Example usage (Note: df_in should be a properly formatted pandas DataFrame)
# df_out = calculate_growth_mortality_optimized(df_in)
# Turn formatting on again
# fmt: on
# -----------------------------------------------------------------------------
def extract_nspecies(df_in, species_vars=["espar_red", "species_lat", "genus_lat"]):
    df_out = pd.DataFrame({"group_id": df_in["group_id"].unique()})
    for var in species_vars:
        df_out[f"n_species_{var}"] = df_in[var].nunique()

    return df_out


# extract_nspecies(df_list[125], ["espar_red", "species_lat", "genus_lat"])
# -----------------------------------------------------------------------------
def extract_statistics_general(df_in, vars_in, fcts_to_apply):
    """
    This function extracts the desired statistics from the input dataframe.

    Returns:
        _type_: _description_
    """

    df_out = pd.DataFrame({"group_id": df_in["group_id"].unique()})

    fct_dict = {
        "mean": np.nanmean,
        "std": np.nanstd,
        "median": np.nanmedian,
        "max": np.nanmax,
        "min": np.nanmin,
        "range": range_func,
        "sum": np.nansum,
        "iqr": iqr_func,
    }

    # Loop through functions
    for my_fct in fcts_to_apply:
        # print(my_fct)
        # Loop through variables
        for var_in in vars_in:
            # print(var_in)
            var_name = var_in + "_" + my_fct

            # If array is empty or if all values are NA, set to NA
            if (len(df_in[var_in]) == 0) or (df_in[var_in].isna().all()):
                my_value = np.nan
            else:
                my_value = fct_dict[my_fct](df_in[var_in])

            # print(my_value)
            df_out[var_name] = my_value

    return df_out


# xxx = extract_statistics_general(
#     df_in=df_list[125],
#     vars_in=["htot", "age13", "ir5", "v", "ba_1", "ba_2"],
#     fcts_to_apply=["mean", "std", "median", "max", "min", "sum", "range", "iqr"],
# ).T


# for i in range(xxx.shape[0]):
#     print(f"{xxx.index[i]:<30} = {xxx.iloc[i].values}")
# -----------------------------------------------------------------------------
def extract_statistics_per_visit_and_tree_state(df_in, verbose=True):
    """
    This function extracts statistics per tree state (alive, dead, rec) for each visit. These metrics provide information on the number and basal area of these tree states in relation to the total area or number of trees in a plot.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    if verbose:
        print("\nExtracting statistics per visit and tree state...")

    df_out = pd.DataFrame({"group_id": df_in["group_id"].unique()})

    for i_visit in [1, 2]:
        var_ba = f"ba_{i_visit}"
        var_vis = f"v{i_visit}"
        var_state = f"tree_state_{i_visit}"

        # Remove recruits because they are technically not there yet
        # This only applies to the first visit group
        i_df = df_in.query(f"{var_state} != 'new'")

        groups = i_df.groupby(var_state, as_index=False, observed=False)

        df_out[f"static_full_stand_ba_at_{var_vis}"] = i_df[var_ba].sum()
        df_out[f"static_full_stand_nt_at_{var_vis}"] = i_df.shape[0]

        # * When adding new variables, make sure to add them to the if and else statement sections below!
        for gname, group in groups:
            # Skip group for recruits
            if gname == "new":
                continue

            if group.shape[0] > 0:
                # Total Area and Ntrees per group
                df_out[f"static_{gname}_ba_at_{var_vis}"] = group[var_ba].sum()
                df_out[f"static_{gname}_nt_at_{var_vis}"] = group.shape[0]

                # Percentage of total area and Number of trees per group
                if df_out[f"static_full_stand_ba_at_{var_vis}"].iloc[0] == 0:
                    df_out[f"static_{gname}_ba_at_{var_vis}_perc"] = 0
                else:
                    df_out[f"static_{gname}_ba_at_{var_vis}_perc"] = (
                        group[var_ba].sum()
                        / df_out[f"static_full_stand_ba_at_{var_vis}"].iloc[0]
                    )

                if df_out[f"static_full_stand_nt_at_{var_vis}"].iloc[0] == 0:
                    df_out[f"static_{gname}_nt_at_{var_vis}_perc"] = 0
                else:
                    df_out[f"static_{gname}_nt_at_{var_vis}_perc"] = (
                        group.shape[0]
                        / df_out[f"static_full_stand_nt_at_{var_vis}"].iloc[0]
                    )

            else:
                # Return zeros
                df_out[f"static_{gname}_ba_at_{var_vis}"] = 0
                df_out[f"static_{gname}_nt_at_{var_vis}"] = 0
                df_out[f"static_{gname}_ba_at_{var_vis}_perc"] = 0
                df_out[f"static_{gname}_nt_at_{var_vis}_perc"] = 0

    # # Quality Control
    # espected_cols = len(v1_groups) * 8 + 1  # group_id + 8 variables * number of groups
    # actual_cols = df_out.shape[1]
    # if actual_cols != espected_cols:
    #     raise ValueError(
    #         f"Expected output shape is not correct! Espected: {espected_cols} | Actual: {actual_cols}"
    #     )

    return df_out


# xxx = extract_statistics_per_visit_and_tree_state(df_list[125]).T
# for i in range(xxx.shape[0]):
#     print(f"{xxx.index[i]:<30} = {xxx.iloc[i].values}")
# -----------------------------------------------------------------------------§
def extract_stand_characterstics_for_all_and_topn_species(
    df_in,
    vars_in=["htot", "age13", "ir5", "v", "ba_1", "ba_2", "ba_change_perc_yr"],
    fcts_to_apply=["mean", "std", "median", "max", "min", "sum", "range", "iqr"],
    species_var="espar_red",
    n_groups=3,
    verbose=True,
):

    if verbose:
        print("\n Extracting stand characteristics using all trees...")
        print(
            "... note that the top-n-species specific extraction has been removed again."
        )

    # Ignore certain warnings caused by empty dfs or NA slices
    warnings.filterwarnings("ignore", message="All-NaN")
    warnings.filterwarnings("ignore", message="Degrees")
    warnings.filterwarnings("ignore", message="Mean")

    # Get output df
    df_out = pd.DataFrame({"group_id": df_in["group_id"].unique()})

    # Extraction for all species ---------------------------------------------
    df_add = extract_nspecies(df_in).drop("group_id", axis=1)
    df_out = pd.concat([df_out, df_add], axis=1)
    df_add = extract_statistics_general(
        df_in=df_in,
        vars_in=vars_in,
        fcts_to_apply=fcts_to_apply,
    )

    # Attach prefix to column names
    df_add = df_add.add_prefix("allspecies_")
    df_add = df_add.drop(columns=["allspecies_group_id"])

    df_out = pd.concat([df_out, df_add], axis=1)

    # Extraction per top n species -------------------------------------------
    # Get sum of ba_1 per group

    # top3species_groups = (
    #     (
    #         df_in.groupby(species_var)
    #         .agg({"ba_1": "sum"})
    #         .reset_index()
    #         .sort_values(by="ba_1", ascending=False)
    #     )
    #     .head(n_groups)[species_var]
    #     .tolist()
    # )

    # # Prepare empty df
    # for i in range(n_groups):
    #     df_i = df_in.query(f"{species_var} == @top3species_groups[{i}]")

    #     # If df is empty, set species to None
    #     if df_i.shape[0] == 0:
    #         top3species_groups[i] = "None"

    #     # display(df_i)
    #     df_i = extract_statistics_general(
    #         df_in=df_i,
    #         vars_in=vars_in,
    #         fcts_to_apply=fcts_to_apply,
    #     )

    #     # Attach prefix to column names
    #     df_i = df_i.add_prefix(f"top{i+1}_")
    #     df_i[f"top{i+1}_{species_var}"] = str(top3species_groups[i])
    #     df_i = df_i.rename(columns={f"top{i+1}_group_id": "group_id"})
    #     df_i = df_i.drop(columns=["group_id"])

    #     df_out = pd.concat([df_out, df_i], axis=1)

    # df_out = df_out.merge(df_i, on="group_id", how="left")
    # display(df_out)

    # Reset warnings
    warnings.filterwarnings("default")

    return df_out


def share_alive_trees(df):

    if df.idp.nunique() > 1:
        raise ValueError("More than idp detected - not allowed. Do per site!")

    n_alive = df.query("tree_state_1 == 'alive'").shape[0]
    n_dead = df.query("tree_state_1 == 'dead'").shape[0]
    n_tot = n_alive + n_dead

    if n_tot == 0:
        share_alive = np.nan
    else:
        share_alive = n_alive / n_tot

    df_out = pd.DataFrame(
        {
            "idp": df.idp.unique(),
            "share_alive": share_alive,
        },
        index=[0],
    )

    return df_out


def share_alive_trees_loop(df):
    idps = df.idp.unique()
    df_out = []
    for i in idps:
        df_i = df.query("idp == @i")
        df_i = share_alive_trees(df_i)
        df_out.append(df_i)

    df_out = pd.concat(df_out)
    return df_out


def share_alive_trees_mp(df):
    df_list = split_df_into_list_of_group_or_ns(df, 10, "idp")
    df_list = run_mp(
        share_alive_trees_loop,
        df_list,
        num_cores=10,
        progress_bar=True,
    )
    df_list = pd.concat(df_list)
    return df_list


def share_larger75dbh_trees(df):

    if df.idp.nunique() > 1:
        raise ValueError("More than idp detected - not allowed. Do per site!")

    n_small = df.query("tree_state_1 == 'alive' and dbh_1 < 0.075").shape[0]
    n_large = df.query("tree_state_1 == 'alive' and dbh_1 >= 0.075").shape[0]
    n_tot = n_small + n_large

    if n_tot == 0:
        share_larger75dbh = np.nan
    else:
        share_larger75dbh = n_large / n_tot

    df_out = pd.DataFrame(
        {
            "idp": df.idp.unique(),
            "share_larger75dbh": share_larger75dbh,
        },
        index=[0],
    )

    return df_out


def share_larger75dbh_trees_loop(df):
    idps = df.idp.unique()
    df_out = []
    for i in idps:
        df_i = df.query("idp == @i")
        df_i = share_larger75dbh_trees(df_i)
        df_out.append(df_i)

    df_out = pd.concat(df_out)
    return df_out


def share_larger75dbh_trees_mp(df):
    df_list = split_df_into_list_of_group_or_ns(df, 10, "idp")
    df_list = run_mp(
        share_larger75dbh_trees_loop,
        df_list,
        num_cores=10,
        progress_bar=True,
    )
    df_list = pd.concat(df_list)
    return df_list


# xxx = extract_stand_characterstics_for_all_and_topn_species(
#     df_in=df_list[125],
#     vars_in=["htot", "age13", "ir5", "v", "ba_1"],
#     fcts_to_apply=["mean", "std", "median", "max", "min", "sum", "range", "iqr"],
#     species_var="genus_lat",
# ).T.sort_index()
# for i in range(xxx.shape[0]):
#     print(f"{xxx.index[i]:<25} = {xxx.iloc[i].values}")
# -----------------------------------------------------------------------------
def extract_tree_damage_info(df_in, verbose=True):
    """
    For all trees that are alive, assess the damage that they have at first and at second visit.
    """

    if verbose:
        print("\nExtracting tree damage information...")

    # Define output df
    df_out = pd.DataFrame({"group_id": df_in["group_id"].unique()})

    for tree_condition in ["alive", "died", "survived"]:
        # alive = looking at all trees that are alive at first visit or at second visit
        # died = looking at all trees that died between first and second visit
        # survived = looking at all trees that survived between first and second visit

        # Get subset of alive trees
        if tree_condition == "alive":
            # Here we need to separate because we want to include recruits in
            # the second visit site assessment.
            df_1 = df_in.query("tree_state_1 == 'alive'")
            df_2 = df_in.query("tree_state_2 == 'alive'")

        if tree_condition == "died":
            # Here we do not need to separate because we are focusing already on
            # trees that died between first and second visit.
            df_1 = df_in.query("tree_state_change == 'alive_dead'")
            df_2 = df_in.query("tree_state_change == 'alive_dead'")

        if tree_condition == "survived":
            # Here we do not need to separate because we are focusing already on
            # trees that survived between first and second visit.
            df_1 = df_in.query("tree_state_change == 'alive_alive'")
            df_2 = df_in.query("tree_state_change == 'alive_alive'")

        nt_1 = df_1.shape[0]
        nt_2 = df_2.shape[0]

        # Print for debug -------------------------------------------------------------
        # ddebug = df_1[
        #         [
        #             "tree_state_change",
        #             "acci",
        #             "deggib",
        #             "sfpied",
        #             "sfcoeur",
        #             "sfgui_1",
        #             "sfgeliv_1",
        #             "sfdorge_1",
        #             "mortb_1",
        #             "mortb_2",
        #         ]
        #     ]

        # display(f"{ddebug.dtypes.to_dict()}")
        # display(ddebug)

        # print(df_1.shape)
        # print(nt_1)
        # print(nt_2)

        # First visit -------------------------------------------------------------
        # Structural damange
        n_struct = df_1.query("acci in ['1', '2', '3']").shape[0]
        df_out[f"trees_{tree_condition}_with_structdmg_at_v1_in_perc"] = (
            np.nan if nt_1 == 0 else n_struct / nt_1
        )

        # Fire
        n_fire = df_1.query("acci == '4'").shape[0]
        df_out[f"trees_{tree_condition}_with_burn_at_v1_in_perc"] = (
            np.nan if nt_1 == 0 else n_fire / nt_1
        )

        # Game damage
        n_game = df_1.query("deggib not in ['0', 'Missing']").shape[0]
        df_out[f"trees_{tree_condition}_with_game_at_v1_in_perc"] = (
            np.nan if nt_1 == 0 else n_game / nt_1
        )

        # Foot damage
        n_foot = df_1.query("sfpied not in ['0', 'Missing']").shape[0]
        df_out[f"trees_{tree_condition}_with_foot_at_v1_in_perc"] = (
            np.nan if nt_1 == 0 else n_foot / nt_1
        )

        # Rot core
        n_rot = df_1.query("sfcoeur not in ['0', 'Missing']").shape[0]
        df_out[f"trees_{tree_condition}_with_rot_at_v1_in_perc"] = (
            np.nan if nt_1 == 0 else n_rot / nt_1
        )

        # Mistletoe (0 = none, 1 = 1 or two, 2 = 3 or 5, 3 = 6 or more)
        n_mist = df_1.query("sfgui_1 > 0").shape[0]
        df_out[f"trees_{tree_condition}_with_mistl_at_v1_in_perc"] = (
            np.nan if nt_1 == 0 else n_mist / nt_1
        )

        # Frost damage
        n_frost = df_1.query("sfgeliv_1 > 0").shape[0]
        df_out[f"trees_{tree_condition}_with_frost_at_v1_in_perc"] = (
            np.nan if nt_1 == 0 else n_frost / nt_1
        )

        # Fir rust (0 = none, 1 = 1 frost damage, 2 = 2 two or more)
        n_firrust = df_1.query("sfdorge_1 not in ['0', 'Missing']").shape[0]
        df_out[f"trees_{tree_condition}_with_firrust_at_v1_in_perc"] = (
            np.nan if nt_1 == 0 else n_firrust / nt_1
        )

        # Branch mortality
        n_mortb = df_1.query("mortb_1 > 0").shape[0]
        df_out[f"trees_{tree_condition}_with_branchdmg_at_v1_in_perc"] = (
            np.nan if nt_1 == 0 else n_mortb / nt_1
        )

        # Any damage
        n_any = df_1.query(
            "acci in ['1', '2', '3'] | acci == '4' | deggib not in ['0', 'Missing'] | sfpied not in ['0', 'Missing'] | sfcoeur not in ['0', 'Missing'] | sfgui_1 > 0 | sfgeliv_1 > 0 | sfdorge_1 not in ['0', 'Missing'] | mortb_1 > 0"
        ).shape[0]

        df_out[f"trees_{tree_condition}_with_anydmg_at_v1_in_perc"] = (
            np.nan if nt_1 == 0 else n_any / nt_1
        )

        # Second visit -----------------------------------------------------------
        # Frost damage
        n_frost = df_2.query("sfgeliv_2 > 0").shape[0]
        df_out[f"trees_{tree_condition}_with_frost_at_v2_in_perc"] = (
            np.nan if nt_2 == 0 else n_frost / nt_2
        )

        # Mistletoe (0 = none, 1 = 1 or two, 2 = 3 or 5, 3 = 6 or more)
        n_mist = df_2.query("sfgui_2 > 0").shape[0]
        df_out[f"trees_{tree_condition}_with_mistl_at_v2_in_perc"] = (
            np.nan if nt_2 == 0 else n_mist / nt_2
        )

        # Fir rust (0 = none, 1 = 1 frost damage, 2 = 2 two or more)
        n_firrust = df_2.query("sfdorge_2 not in ['0', 'Missing']").shape[0]
        df_out[f"trees_{tree_condition}_with_firrust_at_v2_in_perc"] = (
            np.nan if nt_2 == 0 else n_firrust / nt_2
        )

        # Branch mortality
        n_mortb = df_2.query("mortb_2 > 0").shape[0]
        df_out[f"trees_{tree_condition}_with_branchdmg_at_v2_in_perc"] = (
            np.nan if nt_2 == 0 else n_mortb / nt_2
        )

        # Any damage
        n_any = df_2.query(
            "sfgeliv_2 > 0 | sfgui_2 > 0 | sfdorge_2 not in ['0', 'Missing'] | mortb_2 > 0"
        ).shape[0]
        df_out[f"trees_{tree_condition}_with_anydmg_at_v2_in_perc"] = (
            np.nan if nt_2 == 0 else n_any / nt_2
        )

    # Check if any value except group_id is above 1.0
    if (sum(df_out[col] > 1.0 for col in df_out.columns if col != "group_id") != 0)[0]:
        raise ValueError(
            "Invalid value found in df_out. At least one value is above 1.0!"
        )

    return df_out


# extract_tree_damage_info(df_list[666])
# xxx = extract_tree_damage_info(df_in).T
# for i in range(xxx.shape[0]):
#     print(f"{xxx.index[i]:<40} = {xxx.iloc[i].values}")
# -----------------------------------------------------------------------------
def aggregate_tree_info_to_site_level(
    df_in, vars_in, fcts_to_apply, species_var=None, verbose=True
):

    # Extract stand characteristics
    df_1 = extract_stand_characterstics_for_all_and_topn_species(
        df_in=df_in,
        vars_in=vars_in,
        fcts_to_apply=fcts_to_apply,
        species_var=species_var,
        n_groups=species_var,
        verbose=verbose,
    )

    # Extract stand statistics per visit
    df_2 = extract_statistics_per_visit_and_tree_state(df_in, verbose=verbose)

    # Extract tree damage info
    df_3 = extract_tree_damage_info(df_in, verbose=verbose)

    df_out = pd.concat(
        [
            df_1,
            df_2.drop("group_id", axis=1),
            df_3.drop("group_id", axis=1),
        ],
        axis=1,
    )

    return df_out


# -----------------------------------------------------------------------------
def attach_regional_information(df_in, verbose=False):
    # Load shapefiles
    # * All geojson files are in CRS 2154, so using lat_fr and lon_fr for referencing!
    shp_reg = gpd.read_file(here("data/raw/maps/france_geojson/reg.geojson"))
    shp_dep = gpd.read_file(here("data/raw/maps/france_geojson/dep.geojson"))
    shp_gre = gpd.read_file(here("data/raw/maps/france_geojson/gre.geojson"))
    shp_ser = gpd.read_file(here("data/raw/maps/france_geojson/ser.geojson"))
    shp_hex = load_hexmap()

    # Add variables
    df_in["reg"] = "Missing"
    df_in["dep"] = "Missing"
    df_in["gre"] = "Missing"
    df_in["ser"] = "Missing"
    df_in["hex"] = "Missing"

    # Import Point function
    from shapely.geometry import Point

    # TODO: Ignore chained assignment warning. Not sure if this is the best way to do it...
    pd.options.mode.chained_assignment = None

    # Loop over every site and get the corresponding regional code
    for i in tqdm(range(df_in.shape[0]), disable=not verbose):
        # Get site coordinates
        lat = df_in["lat_fr"].iloc[i]
        lon = df_in["lon_fr"].iloc[i]

        # Add information from closest polygon to df
        df_in["reg"].iloc[i] = shp_reg.iloc[shp_reg.distance(Point(lon, lat)).idxmin()][
            "reg"
        ]
        df_in["dep"].iloc[i] = shp_dep.iloc[shp_dep.distance(Point(lon, lat)).idxmin()][
            "dep"
        ]
        df_in["gre"].iloc[i] = shp_gre.iloc[shp_gre.distance(Point(lon, lat)).idxmin()][
            "gre"
        ]
        df_in["ser"].iloc[i] = shp_ser.iloc[shp_ser.distance(Point(lon, lat)).idxmin()][
            "ser"
        ]
        df_in["hex"].iloc[i] = shp_hex.iloc[shp_hex.distance(Point(lon, lat)).idxmin()][
            "hex"
        ]

    return df_in


# -----------------------------------------------------------------------------
def split_df_into_list_of_group_or_ns(df_in, group_variable=10, group_by_var=None):
    """
    args:
    - df_in: pandas.DataFrame, the data to split
    - group_variable: int or str, the number of groups to split the data into
    - group_by_var: str, the column to group by, if group variable is an integer
    """

    if type(group_variable) == int:

        if group_by_var is None:
            print(f" - Splitting df into {group_variable} random groups")
            df_list = np.array_split(df_in, group_variable)
        else:
            print(
                f" - Splitting df into {group_variable} groups, grouped by {group_by_var}"
            )

            # Group dataframe by idp and split into groups
            # group_by_var = "idp"
            groups = df_in[f"{group_by_var}"].unique().tolist()
            groups = np.array_split(groups, group_variable)

            # Initiate list
            df_list = []

            # Loop over each group and put subitems of that group into list
            for group in groups:
                df_i = df_in[df_in[f"{group_by_var}"].isin(group)]
                df_list.append(df_i)

    else:
        # Check if group_variable is in df
        if group_variable not in df_in.columns:
            raise ValueError(f"Group variable {group_variable} not in df!")

        # Group by group_variable and turn into list
        df_list = [df for _, df in df_in.groupby(group_variable, observed=False)]

    return df_list


# -----------------------------------------------------------------------------
def calculate_percentage_per_group(df, col_a, col_b, col_c):
    """
    Calculates the percentage of a specified column (col_c) for each group defined by col_a-col_b
    relative to the total in col_a, and slices the DataFrame to return only the group with
    the largest percentage.

    Parameters:
    - df: pandas.DataFrame containing the data.
    - col_a: str, the column name to group by first.
    - col_b: str, the column name to group by second.
    - col_c: str, the column name of the values to calculate percentages for.

    Returns:
    - A pandas DataFrame with the row corresponding to the col_a-col_b group with the largest
      percentage of col_c.
    """

    print(f"- Calculate percentage of groups {col_a}-{col_b} based on {col_c}...")

    # Sum of col_c for each col_a-col_b group
    group_sum = (
        df.groupby([col_a, col_b], observed=False)[col_c]
        .sum()
        .reset_index(name="group_c_sum")
    )

    # Total of col_c for each col_a
    total_c_by_a = (
        df.groupby(col_a, observed=False)[col_c].sum().reset_index(name="total_c_by_a")
    )

    # Merge the sums back to calculate the percentage
    result = pd.merge(group_sum, total_c_by_a, on=col_a)
    result["percentage"] = (result["group_c_sum"] / result["total_c_by_a"]) * 100

    # Finding the group with the largest percentage
    # result = (
    #     result.sort_values("percentage", ascending=False).groupby(col_a).head(1)
    # )

    result = (
        result[[col_a, col_b, "percentage"]]
        .rename(columns={"percentage": f"{col_c}_perc_of_{col_b}"})
        .reset_index(drop=True)
    )

    # return result
    return result


# ttt = calculate_percentage_per_group(subset, "idp", "genus_lat", "ba_1")
# ttt.query("idp == 500137")


# -----------------------------------------------------------------------------
def get_dominant_species(df_in, species_var):

    df_dom = df_in.groupby(["idp", species_var], observed=False).agg(
        {"ba_1": "sum", "lib": "sum", "htot_final": "sum"}
    )

    # ! Simply basal area based dominance metric
    df_dom["balib"] = df_dom["ba_1"]
    print(f"- Calculate dominant group for {species_var} based on basal area only...")

    # ! Dominance metric based on basal area, canopy size, and height
    # df_dom["balib"] = df_dom["ba_1"] * df_dom["lib"] + df_dom["htot_final"]
    # print(
    #     f"\nCalculate dominant group for {species_var} based on basal area, canopy size, and height..."
    # )

    df_dom = df_dom.loc[df_dom.groupby("idp", observed=False)["balib"].idxmax()]
    df_dom = df_dom.reset_index()[["idp", species_var]]

    df_perc = calculate_percentage_per_group(df_in, "idp", species_var, "ba_1")
    df_all = pd.merge(df_dom, df_perc, on=["idp", species_var], how="left")
    df_all = df_all.rename(columns={species_var: f"dom_{species_var}"})

    return df_all


def calculate_shannon_diversity(df, site_var, tree_var, species_var):
    """
    Calculate the Shannon Diversity Index for each site in the DataFrame.

    Parameters:
    - df: pandas DataFrame containing the dataset.
    - site_var: string, the name of the column in df that groups trees by site.
    - tree_var: string, the name of the column in df that holds the tree ID.
    - species_var: string, the name of the column in df that holds tree species information.

    Returns:
    - A pandas DataFrame with each site and its corresponding Shannon Diversity Index.
    """
    print(f"Calculate Shannon Diversity Index for each site based on {species_var}...")

    # Group by site and species to count the number of trees of each species at each site
    species_counts = (
        df.groupby([site_var, species_var])[tree_var].count().reset_index(name="count")
    )

    # Total number of trees at each site
    total_trees_per_site = (
        species_counts.groupby(site_var)["count"].sum().reset_index(name="total")
    )

    # Merge counts with totals to calculate proportions
    merged = pd.merge(species_counts, total_trees_per_site, on=site_var)
    merged["proportion"] = merged["count"] / merged["total"]

    # Calculate Shannon Diversity Index for each site
    merged["shannon_contrib"] = -merged["proportion"] * np.log(merged["proportion"])
    shannon_index = (
        merged.groupby(site_var)["shannon_contrib"]
        .sum()
        .reset_index(name=f"biodiv_shan_{species_var}")
    )

    return shannon_index


def calculate_simpson_diversity(df, site_var, species_var):
    """
    Calculate the Simpson Diversity Index for each site in the DataFrame.

    Parameters:
    - df: pandas DataFrame containing the dataset.
    - site_var: string, the name of the column in df that groups data by site.
    - species_var: string, the name of the column in df that holds species information.

    Returns:
    - A pandas DataFrame with each site and its corresponding Simpson Diversity Index.
    """
    print(f"Calculate Simpson Diversity Index for each site based on {species_var}...")

    # Initialize an empty list to store results
    results_list = []

    for site, group in tqdm(df.groupby(site_var)):
        # Count the total number of occurrences of each species in the site
        species_counts = group[species_var].value_counts()
        total_count = species_counts.sum()

        # Calculate the proportion of each species
        proportions = species_counts / total_count

        # Calculate the Simpson Index
        simpson_index = sum(proportions**2)

        # Optionally calculate 1 - D to express diversity where higher values indicate higher diversity
        diversity_score = 1 - simpson_index

        # Append the results
        results_list.append(
            pd.DataFrame(
                {
                    site_var: [site],
                    f"biodiv_simpson_index_{species_var}": [simpson_index],
                    f"biodiv_simpson_score_{species_var}": [diversity_score],
                }
            )
        )

    # Concatenate all DataFrame objects in the list into a single DataFrame
    diversity_results = pd.concat(results_list, ignore_index=True)

    return diversity_results


def calculate_competition_metrics(df_in, tree_size_var="htot_final", verbose=True):
    """
    Calculate competition metrics for each tree in the DataFrame.

    Parameters:
    - df_in: pandas DataFrame containing the data.
    - tree_size_var: str, the variable on which light competition should be based on (htot_final or ba_1)

    Returns:
    - A pandas DataFrame with competition metrics for each tree.
    """

    # Initiate empty list
    df_list = []

    for tree in tqdm(df_in.tree_id.unique()):

        # Tree metrics
        site = df_in[df_in.tree_id == tree].idp.values[0]
        size = df_in[df_in.tree_id == tree][f"{tree_size_var}"].values[0]
        species = df_in[df_in.tree_id == tree].species_lat.values[0]

        # Competition metrics
        competition_total = df_in.query("idp == @site")["ba_1"].sum()

        # Light competition
        competition_larger = df_in.query(f"idp == @site and {tree_size_var} > @size")[
            "ba_1"
        ].sum()
        competition_same_species = df_in.query(
            "idp == @site and species_lat == @species"
        )["ba_1"].sum()

        competition_other_species = df_in.query(
            "idp == @site and species_lat != @species"
        )["ba_1"].sum()

        df_list.append(
            pd.DataFrame(
                {
                    "tree_id": [tree],
                    "competition_total": [competition_total],
                    "competition_larger": [competition_larger],
                    "competition_larger_rel": [competition_larger / competition_total],
                    "competition_same_species": [competition_same_species],
                    "competition_same_species_rel": [
                        competition_same_species / competition_total
                    ],
                    "competition_other_species": [competition_other_species],
                    "competition_other_species_rel": [
                        competition_other_species / competition_total
                    ],
                }
            )
        )

    # Concatenate all DataFrames in the list
    df_competition = pd.concat(df_list)

    return df_competition


def calculate_competition_metrics_mp(df_in, tree_size_var="htot_final", verbose=True):

    df_list = split_df_into_list_of_group_or_ns(df_in, 10, "idp")
    df_mp = run_mp(
        calculate_competition_metrics,
        df_list,
        combine_func=pd.concat,
        num_cores=10,
        progress_bar=True,
    )
    return df_mp


def calculate_gini_coefficient(df, site_var, metric_var):
    """
    Calculate the Gini coefficient for a specified metric across each site using pd.concat.
    0 = Extreme equality. All trees are of similar size.
    1 = Extreme inequality. One tree is much larger than all others.

    Parameters:
    - df: pandas DataFrame containing the dataset.
    - site_var: string, the name of the column in the DataFrame that identifies the site.
    - metric_var: string, the name of the column in the DataFrame that contains the metric used for calculating the Gini coefficient.

    Returns:
    - A pandas DataFrame with each site and its corresponding Gini coefficient.
    """

    print(f"\nCalculate Gini-Coefficient per site for {metric_var}")

    gini_list = []  # Initialize a list to store intermediate DataFrame objects

    for site, group in tqdm(df.groupby(site_var)):
        sorted_metric = np.sort(group[metric_var].values)

        # Skip sites with non-positive metric values
        if any(sorted_metric <= 0):
            continue

        cumsum_metric = np.cumsum(sorted_metric)
        total_sum = cumsum_metric[-1]

        n = len(sorted_metric)
        relative_mean_difference = (2 * np.sum(np.arange(1, n + 1) * sorted_metric)) / (
            n * total_sum
        ) - (n + 1) / n
        gini_coefficient = 1 - relative_mean_difference

        # Store each site's Gini coefficient in a DataFrame and add it to the list
        gini_list.append(
            pd.DataFrame({site_var: [site], f"gini_{metric_var}": [gini_coefficient]})
        )

    # Concatenate all DataFrame objects in the list into a single DataFrame
    gini_results = pd.concat(gini_list, ignore_index=True)

    return gini_results


# -----------------------------------------------------------------------------
def ___DIGITALIS___():
    pass


def get_anomaly_metrics(df_in, dataset, years_before_second_visit):
    # Check if input data has more than one site, error if so
    if df_in.idp.nunique() > 1:
        raise ValueError("Input data has more than one site!")

    # Get site specific variables
    idp = df_in.idp.unique()[0]
    first_year = df_in.first_year.unique()[0]
    second_year = df_in.first_year.unique()[0] + 5
    start_year = second_year - years_before_second_visit

    # Prepare output df
    df_list = []

    # Loop over seasons
    for s in ["win", "spr", "sum", "aut", "ann"]:

        if s == "win":
            df = df_in.query("month in ['12', '01', '02']").copy()
        elif s == "spr":
            df = df_in.query("month in ['03', '04', '05']").copy()
        elif s == "sum":
            df = df_in.query("month in ['06', '07', '08']").copy()
        elif s == "aut":
            df = df_in.query("month in ['09', '10', '11']").copy()
        elif s == "ann":
            df = df_in.copy()

        # Get fresh df
        df_out = pd.DataFrame(
            index=[0],
        )

        # Normalize data
        df["value"] = (df["value"] - df["value"].mean()) / df["value"].std()

        # Get linear regression slope
        X = sm.add_constant(df["month_count"])  # Add a constant term for the intercept
        y = df["value"]
        model = sm.OLS(y, X)
        results = model.fit()
        # print(results.summary())

        df_out["slope"] = results.params["month_count"]

        # Get mean anomaly, min, max n years before second measurement
        df_red = df.query("@start_year <= year <= @second_year")
        df_out[f"mean_anomaly"] = df_red["value"].mean()
        df_out[f"min_anomaly"] = df_red["value"].min()
        df_out[f"max_anomaly"] = df_red["value"].max()

        # Add prefix based on variable and based on season
        df_out = df_out.add_prefix(f"{dataset}_{s}_")
        df_list.append(df_out)

    # Concatenate all dfs
    df_out = pd.concat(df_list, axis=1)

    # Add site specific variables
    df_out["idp"] = idp
    df_out["first_year"] = first_year
    df_out["yrs_before_second_visit"] = years_before_second_visit
    df_out = move_vars_to_front(
        df_out, ["idp", "first_year", "yrs_before_second_visit"]
    )

    # Return df
    return df_out


def get_anomaly_metrics_loop(df_in, dataset, years_before_second_visit):
    df_list = []

    for idp in df_in.idp.unique():
        df = df_in.query("idp == @idp").sort_values(by=["date"]).copy()
        df_out = get_anomaly_metrics(df, dataset, years_before_second_visit)
        df_list.append(df_out)

    df_out = pd.concat(df_list)

    return df_out


def get_anomaly_metrics_loop_mp(dataset, years_before_second_visit, source):

    # Load data
    filepath = f"/Volumes/WD - ExFat/IFNA/digitalis_v3/processed/1km/{source}/{dataset}.feather"
    if not os.path.exists(filepath):
        raise ValueError(f"Filepath {filepath} does not exist!")
    
    print(f"Loading data from {filepath}...")
    df = pd.read_feather(filepath)

    # Drop non-numeric values in the 'month' column
    print("Cleaning data...")
    df["value"] = df["value"]/10 # Original data is 10x
    df = df[pd.to_numeric(df["month"], errors="coerce").notna()]
    df = (
        df.query("month not in ['au', 'et', 'hi', 'pr', '13']")
        .assign(
            year=lambda x: pd.to_numeric(x["year"]),
            month=lambda x: pd.to_numeric(x["month"]).apply(lambda m: str(m).zfill(2)),
            date=lambda x: x["year"].astype(str) + "-" + x["month"].astype(str) + "-01",
            month_count=lambda x: df["month"].astype(int)
            * (df.year.astype(int) - 1960)
            / 12,
        )
        .assign(date=lambda x: pd.to_datetime(x["date"], errors="coerce"))
        .dropna()
        .loc[lambda x: x["year"].between(1961, 2020)]
    )

    df.sort_values(by=["idp", "first_year", "date"])

    # Split into list
    df_list = split_df_into_list_of_group_or_ns(df, 10, "idp")

    # Run mp
    print("Extracting anomalies...")
    df_mp = run_mp(
        get_anomaly_metrics_loop,
        df_list,
        combine_func=pd.concat,
        dataset=dataset,
        years_before_second_visit=years_before_second_visit,
        num_cores=10,
    )
    return df_mp


def calc_spei_trend(df_in):
    # Check that only one site inputed
    assert len(df_in.idp.unique()) == 1

    # Sort by date and attach index as time variable
    df_in = df_in.copy().sort_values("date").reset_index(drop=True)
    df_in["time"] = df_in.index

    # Prepare output df
    df_out = pd.DataFrame()
    df_out["idp"] = df_in.idp.unique()
    new_cols = []

    # Get spei columns
    spei_cols = [col for col in df_in.columns if "spei" in col]

    # Calculate the trend for each SPEI variable and each month
    months = np.arange(1, 14, 1)

    for month in months:

        if month != 13:
            df_i = df_in.copy().query(f"month == {month}").dropna()
        else:
            df_i = df_in.copy().dropna()

        for col in spei_cols:
            x = df_i["time"]
            y = df_i[col]
            trend = np.polyfit(x, y, 1)[0]
            new_cols.append(pd.DataFrame({f"{col}-{month}_trend": [trend]}, index=[0]))

    df_out = pd.concat([df_out, pd.concat(new_cols, axis=1)], axis=1)

    return df_out


def calc_spei_trend_loop(df_in):
    # Loop over all sites
    df_out = []

    for idp in df_in.idp.unique():
        df_i = calc_spei_trend(df_in[df_in.idp == idp])
        df_out.append(df_i)

    return pd.concat(df_out, axis=0)


def calc_spei_trend_loop_mp(df_in):

    # Split into list
    df_in = split_df_into_list_of_group_or_ns(df_in, 10, "idp")

    # Run mp
    df_mp = run_mp(
        calc_spei_trend_loop,
        arg_list=df_in,
        combine_func=pd.concat,
        num_cores=10,
        progress_bar=True,
    )

    chime.success()

    return df_mp


def calc_spei_min_mean(df_in, years_before_second_visit):
    # Check that only one site inputed
    assert len(df_in.idp.unique()) == 1

    # Filter dates
    first_year = df_in.first_year.unique()[0] + 5 - years_before_second_visit
    last_year = df_in.first_year.unique()[0] + 5

    # Starting and ending first day of hydrological year
    first_day = first_year.astype("str") + "-08-31"
    last_day = last_year.astype("str") + "-08-01"

    first_day = pd.to_datetime(first_day)
    last_day = pd.to_datetime(last_day)

    df_in = df_in.copy().query(f"'{first_day}' <= date <= '{last_day}'")
    df_in["time"] = df_in.index

    # Prepare output df
    df_out = pd.DataFrame()
    df_out["idp"] = df_in.idp.unique()
    new_cols = []

    # Get spei columns
    spei_cols = [col for col in df_in.columns if "spei" in col]

    # Calculate for each month
    months = np.arange(1, 14, 1)

    # Calculate the mean and min for each SPEI variable
    for month in months:
        if month != 13:
            df_i = df_in.copy().query(f"month == {month}").dropna()
        else:
            df_i = df_in.copy().dropna()

        for col in spei_cols:
            new_cols.append(
                pd.DataFrame(
                    {
                        f"{col}-{month}_min": df_i[col].min(),
                        f"{col}-{month}_max": df_i[col].max(),
                        f"{col}-{month}_mean": df_i[col].mean(),
                    },
                    index=[0],
                )
            )

    df_out = pd.concat([df_out, pd.concat(new_cols, axis=1)], axis=1)

    return df_out


def calc_spei_min_mean_loop(df_in, years_before_second_visit):
    # Loop over all sites
    df_out = []

    for idp in df_in.idp.unique():
        df_i = calc_spei_min_mean(df_in[df_in.idp == idp], years_before_second_visit)
        df_out.append(df_i)

    return pd.concat(df_out, axis=0)


def calc_spei_min_mean_loop_mp(df_in, years_before_second_visit):

    # Split into list
    df_in = split_df_into_list_of_group_or_ns(df_in, 10, "idp")

    # Run mp
    df_mp = run_mp(
        calc_spei_min_mean_loop,
        arg_list=df_in,
        combine_func=pd.concat,
        num_cores=10,
        progress_bar=True,
        years_before_second_visit=years_before_second_visit,
    )

    chime.success()

    return df_mp


# -----------------------------------------------------------------------------
def ___MAKE_MAPS___():
    pass


def make_plots_per_file_parallel(
    file_group,
    run_only_subset=True,
    subset_fraction=None,
    method=None,
    verbose=False,
    skip_existing=True,
):
    for my_file in file_group:
        make_plots_per_file(
            my_file=my_file,
            method=method,
            run_only_subset=run_only_subset,
            subset_fraction=subset_fraction,
            verbose=verbose,
            skip_existing=skip_existing,
        )


def make_plots_per_file(
    my_file,
    method,
    run_only_subset,
    subset_fraction,
    verbose=False,
    skip_existing=True,
):
    # ! Preparation ---------------------------------------------------------------
    if method != "direct":
        raise ValueError("Only direct method is currently implemented")

    sp_france = get_shp_of_region("cty", make_per_year=False)  # Get shapefile of france

    # Extract species and region from filename
    my_species = my_file.split("/")[-1].split("_")[1].split("-")[0]
    my_region = my_file.split("/")[-1].split("_")[-1].split(".")[0].split("-")[1]

    # Set dir for plots
    # ? Debugging: Change output folder...
    if run_only_subset:
        my_dir = here(
            f"notebooks/00_process_nfi_data/maps_of_change/{method}/subset_{round(subset_fraction*100)}%_of_sites/species-{my_species}_region-{my_region}"
        )
    else:
        my_dir = here(
            f"notebooks/00_process_nfi_data/maps_of_change/{method}/species-{my_species}_region-{my_region}"
        )

    # Create dir if not exists
    os.makedirs(my_dir, exist_ok=True)

    # ! Read file  ---------------------------------------------------------------
    df_loop = pd.read_feather(my_file)

    # min_sites_per_pixel = 10
    # warnings.warn(f"Debugging: Removing pixels with less than {min_sites_per_pixel} sites!")
    # df_loop = df_loop.query("n_plots >= @min_sites_per_pixel").copy().reset_index()

    # todo: If method is not direct, then I have to rename variables so that _sd are dropped and _mean removed from variable name!

    # Plot distribution of n_plots per region_year
    fig, ax = plt.subplots(figsize=(12, 4))
    df_loop["n_plots"].plot(kind="hist", bins=50, ax=ax).set_title(
        f"Distribution of n_plots per {my_region} and year"
    )
    plt.savefig(
        f"{my_dir}/_distribution_of_nplots.png",
        dpi=300,
    )
    plt.close(fig)

    # ! Prepare plotting data ---------------------------------------------------------------
    # Get shapefile and clean merging variables
    sp_loop = get_shp_of_region(my_region, make_per_year=True)
    sp_loop["year"] = sp_loop["year"].astype(int)
    sp_loop[my_region] = sp_loop[my_region].astype(str)
    sp_loop[f"{my_region}_year"] = (
        sp_loop[f"{my_region}"] + "_" + sp_loop["year"].astype(str)
    )
    sp_loop = sp_loop.reset_index(drop=True)

    # Clean merging variables in df_loop
    df_loop[my_region] = df_loop[f"{my_region}_year"].str.split("_").str[0]
    df_loop["year"] = df_loop[f"{my_region}_year"].str.split("_").str[1].astype(int)
    df_loop[f"{my_region}_year"] = (
        df_loop[f"{my_region}"] + "_" + df_loop["year"].astype(str)
    )
    df_loop = df_loop.reset_index(drop=True)

    # Attach df to shapefile
    df_loop = df_loop.merge(
        sp_loop, how="right", on=[f"{my_region}_year", "year", my_region]
    )
    df_loop = df_loop.reset_index(drop=True)
    df_loop = gpd.GeoDataFrame(df_loop, geometry="geometry")

    # ! Calculate change relative to 2010 ---------------------------------------------------
    # Get metrics
    all_metrics = []
    all_metrics = all_metrics + [x for x in df_loop.columns if "mort" in x]
    # all_metrics = all_metrics + [x for x in df_loop.columns if "grwt" in x]
    # all_metrics = all_metrics + [x for x in df_loop.columns if "change" in x]
    # ? Debug
    all_metrics = ["mort_nat_stems_prc_yr", "mort_nat_vol_yr"]
    warnings.warn(f"Debugging: Only using selected metrics! {all_metrics}")

    # Loop over every region-tile
    diff_abs = pd.DataFrame()
    diff_rel = pd.DataFrame()

    for i_region in df_loop[my_region].unique():
        # Get the data for the current region-tile
        idf_region = df_loop[df_loop[my_region] == i_region]
        # display(idf_region.n_plots.isna().sum())
        # display(idf_region.shape[0])

        # Get the data for the year 2010, subset to metrics
        data_2010 = idf_region[idf_region["year"] == 2010].reset_index(drop=True)[
            all_metrics
        ]

        # If there is no data for 2010, skip the region-tile
        # if data_2010.dropna().shape[0] == 0:
        #     continue

        # Loop over all years
        for i_year in df_loop["year"].unique():
            # Skip the year 2010
            if i_year == 2010:
                continue
            # Get the data for the current year, subset to metrics
            data_year = idf_region[idf_region["year"] == i_year].reset_index(drop=True)[
                all_metrics
            ]

            # If there is no data for the current year, skip it
            if data_year.dropna().shape[0] == 0:
                continue

            # if i_year == 2012:
            #     raise ValueError("Debugging Stop")

            # Calculate the absolute and relative differences
            i_abs = data_year.subtract(data_2010)
            i_rel = i_abs.divide(data_2010) * 100

            # Add the region and year to the results
            i_abs[my_region] = i_region
            i_abs["year"] = i_year
            i_abs[f"{my_region}_year"] = f"{i_region}_{i_year}"

            i_rel[my_region] = i_region
            i_rel["year"] = i_year
            i_rel[f"{my_region}_year"] = f"{i_region}_{i_year}"

            # Add the results to the main DataFrames
            diff_abs = pd.concat([diff_abs, i_abs], ignore_index=True)
            diff_rel = pd.concat([diff_rel, i_rel], ignore_index=True)

    # Clean infities and index
    diff_rel = diff_rel.replace([np.inf, -np.inf], np.nan)
    diff_abs.reset_index(drop=True)
    diff_rel.reset_index(drop=True)

    # Subset shapefile
    sp_tmp = sp_loop.query("year != 2010")[["year", f"{my_region}_year", "geometry"]]
    sp_tmp = sp_tmp.sort_values(f"{my_region}_year").reset_index(drop=True)

    diff_abs_sp = diff_abs.merge(sp_tmp, on=["year", f"{my_region}_year"], how="right")
    # diff_rel_sp = diff_rel.merge(sp_tmp, on=["year", f"{my_region}_year"], how="right")

    if diff_abs_sp.shape[0] != sp_tmp.shape[0]:
        raise ValueError(
            "Shape of diff_abs_sp is not correct! Something failed with the merge..."
        )

    # Attach how many site there are per pixel
    diff_abs_sp = pd.merge(
        df_loop[[f"{my_region}_year", "n_plots"]],
        diff_abs_sp,
        on=f"{my_region}_year",
        how="right",
    )
    # diff_rel_sp = pd.merge(df_loop[[f"{my_region}_year",  "n_plots"]], diff_rel_sp, on=f"{my_region}_year", how="right")

    # ! Loop over all metrics ---------------------------------------------------------------
    idebug = 0
    for my_var in tqdm(all_metrics, disable=True):
        # print(f" - Working on: {my_var} | {my_dir}", end=" | ")

        # Get figure dictionary for variable
        fig_dic = figure_dictionary_for_variable(my_var, my_species)

        # * Plot normal yearly figure
        # make_map_of_change(
        #     df_loop,
        #     fig_dic,
        #     sp_france,
        #     my_dir,
        #     skip_existing=skip_existing
        # )

        # * Absolute change
        make_map_of_change(
            diff_abs_sp,
            figure_dictionary_for_variable(
                my_var, my_species, change_to_2010="absolute"
            ),
            sp_france,
            my_dir,
            skip_existing=skip_existing,
        )

        # * Relative change
        # make_map_of_change(
        #     diff_rel_sp,
        #     figure_dictionary_for_variable(
        #         my_var, my_species, change_to_2010="relative"
        #     ),
        #     sp_france,
        #     my_dir,
        #     skip_existing=skip_existing
        # )

        idebug = idebug + 1
        # if idebug > 3:
        #     break
        # raise ValueError("Debugging Stop")


# -----------------------------------------------------------------------------
def figure_dictionary_for_variable(my_var, my_species, change_to_2010=None):
    # Variable
    fig_dic = {"var": my_var}

    # * Title
    # Species
    if my_species == "all":
        fig_dic["species"] = "All Species"
    else:
        fig_dic["species"] = my_species.capitalize()

    # Mortality
    if "mort" in my_var:
        if change_to_2010 is None:
            fig_dic["cmap"] = plt.cm.Reds
        else:
            fig_dic["cmap"] = plt.cm.RdBu_r
        if "tot" in my_var:
            fig_dic["change"] = "Total Loss"
            fig_dic["main"] = fig_dic["change"] + " of " + fig_dic["species"]
        elif "nat" in my_var:
            fig_dic["change"] = "Mortality"
            fig_dic["main"] = fig_dic["change"] + " of " + fig_dic["species"]
        elif "cut" in my_var:
            fig_dic["change"] = "Harvest"
            fig_dic["main"] = fig_dic["change"] + " of " + fig_dic["species"]

    # Growth
    if "grwt" in my_var:
        fig_dic["change"] = "Gain"
        if change_to_2010 is None:
            fig_dic["cmap"] = plt.cm.Greens
        else:
            fig_dic["cmap"] = plt.cm.RdBu
        if "tot" in my_var:
            fig_dic["main"] = f"Total Growth of " + fig_dic["species"]
        elif "sur" in my_var:
            fig_dic["main"] = f"Survivor Growth of " + fig_dic["species"]
        elif "rec" in my_var or "stems" in my_var:
            fig_dic["main"] = f"Recruits Growth of " + fig_dic["species"]

    # Change
    if "change" in my_var:
        fig_dic["change"] = "Change"
        fig_dic["cmap"] = plt.cm.RdBu
        fig_dic["main"] = f"Total Change of Alive Biomass of " + fig_dic["species"]
        if change_to_2010 in ["absolute", "relative"]:
            fig_dic["main"] = (
                "Change of Change in Biomass (2nd derivative, not sure how to interpret...)"
            )

    # If change to 2010, then change title
    if change_to_2010 == "absolute":
        fig_dic["main"] = "Absolute Change of " + fig_dic["main"]

    if change_to_2010 == "relative":
        fig_dic["main"] = "Relative Change of " + fig_dic["main"]

    # * Colorbar
    if "change" in my_var or change_to_2010 is not None:
        fig_dic["default_cbar"] = False
    else:
        fig_dic["default_cbar"] = True

    # * Units
    if change_to_2010 is None:
        pref = ""
        prct = ""
    elif change_to_2010 == "absolute":
        pref = ""  # Not using "Absolute" because should be self-explanatory...
        prct = ""
    elif change_to_2010 == "relative":
        pref = "Relative"
        prct = "% of "
    else:
        raise ValueError(f"Value for change_to_2010 {change_to_2010} is invalid!")

    if "stems_prc_yr" in my_var:
        fig_dic["unit"] = f"[{prct}%-stems yr$^{-1}$]"
    elif "ba_yr" in my_var:
        fig_dic["unit"] = f"[{prct}m$^{2}$ yr$^{-1}$]"
    elif "ba_prc_yr" in my_var:
        fig_dic["unit"] = f"[{prct}%-m$^{2}$ yr$^{-1}$]"
    elif "vol_yr" in my_var:
        fig_dic["unit"] = f"[{prct}m$^{3}$ yr$^{-1}$]"
    elif "vol_prc_yr" in my_var:
        fig_dic["unit"] = f"[{prct}%-m$^{3}$ yr$^{-1}$]"

    # * Legend
    if change_to_2010 is None:
        fig_dic["legend"] = fig_dic["change"] + " " + fig_dic["unit"]
    else:
        fig_dic["legend"] = (
            pref + " Change of " + fig_dic["change"] + " " + fig_dic["unit"]
        )

    # * Plot directory
    fig_dic["dir"] = "no_change"
    if change_to_2010 == "absolute":
        fig_dic["dir"] = "absolute_change"
    if change_to_2010 == "relative":
        fig_dic["dir"] = "relative_change"

    # * Define color axis limits
    fig_dic["color_axis_limits"] = None
    if change_to_2010 == "absolute":
        if "mort_nat" in my_var and "prc" in my_var:
            fig_dic["color_axis_limits"] = 1.5

        if "mort_nat_vol_yr" == my_var:
            fig_dic["color_axis_limits"] = 3

    return fig_dic


# -----------------------------------------------------------------------------
def make_map_of_change(
    df_in,
    fig_dic,
    sp_france,
    fig_dir,
    skip_existing=False,
):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import geopandas as gpd
    import numpy as np
    import matplotlib.colors as colors
    import matplotlib.colors as mcolors
    from matplotlib.colors import LinearSegmentedColormap

    # ! DEBUGGING ---------------------------------------------------------------
    # check if file already exists
    # Update input dictionary
    fig_dir = f"{fig_dir}/{fig_dic['dir']}"
    filepath = f"{fig_dir}/{fig_dic['var']}.png"
    if os.path.isfile(filepath) and skip_existing:
        print(f"\t\t - File already exists: {filepath}, skipping it")
        return

    # * Load the data
    gdf = df_in.copy()

    # Make sure gdf is a GeoDataFrame
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry")

    # Unique years to create subplots for
    unique_years = sorted(gdf["year"].unique())
    n_years = len(unique_years)

    # Use some nice font
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"

    # Set up figure and GridSpec
    n_cols = int(np.ceil(n_years / 2))  # Make sure to fit on two rows
    fig = plt.figure(figsize=(15, 8))

    # Allocate the last column for the colorbar and use 2x2 grid for the rest
    gs = gridspec.GridSpec(
        2,
        n_cols + 1,
        height_ratios=[1, 1],
        width_ratios=np.repeat(1, n_cols).tolist() + [0.025],
    )

    # ! Color normalization and colormap ------------------------------------------
    if fig_dic["default_cbar"]:
        # Default

        # Taking 95 percentile to avoid outliers dominating coloring
        # data_max = gdf[fig_dic["var"]].max()
        data_max = np.percentile(gdf[fig_dic["var"]].dropna(), 95)
        data_min = 0

        cbar_extend = "max"
        norm = colors.Normalize(vmin=data_min, vmax=data_max)
        sm = plt.cm.ScalarMappable(cmap=fig_dic["cmap"], norm=norm)

    else:
        # Taking 5 and 95 percentile to avoid outliers dominating coloring
        # data_min = gdf[fig_dic["var"]].min()
        # data_max = gdf[fig_dic["var"]].max()
        if fig_dic["color_axis_limits"] is not None:
            data_min = -fig_dic["color_axis_limits"]
            data_max = fig_dic["color_axis_limits"]
            abs_max = fig_dic["color_axis_limits"]
        else:
            data_min = np.percentile(gdf[fig_dic["var"]].dropna(), 5)
            data_max = np.percentile(gdf[fig_dic["var"]].dropna(), 95)
            abs_max = max(abs(data_max), abs(data_min))

        cbar_extend = "both"
        norm = colors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
        sm = plt.cm.ScalarMappable(cmap=fig_dic["cmap"], norm=norm)

    # ! Iterate over the years and create a subplot for each -----------------------
    for i, year in enumerate(unique_years):
        ax = fig.add_subplot(gs[i // n_cols, i % n_cols])

        # Filter the data for the year and plot
        data_for_year = gdf[gdf["year"] == year]
        # Plot it
        plot = data_for_year.plot(
            column=fig_dic["var"],
            edgecolor="face",
            linewidth=0.5,
            ax=ax,
            cmap=fig_dic["cmap"],
            norm=norm,
            missing_kwds={
                "color": "darkgrey",
                "edgecolor": "darkgrey",
                "linewidth": 0.5,
            },
        )

        # Hatch regions where n_plots is below 10
        data_for_year.loc[data_for_year["n_plots"] < 10].plot(
            hatch="///",
            # edgecolor="black",
            alpha=0.5,
            linewidth=0,
            facecolor="none",
            ax=ax,
        )

        # Add countour of France
        sp_france.plot(ax=ax, color="none", edgecolor="black", linewidth=0.5)

        # Remove axis
        ax.set_axis_off()

        # Add year as text below the map
        ax.text(
            0.5, 0, str(year), transform=ax.transAxes, ha="center", fontweight="bold"
        )

    # ! Add colorbar --------------------------------------------------------------
    # Create a colorbar in the space of the last column of the first row
    # Span both rows in the last column for the colorbar
    cbar_ax = fig.add_subplot(gs[0:2, n_cols])
    cbar = fig.colorbar(sm, cax=cbar_ax, extend=cbar_extend)
    cbar.set_label(fig_dic["legend"])

    # ! Finish up -----------------------------------------------------------------
    # Adjust layout to accommodate the main title and subplots
    plt.tight_layout(rect=[0, 0, 1, 1])

    # After creating your subplots and before showing or saving the figure
    fig.suptitle(fig_dic["main"], fontsize=16, fontweight="bold", position=(0.5, 1.05))

    # Show/save the figure
    # plt.show()

    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(filepath, bbox_inches="tight", pad_inches=0.1, dpi=300)
    plt.close()
    print(f"\t\t - Map saved to: {filepath}")


# ------------------------------------------------------------------------------------------
def get_shp_of_region(region, make_per_year=None, make_per_group=None):

    # Get corresponding shapefile
    if region == "cty":
        sp = gpd.read_file(here("data/raw/maps/france_geojson/cty.geojson"))
        sp.rename(columns={"geounit": "cty"}, inplace=True)
    if region == "reg":
        sp = gpd.read_file(here("data/raw/maps/france_geojson/reg.geojson"))
    if region == "dep":
        sp = gpd.read_file(here("data/raw/maps/france_geojson/dep.geojson"))
    if region == "gre":
        sp = gpd.read_file(here("data/raw/maps/france_geojson/gre.geojson"))
    if region == "ser":
        sp = gpd.read_file(here("data/raw/maps/france_geojson/ser_dissolved.geojson"))
    if region == "hex":
        sp = load_hexmap()

    # If one region is split into multiple polygons, then merge them
    # ! This was only an issue for ser, but I created a new shapefile for this
    # ! It is loaded directly above.
    # if sp.shape[0] != sp[region].nunique():
    #     myfile = here(f"data/raw/maps/france_geojson/{region}_dissolved.geojson")
    #     if os.path.isfile(myfile):
    #         # print("[>] get_shp_of_region(): Dissolved file already exists, loading it!")
    #         sp = gpd.read_file(myfile)
    #     else:
    #         print("[>] get_shp_of_region(): Need to dissolve into mulitpolygon which takes time...")
    #         # Dissolve into multipolygon
    #         diss = sp[[f"{region}", "geometry"]].dissolve(by=region, aggfunc="sum").reset_index()
    #         # Attach remaining info back
    #         sp = sp.drop("geometry", axis=1).drop_duplicates().merge(diss, on=region, how="left")
    #         # Save it
    #         sp = gpd.GeoDataFrame(sp)
    #         sp.to_file(myfile, driver="GeoJSON")

    # > NFI data did not differentiate between north and south corse
    # Instead of matching coordinates to respective polygon, I just merge corse to one polygon
    # Should not matter because corse is small and less important anyways.
    if region == "dep":
        # Load region shapefile and extract geometry for Corse
        add_corse = (
            get_shp_of_region("reg", make_per_year=None)
            .rename(columns={"reg": "dep"})
            .query("dep == '94'")
            .replace({"dep": {"94": "2"}})
        )
        add_corse["dep_name"] = "Corse"
        # Remove old corse
        sp.drop(sp[sp["dep"] == "2"].index, inplace=True)
        # Attach corse to sp and continue
        sp = pd.concat([sp, add_corse]).sort_values("dep").reset_index(drop=True)

    if region == "hex":
        sp["hex_name"] = sp["hex"].astype(str)

    # Sanity check that is working now
    if sp.shape[0] != sp[region].nunique():
        raise ValueError("Something is wrong with the shapefile!")

    # ! Make region_year pairs
    if make_per_year is not None:
        spyear = pd.DataFrame()
        myrange = range(make_per_year[0], make_per_year[-1] + 1)

        for year in myrange:
            sp_i = sp.copy()
            sp_i["year"] = year
            spyear = pd.concat([spyear, sp_i])

        # Sanity check that is working now
        if spyear.shape[0] != (spyear[region].nunique() * myrange.__len__()):
            raise ValueError("Something is wrong with the shapefile!")
        sp = spyear.copy()

    sp = sp.drop_duplicates().reset_index(drop=True)

    # ! Make region_group pairs
    if make_per_group is not None:
        spgroup = pd.DataFrame()
        for group in make_per_group:
            sp_i = sp.copy()
            sp_i["group"] = group
            spgroup = pd.concat([spgroup, sp_i])

        # Sanity check that is working now
        if make_per_year is not None:
            expected_shape = (sp[region].nunique() * myrange.__len__()) * len(
                make_per_group
            )
        else:
            expected_shape = sp[region].nunique() * len(make_per_group)

        if spgroup.shape[0] != expected_shape:
            raise ValueError("Something is wrong with the shapefile!")
        sp = spgroup.copy()

    sp = gpd.GeoDataFrame(sp)
    return sp


def ___RASTER___():
    pass


# ------------------------------------------------------------------------------------------


def ndvi_extract_zonal_mean(
    path_files, path_buffer, force_run=False, return_df=True, verbose=True
):

    # path_files input needs to have the following columns: input_file, output_file
    if type(path_files) != pd.core.frame.DataFrame:
        raise ValueError("path_files must be a DataFrame.")

    if "input_file" not in path_files.columns:
        raise ValueError("'input_file' column not found in path_files.")

    if "output_file" not in path_files.columns:
        raise ValueError("'output_file' column not found in path_files.")

    buffer = gpd.read_file(path_buffer)

    # path_buffer needs to have the following columns: idp, first_year
    if "idp" not in buffer.columns:
        raise ValueError("'idp' column not found in buffer.")

    if "first_year" not in buffer.columns:
        raise ValueError("'first_year' column not found in buffer.")

    # Define output df
    df_out = buffer[["idp", "first_year"]]

    # Loop through all path_files
    extractions_done = 0
    for i in tqdm(range(len(path_files)), disable=not verbose):

        # print(f"Processing {i+1} of {len(path_files)}...")

        # Skip if already extracted
        if os.path.exists(path_files.output_file.iloc[i]) and not force_run:

            if return_df:
                print(
                    f" - File {path_files.output_file.iloc[i]} already exists, returning it!"
                )
                df_save = pd.read_csv(path_files.output_file.iloc[i])
                return df_save

            print(
                f" - File {path_files.output_file.iloc[i]} already exists, skipping it!"
            )
            next

        else:
            # Check if file has same CRS as buffer
            with rasterio.open(path_files.input_file.iloc[i]) as src:
                crs = src.crs
                if crs != buffer.crs:
                    raise ValueError(
                        f"CRS of input file {path_files.input_file.iloc[i]} does not match buffer CRS: {buffer.crs}!"
                    )

            # Extract
            df_zonal = zonal_stats(
                path_buffer,
                path_files.input_file.iloc[i],
                stats="count mean sum std nodata",
                nodata=-999,
            )
            df_zonal = pd.DataFrame(df_zonal)

            # Save
            df_save = pd.concat([df_out, df_zonal], axis=1)
            df_save.to_csv(path_files.output_file.iloc[i])

            # Update counter
            extractions_done = extractions_done + 1

    # Return
    if return_df:
        if extractions_done == 0:
            print(f" - No extractions done, returning None!")
            return None
        else:
            print(
                f" - {extractions_done} extractions done, returning last df from file: {path_files.output_file.iloc[i]}..."
            )
            return df_save


def ndvi_extract_zonal_mean_mp(
    path_files, path_buffer, force_run=False, return_df=True
):

    path_files = split_df_into_list_of_group_or_ns(path_files, "input_file")

    run_mp(
        ndvi_extract_zonal_mean,
        arg_list=path_files,
        num_cores=10,
        progress_bar=True,
        path_buffer=path_buffer,
        force_run=force_run,
        return_df=return_df,
        verbose=False,
    )


# ------------------------------------------------------------------------------------------


def extract_raster_values(
    tiff_file,
    variable_name,
    latitudes,
    longitudes,
    progress_bar=False,
    expected_crs=None,
):
    """
    Extracts raster values from a TIFF file at specified latitude and longitude coordinates.

    Args:
        tiff_file (str): The path to the TIFF file.
        variable_name (str): The name of the variable being extracted.
        latitudes (list): A list of latitude coordinates.
        longitudes (list): A list of longitude coordinates.
        progress_bar (bool, optional): Whether to display a progress bar during extraction. Defaults to False.
        expected_crs (str, optional): The expected Coordinate Reference System (CRS) of the input coordinates. Defaults to None.

    Returns:
        pandas.DataFrame: A DataFrame containing the extracted values, latitudes, and longitudes.
    """

    if expected_crs is None:
        print(
            f"\t🚧 WARNING: No CRS specified. Make sure inputed coordinates have matching CRS!."
        )
        print(f"\t Returning None!")
        return None

    # Open the TIFF file
    with rasterio.open(tiff_file) as src:
        # Print the CRS (Coordinate Reference System) for quality control
        # Get temporary CRS
        tmp_crs = str(src.crs)

        # Bugfix for Agroparistech data
        if "RGF_1993_Lambert_Conformal" in tmp_crs:
            tmp_crs = "EPSG:2154"

        if not expected_crs in tmp_crs:
            print(
                f"\t🚧 WARNING: The CRS is {tmp_crs} and not {expected_crs}. Make sure inputed coordinates have matching CRS!."
            )
            print(f"\t Returning None!")
            return None

        # Create empty lists to store the extracted values and coordinates
        raster_values = []
        latitudes_out = []
        longitudes_out = []

        # Iterate over the latitudes and longitudes with or without progress bar
        if progress_bar:
            dummy = tqdm(zip(latitudes, longitudes), total=len(latitudes))
        else:
            dummy = zip(latitudes, longitudes)

        for lat, lon in dummy:
            # Get the row and column indices corresponding to the latitude and longitude coordinates
            row, col = src.index(lon, lat)
            # Check if the row and column indices are within the raster extent
            if row < 0 or row >= src.height or col < 0 or col >= src.width:
                # print(
                #     f"\t 🚧 WARNING: The lat/lon {lat}/{lon} are outside the extent of the TIFF file."
                # )
                raster_values.append(np.nan)
            else:
                # Read the raster value at the specified indices
                raster_value = src.read(1)[row, col]
                raster_values.append(raster_value)

            # Append the coordinates to the respective lists
            latitudes_out.append(lat)
            longitudes_out.append(lon)

        # Create a dataframe with the coordinates and the extracted values
        # Turn all elements in raster_values to float to avoid issues merging integer with float-NaN
        raster_values = [float(i) for i in raster_values]

        df = pd.DataFrame(
            {
                variable_name: raster_values,
                "Latitude": latitudes_out,
                "Longitude": longitudes_out,
            }
        )

        return df


# ------------------------------------------------------------------------------------------
def extract_raster_values_loop(files, coords, expected_crs, verbose=False):

    # Checks
    if expected_crs == "2154":
        lat = "y_fr"
        lon = "x_fr"
    elif expected_crs == "4326":
        lat = "y"
        lon = "x"
    else:
        raise ValueError("Expected CRS must be 2154 or 4326.")

    if lat and lon not in coords.columns:
        raise ValueError("Latitude and Longitude columns not found in coords.")

    if type(files) != pd.core.frame.DataFrame:
        raise ValueError("Files must be a DataFrame.")

    if "path" not in files.columns:
        raise ValueError("'path' column not found in files.")

    if "filename" not in files.columns:
        raise ValueError("'filename' column not found in files.")

    # Create output list
    list_out = []

    # Start loop
    for i in range(len(files)):
        # Verbose
        if verbose:
            print(f"Processing {i+1} of {len(files)}")

        # Extract raster values
        df_values = extract_raster_values(
            tiff_file=files.path.iloc[i],
            variable_name=files.filename.iloc[i],
            latitudes=coords[lat],
            longitudes=coords[lon],
            expected_crs=expected_crs,
            progress_bar=verbose,
        )

        list_out.append(df_values[[files.filename.iloc[i]]])

    df_out = pd.concat(list_out, axis=1)
    df_out = pd.concat([coords, df_out], axis=1)

    return df_out


# ------------------------------------------------------------------------------------------
def extract_raster_values_loop_mp(
    files, coords, expected_crs, ncores=10, progress_bar=True
):

    df_mp = run_mp(
        map_func=extract_raster_values_loop,
        arg_list=files,
        num_cores=ncores,
        progress_bar=progress_bar,
        # kwargs:
        coords=coords,
        expected_crs=expected_crs,
        verbose=False,
    )

    return df_mp


# ------------------------------------------------------------------------------------------
def extract_zonal_mean_digitalis(files, buffer, force_run):

    df_out = buffer[["idp", "first_year"]]

    # ! New version, saving files
    for f in files:

        # Get filename
        variable = f.split("/")[-1].split(".")[0]
        dir_save = "../../data/final/digitalis/1km/python_extracted/2023_revisits_only/"
        filename = f"{dir_save}{variable}.feather"

        # Check if directory exists
        if not os.path.exists(dir_save):
            os.makedirs(dir_save)
        
        # Skip if already extracted
        if os.path.exists(filename) and not force_run:
            next
        else:
            # Extract
            df_zonal = zonal_stats(buffer, f, stats="mean")
            df_zonal = pd.DataFrame(df_zonal).rename(columns={"mean": variable})

            # Save
            df_save = pd.concat([df_out, df_zonal], axis=1)
            df_save.to_feather(filename)

def extract_zonal_mean(
    files=None,
    buffer=None,
    save_dir=None,
    force_run=None,
):

    # Check if any of the inputs are None
    if any([files is None, buffer is None, save_dir is None, force_run is None]):
        raise ValueError(" - All input arguments must be specified: files, buffer, save_dir, force_run")

    # Check if files has filename and path as columns
    if not all([col in files.columns for col in ["filename", "path"]]):
        raise ValueError(" - Files must have filename and path as columns")

    # Check if save_dir exists
    if not os.path.exists(save_dir):
        print(" - Creating directory: ", save_dir)
        os.makedirs(save_dir)

    # Loop over files
    for f in files.iterrows():
        # Get filename and path
        filename = f[1]["filename"]
        path = f[1]["path"]

        # Get filename of output
        output_filename = f"{save_dir}/{filename}.feather"

        # Check if file exists
        if os.path.exists(output_filename) and not force_run:
            # print(" - File already exists: ", filename, path, output_filename)
            continue

        # Extract data
        df_zonal = zonal_stats(buffer, path, stats="mean")
        df_zonal = pd.DataFrame(df_zonal).rename(columns={"mean": filename}).reset_index(drop=True)

        # Save
        df_save = pd.concat([buffer[["idp", "first_year"]].reset_index(drop=True), df_zonal], axis=1)
        df_save.to_feather(output_filename)


# ------------------------------------------------------------------------------------------
def parallel_agroparistech_extraction(
    group_in, df_coords, progress_bar=False, verbose=True, concat_by_axis=1
):
    # Get df_all for column-wise merging
    df_all = df_coords[["idp", "x_fr", "y_fr"]]

    # Get df_row for row-wise merging
    df_row = pd.DataFrame()

    # Print working directory
    # print("Working directory: ", os.getcwd())

    for i in range(len(group_in)):
        # Print progress
        if verbose:
            print(
                f"\nGroup {group_in['group'].iloc[i]} \t | {i+1}/{len(group_in)} | {group_in['variables'].iloc[i]}.tif | {group_in['crs'].iloc[i]} | {group_in['files'].iloc[i]}",
                end="\t",
            )

        # Extract values from raster
        df = extract_raster_values(
            group_in["files"].iloc[i],
            group_in["variables"].iloc[i],
            df_coords["y_fr"],
            df_coords["x_fr"],
            progress_bar=progress_bar,
            expected_crs=group_in["crs"].iloc[i],
        )

        # Merge with outgoing df
        # Depends on whether we want to concatenate by row or column (row = slow files, column = fast files)

        if concat_by_axis == 0:
            # Fix coordinates naming to fit input
            df = df.rename(columns={"Latitude": "y_fr", "Longitude": "x_fr"})

            # Attach idp back again to avoid issues when merging
            df["idp"] = df_all.copy()["idp"]

            # Then drop all na values do avoid duplicating sites when row-wise concatenating
            # Plus, reset index just in case
            df = df.dropna().reset_index(drop=True)
            df_row = df_row.reset_index(drop=True)

            # Finally, concatenate by row
            df_row = pd.concat([df_row, df], axis=concat_by_axis)

        else:
            df = df.reset_index(drop=True)
            df_all = df_all.reset_index(drop=True)
            df_all = pd.concat(
                [df_all, df[group_in["variables"].iloc[i]]], axis=concat_by_axis
            )

    if concat_by_axis == 0:
        return df_row
    else:
        return df_all


# -----------------------------------------------------------------------------------
def extract_closest_soil_polygon_parallel(group_in, soil_sites):
    df_all = pd.DataFrame()

    # Loop over every location in the group
    for i in tqdm(range(len(group_in))):
        # for i in range(len(group_in)):

        # Get copy of soil data
        tmp_soils = soil_sites.copy().reset_index(drop=True)

        # Slice group_in at ith location
        df_ith = pd.DataFrame(group_in.iloc[i]).T.reset_index(drop=True)

        # Turn into geodataframe
        df_ith = gpd.GeoDataFrame(
            df_ith,
            geometry=gpd.points_from_xy(df_ith.x_fr, df_ith.y_fr),
            crs="EPSG:2154",
        )[["idp", "geometry"]]

        # Turn to meter projection
        df_ith.to_crs(epsg=3857, inplace=True)
        tmp_soils.to_crs(epsg=3857, inplace=True)

        # Calculate distances and find the minimum
        point_to_compare = df_ith.geometry[0]
        distances = tmp_soils.distance(point_to_compare, align=False)
        min_dist_index = distances.idxmin()

        # Extract the row with the minimum distance
        closest_row = pd.DataFrame(tmp_soils.loc[min_dist_index]).T
        df_ith["rmqs_distance"] = distances[min_dist_index]

        # print(f"----------------------------------")
        # print(f"point_to_compare idp {df_ith['idp'][0]}: ", end="\t")
        # print(point_to_compare)
        # print(
        #     f"closest point site_id {closest_row.iloc[0]['id_site']}:\t{closest_row.iloc[0]['geometry']} is {df_ith.iloc[0]['rmqs_distance']} away"
        # )
        # display(closest_row)

        # Reset index and concatenate (drop all geometry columns because they are in wrong EPSG)
        closest_row = closest_row.drop(columns=["geometry"]).reset_index(drop=True)
        df_ith = df_ith.reset_index(drop=True)

        df_ith = pd.concat([df_ith[["idp", "rmqs_distance"]], closest_row], axis=1)

        # Attach to df_all
        df_all = pd.concat([df_all, df_ith], axis=0)

        # if i == 3:
        #     break

    return df_all


def wrapper_for_large_files(
    group_in, tif_in, var_in, crs_in, progress_bar=False, verbose=False
):
    # Start verbose output
    if verbose:
        print(
            f"wrapper_for_large_files():\n - Working on group {group_in['group'].unique()[0]}..."
        )

    # Need to create a separate tif_in for each group_in
    # Else, parallelization does not work

    tif_in_tmp = tif_in + "_" + group_in["group"].unique()[0].astype(str)
    tif_in_tmp = tif_in
    if verbose:
        print(f" - Creating temporary tif: {tif_in_tmp}...")
    # shutil.copy(tif_in, tif_in_tmp)

    # Run extraction
    if verbose:
        print(" - Extracting values from raster...")
    df = extract_raster_values(
        tiff_file=tif_in_tmp,
        variable_name=var_in,
        latitudes=group_in["y"],
        longitudes=group_in["x"],
        progress_bar=progress_bar,
        expected_crs=crs_in,
    )

    # Merge extracted data
    df = df.drop(columns=["Latitude", "Longitude"]).reset_index(drop=True)
    df = pd.concat([group_in, df], axis=1)

    # Delete temporary tif
    if verbose:
        print(" - Deleting temporary tif...")
    # os.remove(tif_in_tmp)

    return df


# ------------------------------------------------------------------------------------------
# FUNCTIONS TO EXTRACT EDO DATA
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
def parallel_edo_extraction(
    group_in, df_sites, progress_bar=False, debug=False, expected_crs=None
):
    """
    Extracts EDO data for all sites in df_sites for all files in group_in.
    """
    # Create empty dataframe to store results
    df_all_dates = pd.DataFrame()

    # Loop over all files in group_in
    for i in tqdm(range(len(group_in)), disable=not progress_bar):
        # Get current file and date
        current_file = group_in.iloc[i]["file"]
        current_date = group_in.iloc[i]["date"]
        current_var = group_in.iloc[i]["variable"]

        # Get current sites, meaning sites for which the current date is within the start and end year
        current_sites = df_sites.copy().reset_index(drop=True)
        current_sites = current_sites.query(
            "start_year <= @current_date.year <= end_year"
        ).reset_index(drop=True)

        # Print progress
        if progress_bar:
            print(
                f"\nProgress: {i+1}/{len(group_in)} | Variable: {current_var} | Date: {current_date.strftime('%Y-%m-%d')}",
                end="\t",
            )

        # Extract raster values
        df = extract_raster_values(
            tiff_file=current_file,
            variable_name=current_var,
            latitudes=current_sites.y,
            longitudes=current_sites.x,
            progress_bar=False,
            expected_crs=expected_crs,
        )

        # Attach date to df
        df["date"] = current_date
        # Match extracted value with sites
        df = pd.concat([df[[current_var, "date"]], current_sites], axis=1)
        # Combine all data of current variable row-wise
        df_all_dates = pd.concat([df_all_dates, df])

        # Debugging
        if debug:
            if i == 10:
                break

    return df_all_dates


# ------------------------------------------------------------------------------------------
def seasonal_aggregation_per_site(
    df_in, current_var, fcts_to_apply, progress_bar=False
):
    grouped = df_in.groupby("idp", observed=False)  # Group by idp
    df_list = [group for name, group in grouped]  # Create list
    df_out = pd.DataFrame()  # Create empty dataframe for output

    for i in tqdm(range(len(df_list)), disable=not progress_bar):
        current_group = df_list[i].copy()[
            ["idp", "date", "first_year", "season", current_var]
        ]
        current_idp = df_list[i].idp.unique()[0]

        df_i = get_seasonal_aggregates(
            df_in=current_group,
            timescale_days_to_months="fall cut-off",
            fcts_to_apply=fcts_to_apply,
            debug=False,
            verbose=False,
        )

        df_i["idp"] = current_idp
        df_i.insert(0, "idp", df_i.pop("idp"))

        df_out = pd.concat([df_out, df_i])

    return df_out


# ------------------------------------------------------------------------------------------
def get_seasonal_aggregates(
    df_in=None,
    timescale_days_to_months="fall cut-off",
    fcts_to_apply=None,
    verbose=False,
    debug=False,
):
    """
    Compute seasonal aggregates of variables in a given dataframe.
    ❗ Function calculates the five years following the `first year` variable in the dataframe.
    ❗ If the previous x years are needed, shift input `first_year` by x years.

    Args:
    - df_in: pandas DataFrame containing the data to aggregate.
    - timescale_days_to_months: str, optional. The timescale to use for aggregation. Default is "fall cut-off".
    - fcts_to_apply: list of str, optional. The functions to apply for aggregation. Default is ["mean", "std"].
    - verbose: bool, optional. Whether to print information about the number of variables created. Default is True.
    - debug: bool, optional. Whether to print debug information. Default is False.

    Returns:
    - df_outside: pandas DataFrame containing the seasonal aggregates of the variables in df_in.
    """
    # Imports
    import numpy as np
    import pandas as pd
    import datetime as dt
    from os import error

    # Checks
    if df_in is None:
        error("No dataframe provided.")

    supported_fcts = ["mean", "std", "median", "max", "min", "range", "sum", "iqr"]
    default_fcts = ["mean", "std"]

    if fcts_to_apply is None:
        print(
            f"No functions provided! Using default: {default_fcts}.",
            f"Supported functions are: {supported_fcts}",
        )

    # Settings
    # timescale_days_to_months = "fall cut-off"
    # fcts_to_apply     = ['mean', 'std']
    # fcts_to_apply     = ['mean', 'std', 'median', 'max', 'min']

    first_year = df_in["first_year"].unique()[0]

    vars_to_aggregate = df_in.drop(
        columns=["date", "idp", "SiteID", "season", "first_year"],
        errors="ignore",
    ).columns

    # Reduce dataframe to relevant time period
    if timescale_days_to_months == "fall cut-off":
        # Set first and last day of time period
        # fall cut-off means that the first year of impact starts in September
        cut_off_date = "-09-01"

        first_day = str(first_year) + cut_off_date
        last_day = str(first_year + 5) + cut_off_date

        df_filtered_daterange = df_in.query("date >= @first_day and date < @last_day")

    # Create output dataframe
    df_outside = pd.DataFrame(
        {"nan": [np.nan]}
    )  # For some reason, I need to add an NA entry to attach new values in the loop...
    i = 0  # Set counter to 0

    # Define dictionary with functions
    fct_dict = {
        "mean": np.nanmean,
        "std": np.nanstd,
        "median": np.nanmedian,
        "max": np.nanmax,
        "min": np.nanmin,
        "range": range_func,
        "sum": np.nansum,
        "iqr": iqr_func,
    }

    # Loop through functions
    for my_fct in fcts_to_apply:
        # print(my_fct)
        # Loop through variables
        for my_var in vars_to_aggregate:
            # print(my_var)
            # my_var = my_var[0]
            df_tmp = df_filtered_daterange.groupby("season", observed=False)[
                my_var
            ].agg(fct_dict[my_fct])

            # Loop through seasons
            for my_season in df_in["season"].unique():
                var_name = my_var + "_" + my_fct + "_in_" + my_season
                my_value = df_tmp[my_season]
                # print(var_name, ':', my_value)
                df_outside[var_name] = my_value

                i = i + 1

    if verbose:
        print(f"Number of variables created: {i}")

    # Drop NA column again
    df_outside = df_outside.drop(columns=["nan"])

    return df_outside


# Define custom aggregation functions
def range_func(x):
    x = x.dropna()
    return x.max() - x.min()


def iqr_func(x):
    x = x.dropna()
    return x.quantile(0.75) - x.quantile(0.25)


# ------------------------------------------------------------------------------------------
# FUNCTIONS FOR EXTRACTING EXTREME EVENTS FROM EDO DATA


def extract_extreme_events_per_idp(df_in):
    # Get df with idp
    df_befaft_all = pd.DataFrame({"idp": df_in["idp"].unique()})
    df_grouped_by_befaft = df_in.groupby("before_first_year")

    for _, g_befaft in df_grouped_by_befaft:
        # Get metrics on extreme events
        df_metrics_all = pd.DataFrame()
        # df_metrics_all = pd.DataFrame({"idp": g_befaft["idp"].unique()})

        # Loop over groups heat and cold waves
        df_grouped_by_wave = g_befaft.groupby("heat_or_cold")
        for _, g_wave in df_grouped_by_wave:
            # Extract metrics
            df_metrics = extract_extreme_wave_metrics(g_wave)
            # Merge metrics
            df_metrics_all = pd.concat([df_metrics_all, df_metrics], axis=1)

        # Add suffix based on before or after
        is_before = g_befaft["before_first_year"].unique()[0]
        # display(is_before)
        if is_before:
            df_metrics_all = df_metrics_all.add_suffix("_tmin5")
        else:
            df_metrics_all = df_metrics_all.add_suffix("_tpls5")

        # display(df_metrics_all)

        # Concatenate to df_befaft_all column-wise
        df_befaft_all = pd.concat([df_befaft_all, df_metrics_all], axis=1)

    return df_befaft_all


# ------------------------------------------------------------------------------------------
def extract_extreme_wave_metrics(group_in):
    # Input
    wave_type = group_in["heat_or_cold"].unique()[0]
    # display(wave_type)

    # Prepare empty dataframe
    df_out = pd.DataFrame(
        {
            "n_events": np.nan,
            "temp_min": np.nan,
            "temp_max": np.nan,
            "temp_mean": np.nan,
            "days_btwn_events_mean": np.nan,
            "days_btwn_events_min": np.nan,
            "duration_max": np.nan,
            "duration_mean": np.nan,
            "duration_sum": np.nan,
        },
        index=[0],
    )

    ## Attach grouping variable for each wave segment
    # Create a boolean series where True indicates non-NA values
    non_na = group_in["heatw"].notna()

    # Use cumsum on the negated non_na series to form groups
    group = (~non_na).cumsum()

    # Retain group numbers only for non-NA rows
    group_in["group"] = group.where(non_na, np.nan)

    # Drop rows where '"heatw"is NaN
    filtered_df = group_in.dropna(subset=["heatw"]).copy()

    # Group by 'wave_id'
    grouped = filtered_df.groupby("group")
    # for g in grouped:
    # display(grouped.get_group(g[0]))

    # Get number of events
    n_events = len(grouped)

    if len(filtered_df) == 0:
        df_out["n_events"] = 0

    else:
        # Calculate the size of each group
        wave_sizes = grouped.size()

        # Calculate required statistics
        duration_max = wave_sizes.max()
        duration_mean = wave_sizes.mean()
        duration_sum = wave_sizes.sum()

        # Calculate the difference in days between consecutive waves
        start_dates = pd.DataFrame(grouped["date"].min()).rename(
            columns={"date": "start_date"}
        )
        end_dates = pd.DataFrame(grouped["date"].max()).rename(
            columns={"date": "end_date"}
        )
        # Shift rows of end dates one down to calculate time from last end to next start easier
        end_dates = end_dates.shift(1)

        df_dates = pd.concat([start_dates, end_dates], axis=1)[
            ["start_date", "end_date"]
        ]

        # Get time differences from end_date to start_date
        df_dates["time_diff"] = df_dates["start_date"] - df_dates["end_date"]

        # Turn duration into integer
        df_dates["time_diff"] = df_dates["time_diff"].dt.days

        # Calculate the average and minimum time interval
        days_btwn_events_mean = df_dates["time_diff"].mean()
        days_btwn_events_min = df_dates["time_diff"].min()

        # Extract the mean and max extreme temperature
        if wave_type == "heatwave":
            temp_var = "maxtmp"
        else:
            temp_var = "mintmp"

        temp_min = filtered_df[temp_var].min()
        temp_max = filtered_df[temp_var].max()
        temp_mean = filtered_df[temp_var].mean()

        # Overwrite data
        df_out["n_events"] = n_events

        df_out["temp_min"] = temp_min
        df_out["temp_max"] = temp_max
        df_out["temp_mean"] = temp_mean

        df_out["duration_max"] = duration_max
        df_out["duration_mean"] = duration_mean
        df_out["duration_sum"] = duration_sum

        df_out["days_btwn_events_mean"] = days_btwn_events_mean
        df_out["days_btwn_events_min"] = days_btwn_events_min

    # Add prefix based on wave type
    if wave_type == "heatwave":
        df_out = df_out.add_prefix("hw_")
    else:
        df_out = df_out.add_prefix("cw_")

    return df_out


# Function to divide the list of dataframes into 10 nearly equal parts
def divide_chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


# Function to put the dataframes into 10 lists by keeping the idp together
def put_df_into_10lists_by_keeping_idp_together(df_in):
    dfs = [g for _, g in df_in.groupby("idp")]

    # Size of each chunk (100 for 1000 dataframes)
    chunk_size = len(dfs) // 10

    # Splitting the list of dataframes into 10 parts
    dfs_chunks = list(divide_chunks(dfs, chunk_size))

    # Concatenate dataframes within each chunk
    concatenated_dfs = [pd.concat(chunk, ignore_index=True) for chunk in dfs_chunks]

    return concatenated_dfs


# ==========================================================================================
def ___SAFRAN___():
    pass


# ==========================================================================================
def safran_extract_value(ds_in, lat_in, lon_in, return_fig=False):
    """
    Extracts a value from a safran netcdf dataset (subsetted to variable AND time) at a given lat/lon point.
    🔗 Function is based on this StackOverflow: https://stackoverflow.com/questions/58758480/xarray-select-nearest-lat-lon-with-multi-dimension-coordinates?rq=1
    """

    # First, find the index of the grid point nearest the inputted lat/lon.
    abslat = np.abs(ds_in.lat - lat_in)
    abslon = np.abs(ds_in.lon - lon_in)
    c = np.maximum(abslon, abslat)
    ([latloc], [lonloc]) = np.where(c == np.min(c))

    # Now I can use that index location to get the values at the x/y diminsion
    point_ds = ds_in.isel(lon=lonloc, lat=latloc)

    if return_fig:
        # Make plot with two figures side by side
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        # Plot for full France
        # axs[0].scatter(ds_in.lon, ds_in.lat)
        ds_in.plot(x="lon", y="lat", ax=axs[0])
        # Plot requested lat/lon point blue
        axs[0].scatter(lon_in, lat_in, color="b")
        axs[0].text(lon_in, lat_in, "requested")
        # Plot nearest point in the array red
        axs[0].scatter(point_ds.lon, point_ds.lat, color="r")
        axs[0].text(point_ds.lon, point_ds.lat, "nearest")
        axs[0].set_title("Value at nearest point: %s" % point_ds.data)

        # Plot zoomed into extracted point
        # Plot grid around extracted poin
        ds_sub = ds_in[latloc - 4 : latloc + 4, lonloc - 4 : lonloc + 4]
        ds_sub.plot(x="lon", y="lat", ax=axs[1])
        axs[1].scatter(ds_sub.lon, ds_sub.lat, color="white")
        # Plot requested lat/lon point blue
        axs[1].scatter(lon_in, lat_in, color="b")
        axs[1].text(lon_in, lat_in, "requested")
        # Plot nearest point in the array red
        axs[1].scatter(point_ds.lon, point_ds.lat, color="r")
        axs[1].text(point_ds.lon, point_ds.lat, "nearest")
        axs[1].set_title("Value at nearest point: %s" % point_ds.data)

    return point_ds.data.item()


def safran_get_closest_point(ds_in, lat_in, lon_in):
    """
    Extracts a value from a safran netcdf dataset (subsetted to variable AND time) at a given lat/lon point.
    """

    # First, find the index of the grid point nearest the inputted lat/lon.
    abslat = np.abs(ds_in.lat - lat_in)
    abslon = np.abs(ds_in.lon - lon_in)
    c = np.maximum(abslon, abslat)
    ([latloc], [lonloc]) = np.where(c == np.min(c))

    return latloc, lonloc


def safran_extract_from_index(ds_in, latloc, lonloc):
    """
    Extracts a value from a safran netcdf dataset (subsetted to variable AND time) at a given lat/lon point.
    """

    # Now I can use that index location to get the values at the x/y diminsion
    point_ds = ds_in.isel(lon=lonloc, lat=latloc)

    return point_ds.data.item()


def safran_extract_data_per_site(sites_in, nc_filepath, timestep, verbose=False):
    # Open netcdf dataset
    ds_org = xr.open_dataset(nc_filepath, engine="netcdf4")
    # List of all variables
    # - Tair
    # - Qair
    # - PSurf
    # - Wind
    # - Rainf
    # - Snowf
    # - SWdown
    # - LWdown

    # Attach new variables
    if verbose:
        print(" - Attaching new variables...")
    # - Total Precipitation
    ds_org["Precip"] = ds_org["Rainf"] + ds_org["Snowf"]
    # - Saturation Vapor Pressure
    ds_org["SVP"] = 0.6108 * np.exp((17.27 * ds_org["Tair"]) / (ds_org["Tair"] + 237.3))
    # - Actual Vapor Pressure
    ds_org["AVP"] = (ds_org["Qair"] * ds_org["PSurf"]) / (
        0.622 + 0.378 * ds_org["Qair"]
    )
    # - Vapor Pressure Deficit
    ds_org["VPD"] = ds_org["SVP"] - ds_org["AVP"]

    # Aggregate to timestep of interest
    # Ignore RuntimeWarning
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if verbose:
        print(f" - Aggregating data to {timestep}...")
    # Get ds of means
    ds_means = (
        ds_org[
            [
                "Tair",
                "VPD",
                "SWdown",
                "LWdown",
                "PSurf",
                "Wind",
            ]
        ]
        .groupby(f"tstep.{timestep}")
        .mean("tstep")
    )

    # Get ds of min
    ds_min = (
        ds_org[
            [
                "Tair",
                "PSurf",
            ]
        ]
        .groupby(f"tstep.{timestep}")
        .min("tstep")
    )

    # Get ds of max
    ds_max = (
        ds_org[
            [
                "Tair",
                "VPD",
                "PSurf",
                "Wind",
            ]
        ]
        .groupby(f"tstep.{timestep}")
        .max("tstep")
    )

    # Get ds of sum
    ds_sum = (
        ds_org[
            [
                "Precip",
            ]
        ]
        .groupby(f"tstep.{timestep}")
        .sum("tstep")
    )

    # Reset warnings
    warnings.resetwarnings()

    # Loop over coordinates
    if verbose:
        print(f" - Looping over {sites_in.shape[0]} sites...")

    df_final = pd.DataFrame()
    for i in tqdm(range(sites_in.shape[0]), disable=not verbose):
        # Get closest point
        lat = sites_in["y"].iloc[i]
        lon = sites_in["x"].iloc[i]
        latloc, lonloc = safran_get_closest_point(ds_org, lat, lon)

        # Extract data
        # - Mean Values
        df_mean = pd.DataFrame()
        for variable in list(ds_means.data_vars):
            df_i = safran_extract_all_timesteps(
                ds_means[variable],
                variable + "_mean",
                timestep,
                latloc,
                lonloc,
                verbose=False,
            )

            if df_mean.empty:
                df_mean = df_i.copy()
            else:
                df_mean = df_mean.merge(df_i, on=timestep, how="outer")

        # - Min Values
        df_min = pd.DataFrame()
        for variable in list(ds_min.data_vars):
            df_i = safran_extract_all_timesteps(
                ds_min[variable],
                variable + "_min",
                timestep,
                latloc,
                lonloc,
                verbose=False,
            )

            if df_min.empty:
                df_min = df_i.copy()
            else:
                df_min = df_min.merge(df_i, on=timestep, how="outer")

        # - Max Values
        df_max = pd.DataFrame()
        for variable in list(ds_max.data_vars):
            df_i = safran_extract_all_timesteps(
                ds_max[variable],
                variable + "_max",
                timestep,
                latloc,
                lonloc,
                verbose=False,
            )

            if df_max.empty:
                df_max = df_i.copy()
            else:
                df_max = df_max.merge(df_i, on=timestep, how="outer")

        # - Sum Values
        df_sum = pd.DataFrame()
        for variable in list(ds_sum.data_vars):
            df_i = safran_extract_all_timesteps(
                ds_sum[variable],
                variable + "_sum",
                timestep,
                latloc,
                lonloc,
                verbose=False,
            )

            if df_sum.empty:
                df_sum = df_i.copy()
            else:
                df_sum = df_sum.merge(df_i, on=timestep, how="outer")

        # Merge all variables of that site
        df_min = df_min.reset_index(drop=True)
        df_max = df_max.reset_index(drop=True)
        df_sum = df_sum.reset_index(drop=True)

        df_site = pd.concat(
            [
                df_mean,
                df_min.drop(timestep, axis=1),
                df_max.drop(timestep, axis=1),
                df_sum.drop(timestep, axis=1),
            ],
            axis=1,
        )
        # Add site id
        df_site["idp"] = sites_in["idp"].iloc[i]

        # Add to df_final
        if df_final.empty:
            df_final = df_site.copy()
        else:
            df_final = pd.concat([df_final, df_site], axis=0)
    # End of loop over sites

    # Move idp to first column
    df_final.insert(0, "idp", df_final.pop("idp"))

    # Reset index
    df_final = df_final.reset_index(drop=True)

    # Close all datasets
    ds_org.close()
    ds_means.close()
    ds_min.close()
    ds_max.close()
    ds_sum.close()

    return df_final


def safran_extract_all_timesteps(
    ds_var, my_var, time_var, latloc, lonloc, verbose=False
):
    """
    Extract the data for a given variable at all timesteps.
    Input data must be reduced to one variable.
    time_var is the variable name for the time dimension.
    my_var is the variable name (only used to label df_out column).
    latloc and lonloc are the indeces for the closest pixel to the site.
    """
    # Set up output dataframe
    df_out = pd.DataFrame(columns=[time_var, my_var])
    # Get time steps
    all_tsteps = ds_var.get_index(time_var).to_list()
    # Loop through all timesteps)
    for i in tqdm(
        range(len(all_tsteps)),
        disable=not verbose,
        desc=f"        - Working on timestep ({time_var}): ",
    ):
        # Get ds for current timestep
        ds_var_tstep = ds_var[i, :, :]
        df_i = pd.DataFrame(
            {
                time_var: [all_tsteps[i]],
                my_var: [safran_extract_from_index(ds_var_tstep, latloc, lonloc)],
            }
        )

        df_out = pd.concat([df_out, df_i], axis=0)

    return df_out


# -----------------------------------------------------------------------------------
def ___TEMPORAL_AGGREGATION___():
    pass


def aggregate_annual_values_to_before_and_after(df_in, vars):
    # Input dataframe is only allowed to contain the following variables
    allowed_vars = ["idp", "year", "first_year"] + vars
    df_in = df_in[allowed_vars]

    # Define own functions to be used within aggregation
    # todo: not working for now... iqr has issues, range is calculated later from min man
    # def iqr_func(x):
    #     q75, q25 = np.percentile(x.dropna(), [75, 25])
    #     iqr = q75 - q25
    #     return iqr

    # def range_func(x):
    #     x = x.dropna()
    #     return x.max() - x.min()
    # stats = ["mean", "std", "min", "max", "median", iqr_func, range_func]

    # Calculate statistics for each year
    stats = ["mean", "std", "min", "max", "median"]

    # Get aggregation metric so that each variable is aggregated by each statistic
    my_agg = {}
    for v in vars:
        my_agg[v] = stats

    # Split data into 5 years before and after first census year
    df_before = df_in.query("year >= first_year - 5 and year < first_year")
    df_after = df_in.query("year <= first_year + 5 and year > first_year")

    # Group data by idp and aggregate
    df_before = df_before.groupby("idp").agg(my_agg)
    df_after = df_after.groupby("idp").agg(my_agg)

    # Flatten index
    df_before.columns = ["_".join(col) for col in df_before.columns]
    df_after.columns = ["_".join(col) for col in df_after.columns]

    # Reset index
    df_before = df_before.reset_index()
    df_after = df_after.reset_index()

    # Attach range if min and max are in stats
    if "min" in stats and "max" in stats:
        for v in vars:
            df_before[v + "_range"] = df_before[v + "_max"] - df_before[v + "_min"]
            df_after[v + "_range"] = df_after[v + "_max"] - df_after[v + "_min"]

    # Remove idp and year before adding suffix to rest
    idp_vector = df_before["idp"]
    df_before = df_before.drop("idp", axis=1)
    df_after = df_after.drop("idp", axis=1)

    # Add suffix to columns
    df_before = df_before.add_suffix("_over_tmin5")
    df_after = df_after.add_suffix("_over_tpls5")

    # Add idp back again
    df_before.insert(0, "idp", idp_vector)
    df_after.insert(0, "idp", idp_vector)

    # Merge data back together
    df_merged = pd.merge(df_before, df_after, on="idp")

    # Return
    return df_merged


# ------------------------------------------------------------------------------------------
def region_code_to_name(region):
    if region == "gre":
        code = {
            "A": "Grand Ouest",
            "B": "Centre Nord",
            "C": "Grand Est",
            "D": "Vosges",
            "E": "Jura",
            "F": "Sud Ouest",
            "G": "Massif Central",
            "H": "Alpes",
            "I": "Pyrénées",
            "J": "Méditerranée",
            "K": "Corse",
        }
    elif region == "reg":
        # INSEE Code
        # https://en.wikipedia.org/wiki/Regions_of_France
        # https://www.insee.fr/fr/information/2560563#titre-bloc-29
        code = {
            "11": "Île-de-France",
            "24": "Centre-Val de Loire",
            "27": "Bourgogne-Franche-Comté",
            "28": "Normandie",
            "32": "Hautes-de-France",
            "44": "Grand Est",
            "52": "Pay de la Loire",
            "53": "Bretagne",
            "75": "Nouvelle-Aquitaine",
            "76": "Occitanie",
            "84": "Auvergne-Rhône-Alpes",
            "93": "Provence-Alpes-Côte d'Azur",
            "94": "Corse",
        }

    return code


def bootstrap_ids_per_group(df_in, target, group_id, id, n_bootstraps, verbose=True):
    """
    Function repeats a dataframe n times by taking a random sample of trees within a group.
    The sampling of trees is done by taking the same number of trees per group as in the original population, with replacement.
    This sampling per group is repeated n times, so that the mean and std of that group can be calculated directly from the random population of trees.

    Args:
        df_in (_type_): Input dataframe
        group_id (_type_): Should be group_year
        id (_type_): Should be tree_id
        n_bootstraps (int, optional): Number of bootstraps.

    Returns:
        _type_: _description_
    """
    # df_in = xxx.copy()
    # group_id = "group_year"
    # id = "tree_id"
    # n_bootstraps = 10

    # Verbose
    if verbose:
        print("\nBootstrapping data...")

    # Subset input to bootstrap: group_id is the group within which id is sampled.
    # id is then used to attach information on that id below again.
    if target is None:
        df_to_bootstrap = df_in[
            [group_id, id]
        ]  # ! Original, subsetting to vars relevant for bs
    else:
        df_to_bootstrap = df_in[[group_id, id, target]]

    # Group by group_id to bootstrap within groups
    df_list_to_bootstrap = split_df_into_list_of_group_or_ns(df_to_bootstrap, group_id)

    # # Get empty list to bs
    df_list_bootstrapped = []

    # Do bs
    np.random.seed(42)
    for i in tqdm(range(n_bootstraps), disable=not verbose):
        for i_df in df_list_to_bootstrap:
            i_df = i_df.sample(frac=1, replace=True)
            i_df["bs_id"] = i + 1
            df_list_bootstrapped.append(i_df)

    # Concatenate lists rowwise
    if verbose:
        print(f"... concatenating columns: {i_df.columns.to_list()}")
    df_bs = pd.concat(df_list_bootstrapped)

    # ! Uncommented to directly return df_bs
    # Remove old group_id and id from df_in
    if target is None:
        if verbose:
            print("... merging")

        df_bs = df_bs.merge(df_in, on=[group_id, id], how="left")

    # Report
    if verbose:
        print(
            f"... original df had \t {df_in.shape[0]} rows with {df_in[group_id].nunique()} groups and {df_in.shape[1]} columns."
        )
        print(
            f"... bootstrapped df has\t {df_bs.shape[0]} rows with {df_in[group_id].nunique()} groups and {df_bs.shape[1]} columns."
        )

    # Return
    return df_bs

# ------------------------------------------------------------------------------------------
def ___RANDOM_FOREST___():
    pass


# ------------------------------------------------------------------------------------------
# Remove share of 0s in target variable
def remove_na_and_reduce_zero_share(df, target_column, max_zero_share, verbose=False):
    # Make a copy of the original dataframe
    df_org = df.copy()

    # ! Remove NA values
    df_nona = df.dropna(subset=[target_column])

    if verbose:
        print(f"\nRemoving NA values from target variable '{target_column}'")
        print(
            f" - Shape of data before removing NA values:\t {df_org.shape} \t | % of NAs in target:\t {(df_org[target_column] == np.nan).mean():.2%}"
        )
        print(
            f" - Shape of data after removing NA values:\t {df_nona.shape} \t | % of NAs in target:\t {(df_nona[target_column] == np.nan).mean():.2%}"
        )
        print(f" - Number of NA values removed:\t\t\t {len(df_org) - len(df_nona)}")

    # ! Calculate the current share of 0 values in the target column
    df = df_nona.copy()
    zero_share = (df[target_column] == 0).mean()

    if max_zero_share == 0:
        df = df[df[target_column] != 0]
    else:
        # Check if the current share exceeds the maximum allowed share
        while zero_share > max_zero_share * 1.025:
            # Calculate the number of 0 values to remove
            num_zeros_to_remove = int((zero_share - max_zero_share) * len(df))

            # Get the indices of the 0 values
            zero_indices = df[df[target_column] == 0].index

            # Randomly select indices to remove
            indices_to_remove = np.random.choice(
                zero_indices, size=num_zeros_to_remove, replace=False
            )

            # Remove the selected indices from the dataframe
            df = df.drop(indices_to_remove)

            # Recalculate the share of 0 values in the target column
            zero_share = (df[target_column] == 0).mean()

    # Verbose
    if verbose:
        print(
            f"\nRemoving 0 values from target variable '{target_column}' to {max_zero_share}"
        )
        print(
            f" - Shape of data before removing 0s:\t {df_nona.shape} \t | % of 0s in target:\t {(df_nona[target_column] == 0).mean():.2%}"
        )
        print(
            f" - Shape of data after removing 0s:\t {df.shape} \t | % of 0s in target:\t {(df[target_column] == 0).mean():.2%}"
        )
        print(f" - Number of 0s removed:\t\t {len(df_nona) - len(df)}")

    return df


# ------------------------------------------------------------------------------------------
# Run a small random forest to get feature importances for all variables
def run_basic_rf(df_in, target_of_interest, dir_rf):

    os.makedirs(dir_rf, exist_ok=True)
    # ! Data Preparation ------------------------------------------------------
    print(f" - Prepare data...")
    # Remove NAs in target
    df_in = df_in.dropna(subset=target_of_interest)

    # One-hot encode categorical variables
    df_in = pd.get_dummies(df_in)

    # Get X and y
    X = df_in.drop(target_of_interest, axis=1)
    y = df_in[target_of_interest]

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )

    # Replace NA values in train data with -9999
    # X_train = X_train.fillna(-9999)
    # X_test = X_test.fillna(-9999)
    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_train.mean())

    # ! Fit ------------------------------------------------------
    print(f" - Fit random forest...")
    # Fit a random forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=5, n_jobs=9)
    # Fit the model
    rf.fit(X_train, y_train)

    # ! Evaluate ------------------------------------------------------
    print(f" - Feature Importance...")
    # Get feature importances
    show_top_predictors(X_train, rf_model=rf, current_dir=dir_rf)

    print(f" - Partial Dependence...")
    # Get the top 10 predictors
    feature_importances = pd.Series(
        rf.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False)
    top_predictors = feature_importances[:10].index.tolist()
    # Create a 2x5 grid for subplots
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    # Iterate over the top predictors and plot the PDP for each one
    for i, predictor in enumerate(top_predictors):
        # Create the PartialDependenceDisplay object
        pdp_display = PartialDependenceDisplay.from_estimator(
            estimator=rf,
            X=X_train,
            features=[predictor],
            response_method="auto",
            n_jobs=9,
            n_cols=5,
            ax=axes[i // 5, i % 5],
        )
        # Plot the PDP
        # pdp_display.plot()

    # Adjust the spacing between subplots
    fig.tight_layout()

    # Save the plot
    fig.savefig(f"{dir_rf}/pdp_plot.png")
    plt.close(fig)

    # Compare predicted and actual values for both, train and test
    print(f" - Modobs...")
    model_evaluation_regression(
        rf, X_train, y_train, X_test, y_test, save_directory=dir_rf
    )


def list_nfi_site_level_datasets(all_species_subsets=False, all_height_subsets=False):
    print("NFI site level datasets available:")
    my_dir = here("data/tmp/nfi/growth_and_mortality_data/idp")
    # Get all files
    all_files = os.listdir(my_dir)
    # Remove path from files
    all_files = [f.split("/")[-1] for f in all_files]
    # Keep only .feather files
    all_files = [f for f in all_files if f.endswith(".feather")]
    # Filter if required
    pattern = None
    if all_species_subsets and all_height_subsets:
        print(" - Showing all species and height subsets...")

    else:
        if not all_species_subsets:
            print(" - Showing no species subsets...")
            pattern = "species_all"
            all_files = [f for f in all_files if pattern in f]
        if not all_height_subsets:
            print(" - Showing no height subsets...")
            pattern = "height_all"
            all_files = [f for f in all_files if pattern in f]

    # Print files
    for i, file in enumerate(sorted(all_files)):
        print(f"   {i}. {file}")


def load_data_of_change(subset=None, verbose=True):

    if verbose:
        print(f"\nLoading data of change for subset {subset}...")
        print(
            " - Note: I changed this function to simply load the per_site calculations as done in '01_calculate_growth_and_morality'!"
        )

    # ! No subset
    if subset is None:
        filename = here(
            "notebooks/00_process_nfi_data/data_of_change/per_site/gre_subset-none.feather"
        )
    # ! Regional subsets
    elif subset == "gre":
        filename = here(
            "notebooks/00_process_nfi_data/data_of_change/per_site/gre_subset-none.feather"
        )
    elif subset == "ser":
        filename = here(
            "notebooks/00_process_nfi_data/data_of_change/per_site/ser_subset-none.feather"
        )
    elif subset == "dep":
        filename = here(
            "notebooks/00_process_nfi_data/data_of_change/per_site/dep_subset-none.feather"
        )
    elif subset == "reg":
        filename = here(
            "notebooks/00_process_nfi_data/data_of_change/per_site/reg_subset-none.feather"
        )
    # ! Species subsets
    elif subset == "genus_lat":
        filename = here(
            "notebooks/00_process_nfi_data/data_of_change/per_site/genus_lat_subset-none.feather"
        )
    elif subset == "species_lat":
        filename = here(
            "notebooks/00_process_nfi_data/data_of_change/per_site/species_lat_subset-none.feather"
        )
    # ! Size subsets
    elif subset == "tree_height_class":
        filename = here(
            "notebooks/00_process_nfi_data/data_of_change/per_site/tree_height_class_subset-none.feather"
        )
    elif subset == "tree_circumference_class":
        filename = here(
            "notebooks/00_process_nfi_data/data_of_change/per_site/tree_circumference_class_subset-none.feather"
        )
    else:
        raise ValueError(f"Unknown subset requested: {subset}")

    data = load_file_and_report_recency(filename.__str__(), verbose=verbose)
    data["idp"] = 0
    data["idp"] = data["group_year_site"].str.extract(r"(\d+)$").astype(int)
    data = move_vars_to_front(
        data, ["group_year_site", "group_year", "group", "year", "idp"]
    )

    if subset is None:
        data = data.drop(columns=["group_year_site", "group_year", "group", "year"])

    return data

    # ! old structure:
    # def load_data_of_change(species_subset = None, height_subset = None, region_subset = None):
    # Get subset patterns
    # species_pattern = "species_all"
    # height_pattern = "height_all"
    # region_pattern = "reg_all"

    # if species_subset is not None:
    #     species_pattern = f"species_{species_subset}"
    # if height_subset is not None:
    #     height_pattern = f"height_{height_subset}"
    # if region_subset is not None:
    #     raise ValueError("Not implemented yet!")
    #     region_pattern = f"reg_{region_subset}"

    # my_dir = here("data/tmp/nfi/growth_and_mortality_data/idp")
    # my_fil = f"{my_dir}/{species_pattern}-{height_pattern}.feather"

    # if os.path.exists(my_fil):
    #     print(f"Loading file for: {species_pattern} | {height_pattern}")
    #     df = pd.read_feather(my_fil)
    #     return df
    # else:
    #     chime.error()
    #     raise ValueError(f"Subset does not exist: {species_pattern} | {height_pattern}")


# ------------------------------------------------------------------------------------------


def list_predictor_datasets(return_list=False):
    # Show all available datasets
    dir_predictors = here("data/final/predictor_datasets")
    predictor_datasets = [
        f
        for f in os.listdir(dir_predictors)
        if not f.startswith(".") and not f.endswith("archive")
    ]
    predictor_datasets = [re.sub(".feather", "", f) for f in predictor_datasets]
    if return_list:
        return predictor_datasets

    print("Predictor datasets available:")
    for i, dataset in enumerate(sorted(predictor_datasets)):
        date_created = os.path.getmtime(f"{dir_predictors}/{dataset}.feather")
        date_created = datetime.datetime.fromtimestamp(date_created)
        days_ago = (datetime.datetime.now() - date_created).days

        print(
            f" {i+1}.\t {dataset:25} \t | Created: {days_ago:>5} days ago \t({date_created.date()})"
        )


def attach_or_load_predictor_dataset(dataset, df_input=None, verbose=True):
    # Check if dataset exists at all
    filename = here(f"data/final/predictor_datasets/{dataset}.feather")
    if not os.path.isfile(filename):
        chime.error()
        raise ValueError(f"Dataset not found:   {filename}")

    df_add = pd.read_feather(filename)
    if df_input is None:
        print(f"Returning dataset:   {dataset}")
        return df_add

    # Sanity checks
    # Check if input df has idp
    if "idp" not in df_input.columns:
        chime.error()
        raise ValueError("Input df does not have idp variable!")

    # Check if input df already has one of the variables
    for col in df_add.columns:
        if col in ["idp", "tree_id"]:
            continue
        if col in df_input.columns:
            df_add = df_add.drop(columns=col)
            print(
                f"Input df already has variable: {col}. Dropping it to avoiding adding twice."
            )

    # Check if dataset lacks idp that are in input df
    input_nsites = df_input.idp.nunique()
    pred_nsites = df_add.idp.nunique()
    if input_nsites > pred_nsites:
        print(
            f"⚠️ Careful, input df has {input_nsites-pred_nsites} sites more than predictor dataset. These are filled with NAs !"
        )

    # Merge data
    if "tree_id" in df_add.columns:
        mergers = ["idp", "tree_id"]
    else:
        mergers = ["idp"]
    df_output = df_input.merge(df_add, how="left", on=mergers)

    # Sanity check
    if df_output.shape[0] != df_input.shape[0]:
        chime.error()
        raise ValueError(f"Shape after merging: {df_output.shape}")

    # Verbose
    if verbose:
        print(
            f"Attaching dataset:    {dataset}",
            f"\n - Shape of predictor:     {df_add.shape}",
            f"\n - Shape before merging: {df_input.shape}",
            f"\n - Shape after merging:  {df_output.shape}\n",
        )

    return df_output


def match_variables(df, patterns):
    matched_variables = []
    for column in df.columns:
        for pattern in patterns:
            if re.match(pattern.replace("*", ".*"), column):
                matched_variables.append(column)
                break
    return matched_variables


def ___TREND_UTILS___():
    pass


def calculate_change_per_group_and_year(
    df=None,
    my_grouping=None,
    top_n_groups=None,
    my_method=None,
    n_bootstraps_samples=None,
    load_from_file=False,
    file_suffix="",
):

    # If any input none, raise error
    # ! Checks ---------------------------
    if None in [
        my_grouping,
        my_method,
    ]:
        raise ValueError("One of required inputs is None")

    if my_method == "direct_bs" and n_bootstraps_samples is None:
        raise ValueError("n_bootstraps_samples is None")

    # Function Start
    all_methods = ["direct", "direct_bs", "per_site", "per_site_bs"]
    if my_method not in all_methods:
        raise ValueError(f"Invalid method: {my_method}, use one of: {all_methods}")

    for igroup in my_grouping:
        if igroup not in df.columns:
            raise ValueError(f"Invalid grouping: {igroup}, not in df_in")

    if my_method == "per_site_bs":
        my_method = "per_site"

    # Sort grouping variable alphabetically to ensure consistency
    def custom_sort_key(element):
        priority_order = ["gre", "ser", "hex", "reg", "dep"]
        if element in priority_order:
            return (priority_order.index(element), element)
        else:
            return (len(priority_order), element)

    my_grouping = sorted(my_grouping, key=custom_sort_key)

    print(
        f"\nCalculate change for {my_grouping} using method: {my_method}\n... additional settings: {n_bootstraps_samples} bootstraps, load_from_file is {load_from_file}, selecting top {top_n_groups} groups"
    )

    # ! Wrangle input df ---------------------------
    # Add grouping variable
    df["year"] = df["campagne_1"].astype("string")
    df["group"] = df[my_grouping].astype(str).agg("&".join, axis=1)  # Attach all groups
    group_vars = []
    for i in range(len(my_grouping)):
        df[f"group_{i}"] = df[my_grouping[i]].astype("string")
        group_vars.append(f"group_{i}")

    df["group_year"] = df["group"] + "_" + df["year"]
    df["group_year_site"] = (
        df["group"] + "_" + df["year"] + "_" + df["idp"].astype("string")
    )

    # Keep only relevant columns
    relevant_cols = [
        "year",
        "idp",
        "tree_id",
        "group",
        "group_year",
        "group_year_site",
        "tree_state_1",
        "tree_state_2",
        "tree_state_change",
        "ba_1",
        "ba_2",
        "v",
        "genus_lat",
        "species_lat",
        "species_lat2",
        "species_lat_short",
        "tree_height_class",
        "tree_circumference_class",
        "hex",
        "ser",
        "gre",
        "reg",
        "dep",
    ] + group_vars

    df = df[relevant_cols]

    # Remove rows where group holds "missing"
    df = df[~df["group"].str.contains("Missing")]

    # Keep only most common 20 species
    if "species_lat2" in df.columns:
        # print(f"... keeping only the top 20 species 🚨🚨🚨")
        print(f"... running for all species 🚨🚨🚨")
        df["species_lat2"] = df["species_lat2"].astype(str)
        # top_species = df["species_lat2"].value_counts().sort_values(ascending=False).head(20).index
        # df = df[df["species_lat2"].isin(top_species)]

    # Reset index
    df = df.reset_index(drop=True)

    # ! Check grouping variable ---------------------------
    # Check for problematic group levels
    for c in df["group"].unique():
        if "_" in c:
            raise ValueError(
                "Column name contains underscore, messing with the code {}".format(c)
            )

    # Focus only on the n most frequent groups
    n_all_groups = df["group"].nunique()  # get number of unique groups

    if top_n_groups is None:
        top_n_groups = n_all_groups
        print(
            f"... {my_grouping}: top_n_groups is None, setting top_n_groups = {n_all_groups}"
        )

    elif top_n_groups >= n_all_groups:
        if len(my_grouping) > 1:
            raise ValueError(
                "Currently only single grouping is supported for selecting top n groups."
            )

        top_n_groups = n_all_groups
        print(
            f"... {my_grouping}: top_n_groups >= n_all_groups, setting top_n_groups = {n_all_groups}"
        )
    else:
        print(
            f"... {my_grouping}: top_n_groups < n_all_groups, keeping top_n_groups = {top_n_groups} of {n_all_groups} groups."
        )

    # Record how many trees that were alive at first visit (removing those that were dead)
    list_counts = []
    for igroup in group_vars + ["group"]:
        df_counts = count_alive_trees_and_plots_per_group(df, igroup)
        list_counts.append(df_counts)
    df_counts = pd.concat(list_counts)

    # Subset to desired groups
    df = df[
        df["group"].isin(df["group"].value_counts().head(top_n_groups).index)
    ]  # keep only the top n groups

    # ! Get filename ---------------------------
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    my_dir = here(
        f"notebooks/00_process_nfi_data/data_of_change/{my_method}/"
    ).__str__()
    for i, igroup in enumerate(my_grouping):
        if i == 0:
            my_dir += f"/{igroup}"
        else:
            my_dir += f"&{igroup}"
            
    if file_suffix != "":
        my_dir = my_dir.replace("data_of_change", f"specific_runs/{file_suffix} - {today}/data_of_change")

    os.makedirs(my_dir, exist_ok=True)
    file_name = f"{my_dir}/"

    if top_n_groups == n_all_groups:
        file_name += f"/subset-none"
    else:
        file_name += f"/subset-top-{top_n_groups}-of-{n_all_groups}-groups"

    if my_method == "direct_bs":
        file_name += f"_bs-{n_bootstraps_samples}x"

    file_name += ".feather"

    # ! Check if file exists ---------------------------
    if load_from_file:
        if os.path.isfile(file_name):
            print(f"... file already exists, loading it...  {file_name}")
            df_gm = pd.read_feather(file_name)
            return df_gm, df_counts

        print(f"... file does not exist!: {file_name}")

    # ! Calculate change ---------------------------
    # > Direct method: The method directly calculates the change for each group_year
    if my_method == "direct":
        print(
            f"... number of groups for which change is calculated: {df.group_year.nunique()}\n"
        )
        df_gm = mp_calculate_growth_mortality(df, "group_year")

    # > Direct bootstrapped method: Bootstrap the trees within each group_year
    elif my_method == "direct_bs":
        if n_bootstraps_samples is None:
            raise ValueError("n_bootstraps_samples is 'None'")

        # I can only bootstrap trees that were alive in the first year!
        df = df.query("tree_state_1 == 'alive'")
        
        # Check if too many groups are present
        n_groups = df.group_year.nunique()
        n_max = 1500

        print(f"... number of groups for which change is calculated: {df.group_year.nunique()} x {n_bootstraps_samples}\n")
        
        if n_groups > n_max:
            # Verbose
            print(
                f"... 🚨 Too many groups ({n_groups}) to run at once, so splitting them up into groups of {n_max}s"
            )
            # Split into groups
            group_years = df.group_year.unique()
            n_splits = int(np.ceil(n_groups / n_max))
            l_splits = split_df_into_list_of_group_or_ns(df, n_splits, "group_year")

            # Loop over splits
            l_runs = []
            for i in range(n_splits):
                # Check if file exists
                file_extension = f"_split_{i+1}_of_{n_splits}.feather"
                i_file = file_name.replace(".feather", file_extension)
                
                if load_from_file and os.path.isfile(i_file):
                    print(f" --- Loading split {i+1}/{n_splits} from file ---")
                    l_runs.append(pd.read_feather(i_file))
                    continue
                print(f"\t--- Running split {i+1}/{n_splits} ---")
                
                # Filter data
                df_split = l_splits[i]

                # Do bootstrap
                print(f" - Bootstrapping...")
                df_split = bootstrap_ids_per_group(
                    df_split,
                    None,
                    "group_year",
                    "tree_id",
                    n_bootstraps_samples,
                    verbose=False,
                )
                df_split["group_year_site"] = (
                    df_split["group_year"] + "_" + df_split["bs_id"].astype("string")
                )

                print(f" - Calculating...")
                df_split = mp_calculate_growth_mortality(df_split, "group_year_site")
                l_runs.append(df_split)

                # Save to file
                print(f" - Saving...")
                df_split.to_feather(i_file)
                    
            df_gm = pd.concat(l_runs, axis=0).reset_index(drop=True)
            
            # Cleaning up variables
            df_gm["group"] = df_gm["group_year_site"].str.split("_").str[0]
            df_gm["year"] = df_gm["group_year_site"].str.split("_").str[1]
            df_gm["group_year"] = df_gm[["group", "year"]].apply(lambda x: "_".join(x), axis=1)
        
        else:
            # If not too many groups, run directly
            
            # Bootstrap trees per group
            # The bs id is the variable over which the change is calculated. I.e., the group_year_site.
            df = bootstrap_ids_per_group(
                df, None, "group_year", "tree_id", n_bootstraps_samples
            )
            df["group_year_site"] = (
                df["group_year"] + "_" + df["bs_id"].astype("string")
            )

            print(
                f"... number of groups for which change is calculated: {df.group_year_site.nunique()}\n"
            )
            
            df_gm = mp_calculate_growth_mortality(df, "group_year_site")

    # > Per site method: The method calculates the change for each group_year_site
    # > (filtering for trees per site and sites per year required later)
    elif my_method == "per_site":
        print(
            f"... number of sites for which change is calculated: {df.group_year_site.nunique()}"
        )
        df_gm = mp_calculate_growth_mortality(df, "group_year_site")

    else:
        raise ValueError(f"Invalid method: {my_method}")

    # ! Reattach grouping variable ---------------------------
    if my_method != "direct" and "group" not in df_gm.columns:
        df_gm = df_gm.merge(
            df[["year", "group", "group_year", "group_year_site"]].drop_duplicates(),
            on="group_year_site",
            how="left",
        )

    # Safety check
    expected_shape = (
        df.group_year.nunique()
        if my_method == "direct"
        else df.group_year_site.nunique()
    )

    # if df_gm.shape[0] != expected_shape:
    #     chime.error()
    #     print("🚨🚨🚨 Number of rows in df_gm is not equal to number of sites 🚨🚨🚨")
    #     print(
    #         f" - Expected: {expected_shape} based on grouping variable:"
    #     )
    #     return df_gm, df_counts

    # Save file
    df_gm.to_feather(file_name)
    print(f"... file saved: {file_name}")

    return df_gm, df_counts  # , file_name  # ! RETURN


def cilower(x):
    if x.dropna().empty:
        return None
    else:
        return np.percentile(x.dropna(), [2.5])[0]


def ciupper(x):
    if x.dropna().empty:
        return None
    else:
        return np.percentile(x.dropna(), [97.5])[0]


def iqr(x):
    if x.dropna().empty:
        return None
    else:
        return np.percentile(x.dropna(), [75])[0] - np.percentile(x.dropna(), [25])[0]


def extract_mean_change_over_sites(
    df_in, grouping_var="none", weigh_by_trees_or_sites="trees", verbose=True
):

    if verbose:
        print("\nCalculating mean change over sites...")

    # df_in = df_gm.copy() # For debuggin
    grouping_var = "group_year"

    if grouping_var not in df_in.columns:
        raise ValueError(
            f"group_year not in df!\n - Probably not using a df that has metrics of change? Or false grouping variable inputed?\n - Variable at first location is called: {df_in.columns[0]}"
        )

    # Drop rows that have NAs in them
    df_tmp = df_in.copy()
    # df_tmp = df_tmp.dropna()

    # display(df_tmp)

    vars = []

    # Add only variables which should be aggregated
    for c in df_tmp.columns:
        if c in [grouping_var, "bs_id", "n_a1"]:
            pass

        elif not pd.api.types.is_numeric_dtype(df_tmp[c]):
            # warnings.warn(f"Column {c} is not numeric, dropping it!")
            df_tmp = df_tmp.drop(columns=c)
        else:
            vars += [c]

    # Group by group_id and calculate mean, std, and 95% CI
    # Except for number of plots, which should be summed up
    my_agg = {
        var: ["std", "mean", cilower, ciupper, "median", iqr, "sem"]
        for var in vars
        if var not in [grouping_var, "n_plots"]
    }

    if "bs_id" in df_tmp.columns:
        my_agg["bs_id"] = "nunique"

    # ! Do aggregation
    df_grouped = df_tmp.groupby(grouping_var).agg(my_agg)
    df_grouped.columns = ["_".join((var, stat)) for var, stat in df_grouped.columns]
    df_grouped = df_grouped.reset_index()

    # ! Weighted mean
    # Add calculation of weighing rates by number of trees per plot (no idea how to do this as aggregation)
    # Loop over vars and groups
    if weigh_by_trees_or_sites != "none":
        if weigh_by_trees_or_sites == "trees":
            weight_variable = "n_a1"
        elif weigh_by_trees_or_sites == "sites":
            weight_variable = "n_plots"
        else:
            raise ValueError("weigh_by_trees_or_sites not in ['trees', 'sites']")

        rows = []
        for v in vars:
            for g in df_tmp[grouping_var].unique():
                # Filter the data for the current group
                df_group = df_in[df_in[grouping_var] == g]

                # Calculate weighted mean, var, std
                weighted_mean = np.average(
                    df_group[v], weights=df_group[weight_variable]
                )
                weighted_vari = np.average(
                    (df_group[v] - weighted_mean) ** 2,
                    weights=df_group[weight_variable],
                )
                weighted_std = math.sqrt(weighted_vari)

                # Create a dictionary representing the row and add it to the list
                rows.append(
                    {
                        grouping_var: g,
                        "variable": v,
                        "mean_weighted": weighted_mean,
                        "std_weighted": weighted_std,
                    }
                )

        # Create a DataFrame from the list of rows
        df_weighted = pd.DataFrame(rows)
        # Pivot the DataFrame
        df_weighted_pivot = df_weighted.pivot(
            index=grouping_var,
            columns="variable",
        )
        # Flatten the MultiIndex columns
        df_weighted_pivot.columns = [
            "_".join(col[::-1]).strip() for col in df_weighted_pivot.columns.values
        ]
        # Reset the index
        df_weighted_pivot = df_weighted_pivot.reset_index()

        # ! Merge and return
        df_grouped = pd.merge(
            df_grouped, df_weighted_pivot, how="left", on=grouping_var
        )

    return df_grouped


def extract_mean_change_over_sites_OLD(df_in, grouping_var="none", verbose=True):

    if verbose:
        print("\nCalculating mean change over sites...")

    # df_in = df_gm.copy() # For debuggin
    grouping_var = "group_year"

    if grouping_var not in df_in.columns:
        raise ValueError(
            f"group_year not in df!\n - Probably not using a df that has metrics of change? Or false grouping variable inputed?\n - Variable at first location is called: {df_in.columns[0]}"
        )

    vars = []

    for c in df_in.columns:
        if c in [grouping_var, "bs_id"]:
            pass

        elif not pd.api.types.is_numeric_dtype(df_in[c]):
            # warnings.warn(f"Column {c} is not numeric, dropping it!")
            df_in = df_in.drop(columns=c)
        else:
            vars += [c]

    # Group by group_id and calculate mean, std, and 95% CI
    # Except for number of plots, which should be summed up
    my_agg = {
        var: ["std", "mean", cilower, ciupper, "median", iqr, "sem"]
        for var in vars
        if var not in [grouping_var, "n_plots"]
    }

    if "bs_id" in df_in.columns:
        my_agg["bs_id"] = "nunique"

    # Do aggregation
    df_grouped = df_in.groupby(grouping_var).agg(my_agg)
    df_grouped.columns = ["_".join((var, stat)) for var, stat in df_grouped.columns]
    df_grouped = df_grouped.reset_index()

    # All sites that

    return df_grouped


def count_alive_trees_and_plots_per_group(df, my_grouping):
    df_nplots_per_group = (
        df.query("tree_state_1 == 'alive'")[[my_grouping, "idp"]]
        .drop_duplicates()
        .value_counts(my_grouping)
        .to_frame()
        .reset_index(names=my_grouping)
        .rename(columns={"count": "n_plots"})
    )

    df_trees_per_group = (
        df.query("tree_state_1 == 'alive'")[[my_grouping, "tree_id"]]
        .value_counts(my_grouping)
        .to_frame()
        .reset_index(names=my_grouping)
        .rename(columns={"count": "n_trees"})
    )

    df_counts = pd.merge(df_nplots_per_group, df_trees_per_group, on=my_grouping)
    df_counts = df_counts.rename(columns={my_grouping: "group"})

    # Drop groups where there are no sites for
    df_counts = df_counts.query("n_plots > 0")
    return df_counts


def filter_for_enough_trees_and_sites_per_group(
    df_in,
    min_trees_per_site,
    min_sites_per_group_year,
    verbose=True,
):
    # todo Alternative for later: Only keep sites where desired group is dominant (needs separate assessment...)

    if verbose:
        print(
            f"\nFilter for sites that have {min_trees_per_site}+ trees and groups that have {min_sites_per_group_year}+ sites... "
        )

    # Input
    df = df_in.copy()
    min_trees_per_site = max(
        min_trees_per_site, 1
    )  # Safety check to have at least 1 tree per site
    min_sites_per_group_year = max(
        min_sites_per_group_year, 1
    )  # Safety check to have at least 1 site per group_year

    # Remove sites where not enough tree of the group where present
    df = df.query("n_a1 >= @min_trees_per_site")

    # Get group_years with enough sites
    group_year_to_keep = (
        df[["group_year", "group_year_site"]]
        .drop_duplicates()
        .value_counts("group_year")
        .reset_index()
        .query("count >= @min_sites_per_group_year")["group_year"]
    )

    # Filter for group_years with enough sites
    df = df.query("group_year in @group_year_to_keep")

    # Calculate number of trees and sites per group (over all years, for reporting in the legend)
    tree_count = (
        df[["group", "n_a1"]]
        .groupby("group")
        .sum()
        .reset_index()
        .rename(columns={"n_a1": "n_trees"})
    )

    site_count = (
        df[["group", "group_year_site"]]
        .drop_duplicates()
        .groupby("group")
        .size()
        .reset_index()
        .rename(columns={0: "n_plots"})
    )

    df_counts = pd.merge(tree_count, site_count, on="group")

    if verbose:
        original_rows = len(df_in)
        reduced_rows = len(df)
        reduction_percentage = (original_rows - reduced_rows) / original_rows * 100
        print(
            f"... the original dataframe was reduced by {reduction_percentage:.2f}% (from {original_rows} to {reduced_rows} sites)"
        )

    return df, df_counts

def normalize_to_first_year(
    df, center=True, normalize=True, min_trees_first_year_group=10
    ):
    
    print(f"... normalizing data by: Centering = {center}, Normalizing = {normalize}")

    # Identify numeric columns that are not to be processed
    exclude_columns = {"group", "group_year", "group_year_site", "year"}
    numeric_columns = [
        col
        for col in df.columns
        if col not in exclude_columns
        and not col.startswith("n_")
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    # Initialize an empty DataFrame to store results
    df_edited = pd.DataFrame()

    # Group by 'group' and process each group
    for group_name, group_df in df.groupby("group"):
        group_df = group_df.copy()

        # ! DEBUG
        # if group_name == "A&Quercus ilex":
            # group_df.query(f"group_year == '{group_name}2009'").n_plots.unique()[0]
            # pass

        # Check if first year is 2009 and if there are enough trees in the first year
        first_year = group_df["year"].min()
        ntrees_first_year_group = (
            group_df.query(f"group_year == '{group_name}_2010'")
            .n_plots.unique()
            .tolist()
        )

        if int(first_year) != 2009 or ntrees_first_year_group.__len__() == 0:
            # Set first year mean values to na
            first_year_means = pd.Series(
                [np.nan] * len(numeric_columns), index=numeric_columns
            )
        elif ntrees_first_year_group[0] < min_trees_first_year_group:
            # Set first year mean values to na
            first_year_means = pd.Series(
                [np.nan] * len(numeric_columns), index=numeric_columns
            )
        else:
            # Get the first year mean values
            first_year_means = group_df[group_df["year"] == first_year][
                numeric_columns
            ].mean()

        # Apply centering and normalization
        if center:
            group_df[numeric_columns] = group_df[numeric_columns] - first_year_means
        if normalize:
            group_df[numeric_columns] = (
                group_df[numeric_columns].div(first_year_means) * 100
            )

        df_edited = pd.concat([df_edited, group_df], axis=0)

    # Reset index
    df_edited.reset_index(drop=True, inplace=True)
    # Replace infinite values with NaN
    df_edited.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Fill NA values with 0
    # df_edited.fillna(0, inplace=True)

    # Ensure the shape remains the same
    if df_edited.shape != df.shape:
        raise ValueError(
            "Center and scaling did not work properly! Resulting df has not the same shape as the original df."
        )

    return df_edited

def plot_temporal_trend(
    df_grouped,
    df_counts,
    my_method,
    my_grouping,
    my_metric,
    uncertainty_representation,
    uncertainty_variable,
    aggregation_variable,
    top_n_metric,
    facet_label_trees_or_sites="trees",
    set_ylim=None,  # None or [min, max]
    center_to_first_year=True,
    normalize_to_first_year=False,
    top_n_groups=None,
    n_bootstraps=None,
    save_plot=True,
    file_suffix="",
):

    from matplotlib import patches as mpatches

    print("\nPlotting temporal trend...")

    # ! Checks ---------------------------------------------------------------
    if my_method not in ["direct", "direct_bs", "per_site", "per_site_bs"]:
        raise ValueError(f"Invalid method: {my_method}")

    if my_grouping not in [
        "gre",
        "ser",
        "reg",
        "dep",
        "genus_lat",
        "species_lat",
        "species_lat2",
        "species_lat_short",
        "tree_height_class",
        "tree_circumference_class",
    ]:
        raise ValueError(f"Invalid grouping: {my_grouping}")

    if my_method == "direct":
        if my_metric not in df_grouped.columns:
            raise ValueError(f"Missing metric: {my_metric}")
    else:

        if f"{my_metric}_{aggregation_variable}" not in df_grouped.columns:
            raise ValueError(f"Missing metric: {my_metric}_{aggregation_variable}")

        if uncertainty_variable not in ["std", "sem", "iqr", "ci", "std_weighted"]:
            raise ValueError(f"Missing uncertainty_variable: {uncertainty_variable}")

        if aggregation_variable == "median" and uncertainty_variable != "iqr":
            uncertainty_variable = "iqr"
            print(
                f"... aggregation_variable is median but uncertainty_variable is not iqr! -> Setting it to iqr."
            )

        if aggregation_variable == "mean" and uncertainty_variable == "iqr":
            print(
                f"... aggregation_variable is median mean uncertainty_variable is iqr! -> Setting it to std instead."
            )

        if uncertainty_representation not in ["band", "bar", "none"]:
            raise ValueError(
                f"Invalid uncertainty_representation: {uncertainty_representation}"
            )

    if top_n_metric not in ["most_observations", "highest_final_mortality"]:
        raise ValueError(f"Invalid top_n_metric: {top_n_metric}")

    # ! Prepare dataset for plotting ---------------------------------------------

    # Get data and clean names
    df = df_grouped.copy()
    df["group"] = df["group_year"].str.split("_", expand=True)[0]
    df["year"] = df["group_year"].str.split("_", expand=True)[1]

    # Fix to show second visit year
    df["year"] = df["year"].astype(int) + 5  # 5 years between visits
    df["year"] = df["year"].astype(str)

    # Add group from group_year
    df["group"] = df["group_year"].str.split("_", expand=True)[0]

    # Add information on how many trees and sites per group
    df = df.drop(columns=["n_plots"], errors="ignore").merge(df_counts, on="group")

    # Add number of trees or sites, ticks to separate thousands
    df["n_trees"] = (
        df["n_trees"].astype(int).apply(lambda x: format(x, ",").replace(",", "'"))
    )
    df["n_plots"] = (
        df["n_plots"].astype(int).apply(lambda x: format(x, ",").replace(",", "'"))
    )

    # ! Get dictionary for metric and species ------------------------------------

    # Attach facet label
    if facet_label_trees_or_sites == "trees":
        facet_label = "Trees"
        df["facet"] = df["group"] + "  (" + df["n_trees"].astype(str) + ")"
    else:
        facet_label = "Sites"
        df["facet"] = df["group"] + "  (" + df["n_plots"].astype(str) + ")"

    # Make sure year is int
    df["year"] = df["year"].astype(int)

    # ! Set target, lower and upper ----------------------------------------------
    # Checks
    if my_method in ["per_site", "direct_bs", "per_site_bs"]:

        if "mean" in aggregation_variable:

            df["target"] = df[f"{my_metric}_{aggregation_variable}"]

            if uncertainty_variable in ["std", "std_weighted", "sem"]:
                df["lower"] = (
                    df[f"{my_metric}_{aggregation_variable}"]
                    - df[f"{my_metric}_{uncertainty_variable}"]
                )
                df["upper"] = (
                    df[f"{my_metric}_{aggregation_variable}"]
                    + df[f"{my_metric}_{uncertainty_variable}"]
                )
            elif uncertainty_variable == "ci":
                df["lower"] = df[f"{my_metric}_cilower"]
                df["upper"] = df[f"{my_metric}_ciupper"]

        elif aggregation_variable == "median":
            df["target"] = df[f"{my_metric}_median"]
            df["lower"] = df[f"{my_metric}_median"] - df[f"{my_metric}_iqr"] / 2
            df["upper"] = df[f"{my_metric}_median"] + df[f"{my_metric}_iqr"] / 2

    elif my_method == "direct":
        df["target"] = df[f"{my_metric}"]
        df["lower"] = np.nan
        df["upper"] = np.nan

    else:
        raise ValueError(f"Invalid method: {my_method}")

    df = df[["facet", "group", "year", "target", "lower", "upper"]]

    # ! Subset for top n groups ---------------------------------------------------
    if top_n_groups is not None:

        df_counts = df_counts.drop_duplicates()

        if top_n_metric == "most_observations":
            top_groups = (
                df_counts.sort_values("n_trees", ascending=False)
                .head(top_n_groups)["group"]
                .tolist()
            )
            print(
                f"... selecting {top_n_groups} most common groups with most trees: {top_groups}"
            )
            df = df.query("group in @top_groups")

        elif top_n_metric == "highest_final_mortality":
            top_groups = (
                df.query("year == @df['year'].max()")
                .sort_values(by="target", ascending=False)
                .head(top_n_groups)
                .group.values.tolist()
            )

            print(
                f"... selecting {top_n_groups} highest mortality groups with most trees: {top_groups}"
            )

            df = df.query("group in @top_groups")

    # Order factor levels to be alphabetically sorted
    df.loc[:, "facet"] = pd.Categorical(
        df["facet"], categories=sorted(df["facet"].unique())
    )

    # ! Plot -----------------------------------------------------------------------
    # Get layout
    change_to_2010 = None
    if center_to_first_year or normalize_to_first_year:
        if center_to_first_year and not normalize_to_first_year:
            change_to_2010 = "absolute"
        elif normalize_to_first_year:
            change_to_2010 = "relative"

    my_dict = figure_dictionary_for_variable(
        my_metric, "all", change_to_2010=change_to_2010
    )
    my_title = my_dict["legend"]
    # display(my_dict)

    if my_grouping == "gre":
        my_legend = "Greater Ecoregion ($\it{N}$ " + f"{facet_label})"
    elif my_grouping == "ser":
        my_legend = "Smaller Ecoregion ($\it{N}$ " + f"{facet_label})"
    elif my_grouping == "reg":
        my_legend = "Admin. Region ($\it{N}$ " + f"{facet_label})"
    elif my_grouping == "dep":
        my_legend = "Department ($\it{N}$ " + f"{facet_label})"
    elif my_grouping == "genus_lat":
        my_legend = "Genus ($\it{N}$ " + f"{facet_label})"
    elif my_grouping == "species_lat":
        my_legend = "Species ($\it{N}$ " + f"{facet_label})"
    elif my_grouping == "tree_height_class":
        my_legend = "Height ($\it{N}$ " + f"{facet_label})"
    elif my_grouping == "tree_circumference_class":
        my_legend = "Circumference Cateogry ($\it{N}$ " + f"{facet_label})"

    # Setting plot size and style
    plt.figure(figsize=(9, 9))
    sns.set_style("white")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"

    # If centered or normalized, add dotted line at 0
    if center_to_first_year or normalize_to_first_year:
        if center_to_first_year:
            plt.axhline(0, color="black", linestyle="--", linewidth=1)
        elif normalize_to_first_year:
            plt.axhline(100, color="black", linestyle="--", linewidth=1)

    # Creating a color palette with 'cubehelix' scheme
    colors = sns.color_palette("cubehelix", df.facet.nunique())

    # Plotting error band or error bars, line and scatter plot
    for color, (facet, group) in zip(colors, df.groupby("facet", observed=False)):
        if uncertainty_representation == "band":
            plt.fill_between(
                group["year"], group["lower"], group["upper"], color=color, alpha=0.3
            )
        elif uncertainty_representation == "bar":
            plt.errorbar(
                group["year"],
                group["target"],
                yerr=[
                    group["target"] - group["lower"],
                    group["upper"] - group["target"],
                ],
                fmt="none",
                ecolor=color,
                capsize=5,
                elinewidth=1,
            )
        elif uncertainty_representation == "none":
            pass

        plt.plot(group["year"], group["target"], label=f"{facet} Target", color=color)
        plt.scatter(group["year"], group["target"], color=color, marker="o")

    # Setting labels and title with bold font
    plt.xlabel("Year of Revisit", fontweight="bold")
    plt.ylabel(my_title, fontweight="bold")
    plt.title("", fontweight="bold")

    # Creating a custom legend with two rows and five columns
    legend_elements = [
        mpatches.Patch(facecolor=color, edgecolor="none", label=facet)
        for color, facet in zip(colors, df["facet"].unique())
    ]

    if my_grouping in ["species_lat", "species_lat2"] or top_n_groups > 10:

        legend = plt.legend(
            handles=legend_elements,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            ncol=1,
            title=my_legend,
        )

    else:
        legend = plt.legend(
            handles=legend_elements,
            bbox_to_anchor=(0.5, -0.2),
            loc="upper center",
            ncol=5,
            title=my_legend,
        )

    # Customizing legend appearance
    legend.get_frame().set_linewidth(0)
    legend.get_frame().set_facecolor("none")
    plt.setp(legend.get_title(), weight="bold")
    legend._legend_box.align = "center"

    # Setting the title font weight to bold
    legend.set_title(my_legend, prop={"weight": "bold"})

    if set_ylim is not None:
        if len(set_ylim) != 2:
            raise ValueError("ylim must be a list of two values")
        plt.ylim(set_ylim)

    # Removing top and right borders for a cleaner look
    sns.despine()

    # Showing the plot
    # plt.show()

    if save_plot:
        # ! Save plot
        today = datetime.datetime.today().strftime("%Y-%m-%d")
        my_dir = f"temporal_trends/{today}/{my_metric}/{my_grouping}/{aggregation_variable}-{uncertainty_variable}-{uncertainty_representation}/all_in_one/"
        if file_suffix != "":
            my_dir = my_dir.replace("temporal_trends", f"specific_runs/{file_suffix} - {today}/temporal_trends")
        
        if not os.path.exists(my_dir):
            os.makedirs(my_dir, exist_ok=True)

        fname = f"{my_dir}/fig-{my_method}-groups_of_{top_n_metric}-"
        if top_n_groups is None:
            fname += f"all_groups"
        else:
            fname += f"top-{top_n_groups}-groups"

        if "_bs" in my_method:
            fname += f"_{n_bootstraps}-bootstraps"

        if center_to_first_year:
            fname += "_centered"
        if normalize_to_first_year:
            fname += "_normalized"

        fname += ".png"
        plt.savefig(fname, dpi=300, bbox_inches="tight")

        print(f"... plot saved: {fname}")


def plot_temporal_trend_facet(
    df_grouped,
    df_counts,
    my_method,
    my_grouping,
    my_metric,
    uncertainty_representation,
    uncertainty_variable,
    aggregation_variable,
    top_n_metric,
    facet_label_trees_or_sites="trees",
    share_y=False,
    set_ylim=None,  # None or [min, max]
    center_to_first_year=True,
    normalize_to_first_year=False,
    top_n_groups=None,
    n_bootstraps=None,
    save_plot=False,
    file_suffix="",
):

    print("\nPlotting temporal trend...")

    # ! Checks ---------------------------------------------------------------
    if my_method not in ["direct", "direct_bs", "per_site", "per_site_bs"]:
        raise ValueError(f"Invalid method: {my_method}")

    if my_grouping not in [
        "gre",
        "ser",
        "reg",
        "dep",
        "genus_lat",
        "species_lat",
        "species_lat2",
        "species_lat_short",
        "tree_height_class",
        "tree_circumference_class",
    ]:
        raise ValueError(f"Invalid grouping: {my_grouping}")

    if my_method == "direct":
        if my_metric not in df_grouped.columns:
            raise ValueError(f"Missing metric: {my_metric}")
    else:
        if f"{my_metric}_{aggregation_variable}" not in df_grouped.columns:
            raise ValueError(f"Missing metric: {my_metric}_{aggregation_variable}")

        if uncertainty_variable not in ["std", "sem", "iqr", "ci", "std_weighted"]:
            raise ValueError(f"Missing uncertainty_variable: {uncertainty_variable}")

        if aggregation_variable == "median" and uncertainty_variable != "iqr":
            uncertainty_variable = "iqr"
            print(
                f"... aggregation_variable is median but uncertainty_variable is not iqr! -> Setting it to iqr."
            )

        if aggregation_variable == "mean" and uncertainty_variable == "iqr":
            print(
                f"... aggregation_variable is median mean uncertainty_variable is iqr! -> Setting it to std instead."
            )

        if uncertainty_representation not in ["band", "bar", "none"]:
            raise ValueError(
                f"Invalid uncertainty_representation: {uncertainty_representation}"
            )

    if top_n_metric not in ["most_observations", "highest_final_mortality"]:
        raise ValueError(f"Invalid top_n_metric: {top_n_metric}")
    
    if not share_y:
        set_ylim = None

    # ! Prepare dataset for plotting ---------------------------------------------

    # Get data and clean names
    df = df_grouped.copy()
    df["group"] = df["group_year"].str.split("_", expand=True)[0]
    df["year"] = df["group_year"].str.split("_", expand=True)[1]

    # Fix to show second visit year
    df["year"] = df["year"].astype(int) + 5  # 5 years between visits
    df["year"] = df["year"].astype(str)

    # Add group from group_year
    df["group"] = df["group_year"].str.split("_", expand=True)[0]

    # Add information on how many trees and sites per group
    df = df.drop(columns=["n_plots"], errors="ignore").merge(df_counts, on="group")

    # Add number of trees or sites, ticks to separate thousands
    df["n_trees"] = (
        df["n_trees"].astype(int).apply(lambda x: format(x, ",").replace(",", "'"))
    )
    df["n_plots"] = (
        df["n_plots"].astype(int).apply(lambda x: format(x, ",").replace(",", "'"))
    )

    # ! Get dictionary for metric and species ------------------------------------
    # Clean group names for some variables
    if my_grouping == "tree_height_class":
        df["group"] = df["group"].replace(
            {
                "0-10": "0-10m",
                "10-15": "10-15m",
                "15-20": "15-20m",
                "20-25": "20-25m",
                "25+": ">25m",
            }
        )
    elif my_grouping == "gre":
        df["group"] = df["group"].replace(
            {
                "A": "Grand Ouest",
                "B": "Centre Nord",
                "C": "Grand Est",
                "D": "Vosges",
                "E": "Jura",
                "F": "Sud-Ouest",
                "G": "Massif Central",
                "H": "Alpes",
                "I": "Pyrénées",
                "J": "Méditerranée",
                "K": "Corse",
            }
        )
    elif my_grouping == "reg":
        df["group"] = df["group"].replace(
            {
                "11": "Île-de-France",
                "24": "Centre-Val de Loire",
                "27": "Bourgogne-Franche-Comté",
                "28": "Normandie",
                "32": "Hauts-de-France",
                "44": "Grand Est",
                "52": "Pays de la Loire",
                "53": "Bretagne",
                "75": "Nouvelle-Aquitaine",
                "76": "Occitanie",
                "84": "Auvergne-Rhône-Alpes",
                "93": "Provence-Alpes-Côte-d'Azur",
                "94": "Corse",
            }
        )

    # If grouping is species, shorten species name
    if my_grouping in ["species_lat", "species_lat2"]:
        df["group"] = (
            df["group"].str.split(" ").str[0].str[:1].str.title()
            + ". "
            + df["group"].str.split(" ").str[1]
        )
        df = df.replace({np.nan: "Populus spp."})

    # Attach facet label
    if facet_label_trees_or_sites == "trees":
        facet_label = "Trees"
        df["facet"] = df["group"] + "  (N = " + df["n_trees"].astype(str) + ")"
    else:
        facet_label = "Sites"
        df["facet"] = df["group"] + "  (N = " + df["n_plots"].astype(str) + ")"

    # Make sure year is int
    df["year"] = df["year"].astype(int)

    # ! Set target, lower and upper ----------------------------------------------
    # Checks
    if my_method in ["per_site", "direct_bs", "per_site_bs"]:
        if "mean" in aggregation_variable:
            df["target"] = df[f"{my_metric}_{aggregation_variable}"]

            if uncertainty_variable in ["std", "std_weighted", "sem"]:
                df["lower"] = (
                    df[f"{my_metric}_{aggregation_variable}"]
                    - df[f"{my_metric}_{uncertainty_variable}"]
                )
                df["upper"] = (
                    df[f"{my_metric}_{aggregation_variable}"]
                    + df[f"{my_metric}_{uncertainty_variable}"]
                )
            elif uncertainty_variable == "ci":
                df["lower"] = df[f"{my_metric}_cilower"]
                df["upper"] = df[f"{my_metric}_ciupper"]

        elif aggregation_variable == "median":
            df["target"] = df[f"{my_metric}_median"]
            df["lower"] = df[f"{my_metric}_median"] - df[f"{my_metric}_iqr"] / 2
            df["upper"] = df[f"{my_metric}_median"] + df[f"{my_metric}_iqr"] / 2

    elif my_method == "direct":
        df["target"] = df[f"{my_metric}"]
        df["lower"] = np.nan
        df["upper"] = np.nan

    else:
        raise ValueError(f"Invalid method: {my_method}")

    df = df[["facet", "group", "year", "target", "lower", "upper"]]

    # ! Subset for top n groups ---------------------------------------------------
    if top_n_groups is not None:
        df_counts = df_counts.drop_duplicates()
        if top_n_metric == "most_observations":
            top_groups = (
                df_counts.sort_values("n_trees", ascending=False)
                .head(top_n_groups)["group"]
                .tolist()
            )
            df = df.query("group in @top_groups")
        elif top_n_metric == "highest_final_mortality":
            top_groups = (
                df.query("year == @df['year'].max()")
                .sort_values(by="target", ascending=False)
                .head(top_n_groups)
                .group.values.tolist()
            )
            df = df.query("group in @top_groups")

    # Order factor levels to be alphabetically sorted
    df.loc[:, "facet"] = pd.Categorical(
        df["facet"], categories=sorted(df["facet"].unique())
    )

    # ! Plot -----------------------------------------------------------------------
    # sns.set_style("white")
    # plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    sharey_axis = not center_to_first_year
    g = sns.FacetGrid(
        df,
        col="facet",
        col_wrap=5,
        # sharey=sharey_axis,
        sharey=share_y,
        sharex=True,
        height=3.75,
    )
    g.map_dataframe(
        sns.lineplot,
        x="year",
        y="target",
        zorder=2,
        marker="o",
        markers=True,
    )

    # Add uncertainty representation
    if uncertainty_representation == "band":
        g.map_dataframe(
            lambda data, **kwargs: plt.fill_between(
                data["year"], data["lower"], data["upper"], alpha=0.3, zorder=1
            )
        )
    elif uncertainty_representation == "bar":
        g.map_dataframe(
            lambda data, **kwargs: plt.errorbar(
                data["year"],
                data["target"],
                yerr=[data["target"] - data["lower"], data["upper"] - data["target"]],
                fmt="none",
                capsize=5,
                zorder=1,
            )
        )

    # If centered to first year, add a line at 0
    if center_to_first_year:
        g.map(
            lambda **kwargs: plt.axhline(
                0, color="grey", linestyle="dotted", linewidth=1, zorder=0
            )
        )

    # Set ylim if provided
    if set_ylim is not None:
        if len(set_ylim) != 2:
            raise ValueError("ylim must be a list of two values")
        g.set(ylim=set_ylim)

    # if not center_to_first_year:
    # if center_to_first_year:
    #     max_value = df[["upper", "lower"]].abs().max().astype(int).max()
    #     g.set(ylim=(-max_value, max_value))
    # else:
    #     max_value = df[["upper"]].abs().max().astype(int)
    #     g.set(ylim=(0, max_value))

    # Rotate x-tick labels if y-axis is shared because too tight else
    if share_y:
        for ax in g.axes.flat:
            for label in ax.get_xticklabels():
                label.set_rotation(45)

    # ! Labels and positions
    # Remove labels from each facet
    g.set_ylabels("")
    g.set_xlabels("")

    # Give one xlabel to the whole plot
    if center_to_first_year and normalize_to_first_year:
        mytitle = f"Change in tree mortality since 2015 (%)"
    elif center_to_first_year and not normalize_to_first_year:
        mytitle = f"Change in tree mortality since 2015 (%-trees yr$^{-1}$)"
    elif not center_to_first_year and normalize_to_first_year:
        mytitle = f"Tree Mortality relative to 2015 (%)"
    else:
        mytitle = f"Tree Mortality (%-trees yr$^{-1}$)"

    if my_grouping == "species_lat":
        g.fig.text(
            0.5,
            0.04,
            "Year",
            ha="center",
            va="center",
            fontsize=18,
            weight="bold",
        )
        g.fig.text(
            0.04,
            0.5,
            mytitle,
            ha="center",
            va="center",
            rotation="vertical",
            fontsize=18,
            weight="bold",
        )

        g.fig.suptitle(
            f"Tree mortality trends of the 20 most common species in France",
            fontsize=18,
            weight="bold",
        )
    elif my_grouping == "tree_height_class":
        g.fig.text(
            0.5,
            -0.09,
            "Year",
            ha="center",
            va="center",
            fontsize=18,
            weight="bold",
        )

        g.fig.text(
            0.04,
            0.5,
            mytitle,
            ha="center",
            va="center",
            rotation="vertical",
            fontsize=18,
            weight="bold",
        )

        g.fig.suptitle(
            f"Tree mortality trends of the 20 most common species in France, separated by height class",
            fontsize=18,
            weight="bold",
            x=0.5,
            y=1.15,
        )
        
    elif my_grouping == "gre":
        g.fig.text(
            0.5,
            0,
            "Year",
            ha="center",
            va="center",
            fontsize=18,
            weight="bold",
        )

        g.fig.text(
            0.04,
            0.5,
            mytitle,
            ha="center",
            va="center",
            rotation="vertical",
            fontsize=18,
            weight="bold",
        )

        g.fig.suptitle(
            f"Tree mortality trends of the 20 most common species in France, separated by ecological region",
            fontsize=18,
            weight="bold",
            x=0.5,
            y=1,
        )

    elif my_grouping == "reg":
        g.fig.text(
            0.5,
            0,
            "Year",
            ha="center",
            va="center",
            fontsize=18,
            weight="bold",
        )

        g.fig.text(
            0.04,
            0.5,
            mytitle,
            ha="center",
            va="center",
            rotation="vertical",
            fontsize=18,
            weight="bold",
        )

        g.fig.suptitle(
            f"Tree mortality trends of the 20 most common species in France, separated by administrative region",
            fontsize=18,
            weight="bold",
            x=0.5,
            y=1,
        )

    # ! Subtitles
    # Subplot titles
    g.set_titles("\n{col_name}")
    for ax in g.axes.flat:
        ax.set_title(ax.get_title(), fontsize=13, weight="bold")
    
    # ! Ticks
    # Increase tick size
    for ax in g.axes.flat:
        # Set to 4 ticks from min to max
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        
        # Increase size
        ax.tick_params(axis="both", which="major", labelsize=12)

    g.fig.subplots_adjust(bottom=0.09, left=0.075, top=0.925, hspace=0.2)

    # Showing the plot
    plt.show()
    # plt.close()

    # ! Save plot
    if save_plot:
        today = datetime.datetime.today().strftime("%Y-%m-%d")
        my_dir = f"temporal_trends/{today}/{my_metric}/{my_grouping}/{aggregation_variable}-{uncertainty_variable}-{uncertainty_representation}/facet-sharey_{share_y}/"
        
        if file_suffix != "":
            my_dir = my_dir.replace("temporal_trends", f"specific_runs/{file_suffix} - {today}/temporal_trends")
        
        if not os.path.exists(my_dir):
            os.makedirs(my_dir, exist_ok=True)

        fname = f"{my_dir}/fig-{my_method}-groups_of_{top_n_metric}-"
        if top_n_groups is None:
            fname += f"all_groups"
        else:
            fname += f"top-{top_n_groups}-groups"

        if "_bs" in my_method:
            fname += f"_{n_bootstraps}-bootstraps"

        fname += f"-centered_{center_to_first_year}-normalized_{normalize_to_first_year}"
        fname += ".png"
        # base_size = 3.5
        # base_y = base_size * 4
        # base_x = base_size * 5
        # g.fig.set_size_inches((base_x,base_y))
        g.savefig(fname, bbox_inches="tight")

        print(f"... plot saved: {fname}")

    plt.close()
    # plt.show()
    # return g

    # Loop over unique facets and create individual plots
    unique_facets = df["facet"].unique()
    for facet_value in unique_facets:
        facet_data = df[df["facet"] == facet_value]

        plt.figure(figsize=(6, 6))
        sns.lineplot(data=facet_data, x="year", y="target", marker="o", markers=True)

        if uncertainty_representation == "band":
            plt.fill_between(
                facet_data["year"], facet_data["lower"], facet_data["upper"], alpha=0.3
            )
        elif uncertainty_representation == "bar":
            plt.errorbar(
                facet_data["year"],
                facet_data["target"],
                yerr=[
                    facet_data["target"] - facet_data["lower"],
                    facet_data["upper"] - facet_data["target"],
                ],
                fmt="none",
                capsize=5,
            )

        if center_to_first_year:
            plt.axhline(0, color="grey", linestyle="dotted", linewidth=1)

        if set_ylim is not None:
            plt.ylim(set_ylim)

        if my_grouping == "tree_height_class":
            extra_txt = "height group "
        else:
            extra_txt = ""

        plt.xlabel("Year", weight="bold")
        plt.ylabel(mytitle, weight="bold")
        plt.title(f"General trend for {extra_txt}{facet_value}", weight="bold", fontsize=13)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)

        # Save plot
        filename = f"{facet_value.split('(')[0].strip()}-centered_{center_to_first_year}-normalized-{normalize_to_first_year}.png"
        output_path = os.path.join(my_dir, filename)
        plt.savefig(output_path, bbox_inches="tight")
        print(f"... plot saved: {output_path}")
        # plt.show()
        plt.close()
        # break

def start_to_finish_temporal_plot(kwargs, return_before_plotting=False):

    # Calculate change
    df_gm, df_counts = calculate_change_per_group_and_year(
        df=kwargs["df"],
        my_grouping=kwargs["my_grouping"],
        top_n_groups=kwargs["top_n_groups"],
        my_method=kwargs["my_method"],
        n_bootstraps_samples=kwargs["n_bootstraps_samples"],
        load_from_file=kwargs["load_from_file"],
        file_suffix=kwargs["file_suffix"],
    )

    # ! Calculate mean change
    # For direct method, mean is implicitly calculated
    if kwargs["my_method"] == "direct":
        df_grouped = df_gm.copy()

    # For direct bootstrapped method, mean must be calculated
    elif kwargs["my_method"] == "direct_bs":

        # Calculate mean change over sites
        print(
            f"\n... chosen method is {kwargs['my_method']}: calculating mean change over sites or bootstrapped groups directly"
        )

        if kwargs["center_to_first_year"] or kwargs["normalize_to_first_year"]:
            df_gm = normalize_to_first_year(
                df_gm,
                center=kwargs["center_to_first_year"],
                normalize=kwargs["normalize_to_first_year"],
            )

        df_grouped = extract_mean_change_over_sites(
            df_gm,
            grouping_var="group_year",
            weigh_by_trees_or_sites=kwargs["weigh_by_sites_or_trees"],
            verbose=False,
        )

    elif kwargs["my_method"] in ["per_site", "per_site_bs"]:
        print("... calculating mean change over bootstrapped groups")

        if kwargs["reduce_to_dominant_sites"]:
            # Extract site information for merging
            df_gm["site"] = (
                df_gm["group_year_site"].str.split("_", expand=True)[2].astype(int)
            )
            # Save shape for reporting
            shape_before = df_gm.shape
            # Load dominance dataset
            df_spec = attach_or_load_predictor_dataset("forest_biodiv_and_gini")
            # Fix variable names
            df_spec = df_spec.rename(
                columns={"idp": "site", f"dom_{kwargs['my_grouping']}": "group"}
            )[["site", "group"]]
            # Merge to keep only where given group is dominant, drop NA (e.g. group dominant but without mortality)
            df_spec = df_spec.merge(df_gm, how="left", on=["site", "group"])
            df_spec = df_spec.dropna(subset=["group_year_site"])
            # Finally overwrite df_gm
            df_gm = df_spec.copy()
            df_gm["site"] = df_gm["site"].astype(str)
            # Report
            rows_removed = shape_before[0] - df_gm.shape[0]
            perc_removed = rows_removed / shape_before[0] * 100
            perc_removed = round(perc_removed, 2)
            print(
                f"... removing entries where a group is not dominant at a site: Removed {rows_removed} sites (={perc_removed}%)"
            )

        # Filter out sites as desired and re-calculate df_counts
        df_gm, df_counts = filter_for_enough_trees_and_sites_per_group(
            df_gm,
            min_trees_per_site=kwargs["min_trees_per_site"],
            min_sites_per_group_year=kwargs["min_sites_per_group_year"],
        )

        # Filter out sites where no tree was cut or died, if looking at mortality
        shape_before = df_gm.shape
        if "mort_nat" in kwargs["my_metric"]:
            df_gm = df_gm.query("n_ad > 0")
        elif "mort_cut" in kwargs["my_metric"]:
            df_gm = df_gm.query("n_ac > 0")
        elif "mort_tot" in kwargs["my_metric"]:
            df_gm = df_gm.query("(n_ac + n_ad) > 0")

        rows_removed = shape_before[0] - df_gm.shape[0]
        perc_removed = rows_removed / shape_before[0] * 100
        perc_removed = round(perc_removed, 2)

        print(
            f"... removed sites without mortality. Removed {rows_removed} sites (={perc_removed}%)"
        )

        if kwargs["center_to_first_year"] or kwargs["normalize_to_first_year"]:
            df_gm = normalize_to_first_year(
                df_gm,
                center=kwargs["center_to_first_year"],
                normalize=kwargs["normalize_to_first_year"],
            )

        # Do bootstrap over plots if needed
        if kwargs["my_method"] == "per_site":
            df_grouped = extract_mean_change_over_sites(
                df_gm,
                grouping_var="group_year",
                weigh_by_trees_or_sites=kwargs["weigh_by_sites_or_trees"],
                verbose=False,
            )

        elif kwargs["my_method"] == "per_site_bs":

            if kwargs["n_bootstraps_samples"] > 501:
                print("... bootstrapping per sites more than 500 times overloads ram")
                print(
                    "... so looping over each group-year and calculating change separately before merging"
                )

                df_all = pd.DataFrame()

                for i_groupyear in tqdm(df_gm.group_year.unique()):

                    df_loop = bootstrap_ids_per_group(
                        df_gm.query("group_year == @i_groupyear"),
                        None,  # kwargs["my_metric"],
                        "group_year",
                        "group_year_site",
                        n_bootstraps=kwargs["n_bootstraps_samples"],
                        verbose=False,
                    )

                    # Calculate mean change over sites
                    df_loop = extract_mean_change_over_sites(
                        df_loop,
                        grouping_var="group_year",
                        weigh_by_trees_or_sites=kwargs["weigh_by_sites_or_trees"],
                        verbose=False,
                    )
                    df_loop = df_loop.reset_index(drop=True)

                    df_all = pd.concat([df_all, df_loop], axis=0)

                df_grouped = df_all.reset_index(drop=True).copy()

            else:
                # If less than 500 samples, do bs directly

                df_gm = bootstrap_ids_per_group(
                    df_gm,
                    None,  # kwargs["my_metric"],
                    "group_year",
                    "group_year_site",
                    n_bootstraps=kwargs["n_bootstraps_samples"],
                    verbose=True,
                )

                # Calculate mean change over sites
                df_grouped = extract_mean_change_over_sites(
                    df_gm,
                    grouping_var="group_year",
                    weigh_by_trees_or_sites=kwargs["weigh_by_sites_or_trees"],
                    verbose=False,
                )
        else:
            raise ValueError(f"Invalid method: {kwargs['my_method']}")

    # ! Save grouped data
    today = datetime.datetime.today().strftime("%Y-%m-%d")
    my_dir = f"data_of_change/{kwargs['my_method']}/"
    if kwargs['file_suffix'] != "":
        my_dir = my_dir.replace("data_of_change", f"specific_runs/{kwargs['file_suffix']} - {today}/data_of_change")
    
    for i, igroup in enumerate(kwargs['my_grouping']):
        if i == 0:
            my_dir += f"/{igroup}"
        else:
            my_dir += f"&{igroup}"
        
    my_dir += f"/centered_{kwargs['center_to_first_year']}-normalized_{kwargs['normalize_to_first_year']}"

    os.makedirs(my_dir, exist_ok=True)
    file_name = f"{my_dir}/"

    file_name += f"/subset-top-{kwargs['top_n_groups']}"

    if kwargs['my_method'] == "direct_bs":
        file_name += f"_bs-{kwargs['n_bootstraps_samples']}"

    file_name += ".csv"
    df_grouped.to_csv(file_name, index=False)
    print(f"... saved grouped data to {file_name}")
        
    # ! Early return
    if return_before_plotting:
        print("... returning early")
        return df_counts, df_gm, df_grouped

    # Plot
    if len(kwargs["my_grouping"]) > 1:
        raise ValueError(
            "For Temproal Map, currently my grouping cannot be more than 1"
        )

    if kwargs["plot_type"] == "facet":
        plot_temporal_trend_facet(
            df_grouped,
            df_counts,
            my_method=kwargs["my_method"],
            my_grouping=kwargs["my_grouping"][0],
            my_metric=kwargs["my_metric"],
            uncertainty_representation=kwargs["uncertainty_representation"],
            uncertainty_variable=kwargs["uncertainty_variable"],
            aggregation_variable=kwargs["aggregation_variable"],
            facet_label_trees_or_sites=kwargs["facet_label_trees_or_sites"],
            share_y=kwargs["share_y"],
            set_ylim=kwargs["ylim"],
            top_n_metric=kwargs["top_n_metric"],
            top_n_groups=None,
            n_bootstraps=kwargs["n_bootstraps_samples"],
            center_to_first_year=kwargs["center_to_first_year"],
            normalize_to_first_year=kwargs["normalize_to_first_year"],
            save_plot=kwargs["save_plot"],
            file_suffix=kwargs["file_suffix"],
        )
    elif kwargs["plot_type"] == "all":
        plot_temporal_trend(
            df_grouped,
            df_counts,
            my_method=kwargs["my_method"],
            my_grouping=kwargs["my_grouping"][0],
            my_metric=kwargs["my_metric"],
            uncertainty_representation=kwargs["uncertainty_representation"],
            uncertainty_variable=kwargs["uncertainty_variable"],
            aggregation_variable=kwargs["aggregation_variable"],
            facet_label_trees_or_sites=kwargs["facet_label_trees_or_sites"],
            set_ylim=kwargs["ylim"],
            top_n_metric=kwargs["top_n_metric"],
            top_n_groups=kwargs["top_n_groups_plot"],
            n_bootstraps=kwargs["n_bootstraps_samples"],
            center_to_first_year=kwargs["center_to_first_year"],
            normalize_to_first_year=kwargs["normalize_to_first_year"],
            save_plot=kwargs["save_plot"],
            file_suffix=kwargs["file_suffix"],
        )
    else:
        chime.error()
        raise ValueError("🚨🚨🚨 Invalid plot_type")

    return df_counts, df_gm, df_grouped


def make_maps_per_species(
    df_grouped,
    kwargs,
    ci_over_mean_threshold=0.5,
    show=False,
    save_fig=True,
    skip_existing=False,
):

    # ! Local variables
    mm = kwargs["my_metric"]
    center_to_first_year = kwargs["center_to_first_year"]
    normalize_to_first_year = kwargs["normalize_to_first_year"]
    my_method = kwargs["my_method"]

    # ! Wrangle grouped dataframe
    # Get copy
    xxx = df_grouped.copy()
    # Clean up infinity values
    xxx = xxx.replace([np.inf, -np.inf], np.nan)
    # Count total number of trees
    xxx["total_n"] = xxx["n_aa_mean"] + xxx["n_ac_mean"] + xxx["n_ad_mean"]
    # Split key into region, group, and year
    if "&" in xxx["group_year"].iloc[0]:
        xxx["region"] = xxx["group_year"].str.split("&", expand=True)[0]
        xxx["group"] = (
            xxx["group_year"]
            .str.split("&", expand=True)[1]
            .str.split("_", expand=True)[0]
        )
        xxx["year"] = (
            xxx["group_year"]
            .str.split("&", expand=True)[1]
            .str.split("_", expand=True)[1]
        )
        xxx["year"] = xxx["year"].astype(int) + 5
    else:
        xxx["region"] = xxx["group_year"].str.split("_", expand=True)[0]
        xxx["group"] = "All Species"
        xxx["year"] = xxx["group_year"].str.split("_", expand=True)[1]
        xxx["year"] = xxx["year"].astype(int) + 5

    # If centered or normalized, the first year is not needed
    if center_to_first_year or normalize_to_first_year:
        xxx = xxx.query("year > 2015")

    # ! Attach hatching information
    # Use STD for hatching
    # | Alternative:
    #   xxx["ci_width"] = xxx[f"{mm}_ciupper"] - xxx[f"{mm}_cilower"]
    #   xxx["ci_over_mean"] = xxx["ci_width"] / xxx[f"{mm}_mean"]  # Use CI for hatching
    xxx["ci_over_mean"] = xxx[f"{mm}_std"] / xxx[f"{mm}_mean"]
    # If ci_over_mean is NA, set stipples to True
    # If ci_over_mean is above threshold, set stipples to True
    # If standard deviation is 0, set stipples to True
    xxx["hatch"] = 0
    xxx.loc[xxx["ci_over_mean"].isna(), "hatch"] = 1
    xxx.loc[xxx["ci_over_mean"] > ci_over_mean_threshold, "hatch"] = 1
    xxx.loc[xxx[f"{mm}_std"] == 0, "hatch"] = 1

    # Reduce to relevant columns
    xxx = xxx[["group", "year", "region", f"{mm}_mean", "hatch"]]
    
    # Clean up names for height classes
    clean_group = False
    if kwargs["my_grouping"].__len__() > 1:
        if kwargs["my_grouping"][1] == "tree_height_class":
            clean_group=True
    elif kwargs["my_grouping"][0] == "tree_height_class":
        clean_group=True
        
    if clean_group:
        xxx["group"] = xxx["group"].replace(
                    {"0-10" : "0-10m",
                    "10-15" : "10-15m",
                    "15-20" : "15-20m",
                    "20-25" : "20-25m",
                    "25+" : ">25m"}
                    )

    # Get levels to loop over
    n_levels = xxx.group.nunique()
    levels = sorted(xxx.group.unique())

    # ! Create maps per group
    for i, igroup in enumerate(levels):

        # Verbose
        print(f" {i+1}/{n_levels}:\t {igroup}", end="\t")

        # Check if file already exists, if so skip loop
        mydir = f"maps_of_change/{datetime.datetime.now().strftime('%Y-%m-%d')}/{my_method}/{kwargs['my_grouping'][0]}"
        filename = f"{mydir}/{my_method}-centered_{center_to_first_year}_normalized-{normalize_to_first_year}-{igroup}.png"
        if os.path.isfile(filename) and skip_existing:
            print(f"-> Skipped {igroup} because plot already exists!")

        # ! Debug
        # debug_group = "Pinus sylvestris"
        # if igroup != debug_group:
        #     print(f"Skipping {igroup} because it is not {debug_group}")
        #     continue

        # Filter for species
        df_i = xxx.query("group == @igroup")

        # Get shapefile
        df_sf = get_shp_of_region(
            kwargs["my_grouping"][0],
            make_per_year=[df_i.year.min(), df_i.year.max()],
            make_per_group=df_i.group.unique(),
        ).rename(
            columns={
                kwargs["my_grouping"][0]: "region",
                f"{kwargs['my_grouping'][0]}_name": "region_name",
            }
        )

        # Attach geometry
        df_sf["region"] = df_sf["region"].astype(str) # Make sure it is string for merge
        df_i = pd.merge(
            df_i,
            df_sf[["group", "region", "geometry", "region_name", "year"]],
            on=["group", "region", "year"],
            how="right",
        )

        # Clean hatching information because of merge, if NA set to True
        df_i["hatch"] = df_i["hatch"].astype(float).fillna(1)

        # Make sure gdf is a GeoDataFrame
        gdf = gpd.GeoDataFrame(df_i, geometry="geometry").drop_duplicates()

        # Overwrite df_in
        df_in = gdf.copy()

        # ! START PLOT
        # Get shapefile for france
        sp_france = get_shp_of_region("cty", make_per_year=None)

        # Unique years to create subplots for
        unique_years = sorted(df_in["year"].unique())
        n_years = len(unique_years)

        # Use some nice font
        plt.rcParams["font.sans-serif"] = "DejaVu Sans"

        # Set up figure
        fig, axs = plt.subplots(1, n_years, figsize=(40, 25))

        # ! Get colorbar
        # Calculate color normalization parameters
        if center_to_first_year:
            data_min = get_percentile_from_either_df(
                            df_in[f"{mm}_mean"],
                            xxx[f"{mm}_mean"],
                            10,
                        )
            data_max = get_percentile_from_either_df(
                            df_in[f"{mm}_mean"],
                            xxx[f"{mm}_mean"],
                            90,
                        )
            abs_max = np.max([np.abs(data_min), np.abs(data_max)])

            norm = mcolor.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
            cbar_extend = "both"
            cmap = plt.get_cmap("RdBu_r")
            if normalize_to_first_year:
                cbar_label = f"Change in mortality\nsince 2015 (%)"
            else:
                cbar_label = f"Change in mortality\nsince 2015 (% yr$^{{-1}}$)"

        elif normalize_to_first_year:
            data_min = 0
            data_max = get_percentile_from_either_df(
                            df_in[f"{mm}_mean"],
                            xxx[f"{mm}_mean"],
                            90,
                        )

            norm = mcolor.TwoSlopeNorm(vmin=data_min, vcenter=100, vmax=data_max)
            cbar_extend = "max"
            cmap = plt.get_cmap("RdBu_r")
            cbar_label = f"Mortality relative to 2015 (%)"

        else:
            # data_min = np.min(df_in[f"{mm}_mean"].dropna())
            data_min = 0
            data_max = get_percentile_from_either_df(
                            df_in[f"{mm}_mean"],
                            xxx[f"{mm}_mean"],
                            90,
                        )
            norm = mcolor.Normalize(vmin=data_min, vmax=data_max)
            cbar_extend = "max"
            cmap = plt.get_cmap("Reds")
            cbar_label = f"Mortality (% yr$^{{-1}}$)"

        # Create a ScalarMappable for the colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        for i, year in enumerate(unique_years):
            ax = axs[i]

            # Filter the data for the year and plot
            data_for_year = df_in[df_in["year"] == year]
            # display(data_for_year)  # ! Debug

            # Plot it
            data_for_year.plot(
                column=f"{mm}_mean",
                edgecolor="face",
                linewidth=0.5,
                ax=ax,
                cmap=cmap,
                norm=norm,
                missing_kwds={
                    "color": "lightgrey",
                    "edgecolor": "lightgrey",
                    "linewidth": 0.5,
                },
                figsize=(10, 10),
            )

            # Hatch regions where needed
            data_for_year.loc[data_for_year["hatch"] == 1].plot(
                hatch="....",
                edgecolor="lightgrey",
                alpha=0.75,
                linewidth=0,
                facecolor="none",
                ax=ax,
                missing_kwds={
                    "color": "lightgrey",
                    "edgecolor": "lightgrey",
                    "linewidth": 0.5,
                },
                figsize=(10, 10),
            )

            # Add contour of France
            sp_france.plot(ax=ax, color="none", edgecolor="black", linewidth=0.5)

            # Remove axis
            ax.set_axis_off()

            # Add year as text below the map
            ax.text(
                0.5,
                -0.1,
                str(year),
                transform=ax.transAxes,
                ha="center",
                fontweight="bold",
                fontdict={"fontsize": 18},
            )

            if i == 0:
                if igroup == "All Species":
                    ititle = "Spatial distribution of tree mortality in France for the 20 most common species"
                else:
                    ititle = igroup

                ax.set_title(
                    f"{ititle}",
                    fontweight="bold",
                    loc="left",
                    fontdict={"fontsize": 22},
                )

        # Add colorbar to the right of the plots
        cbar = fig.colorbar(
            sm, ax=axs.ravel().tolist(), shrink=0.125, pad=0.01, extend=cbar_extend
        )

        cbar.set_label(
            cbar_label,
            fontsize=16,
            labelpad=10,
        )

        cbar.ax.tick_params(labelsize=13)

        # Adjust layout to fit everything nicely
        # plt.tight_layout()

        if save_fig:

            if not os.path.exists(mydir):
                os.makedirs(mydir)

            print(filename)
            plt.savefig(
                filename,
                dpi=150,
                bbox_inches="tight",
            )
            
            write_txt(f"{mydir}/ci_over_mean_threshold_{ci_over_mean_threshold}.txt")
            
        if show:
            plt.show()
        else:
            plt.close()
        # break

def get_percentile_from_either_df(
    subset,
    fullset,
    percentile,
):
    # Check if subset is all NA
    if subset.isna().all():
       return np.percentile(fullset.dropna(), percentile)
    else:
        return np.percentile(subset.dropna(), percentile)
    

def plot_temporal_trend_facet_height_x_species(
    df_grouped,
    df_counts,
    my_method,
    my_grouping,
    my_metric,
    uncertainty_representation,
    uncertainty_variable,
    aggregation_variable,
    facet_label_trees_or_sites="trees",
    share_y=True,
    set_ylim=None,  # None or [min, max]
    center_to_first_year=True,
    normalize_to_first_year=False,
    n_bootstraps=None,
    save_plot=False,
    file_suffix="",
):

    print("\nPlotting species x tree height class trends...")

    # ! Checks ---------------------------------------------------------------
    if my_method not in ["direct", "direct_bs", "per_site", "per_site_bs"]:
        raise ValueError(f"Invalid method: {my_method}")

    if my_grouping[0] != "species_lat" and my_grouping[0] != "tree_height_class":
        raise ValueError(
            f"Invalid grouping: {my_grouping}, must be ['species_lat', 'tree_height_class']"
        )

    if my_method == "direct":
        if my_metric not in df_grouped.columns:
            raise ValueError(f"Missing metric: {my_metric}")
    else:
        if f"{my_metric}_{aggregation_variable}" not in df_grouped.columns:
            raise ValueError(f"Missing metric: {my_metric}_{aggregation_variable}")

        if uncertainty_variable not in ["std", "sem", "iqr", "ci", "std_weighted"]:
            raise ValueError(f"Missing uncertainty_variable: {uncertainty_variable}")

        if aggregation_variable == "median" and uncertainty_variable != "iqr":
            uncertainty_variable = "iqr"
            print(
                f"... aggregation_variable is median but uncertainty_variable is not iqr! -> Setting it to iqr."
            )

        if aggregation_variable == "mean" and uncertainty_variable == "iqr":
            print(
                f"... aggregation_variable is median mean uncertainty_variable is iqr! -> Setting it to std instead."
            )

        if uncertainty_representation not in ["band", "bar", "none"]:
            raise ValueError(
                f"Invalid uncertainty_representation: {uncertainty_representation}"
            )
            
    if not share_y:
        set_ylim = None

    # ! Prepare dataset for plotting ---------------------------------------------
    import itertools

    df = df_grouped.copy()
    df["group"] = df["group_year"].str.split("_", expand=True)[0]
    df["year"] = df["group_year"].str.split("_", expand=True)[1]
    df["year"] = df["year"].astype(int)

    # Fix to show second visit year
    df["year"] = df["year"].astype(int) + 5  # 5 years between visits
    df["year"] = df["year"].astype(str)

    # Add group from group_year
    df["species"] = df["group"].str.split("&", expand=True)[0]
    df["height"] = df["group"].str.split("&", expand=True)[1]

    # Make df of all possible combinations and merge back to have consistent plotting faceting
    unique_species = df["species"].unique()
    unique_heights = df["height"].unique()
    unique_years = df["year"].unique()
    df_all = pd.DataFrame(
        list(itertools.product(unique_species, unique_heights, unique_years)),
        columns=["species", "height", "year"],
    )

    # Needs some cleaning for proper merging
    df_all["group"] = df_all["species"] + "&" + df_all["height"]
    df["group"] = df["species"] + "&" + df["height"]

    df = df_all.merge(df, how="outer", on=["species", "height", "year", "group"])
    # Add number of trees or sites, ticks to separate thousands
    df = df.drop(columns=["n_plots"], errors="ignore").merge(
        df_counts, on="group", how="left"
    )

    # Clean up height classes
    df["height"] = (
        df["height"]
        .replace(
            {
                "0-10": "0-10m",
                "10-15": "10-15m",
                "15-20": "15-20m",
                "20-25": "20-25m",
                ">25": ">25m",
            }
        )
        .astype(str)
    )

    df["n_trees"] = df["n_trees"].fillna(0)  # Fill 0 where NA was entered from the merge
    df["n_trees"] = (
        df["n_trees"].astype(int).apply(lambda x: format(x, ",").replace(",", "'"))
    )

    # Add facet label (this is the tree height in this case!)
    df["facet"] = df["height"] + "  (N = " + df["n_trees"].astype(str) + ")"

    # Order factor levels to be alphabetically sorted
    df.loc[:, "facet"] = pd.Categorical(
        df["facet"], categories=sorted(df["facet"].unique())
    )

    # ! Set target, lower and upper ----------------------------------------------
    # Checks
    if my_method in ["per_site", "direct_bs", "per_site_bs"]:
        if "mean" in aggregation_variable:
            df["target"] = df[f"{my_metric}_{aggregation_variable}"]

            if uncertainty_variable in ["std", "std_weighted", "sem"]:
                df["lower"] = (
                    df[f"{my_metric}_{aggregation_variable}"]
                    - df[f"{my_metric}_{uncertainty_variable}"]
                )
                df["upper"] = (
                    df[f"{my_metric}_{aggregation_variable}"]
                    + df[f"{my_metric}_{uncertainty_variable}"]
                )

                # Plot line only when std is less than 50% of the mean
                df["plot_line"] = (
                    df[f"{my_metric}_{uncertainty_variable}"]
                    < df[f"{my_metric}_{aggregation_variable}"] * 0.5
                )

            elif uncertainty_variable == "ci":
                df["lower"] = df[f"{my_metric}_cilower"]
                df["upper"] = df[f"{my_metric}_ciupper"]

        elif aggregation_variable == "median":
            df["target"] = df[f"{my_metric}_median"]
            df["lower"] = df[f"{my_metric}_median"] - df[f"{my_metric}_iqr"] / 2
            df["upper"] = df[f"{my_metric}_median"] + df[f"{my_metric}_iqr"] / 2

    elif my_method == "direct":
        df["target"] = df[f"{my_metric}"]
        df["lower"] = np.nan
        df["upper"] = np.nan

    else:
        raise ValueError(f"Invalid method: {my_method}")

    df = df[
        [
            "facet",
            "group",
            "species",
            "year",
            "target",
            "lower",
            "upper",
            "n_trees",
            "plot_line",
        ]
    ]

    # Turn 0 values into NA, because highly unlikely to have 0 mortality
    # and it is more likely to be a missing value
    df["target"] = df["target"].replace(0, np.nan)
    df["lower"] = df["lower"].replace(0, np.nan)
    df["upper"] = df["upper"].replace(0, np.nan)

    # Add whether line should be plotted or not
    # df["plot_line"] = df["target"].isna().apply(lambda x: not x)
    # df["plot_line"] = "1"

    # ! RUN PLOT LOOP -----------------------------------------------------------
    # ! Plot -----------------------------------------------------------------------
    all_species = df["species"].unique()

    for ispecies in all_species:

        idf = df[df["species"] == ispecies]

        sharey_axis = not center_to_first_year
        g = sns.FacetGrid(
            idf,
            col="facet",
            col_wrap=5,
            # sharey=sharey_axis,
            sharey=share_y,
            sharex=True,
            height=4,
        )

        # Add line plot
        g.map_dataframe(
            sns.lineplot,
            x="year",
            y="target",
            # hue="plot_line",
            # palette={True: "#4c72b0", False: "white"},
            markers=True,
            marker="o",
            zorder=2,
        )

        # Add uncertainty representation
        if uncertainty_representation == "band":
            g.map_dataframe(
                lambda data, **kwargs: plt.fill_between(
                    data["year"],
                    data["lower"],
                    data["upper"],
                    alpha=0.3,
                    zorder=1,
                    # color=data["plot_line"].map({True: "#4c72b0", False: "white"}),
                )
            )
        elif uncertainty_representation == "bar":
            g.map_dataframe(
                lambda data, **kwargs: plt.errorbar(
                    data["year"],
                    data["target"],
                    yerr=[data["target"] - data["lower"], data["upper"] - data["target"]],
                    fmt="none",
                    capsize=5,
                    zorder=1,
                    # hue="plot_line",
                )
            )

        # If centered to first year, add a line at 0
        if center_to_first_year:
            g.map(
                lambda **kwargs: plt.axhline(
                    0, color="grey", linestyle="dotted", linewidth=1, zorder=0
                )
            )

        # Set ylim if provided
        if set_ylim is not None:
            if len(set_ylim) != 2:
                raise ValueError("ylim must be a list of two values")
            g.set(ylim=set_ylim)

        # if center_to_first_year:
        #     max_value = df[["upper", "lower"]].abs().max().astype(int)
        #     g.set(ylim=(-max_value, max_value))
        # else:
        #     g.set(ylim=(0, df["upper"].abs().max())*1.1)

        # Rotate x-tick labels
        # for ax in g.axes.flat:
        #     for label in ax.get_xticklabels():
        #         label.set_rotation(45)

        # > Labels and positions
        # Remove labels from each facet
        g.set_ylabels("")
        g.set_xlabels("")

        # Give one xlabel to the whole plot
        if center_to_first_year and normalize_to_first_year:
            mytitle = f"Change in tree mortality since 2015 (%)"
        elif center_to_first_year and not normalize_to_first_year:
            mytitle = f"Change in tree mortality since 2015 (%-trees yr$^{-1}$)"
        elif not center_to_first_year and normalize_to_first_year:
            mytitle = f"Tree Mortality (%)"
        else:
            mytitle = f"Tree Mortality (%-trees yr$^{-1}$)"

        g.fig.text(
            0.5,
            -0.1,
            "Year",
            ha="center",
            va="center",
            fontsize=16,
            weight="bold",
        )

        g.fig.text(
            0.04,
            0.5,
            mytitle,
            ha="center",
            va="center",
            rotation="vertical",
            fontsize=14,
            weight="bold",
        )

        g.fig.suptitle(
            f"Tree mortality trends for {ispecies} by height class",
            fontsize=18,
            weight="bold",
            x=0.5,
            y=1.11,
        )

        # Subplot titles
        g.set_titles("\n{col_name}")
        for ax in g.axes.flat:
            ax.set_title(ax.get_title(), fontsize=12)

        g.fig.subplots_adjust(bottom=0.09, left=0.075, top=0.925, hspace=0.2)

        if save_plot:
            # ! Save data
            today = datetime.datetime.today().strftime("%Y-%m-%d")
            my_dir = f"temporal_trends/{today}/{my_metric}/"
            if file_suffix != "":
                my_dir = my_dir.replace("temporal_trends", f"specific_runs/{file_suffix} - {today}/temporal_trends")

            for i, igroup in enumerate(my_grouping):
                if i == 0:
                    my_dir += f"/{igroup}"
                else:
                    my_dir += f"&{igroup}"

            my_dir += f"/{aggregation_variable}-{uncertainty_variable}-{uncertainty_representation}/facet-sharey_{share_y}/"

            if not os.path.exists(my_dir):
                os.makedirs(my_dir, exist_ok=True)

            fname = f"{my_dir}/fig-{ispecies}-{my_method}-"

            if "_bs" in my_method:
                fname += f"_{n_bootstraps}-bootstraps"

            fname += f"-centered_{center_to_first_year}-normalized_{normalize_to_first_year}"
            fname += ".png"
            g.savefig(fname, dpi=300, bbox_inches="tight")

            print(f"... plot saved: {fname}")

        # plt.show()
        plt.close()
        
        
def clean_group_levels(df, col_name, grouping):
    if grouping == "tree_height_class":
        df[f"{col_name}"] = (
            df[f"{col_name}"]
            .replace(
                {
                    "0-10": "0-10m",
                    "10-15": "10-15m",
                    "15-20": "15-20m",
                    "20-25": "20-25m",
                    ">25": ">25m",
                }
            )
            .astype(str)
        )
    elif grouping == "gre":
        df[f"{col_name}"] = df[f"{col_name}"].replace(
            {
                "A": "Grand Ouest",
                "B": "Centre Nord",
                "C": "Grand Est",
                "D": "Vosges",
                "E": "Jura",
                "F": "Sud-Ouest",
                "G": "Massif Central",
                "H": "Alpes",
                "I": "Pyrénées",
                "J": "Méditerranée",
                "K": "Corse",
            }
        )
    elif grouping == "reg":
        df[f"{col_name}"] = df[f"{col_name}"].replace(
            {
                "11": "Île-de-France",
                "24": "Centre-Val de Loire",
                "27": "Bourgogne-Franche-Comté",
                "28": "Normandie",
                "32": "Hauts-de-France",
                "44": "Grand Est",
                "52": "Pays de la Loire",
                "53": "Bretagne",
                "75": "Nouvelle-Aquitaine",
                "76": "Occitanie",
                "84": "Auvergne-Rhône-Alpes",
                "93": "Provence-Alpes-Côte-d'Azur",
                "94": "Corse",
            }
        )
    return df

def plot_temporal_trend_facet_x_primary(
    df_grouped=None,
    df_counts=None,
    my_method=None,
    my_grouping=None,
    my_metric=None,
    uncertainty_representation=None,
    uncertainty_variable=None,
    aggregation_variable=None,
    facet_label_trees_or_sites=None,
    share_y=True,
    set_ylim=None,
    n_bootstraps=None,
    center_to_first_year=None,
    normalize_to_first_year=None,
    save_plot=None,
    file_suffix="",
    primary_group=None,
    ):
    
    # ! Checks ---------------------------------------------------------------
    if my_method not in ["direct", "direct_bs", "per_site", "per_site_bs"]:
        raise ValueError(f"Invalid method: {my_method}")

    if my_grouping.__len__() != 2:
        raise ValueError(f"Invalid grouping: need two arguments! Given were: {my_grouping}")

    if primary_group not in my_grouping:
        raise ValueError(
            f"Primary group was not found in my_grouping, check again! {primary_group} not in {my_grouping}"
        )

    if my_method == "direct":
        if my_metric not in df_grouped.columns:
            raise ValueError(f"Missing metric: {my_metric}")
    else:
        if f"{my_metric}_{aggregation_variable}" not in df_grouped.columns:
            raise ValueError(f"Missing metric: {my_metric}_{aggregation_variable}")

        if uncertainty_variable not in ["std", "sem", "iqr", "ci", "std_weighted"]:
            raise ValueError(f"Missing uncertainty_variable: {uncertainty_variable}")

        if aggregation_variable == "median" and uncertainty_variable != "iqr":
            uncertainty_variable = "iqr"
            print(
                f"... aggregation_variable is median but uncertainty_variable is not iqr! -> Setting it to iqr."
            )

        if aggregation_variable == "mean" and uncertainty_variable == "iqr":
            print(
                f"... aggregation_variable is median mean uncertainty_variable is iqr! -> Setting it to std instead."
            )

        if uncertainty_representation not in ["band", "bar", "none"]:
            raise ValueError(
                f"Invalid uncertainty_representation: {uncertainty_representation}"
            )
            
    if not share_y:
        set_ylim = None

    # Get faceting variable
    for i, v in enumerate(my_grouping):
        if v == primary_group:
            pos_primary = i
        else:
            facet_var = v
            pos_facet = i

    print(f"\nPlotting trends for {primary_group} x {facet_var}...")

    # ! Prepare dataset for plotting ---------------------------------------------
    import itertools

    df = df_grouped.copy()
    df["group"] = df["group_year"].str.split("_", expand=True)[0]
    df["year"] = df["group_year"].str.split("_", expand=True)[1]
    df["year"] = df["year"].astype(int)

    # Fix to show second visit year
    df["year"] = df["year"].astype(int) + 5  # 5 years between visits
    df["year"] = df["year"].astype(str)

    # Add group from group_year
    df["primary"] = df["group"].str.split("&", expand=True)[pos_primary]
    df["facetvar"] = df["group"].str.split("&", expand=True)[pos_facet]

    # Clean groups in df
    df = clean_group_levels(df, "facetvar", facet_var)
    df = clean_group_levels(df, "primary", primary_group)

    # Make df of all possible combinations and merge back to have consistent plotting faceting
    unique_primary = df["primary"].unique()
    unique_facetvars = df["facetvar"].unique()
    unique_years = df["year"].unique()

    df_all = pd.DataFrame(
        list(itertools.product(unique_primary, unique_facetvars, unique_years)),
        columns=["primary", "facetvar", "year"],
    )

    # Needs some cleaning for proper merging
    df_all["group"] = df_all["primary"] + "&" + df_all["facetvar"]
    df["group"] = df["primary"] + "&" + df["facetvar"]

    # Clean groups in df_count
    df_counts_clean = df_counts.copy()
    df_counts_clean["g1"] = df_counts_clean.group.str.split("&", expand=True)[pos_primary]
    df_counts_clean["g2"] = df_counts_clean.group.str.split("&", expand=True)[pos_facet]
    df_counts_clean = df_counts_clean.dropna()
    df_counts_clean = clean_group_levels(df_counts_clean, "g1", primary_group)
    df_counts_clean = clean_group_levels(df_counts_clean, "g2", facet_var)
    df_counts_clean["group"] = df_counts_clean["g1"] + "&" + df_counts_clean["g2"]

    # Add number of trees or sites, ticks to separate thousands
    df = df_all.merge(df, how="outer", on=["primary", "facetvar", "year", "group"])
    df = df.drop(columns=["n_plots"], errors="ignore").merge(
        df_counts_clean, on="group", how="left"
    )

    df["n_trees"] = df["n_trees"].fillna(0)  # Fill 0 where NA was entered from the merge
    df["n_trees"] = (
        df["n_trees"].astype(int).apply(lambda x: format(x, ",").replace(",", "'"))
    )

    # Add facet label (this is the tree height in this case!)
    df["facet"] = df["facetvar"] + "  (N = " + df["n_trees"].astype(str) + ")"

    # Order factor levels to be alphabetically sorted
    df.loc[:, "facet"] = pd.Categorical(
        df["facet"], categories=sorted(df["facet"].unique())
    )

    # ! Set target, lower and upper ----------------------------------------------
    # Checks
    if my_method in ["per_site", "direct_bs", "per_site_bs"]:
        if "mean" in aggregation_variable:
            df["target"] = df[f"{my_metric}_{aggregation_variable}"]

            if uncertainty_variable in ["std", "std_weighted", "sem"]:
                df["lower"] = (
                    df[f"{my_metric}_{aggregation_variable}"]
                    - df[f"{my_metric}_{uncertainty_variable}"]
                )
                df["upper"] = (
                    df[f"{my_metric}_{aggregation_variable}"]
                    + df[f"{my_metric}_{uncertainty_variable}"]
                )

                # Plot line only when std is less than 50% of the mean
                df["plot_line"] = (
                    df[f"{my_metric}_{uncertainty_variable}"]
                    < df[f"{my_metric}_{aggregation_variable}"] * 0.5
                )

            elif uncertainty_variable == "ci":
                df["lower"] = df[f"{my_metric}_cilower"]
                df["upper"] = df[f"{my_metric}_ciupper"]

        elif aggregation_variable == "median":
            df["target"] = df[f"{my_metric}_median"]
            df["lower"] = df[f"{my_metric}_median"] - df[f"{my_metric}_iqr"] / 2
            df["upper"] = df[f"{my_metric}_median"] + df[f"{my_metric}_iqr"] / 2

    elif my_method == "direct":
        df["target"] = df[f"{my_metric}"]
        df["lower"] = np.nan
        df["upper"] = np.nan

    else:
        raise ValueError(f"Invalid method: {my_method}")

    df = df[
        [
            "facet",
            "group",
            "primary",
            "year",
            "target",
            "lower",
            "upper",
            "n_trees",
            "plot_line",
        ]
    ]

    # Turn 0 values into NA, because highly unlikely to have 0 mortality
    # and it is more likely to be a missing value
    df["target"] = df["target"].replace(0, np.nan)
    df["lower"] = df["lower"].replace(0, np.nan)
    df["upper"] = df["upper"].replace(0, np.nan)

    # Add whether line should be plotted or not
    # df["plot_line"] = df["target"].isna().apply(lambda x: not x)
    # df["plot_line"] = "1"
    # df

    # ! RUN PLOT LOOP -----------------------------------------------------------
    # ! Plot -----------------------------------------------------------------------
    all_primary = df["primary"].unique()

    for iprimary in all_primary:

        idf = df[df["primary"] == iprimary]

        sharey_axis = not center_to_first_year
        g = sns.FacetGrid(
            idf,
            col="facet",
            col_wrap=5,
            # sharey=sharey_axis,
            sharey=share_y,
            sharex=True,
            height=4,
        )

        # Add line plot
        g.map_dataframe(
            sns.lineplot,
            x="year",
            y="target",
            # hue="plot_line",
            # palette={True: "#4c72b0", False: "white"},
            markers=True,
            marker="o",
            zorder=2,
        )

        # Add uncertainty representation
        if uncertainty_representation == "band":
            g.map_dataframe(
                lambda data, **kwargs: plt.fill_between(
                    data["year"],
                    data["lower"],
                    data["upper"],
                    alpha=0.3,
                    zorder=1,
                    # color=data["plot_line"].map({True: "#4c72b0", False: "white"}),
                )
            )
        elif uncertainty_representation == "bar":
            g.map_dataframe(
                lambda data, **kwargs: plt.errorbar(
                    data["year"],
                    data["target"],
                    yerr=[data["target"] - data["lower"], data["upper"] - data["target"]],
                    fmt="none",
                    capsize=5,
                    zorder=1,
                    # hue="plot_line",
                )
            )

        # If centered to first year, add a line at 0
        if center_to_first_year:
            g.map(
                lambda **kwargs: plt.axhline(
                    0, color="grey", linestyle="dotted", linewidth=1, zorder=0
                )
            )

        # Set ylim if provided
        if set_ylim is not None:
            if len(set_ylim) != 2:
                raise ValueError("ylim must be a list of two values")
            g.set(ylim=set_ylim)

        # if center_to_first_year:
        #     max_value = df[["upper", "lower"]].abs().max().astype(int)
        #     g.set(ylim=(-max_value, max_value))
        # else:
        #     g.set(ylim=(0, df["upper"].abs().max())*1.1)

        # Rotate x-tick labels
        # for ax in g.axes.flat:
        #     for label in ax.get_xticklabels():
        #         label.set_rotation(45)

        # > Labels and positions
        # Remove labels from each facet
        g.set_ylabels("")
        g.set_xlabels("")

        # Give one xlabel to the whole plot
        if center_to_first_year and normalize_to_first_year:
            yaxis_label = f"Change in tree mortality since 2015 (%)"
        elif center_to_first_year and not normalize_to_first_year:
            yaxis_label = f"Change in tree mortality since 2015 (%-trees yr$^{-1}$)"
        elif not center_to_first_year and normalize_to_first_year:
            yaxis_label = f"Tree Mortality (%)"
        else:
            yaxis_label = f"Tree Mortality (%-trees yr$^{-1}$)"

        if primary_group == "tree_height_class":
            extra_txt = "height group "
        else:
            extra_txt = ""

        # Get locations
        if facet_var == "tree_height_class":
            yaxis_x = 0.04
            yaxis_y = 0.5

            xaxis_x = 0.5
            xaxis_y = -0.1

            title_x = 0.5
            title_y = 1.11

            title = f"Tree mortality trends for {iprimary} {extra_txt}by tree height"

        elif facet_var == "reg":
            yaxis_x = 0.04
            yaxis_y = 0.5

            xaxis_x = 0.5
            xaxis_y = 0.02

            title_x = 0.5
            title_y = 1

            title = (
                f"Tree mortality trends for {iprimary} {extra_txt}by administrative region"
            )

        elif facet_var == "gre":
            yaxis_x = 0.05
            yaxis_y = 0.5

            xaxis_x = 0.5
            xaxis_y = 0.02

            title_x = 0.5
            title_y = 1

            title = f"Tree mortality trends for {iprimary} {extra_txt}by greater ecoregion"

        elif facet_var == "species_lat":
            yaxis_x = 0.05
            yaxis_y = 0.5

            xaxis_x = 0.5
            xaxis_y = 0.04

            title_x = 0.5
            title_y = 0.98

            title = f"Tree mortality trends for {iprimary} {extra_txt}by species"

        g.fig.text(
            x=xaxis_x,
            y=xaxis_y,
            s="Year",
            ha="center",
            va="center",
            fontsize=16,
            weight="bold",
        )

        g.fig.text(
            x=yaxis_x,
            y=yaxis_y,
            s=yaxis_label,
            ha="center",
            va="center",
            rotation="vertical",
            fontsize=14,
            weight="bold",
        )

        g.fig.suptitle(
            title,
            fontsize=18,
            weight="bold",
            x=title_x,
            y=title_y,
        )

        # Subplot titles
        g.set_titles("\n{col_name}")
        for ax in g.axes.flat:
            ax.set_title(ax.get_title(), fontsize=12)

        g.fig.subplots_adjust(bottom=0.09, left=0.075, top=0.925, hspace=0.2)

        if save_plot:
            # ! Save plot
            today = datetime.datetime.today().strftime("%Y-%m-%d")
            my_dir = f"temporal_trends/{today}/{my_metric}/"
            if file_suffix != "":
                my_dir = my_dir.replace("temporal_trends", f"specific_runs/{file_suffix} - {today}/temporal_trends")

            for i, igroup in enumerate(my_grouping):
                if i == 0:
                    my_dir += f"/{igroup}"
                else:
                    my_dir += f"&{igroup}"

            my_dir += f"/{aggregation_variable}-{uncertainty_variable}-{uncertainty_representation}/facet-sharey_{share_y}/"

            if not os.path.exists(my_dir):
                os.makedirs(my_dir, exist_ok=True)

            fname = f"{my_dir}/fig-{iprimary}-{my_method}-"

            if "_bs" in my_method:
                fname += f"_{n_bootstraps}-bootstraps"

            fname += f"-centered_{center_to_first_year}-normalized_{normalize_to_first_year}"
            fname += ".png"
            g.savefig(fname, dpi=300, bbox_inches="tight")

            print(f"... plot saved: {fname}")

        # plt.show()
        plt.close()
        # break
        
# ----------------        
def ___RF_ANALYSIS___():
    pass

# ---------------------------------------------------------------------------
def load_shap(filepath):
    # Load the explainer object
    with open(filepath, "rb") as file:
        loaded_explainer = pickle.load(file)
    return loaded_explainer
# ---------------------------------------------------------------------------
def get_relevant_feature(df, species, dataset):
    feature = df.query("subset_group == @species")[f"{dataset} - Metrics"].values[0]

    if str(feature) == "nan":
        return None
    else:
        return ast.literal_eval(feature)[0]
    
def get_relevant_category(df, species, dataset):
    feature = df.query("subset_group == @species")[f"{dataset} - Metrics"].values[0]

    if str(feature) == "nan":
        return None
    else:
        return ast.literal_eval(feature)[0]
    
def display_fig(path):
    if not os.path.exists(path):
        raise ValueError(f"🟥 Figure not found: {path}")
    print(path)
    display(Image(path))
    
def get_relevant_figs(df, species):
    # Get directory
    i_dir = df.query("subset_group == @species").loc[:, "dir"].values[0]

    # Load figure
    display(pd.read_csv(f"{i_dir}/final_model_roc_auc_prob-based.csv"))
    display_fig(f"{i_dir}/fig-vip-shap-by_feature.png")
    display_fig(f"{i_dir}/fig-vip-impurity-aggregated.png")
    display_fig(f"{i_dir}/fig-vip-permutation-aggregated.png")
    display_fig(f"{i_dir}/fig_shap_scatter.png")
    display_fig(f"{i_dir}/fig-roc_auc_curves.png")
    display_fig(f"{i_dir}/fig_model_evaluation_test.png")

# ---------------------------------------------------------------------------
def plot_shap_response_interaction(
    df,
    response,
    interaction,
    dir_analysis=None,
    remove_labels=False,
    show=True,
    filesuffix="",
):
    # Clean the dataset so that all rows where response or interaction are na are removed
    remove_these = []
    for species in df["subset_group"].values:

        check1 = get_relevant_feature(df, species, response)
        check2 = get_relevant_feature(df, species, interaction)
        if check1 is None or check2 is None:
            remove_these.append(species)

    # if len(remove_these) > 0:
    #     print(
    #         f" - For {response} ~ {interaction}, the following species are removed due to missing features: {remove_these}"
    #     )

    df_plot = df.query("subset_group not in @remove_these")

    # Inputs
    unique_species = df_plot["subset_group"].unique()
    n_species = len(unique_species)
    if n_species > 20:
        n_cols = 6
        figx=25
    else:
        n_cols = 4
        figx = 15
        
    n_rows = (n_species + n_cols - 1) // n_cols  # Calculate the number of rows needed

    if n_species <= n_cols:
        fig, axs = plt.subplots(
            n_rows, n_cols, figsize=(figx, 4), sharey=False, sharex=False
        )
    elif remove_labels:
        fig, axs = plt.subplots(
            n_rows, n_cols, figsize=(figx, 2 * n_rows), sharey=False, sharex=False
        )
    else:
        fig, axs = plt.subplots(
            n_rows, n_cols, figsize=(figx, 3 * n_rows), sharey=False, sharex=False
        )

    for idx, species in enumerate(unique_species):

        # Verbose
        # print(
        #     f" - Plotting species {species} ({idx + 1}/{n_species}) \t\t |{feat_response} \t| {feat_interaction}"
        # )

        # Get subplot
        row = idx // n_cols
        col = idx % n_cols

        # Get features to plot
        feat_response = get_relevant_feature(df_plot, species, response)
        feat_interaction = get_relevant_feature(df_plot, species, interaction)

        # Load SHAP data
        filepath = df_plot.query("subset_group == @species")["dir"].values[0]
        shap_values = load_shap(filepath + "/shap_values_test.pkl")

        # Plot SHAP values
        if n_species > n_cols:
            ax = axs[row, col]
        else:
            ax = axs[col]
            
        shap.plots.scatter(
            shap_values[:, feat_response][:, 1],
            x_jitter=0,
            alpha=0.5,
            dot_size=10,
            color=shap_values[:, feat_interaction][:, 1],
            ax=ax,
            show=False,
        )

        ax.set_title(f"{species}", weight="bold")
        ax.set_xlabel(feat_response)
        ax.set_ylabel("SHAP Value")

        # If interaction is Carrying Capacity, overwrite coloring to go from 0 to 2+
        if interaction == "Carrying Capacity":
            ax.collections[0].set_clim(0, 2)
            ax.collections[0].colorbar.set_ticks([0, 1, 2])
            ax.collections[0].colorbar.set_ticklabels(["0", "1", "2"])

        if remove_labels:
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticks([])
            ax.set_yticks([])

            # Remove entire colorbar
            ax.collections[0].colorbar.remove()

    # Remove empty subplots if n_species is not a multiple of n_cols
    for i in range(n_species, n_rows * n_cols):
        fig.delaxes(axs.flatten()[i])

    # Give plot a title
    fig.suptitle(
        f"SHAP Scatterplot for {response} and {interaction}\n\n",
        fontsize=20,
        weight="bold",
    )

    plt.tight_layout()

    dir_analysis = dir_analysis + "/shap_interactions_1d"
    if not os.path.exists(dir_analysis):
        os.makedirs(dir_analysis)
    plt.savefig(
        f"{dir_analysis}/shap_scatterplot_{response}-{interaction}_no-labels-{remove_labels}{filesuffix}.png"
    )
    if show:
        plt.show()
    else:
        plt.close()
        
# ---------------------------------------------------------------------------
def plot_shap_interaction2(
    df,
    response,
    interaction,
    dir_analysis=None,
    remove_labels=False,
    show=True,
    filesuffix="",
    make_fig=True,
):
    
    # Make sure the directory exists
    dir_analysis = dir_analysis + "/shap_interactions_2d"
    os.makedirs(dir_analysis, exist_ok=True)
    
    # Clean the dataset so that all rows where response or interaction are na are removed
    remove_these = []
    for species in df["subset_group"].values:

        check1 = get_relevant_feature(df, species, response)
        check2 = get_relevant_feature(df, species, interaction)
        if check1 is None or check2 is None:
            remove_these.append(species)

    # print(
    #     f" - For {response} ~ {interaction}, the following species are removed due to missing features: {remove_these}"
    # )

    df_plot = df.query("subset_group not in @remove_these")

    # Inputs
    df_modelfits=[]
    unique_species = df_plot["subset_group"].unique()
    n_species = len(unique_species)
    if n_species > 20:
        n_cols = 6
        figx=25
    else:
        n_cols = 4
        figx = 15
    n_rows = (n_species + n_cols - 1) // n_cols  # Calculate the number of rows needed

    if make_fig:
        if n_species <= n_cols:
            fig, axs = plt.subplots(
                n_rows, n_cols, figsize=(figx, 4), sharey=False, sharex=False
            )
        elif remove_labels:
            fig, axs = plt.subplots(
                n_rows, n_cols, figsize=(figx, 2 * n_rows), sharey=False, sharex=False
            )
        else:
            fig, axs = plt.subplots(
                n_rows, n_cols, figsize=(figx, 3 * n_rows), sharey=False, sharex=False
            )
        

    for idx, species in enumerate(unique_species):

        # Verbose
        # print(
        #     f" - Plotting species {species} ({idx + 1}/{n_species}) \t\t |{feat_response} \t| {feat_interaction}"
        # )

        # Get subplot
        row = idx // n_cols
        col = idx % n_cols

        # Get features to plot
        feat_response = get_relevant_feature(df_plot, species, response)
        feat_interaction = get_relevant_feature(df_plot, species, interaction)
        
        # Load SHAP data
        filepath = df_plot.query("subset_group == @species")["dir"].values[0]
        shap_values = load_shap(filepath + "/shap_values_interaction_test.pkl")
        X_shap = pd.read_feather(f"{filepath}/X_shap_test.feather")
        feature_names = pd.Series(X_shap.columns)

        # Get values for target == 1
        shap_values = shap_values[:, :, :, 0]

        # Get values for features
        feat_response_values = X_shap[feat_response].values
        feat_interaction_values = X_shap[feat_interaction].values

        pos_response = feature_names[feature_names == feat_response].index[0]
        pos_interaction = feature_names[feature_names == feat_interaction].index[0]

        # Get interaction values for the two features
        shap_values = shap_values[:, pos_response, pos_interaction]
        
        # Normalize all values
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        feat_response_values = scaler.fit_transform(feat_response_values.reshape(-1, 1)).flatten()
        feat_interaction_values = scaler.fit_transform(feat_interaction_values.reshape(-1, 1)).flatten()
        shap_values = scaler.fit_transform(shap_values.reshape(-1, 1)).flatten()
        
        # Inverse SPEI scale to make easier to interpret
        if response == "SPEI":
            feat_response_values = 1 - feat_response_values

        # Get 5 and 95 percentiles for coloring
        # ! Debug: setting from 0 - 1 because of the normalization
        # shap_values_5 = np.percentile(shap_values, 5)
        # shap_values_95 = np.percentile(shap_values, 95)
        shap_values_5 = 0
        shap_values_95 = 1
        
        # Find best fitting model using quadratic lm and interactions
        X = pd.DataFrame({response: feat_response_values, interaction: feat_interaction_values})
        y = shap_values
        modelfit = stepwise_regression(X, y, species)
        df_modelfits.append(modelfit)

        # Plot interaction
        if make_fig:
            if n_species > n_cols:
                ax = axs[row, col]
            else:
                ax = axs[col]

            if response == "Tree Size" or interaction == "Tree Size":
                cmap = plt.colormaps["viridis_r"]
                edgecol = "black"
            else:
                cmap = plt.colormaps["magma_r"]
                # edgecol = None
                edgecol = "black"

            scatter = ax.scatter(
                x=feat_response_values,
                y=feat_interaction_values,
                c=shap_values,
                cmap=cmap,
                edgecolors=edgecol,
                linewidths=0.5,
                alpha=0.9,
                s=40,
            )
            ax.set_xlabel(feat_response)
            ax.set_ylabel(feat_interaction)
            ax.set_title(f"{species}", weight="bold")
            # Add colorbar, centered at 0
            fig.colorbar(scatter, ax=ax, label=f"SHAP Value")
            # absolute_max = np.abs(shap_values).max()
            # scatter.set_clim(-absolute_max, absolute_max)
            # No spines
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # If interaction is Carrying Capacity, overwrite coloring to go from 0 to 2+
            if interaction == "Carrying Capacity":
                scatter.set_clim(0, 1.5)
                scatter.colorbar.set_ticks([0, 0.5, 1, 1.5])
                scatter.colorbar.set_ticklabels(["0", "0.5", "1", "1.5"])
            else:
                scatter.set_clim(shap_values_5, shap_values_95)

            if remove_labels:
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.set_xticks([])
                ax.set_yticks([])

                # Remove entire colorbar
            ax.collections[0].colorbar.remove()

    if make_fig:
        # Remove empty subplots if n_species is not a multiple of n_cols
        for i in range(n_species, n_rows * n_cols):
            fig.delaxes(axs.flatten()[i])
        # Give plot a title
        fig.suptitle(
            f"SHAP Interaction of {response} and {interaction}\n\n",
            fontsize=20,
            weight="bold",
        )
        plt.tight_layout()
        plt.savefig(
            f"{dir_analysis}/{response}-{interaction}_no-labels-{remove_labels}{filesuffix}.png"
        )
        if show:
            plt.show()
        else:
            plt.close()
    else:
        print("🚨 No plots were made because 'make_fig = False' in plot_shap_interaction2()!")
        
    # Save model fits df
    pd.DataFrame(df_modelfits).to_csv(f"{dir_analysis}/{response}-{interaction}_model_fits{filesuffix}.csv", index=False)

# ----------------------------------
def plot_shap_response(
    df,
    response,
    scale_shap=True,
    scale_response=True,
    remove_labels=False,
    dir_analysis=None,
    show=True,
    filesuffix="",
    make_fig=True,
):
    
    # Make dir
    dir_analysis = dir_analysis + "/shap_single"
    os.makedirs(dir_analysis, exist_ok=True)
            
    
    # Clean the dataset so that all rows where response is na are removed
    remove_these = []
    for species in df["subset_group"].values:

        check1 = get_relevant_feature(df, species, response)
        if check1 is None:
            remove_these.append(species)

    # print(
    #     f" - For {response}, the following species are removed due to missing features: {remove_these}"
    # )

    df_plot = df.query("subset_group not in @remove_these")

    # Inputs
    df_modelfits = []
    unique_species = df_plot["subset_group"].unique()
    n_species = len(unique_species)
    if n_species > 20:
        n_cols = 6
        figx=25
    else:
        n_cols = 4
        figx = 15
    n_rows = (n_species + n_cols - 1) // n_cols  # Calculate the number of rows needed

    if make_fig:
        if n_species <= n_cols:
            fig, axs = plt.subplots(
                n_rows, n_cols, figsize=(figx, 4), sharey=False, sharex=False
            )
        elif remove_labels:
            fig, axs = plt.subplots(
                n_rows, n_cols, figsize=(figx, 2 * n_rows), sharey=False, sharex=False
            )
        else:
            fig, axs = plt.subplots(
                n_rows, n_cols, figsize=(figx, 3 * n_rows), sharey=False, sharex=False
            )
                    
    for idx, species in enumerate(unique_species):

        # Verbose
        # print(
        #     f" - Plotting species {species} ({idx + 1}/{n_species}) \t\t |{feat_response}}"
        # )
        
        # Get subplot
        row = idx // n_cols
        col = idx % n_cols

        # Get features to plot
        feat_response = get_relevant_feature(df_plot, species, response)

        # Load SHAP data (using new code in the backend)
        filepath = df_plot.query("subset_group == @species")["dir"].values[0]
        shap_values = load_shap(filepath + "/new_shap/approximated/shap_values_test.pkl")
        X_shap = pd.read_csv(f"{filepath}/final_model/X_test.csv", index_col=0)
        feature_names = pd.Series(X_shap.columns)

        # Get values for feature
        feat_response_values = X_shap[feat_response].values
        pos_response = feature_names[feature_names == feat_response].index[0]
        shap_values = shap_values[:, pos_response, 1].values
        
        # Scale feature and shap values
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        if scale_response:
            feat_response_values = scaler.fit_transform(feat_response_values.reshape(-1, 1)).flatten()
            
            # Inverse SPEI scale to make easier to interpret
            if response == "SPEI":
                feat_response_values = 1 - feat_response_values
        else:
            # Inverse SPEI scale to make easier to interpret
            if response == "SPEI":
                feat_response_values = -feat_response_values
                
        if scale_shap:
            shap_values = scaler.fit_transform(shap_values.reshape(-1, 1)).flatten()
        
        # Get LM for direction of response
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        
        X = pd.DataFrame(feat_response_values)
        y = pd.Series(shap_values)
        
        # ! Fit linear, quadratic, and log-transformed models
        
        
        # Fit linear model
        lin_lm = LinearRegression()
        lin_lm.fit(X, y)
        
        # Get coefficients
        lin_pred = lin_lm.predict(X)
        lin_r2 = lin_lm.score(X, y)
        lin_rmse = np.sqrt(np.mean((lin_pred - y) ** 2))
        
        # -- Code to get pval --
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

        # Calculate residuals
        residuals = y - lin_pred
        n_samples = X.shape[0]

        # Calculate standard errors
        rss = np.sum(residuals ** 2)
        rse = np.sqrt(rss / (n_samples - 2))  # Degrees of freedom: n - 2
        se_slope = rse / np.sqrt(np.sum((X - np.mean(X)) ** 2))

        # Get t-statistic for the slope
        slope = lin_lm.coef_[0]
        t_stat = slope / se_slope
        lin_pval = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n_samples - 2))
        lin_pval = lin_pval[0]
        # -- Code to get pval --
        
        # Quadratic Model
        quad = PolynomialFeatures(degree=2)
        quad_X = quad.fit_transform(X)
        quad_lm = LinearRegression()
        quad_lm.fit(quad_X, y)
        quad_pred = quad_lm.predict(quad_X)
        quad_r2 = quad_lm.score(quad_X, y)
        quad_rmse = np.sqrt(np.mean((quad_pred - y) ** 2))
        
        # Calculate AIC by
        lin_aic = calculate_aic_from_fitted_model(lin_lm, X, y)
        quad_aic = calculate_aic_from_fitted_model(quad_lm, quad_X, y)
        
        # Set solid linetype
        linetype = "solid"
        
        # if lin_r2 > quad_r2:
        if True or lin_aic < quad_aic:
            
            pred = lin_pred
            equation = f"{lin_lm.coef_[0]:.2f}x + {lin_lm.intercept_:.2f} | R² = {lin_r2:.2f} | p = {lin_pval:.2f}"
            
            # Make line dotted if not significant
            if lin_pval > 0.05:
                linetype = "dotted"
            
            idf = pd.DataFrame(
                {
                    "species": species,
                    "model": "linear",
                    "coef_1": lin_lm.coef_[0],
                    "coef_2": 0,
                    "intercept": lin_lm.intercept_,
                    "r2": lin_r2,
                    "rmse": lin_rmse,
                    "pval_1" : lin_pval,
                }, index=[0]
            )
            
        else:
            pred = quad_pred
            equation = f"{quad_lm.coef_[1]:.2f}x + {quad_lm.coef_[2]:.2f}x² + {quad_lm.intercept_:.2f} | R² = {quad_r2:.2f}"
            idf = pd.DataFrame(
                {
                    "species": species,
                    "model": "quadratic",
                    "coef_1": quad_lm.coef_[1],
                    "coef_2": quad_lm.coef_[2],
                    "intercept": quad_lm.intercept_,
                    "r2": quad_r2,
                    "rmse": quad_rmse,
                    "pval_1" : "todo",
                }, index=[0]
            )
        
        # Append to list
        df_modelfits.append(idf)
        
        # ! START PLOT -----------------------------------------------------------
        if make_fig:
            # Remove predictions below 0 because shown data is scaled to 0
            if scale_shap:
                pred[pred < 0] = np.nan
                
            # Plot SHAP values
            if response == "SPEI":
                # color = "gold"
                color = "Greens"
            elif response == "Temperature":
                color = "Oranges"
            else:
                color = "grey"
            
            if n_species > n_cols:
                ax = axs[row, col]
            else:
                ax = axs[col]

            # Add scatter
            # scatter = ax.scatterx.scatter(
            #     x=feat_response_values,
            #     y=shap_values,
            #     edgecolors="black"black",
            #     color=colorr,
            #     linewidths=0.15
            #     alpha=0.5,
            #     s=20=0.5,
            #     s=20,
            # )
            
            # Create 2D density plot
            sns.kdeplot(
                x=feat_response_values,
                y=shap_values,
                cmap=color,
                fill=True,
                ax=ax,
            )
            
            # Add lm line
            df_lm = pd.DataFrame({"x": X[0], "y": pred}).sort_values("x")
            # display(df_lm)
            ax.plot(df_lm["x"], df_lm["y"], color="red", linewidth=1, linestyle=linetype)
            
            # Add lm equation
            ax.text(
                0.5,
                0.1,
                equation,
                transform=ax.transAxes,
                fontsize=10,
                fontweight="bold",
                verticalalignment="bottom",
                horizontalalignment="center",
                color="red",
            )
            
            # Add layout
            ylabel = "SHAP Value"
            xlabel = feat_response
            
            if scale_shap:
                ylabel = "Scaled SHAP Value"
                
            if scale_response:
                xlabel = "Scaled " + xlabel
            else:
                xlabel = "Original " + xlabel    
                if response == "SPEI":
                    xlabel =  xlabel + (" (mirrored by taking negative)")
            
            
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{species}", weight="bold")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if remove_labels:
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.set_xticks([])
                ax.set_yticks([])

    if make_fig:
        # Remove empty subplots if n_species is not a multiple of n_cols
        for i in range(n_species, n_rows * n_cols):
            fig.delaxes(axs.flatten()[i])

        # Give plot a title
        fig.suptitle(
            f"SHAP Scatterplot for {response}\n\n",
            fontsize=20,
            weight="bold",
        )

        plt.tight_layout()
        plt.savefig(
            f"{dir_analysis}/{response}_no-labels-{remove_labels}{filesuffix}.png"
        )
        if show:
            plt.show()
        else:
            plt.close()
    else:
        print("🚨 No plots were made because 'make_fig = False' in plot_shap_response()!")
    
            
    # Save model fits df
    df_modelfits = pd.concat(df_modelfits, axis=0)
    # Add suffix if filesuffix is not empty
    if filesuffix != "":
        df_modelfits["suffix"] = filesuffix
    # Save it
    pd.DataFrame(df_modelfits).to_csv(f"{dir_analysis}/{response}_model_fits{filesuffix}.csv", index=False)
    
    
def calculate_aic_from_fitted_model(model, X, y):
    """
    Calculate the Akaike Information Criterion (AIC) for a fitted linear regression model.

    Parameters:
    model (sklearn.base.RegressorMixin): The fitted model.
    X (np.ndarray): The input feature matrix.
    y (np.ndarray): The target vector.

    Returns:
    float: The AIC value.
    """
    # Number of parameters (including the intercept)
    k = X.shape[1] + 1  # number of features + intercept

    # Calculate the RSS (Residual Sum of Squares)
    y_pred = model.predict(X)
    rss = np.sum((y - y_pred) ** 2)

    # Number of data points
    n = X.shape[0]

    # Calculate the AIC
    aic = n * np.log(rss / n) + 2 * k

    return aic


def stepwise_regression(X, y, species):
    """
    Perform stepwise regression on polynomial features including interaction terms.
    
    Parameters:
    X (pd.DataFrame): Input features with exactly two columns.
    y (pd.Series or np.array): Target variable.

    Returns:
    pd.DataFrame: DataFrame containing model details and coefficients.
    """
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.feature_selection import SequentialFeatureSelector
    from sklearn.metrics import r2_score, mean_squared_error
    
    # Check if X has exactly two columns
    if X.shape[1] != 2:
        raise ValueError("X must have exactly two columns.")

    # Preserve the column names
    col_names = X.columns
    
    # Generate polynomial features up to degree 2
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    # Column names for reference
    poly_features = poly.get_feature_names_out(col_names)
    
    # Manually add interaction terms if necessary
    if f"{col_names[0]} {col_names[1]}" not in poly_features:
        interaction_term = (X.iloc[:, 0] * X.iloc[:, 1]).values.reshape(-1, 1)
        X_poly = np.hstack((X_poly, interaction_term))

    if f"{col_names[0]}^2 {col_names[1]}^2" not in poly_features:
        quadratic_interaction_term = (X.iloc[:, 0] ** 2 * X.iloc[:, 1] ** 2).values.reshape(-1, 1)
        X_poly = np.hstack((X_poly, quadratic_interaction_term))
    
    # Update feature names after adding custom interaction terms
    updated_feature_names = list(poly_features)
    if f"{col_names[0]} {col_names[1]}" not in poly_features:
        updated_feature_names.append(f"{col_names[0]} {col_names[1]}")
    if f"{col_names[0]}^2 {col_names[1]}^2" not in poly_features:
        updated_feature_names.append(f"{col_names[0]}^2 {col_names[1]}^2")
    
    # Convert to DataFrame
    X_poly_df = pd.DataFrame(X_poly, columns=updated_feature_names)
    
    # Perform stepwise regression using SequentialFeatureSelector
    estimator = LinearRegression()
    sfs = SequentialFeatureSelector(estimator, n_features_to_select="auto", direction='forward', cv=5)
    sfs.fit(X_poly_df, y)
    
    # Get selected features
    selected_features = X_poly_df.columns[sfs.get_support()].tolist()
    
    # Ensure the selected features are valid
    X_poly_selected = X_poly_df[selected_features]
    
    # Fit the final model
    final_model = LinearRegression()
    final_model.fit(X_poly_selected, y)
    
    # Calculate metrics
    y_pred = final_model.predict(X_poly_selected)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    # Create a dictionary to store the coefficients
    model_dict = {
        "species": species,
        "intercept": final_model.intercept_,
        "r2": r2,
        "rmse": rmse,
    }
    
    # Add all possible coefficient names, setting to 0 if not included in the final model
    for feature in updated_feature_names:
        if feature in selected_features:
            model_dict[f"{feature}"] = final_model.coef_[selected_features.index(feature)]
        else:
            model_dict[f"{feature}"] = 0
    
    return model_dict

def turn_dictionary_into_df(dic):
    lll = []

    for k in dic.keys():
        lll.append(pd.DataFrame({k: dic[k]}))

    return pd.concat(lll, axis=1)


def bootstrap_optimal_tss(
    y_true,
    y_pred_probs,
    n_bootstraps=100,
    random_state=42,
    thresholds=np.linspace(0.0, 1.0, num=101),
    show=False,
):
    # List to store bootstrapped TSS and specificity values for each threshold
    tss_values = {threshold: [] for threshold in thresholds}
    specificity_values = {threshold: [] for threshold in thresholds}

    rng = np.random.RandomState(random_state)

    for _ in range(n_bootstraps):
        # Stratified Bootstrap sampling
        y_true_boot, y_pred_probs_boot = resample(
            y_true, y_pred_probs, replace=True, stratify=y_true, random_state=rng
        )

        for threshold in thresholds:
            # Binary predictions at the current threshold
            y_pred_boot = (y_pred_probs_boot >= threshold).astype(int)

            # Confusion matrix components
            tn, fp, fn, tp = confusion_matrix(y_true_boot, y_pred_boot).ravel()

            # Calculate Sensitivity and Specificity
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)

            # Calculate TSS
            tss = sensitivity + specificity - 1

            # Append TSS and specificity to the corresponding threshold list
            tss_values[threshold].append(tss)
            specificity_values[threshold].append(specificity)

    # Calculate mean and standard deviation for TSS and specificity at each threshold
    tss_means = {threshold: np.mean(tss_values[threshold]) for threshold in thresholds}
    tss_stds = {threshold: np.std(tss_values[threshold]) for threshold in thresholds}
    specificity_means = {
        threshold: np.mean(specificity_values[threshold]) for threshold in thresholds
    }
    specificity_stds = {
        threshold: np.std(specificity_values[threshold]) for threshold in thresholds
    }

    # Find the threshold with the maximum mean TSS
    optimal_threshold = max(tss_means, key=tss_means.get)
    optimal_tss_mean = tss_means[optimal_threshold]
    optimal_tss_std = tss_stds[optimal_threshold]
    
    optimal_specificity_mean = specificity_means[optimal_threshold]
    optimal_specificity_std = specificity_stds[optimal_threshold]

    if show:
        # Plot TSS and specificity means
        plt.figure(figsize=(12, 6))
        plt.plot(thresholds, list(tss_means.values()), label="TSS (mean)", color="blue")
        plt.plot(
            thresholds,
            list(specificity_means.values()),
            label="Specificity (mean)",
            color="green",
        )
        plt.axvline(
            optimal_threshold,
            color="red",
            linestyle="--",
            label=f"Optimal Threshold = {optimal_threshold:.2f}",
        )
        plt.fill_between(
            thresholds,
            [tss_means[threshold] - tss_stds[threshold] for threshold in thresholds],
            [tss_means[threshold] + tss_stds[threshold] for threshold in thresholds],
            color="blue",
            alpha=0.2,
        )
        plt.fill_between(
            thresholds,
            [
                specificity_means[threshold] - specificity_stds[threshold]
                for threshold in thresholds
            ],
            [
                specificity_means[threshold] + specificity_stds[threshold]
                for threshold in thresholds
            ],
            color="green",
            alpha=0.2,
        )
        plt.xlabel("Threshold")
        plt.ylabel("Metric Value")
        plt.title("TSS and Specificity across Thresholds")
        plt.legend()
        plt.grid(True)
        plt.show()

    return optimal_tss_mean, optimal_tss_std, {"optimal_threshold": optimal_threshold, "optimal_specificity_mean": optimal_specificity_mean, "optimal_specificity_std": optimal_specificity_std}

# -----------------------------------------------------------------------------------------------

def ___maps_evolution_forest_cover___():
    pass

def aggregate_raster_to_csv(
    input_raster_path,
    output_csv_path,
    agg_factor_m,
    save_file=True,
    verbose=False,
):
    """
    Aggregates a raster file to a coarser grid and extracts pixel values and coordinates to save in CSV.

    Parameters:
    input_raster_path (str): Path to the input raster file (original at 1x1km resolution).
    output_csv_path (str): Path to save the output CSV file.
    agg_factor_m (int or float): Aggregation factor in kilometers. For example, 25000 for 25x25 km grid.
    save_file (bool): Whether to save the output CSV file (True by default).

    Returns:
    pandas.DataFrame: The aggregated pixel values and coordinates as a DataFrame.
    """
    # Load the raster
    with rasterio.open(input_raster_path) as src:
        raster_data = src.read(1)  # Reading the first band of the raster
        original_transform = src.transform
        original_width = src.width
        original_height = src.height
        original_res = src.res[0]  # Assuming square pixels, resolution in km
        original_crs = src.crs
        most_common_value = mode(raster_data, axis=None).mode

    # Print the raster properties
    if verbose:
        print(f"Loaded file: {input_raster_path}")
        print(f"Original raster resolution: {original_res:.2f}m")
        print(f"Original raster dimensions: {original_width} x {original_height}")
        print(f"Original CRS: {original_crs}")
        print(f"Most common value in the raster: {most_common_value} - set to NaN!")

    # Set the most common value to NaN
    raster_data = np.where(raster_data == most_common_value, np.nan, raster_data)

    if original_res == 0:
        raise ValueError(
            "Original raster resolution is zero, please check the input file."
        )

    # Calculate aggregation factor in terms of pixels
    agg_factor = agg_factor_m / original_res

    if agg_factor <= 0:
        raise ValueError(
            f"Invalid aggregation factor calculated: {agg_factor}. Ensure agg_factor_m > original_res."
        )

    # Ensure aggregation factor is an integer
    agg_factor = int(np.floor(agg_factor))

    if agg_factor < 1:
        raise ValueError("Aggregation factor is too small. It must be at least 1.")

    # Aggregate the raster data by the aggregation factor
    if verbose:
        print("Aggregate raster data...")

    new_height = original_height // agg_factor
    new_width = original_width // agg_factor

    aggregated_raster = (
        raster_data[: new_height * agg_factor, : new_width * agg_factor]
        .reshape(new_height, agg_factor, new_width, agg_factor)
        .mean(axis=(1, 3))
    )

    # Flatten the aggregated raster for easy export
    if verbose:
        print("Flaten raster data...")
    flattened_values = aggregated_raster.flatten()

    # Create a list to store the coordinates
    coords = []

    # Affine transformation for the new aggregated grid
    if verbose:
        print("Transform raster data...")
    agg_transform = Affine(
        original_transform.a * agg_factor,  # scaled pixel size in X direction
        original_transform.b,
        original_transform.c,
        original_transform.d,
        original_transform.e * agg_factor,  # scaled pixel size in Y direction
        original_transform.f,
    )

    # Calculate centroid coordinates for each aggregated pixel
    if verbose:
        print("Calculate new centroids...")
    for row in range(new_height):
        for col in range(new_width):
            x, y = agg_transform * (col + 0.5, row + 0.5)  # Centroid of each pixel
            coords.append((x, y))

    # Create a dataframe with the flattened values and their corresponding coordinates
    df = pd.DataFrame(coords, columns=["x", "y"])
    df["value"] = flattened_values

    # Save the dataframe to a CSV file if requested
    if save_file:
        df.to_csv(output_csv_path, index=False)

    return df

def get_trend_per_pixel(
    df,
    group_var="group",
    date_var="date",
    value_var="value",
):
    # Check if requested variables are present
    if not all([group_var, date_var, value_var]):
        raise ValueError("Please provide all variables!")

    # Get group
    group = df[group_var].unique()

    # Check if all values are NaN
    if df[value_var].isnull().all():
        return pd.DataFrame({group_var: group, value_var: np.nan}, index=[0])
    # If not, calculate the trend
    lm = LinearRegression()
    lm.fit(df[[date_var]], df[[value_var]])
    return pd.DataFrame({group_var: group, value_var: lm.coef_[0]}, index=[0])

def get_trend_per_pixel_loop(
    df,
    group_var="xy",
    date_var="date",
    value_var="value",
):
    df_list = []
    for group in df[group_var].unique():
        idf = df.query(f"{group_var} == '{group}'").copy()
        idf = get_trend_per_pixel(idf, group_var, date_var, value_var)
        df_list.append(idf)
    df_list = pd.concat(df_list)
    return df_list


def get_trend_per_pixel_loop_mp(
    df,
    group_var="xy",
    date_var="date",
    value_var="value",
    ):
    
    # Split into list
    df_list = split_df_into_list_of_group_or_ns(df, 10, group_var)

    # Run in parallel
    df_mp = run_mp(
        get_trend_per_pixel_loop,
        df_list,
        combine_func=pd.concat,
        group_var=group_var,    
        date_var=date_var,    
        value_var=value_var,    
        num_cores=10,
    )

    return df_mp

def make_map_for_temp_prec_cover(
    df,
    dataset,
    season=None,
    ax=None,
    pixel_res=500j,
    textsize=20,
    cbar_pad=-0.02,
    cbar_fraction = 0.2,
    cbar_shrink = 0.6,
    cbar_aspect = 20,
    contour_levels=7,
    tick_interval=2,
    filepath=None,
    load_file=True,
):

    from scipy.interpolate import griddata
    
    # Check input
    if dataset != "treecover" and season is None:
        raise ValueError("Season needs to be defined now!")
    
    if dataset != "treecover" and season not in ["13", "hi", "pr", "et", "au"]:
        raise ValueError("Season needs to be one of ['13', 'hi', 'sp', 'et', 'au']")

    ax_is_none = ax is None

    # Load the data
    gdf = df.copy()

    # Get local vars
    if dataset == "prec":
        cmap = "RdYlBu"
        legend_label = "mm yr$^{-2}$"
        fformat = "%.0f"
        
        if season == "13":
            # contour_levels = [-8, -4, -2, -1, 0, 1, 2, 4, 8]
            contour_levels = [-8, -4, -2, 0, 2, 4, 8]
        else:
            # contour_levels = [-4, -2, -1, -0.5, 0, 0.5, 1, 2, 4]
            contour_levels = [-3, -2, -1, 0, 1, 2, 3, 4]
        
        extend="both"
        vmin = contour_levels[0]-1
        vmax = contour_levels[-1]+1
        
        # vmin = min(gdf["value"].min(), -gdf["value"].max())
        # vmax = max(gdf["value"].max(), -gdf["value"].min())
        # ticks = np.linspace(vmin, vmax, 3)
    elif dataset == "tmoy":
        cmap = "Reds"
        legend_label = "°C yr$^{-1}$"
        vmin = 0
        vmax = gdf["value"].max()
        fformat = "%.2f"
        
        contour_levels = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
        # if 
        # contour_levels = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
        extend="both"
        vmin = contour_levels[0]
        vmax = contour_levels[-1]
        
        # vmax = max(gdf["value"].max(), -gdf["value"].min())
        # vmin = min(gdf["value"].min(), -gdf["value"].max())
        # ticks = np.linspace(vmin, vmax, 3)
    elif dataset == "treecover":
        cmap = "Greens"
        legend_label = "Forest cover (%)"
        fformat = "%.0f"
        vmin = 0
        vmax = 100
        extend="neither"
        # ticks = np.linspace(vmin, vmax, 3)
    else:
        raise ValueError(f"Dataset not recognized: {dataset}")

    # Define grid for interpolation
    grid_x, grid_y = np.mgrid[
        gdf["x"].min() : gdf["x"].max() : pixel_res,
        gdf["y"].min() : gdf["y"].max() : pixel_res,
    ]

    # Interpolate using scipy's griddata (you can switch to 'cubic' for smoother interpolation)
    file_interpolated = filepath.replace(
        ".png", f"_interpolated_at_{pixel_res}px.feather"
    )
    if os.path.exists(file_interpolated) and load_file:
        grid_z = pd.read_feather(file_interpolated)
        grid_z = grid_z.to_numpy()
    else:
        grid_z = griddata(
            (gdf["x"], gdf["y"]), gdf["value"], (grid_x, grid_y), method="linear"
        )
        pd.DataFrame(grid_z).to_feather(file_interpolated)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    # Plotting the interpolated grid as a smooth heatmap
    # contour_levels = np.linspace(vmin, vmax, num=contour_levels)
    
    ax.contourf(
        grid_x,
        grid_y,
        grid_z,
        levels=contour_levels,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extend=extend,
    )

    # Plot the border of France
    france = get_shp_of_region("cty")
    france.boundary.plot(ax=ax, color="black", linewidth=0.5)

    # Remove axis
    ax.axis("off")

    # Style colorbar
    fig = ax.get_figure()
    cbar = fig.colorbar(
        ax.contourf(grid_x, grid_y, grid_z, levels=contour_levels, cmap=cmap, vmin=vmin, vmax=vmax, extend=extend),
        ax=ax,
        orientation="horizontal",
        pad=cbar_pad,
        fraction=cbar_fraction,
        shrink=cbar_shrink,
        aspect=cbar_aspect,
        format=fformat,
        extend=extend,
    )
    cbar.ax.set_xlabel(
        legend_label,  # Add label to the colorbar
        fontsize=textsize,
        # labelpad=-55,
    )

    # Modify colorbar ticks to show every second tick
    ticks = cbar.get_ticks()  # Get current ticks
    cbar.set_ticks(ticks[::tick_interval])  # Set every second tick
    cbar.ax.tick_params(labelsize=textsize)

    if ax_is_none:
        if filepath is not None:
            plt.savefig(filepath, bbox_inches="tight")
            print(f"Saved fig to {filepath}")
        else:
            plt.show()
        plt.close()
    else:
        return ax
    
def produce_dfs_for_climate_evolution(
    agg_factor_km=25,
    dataset="tmoy",
    first_year=2000,
    last_year=2020,
    season="13",
    load_file = True,
    ):
    # Set local vars
    agg_factor_m = agg_factor_km * 1000
    range_years = np.arange(first_year, last_year + 1)

    # Get files
    files_all = glob.glob(f"/Volumes/WD - ExFat/IFNA/digitalis_v3/raw/1km/{dataset}/*.tif")
    files_all = pd.DataFrame(files_all, columns=["files"])
    # Attach year
    files_all["year"] = files_all["files"].apply(lambda x: x.split("/")[-1].split("_")[1])
    files_all["year"] = files_all["year"].astype(int)
    # Attach month
    files_all["month"] = files_all["files"].apply(
        lambda x: x.split("/")[-1].split("_")[2].split(".")[0]
    )
    # Remove 6120 files
    files_all = files_all.query("year != 6120")
    # Sort by year and month
    files_all = files_all.sort_values(by=["year", "month"]).reset_index(drop=True)

    # Folder to save aggregated files (not really needed because quick enough)
    agg_folder = f"/Volumes/WD - ExFat/IFNA/digitalis_v3/processed/aggregated-to-{agg_factor_km}km/{dataset}"
    os.makedirs(agg_folder, exist_ok=True)
    
    # Get filename
    filename_mp = (
        agg_folder
        + f"/trend_per_pixel-months_{season}-from_{first_year}_to_{last_year}.feather"
    )
    filename_lm = (
        agg_folder
        + f"/trend_per_pixel-months_{season}-from_{first_year}_to_{last_year}_lm.feather"
    )
    
    # Load lm file if exists
    if os.path.exists(filename_lm) and load_file:
        df_lm = pd.read_feather(filename_lm)
        return df_lm

    # Attach csv filenames, make loop to make it easier
    files_all["csv_files"] = ""

    for i, row in files_all.iterrows():
        files_all.loc[i, "csv_files"] = f"{agg_folder}/{row.year}_{row.month}.csv"

    # Filter for season
    files = files_all.query(f"month == '{season}'").reset_index(drop=True)
    # Filter for range of years
    files = files.query("year >= @first_year & year <= @last_year").reset_index(drop=True)
    
    # Aggregate raster to csv
    dfs = []

    for i in tqdm(range(len(files))):
        df = aggregate_raster_to_csv(
            input_raster_path=files.loc[i, "files"],
            output_csv_path=files.loc[i, "csv_files"],
            agg_factor_m=agg_factor_m,
            save_file=False,
        )

        # Get year from filename
        df["year"] = files.loc[i, "year"]
        df["month"] = files.loc[i, "month"]

        # Append to list
        dfs.append(df)

    # Concatenate
    df = pd.concat(dfs).reset_index(drop=True)

    # Attach unique coordinates
    df["xy"] = df["x"].astype(str) + "_" + df["y"].astype(str)

    # Divide values by 10 to get correct value
    df["value"] = df["value"] / 10
    
    # Load if file exists
    if os.path.exists(filename_mp) and load_file:
        df_mp = pd.read_feather(filename_mp)
        chime.info()
        print(f"File loaded from disk!: {filename_mp}")
    else:
        # Calculate trends
        # At resolution of 1km, it takes about 40 minutes
        df_mp = get_trend_per_pixel_loop_mp(
            df.dropna(),
            group_var="xy",
            date_var="year",
            value_var="value",
        )
        chime.success()
        df_mp.to_feather(filename_mp)
        print(f"File saved to disk!: {filename_mp}")
    
    # todo: Cleaning should be part of saving too because takes up some time...        
    # df_mp was calculated with removing NA values, so we need to add these coordinates back
    df_lm = pd.merge(df[["xy"]], df_mp, on="xy", how="left")
    # Attach x and y
    df_lm["x"] = df_lm["xy"].apply(lambda x: x.split("_")[0]).astype(float)
    df_lm["y"] = df_lm["xy"].apply(lambda x: x.split("_")[1]).astype(float)
    
    # Save lm file
    df_lm.to_feather(filename_lm)
    
    return df_lm

# -----------------------------------------------------------------------------------------------
def ___RF_Response_Analysis___():
    pass

def aggregate_spei_temp_responses(my_species, df, roc_threshold=0.6):
    # To be included, the ROC-AUC must be above the threshold
    nrows_before = df.shape[0]
    df = df[df["test_boot_mean"] > roc_threshold].copy()
    nrows_after = df.shape[0]
    if nrows_before != nrows_after:
        print(
            f"🚨 {my_species} - Runs with roc < {roc_threshold} were removed. From {nrows_before} to {nrows_after} runs"
        )

    # Attach importance ratio of SPEI to Temperature
    df["imp_ratio_spei_temp"] = df["SPEI - Importance"] / df["Temperature - Importance"]
    df["imp_ratio_spei_temp"] = df["imp_ratio_spei_temp"].round(1)
    df["imp_sum_spei_temp"] = df["SPEI - Importance"] + df["Temperature - Importance"]

    # Clean df
    df = (
        df[
            [
                "species",
                "subset_group",
                "test_boot_mean",
                "SPEI - Metrics",
                "Temperature - Metrics",
                "imp_ratio_spei_temp",
                "imp_sum_spei_temp",
            ]
        ]
        .dropna()  # Todo
        .reset_index(drop=True)
    )

    # Extract SPEI metrics
    if str(df["SPEI - Metrics"].values[0]) == "nan":
        df["SPEI - Metrics"] = "None"
        df["SPEI - Duration"] = np.nan
        df["SPEI - Season"] = np.nan
        df["SPEI - Season back"] = np.nan
        df["SPEI - Duration Agg."] = np.nan
    else:
        df["SPEI - Metrics"] = df["SPEI - Metrics"].apply(ast.literal_eval)
        df["SPEI - Metrics"] = df["SPEI - Metrics"].apply(lambda x: x[0])
        df["SPEI - Duration"] = df["SPEI - Metrics"].apply(
            lambda x: x.split("spei")[1].split("-")[0]
        )
        df["SPEI - Duration"] = df["SPEI - Duration"].astype(int)
        df["SPEI - Season"] = df["SPEI - Metrics"].apply(
            lambda x: x.split("-")[1].split("_")[0]
        )

        # Rename seasons from month to season
        df["SPEI - Season"] = df["SPEI - Season"].map(
            {
                "feb": "Winter",
                "may": "Spring",
                "aug": "Summer",
                "nov": "Fall",
            }
        )

        # Backcalculate SPEI season from duration
        df["SPEI - Season back"] = ""
        for i in range(df.shape[0]):
            df.loc[i, "SPEI - Season back"] = backcalculate_season(
                df.loc[i, "SPEI - Season"], df.loc[i, "SPEI - Duration"]
            )

        # Merge SPEI durations into short - medium - long
        df["SPEI - Duration Agg."] = df["SPEI - Duration"].map(
            {
                1: "Short",
                3: "Short",
                6: "Short",
                9: "Medium",
                12: "Medium",
                15: "Medium",
                18: "Long",
                21: "Long",
                24: "Long",
            }
        )

    # Extract Temperature metrics
    if str(df["Temperature - Metrics"].values[0]) == "nan":
        df["Temperature - Metrics"] = "None"
        df["Temperature - Season"] = np.nan

    else:
        df["Temperature - Metrics"] = df["Temperature - Metrics"].apply(
            ast.literal_eval
        )
        df["Temperature - Metrics"] = df["Temperature - Metrics"].apply(lambda x: x[0])
        df["Temperature - Season"] = df["Temperature - Metrics"].apply(
            lambda x: x.split("_")[1]
        )

        df["Temperature - Season"] = df["Temperature - Season"].map(
            {
                "win": "Winter",
                "spr": "Spring",
                "sum": "Summer",
                "aut": "Fall",
            }
        )

    # Truncate run name
    df["performance"] = (
        df["subset_group"]
        .str.split(" ")
        .str[1]
        .str.replace("(", "")
        .str.replace(")", "")
    )
    df["subset_group"] = (
        df["subset_group"].str.split(" ").str[0].str.replace("run_", "")
    )

    # ! Group by species, season, and duration
    grouping = [
        "species",
        "Temperature - Metrics",
        "SPEI - Metrics",
        # "Temperature - Season",
        # "SPEI - Duration Agg.",
        # "SPEI - Duration",
        # "SPEI - Season",
        # "SPEI - Season back",
    ]

    # Calculate the group sizes
    group_sizes = df.groupby(grouping).size().reset_index(name="group_size")

    # Aggregate the mean and std of 'test_boot_mean', and aggregate lists for the specified columns
    df_agg = (
        df.groupby(grouping)
        .agg(
            spei_durations=("SPEI - Duration", sorted_list),
            spei_seasons_back=("SPEI - Season back", sorted_list),
            spei_seasons=("SPEI - Season", sorted_list),
            runs=("subset_group", sorted_list),
            spei_metrics=("SPEI - Metrics", sorted_list),
            temp_seasons=("Temperature - Season", sorted_list),
            temp_metrics=("Temperature - Metrics", sorted_list),
            imp_ratio_spei_temp_mean=("imp_ratio_spei_temp", "mean"),
            imp_sum_spei_temp_mean=("imp_sum_spei_temp", "mean"),
            performance_mean=("test_boot_mean", "mean"),
            performance_std=("test_boot_mean", "std"),
            # performances=("performance", sorted_list),
        )
        .reset_index()
    )

    # display(df_agg)

    # Merge the group sizes with the aggregated DataFrame
    df_merged = (
        df_agg.merge(group_sizes, on=grouping)
        .sort_values("group_size", ascending=False)
        .reset_index(drop=True)
    )

    # Add percentage of total runs
    df_merged["perc_runs"] = (
        df_merged["group_size"] / df_merged["group_size"].sum() * 100
    )
    df_merged["perc_runs"] = df_merged["perc_runs"].astype(int)

    df_merged["imp_ratio_spei_temp_mean"] = df_merged["imp_ratio_spei_temp_mean"].round(
        2
    )
    df_merged["imp_sum_spei_temp_mean"] = df_merged["imp_sum_spei_temp_mean"].round(0)
    df_merged["performance_mean"] = df_merged["performance_mean"].round(2)
    df_merged["performance_std"] = df_merged["performance_std"].round(2)

    # Result
    df_merged = move_vars_to_front(df_merged, ["species", "group_size", "perc_runs"])
    return df_merged, df


# ! ---------------------------------------------------------------------
def backcalculate_season(start, months):
    # User input
    valid_seasons = ["Winter", "Spring", "Summer", "Fall"]
    # Make sure the number of months is a multiple of 3
    if months == 1 or months == 3:
        return start
    elif months % 3 != 0:
        raise ValueError("The number of months must be a multiple of 3.")

    # Create a mapping from season name to index and vice versa
    season_to_index = {season: i for i, season in enumerate(valid_seasons)}
    index_to_season = {i: season for i, season in enumerate(valid_seasons)}

    # Get the starting season's index
    start_index = season_to_index[start]

    # Calculate the number of seasons to go back
    seasons_to_go_back = (months // 3) % 4 - 1

    # Calculate the resulting season's index
    final_index = (start_index - seasons_to_go_back) % 4

    # Get the resulting season from the index
    final_season = index_to_season[final_index]

    return final_season


# ! ---------------------------------------------------------------------


def sorted_list(values):
    return sorted(values.unique().tolist())


def make_list(values):
    return values.unique().tolist()


# ! ---------------------------------------------------------------------


def display_fig(path):
    if not os.path.exists(path):
        print(f"🟥 Figure not found: {path}")
    else:
        print(path)
        display(Image(path))


# ! ---------------------------------------------------------------------


def x_get_figs(my_dir, species):
    # Get dir
    i_dir = f"{my_dir}/{species}"
    display_fig(
        f"{i_dir}/by_roc/shap_interactions_2d/SPEI-Temperature_no-labels-False-most_common_runs.png"
    )
    display_fig(f"{i_dir}/by_roc/shap_single/SPEI_no-labels-False-sorted_by_run.png")
    display_fig(
        f"{i_dir}/by_roc/shap_single/Temperature_no-labels-False-sorted_by_run.png"
    )
    display_fig(f"{i_dir}/by_roc/model_performance_sorted_by_test_boot_mean.png")
    display_fig(
        f"{i_dir}/by_roc/feature_importance-shap-by_feature-colored_relative_occurences.png"
    )


def round_half_up(n, decimals=0):
    multiplier = 10**decimals
    return math.floor(n * multiplier + 0.5) / multiplier


def round_to_0_decimals(n, decimals=0):
    for i in (np.arange(15, decimals, -1) - 1).tolist():
        n = round_half_up(n, i)
    return n


# ! ----------------------------------------------------------------------------------------------------------------------------------
def get_spei_table(
    df,
    select_change,
    x_var,
    y_var,
    x_var_order,
    y_var_order,
    method_scaling_importance="sum",
    round_decimals=False,
):

    # Get df
    df = df.query("change == @select_change").copy().reset_index(drop=True)

    # Option to level importance and give each species x pattern pair the same weight
    if method_scaling_importance == "sum":
        df["counts"] = df["group_size"]
    elif method_scaling_importance == "scaled_all":
        df["counts"] = df["group_size_rel_all"]
    elif method_scaling_importance == "scaled_valid":
        df["counts"] = df["group_size_rel_val"]

    df_pivot = df.pivot_table(
        index=y_var,
        columns=x_var,
        values="counts",
        aggfunc="sum",
        fill_value=0,
    )

    # Round accurately
    df_pivot = df_pivot / df_pivot.sum().sum() * 100
    if round_decimals:
        df_pivot = df_pivot.map(round_to_0_decimals).astype(int)

    # Ensure the y-axis goes from 1 to 25 and x-axis is in the order of 'feb', 'may', 'aug', 'nov'
    df_pivot = df_pivot.reindex(
        index=y_var_order,
        columns=x_var_order,
        fill_value=0,
    )

    # display(df_pivot)

    return df_pivot


# ! ----------------------------------------------------------------------------------------------------------------------------------
def plot_bars(
    df,
    change,
    x_var,
    x_var_order,
    y_var,
    y_var_order,
    ax=None,
    ylim=None,
    title=None,
    method_scaling_importance="sum",
):

    # Get df
    pivot_df = get_spei_table(
        df=df,
        select_change=change,
        y_var=y_var,
        y_var_order=y_var_order,
        x_var=x_var,
        x_var_order=x_var_order,
        round_decimals=False,
        method_scaling_importance=method_scaling_importance,
    )

    # display(pivot_df)

    # Define colormap and normalize values
    if change == "warmer_drier" or change == "warmer":
        cmap = plt.get_cmap("Reds_r", len(y_var_order))
    elif change == "warmer_wetter" or change == "cooler":
        cmap = plt.get_cmap("Blues_r", len(y_var_order))
    elif change == "cooler_drier" or change == "drier":
        cmap = plt.get_cmap("Oranges_r", len(y_var_order))
    elif change == "cooler_wetter" or change == "wetter":
        cmap = plt.get_cmap("Purples_r", len(y_var_order))
    elif change == "other":
        cmap = plt.get_cmap("Greys_r", len(y_var_order))
    else:
        raise ValueError(f"🟥 Change not recognized: {change}")

    norm = plt.Normalize(vmin=0, vmax=len(y_var_order) - 1)

    # Plotting with matplotlib
    # plt.figure(figsize=(10, 6))  # Adjust figure size if needed

    # Loop through each metric and plot stacked bars with colormap shades
    labels_legend = []
    bottom = None
    for i, metric in enumerate(y_var_order):
        # Count the number of occurrences for each metric
        metric_perc = pivot_df.sum(axis=1)[metric]
        metric_perc = round_to_0_decimals(metric_perc)
        metric_perc = int(metric_perc)
        labels_legend.append(f"{metric} ({metric_perc}%)")
        values = pivot_df.loc[metric, x_var_order].values
        if bottom is None:
            ax.bar(
                x_var_order,
                values,
                label=metric,
                color=cmap(norm(i)),
                edgecolor="black",
                width=1,
            )
            bottom = values
        else:
            ax.bar(
                x_var_order,
                values,
                bottom=bottom,
                label=metric,
                color=cmap(norm(i)),
                edgecolor="black",
                width=1,
            )
            bottom += values

    # Customize the plot
    # ax.set_title("Stacked Bar Chart of Temp Metrics by Season")
    ax.set_xlabel(x_var)
    ax.set_ylabel("Occurrence per species and run (%)")
    ax.legend(title=y_var, loc="best", labels=labels_legend)

    if ylim is not None:
        ax.set_ylim(ylim)

    if title is not None:
        ax.set_title(title)

    # If spei duration is on the x axis set nice ticks
    if x_var == "spei_duration":
        ax.xaxis.set_major_locator(ticker.MultipleLocator(3))

    # Show plot
    # ax.tight_layout()
    # ax.show()

    return ax


# ! ----------------------------------------------------------------------------------------------------------------------------------
def calculate_weighted_mean_importance(df, importance_columns):
    """
    Calculate the weighted mean importance for each species and each importance column.

    Args:
    df (pd.DataFrame): Input DataFrame containing 'species' and importance columns.
    importance_columns (list): List of importance column names.

    Returns:
    pd.DataFrame: DataFrame with weighted mean importance for each species and importance column.
    """
    # Initialize an empty DataFrame to store the results
    result_df = pd.DataFrame()

    # Iterate over each importance column to calculate the weighted mean importance
    for column in importance_columns:
        # Group by species and calculate the sum of importance and the count of occurrences
        species_grouped = df.groupby("species").agg(
            total_importance=(column, "sum"), count=(column, "count")
        )

        # Calculate the weighted mean importance for each species
        species_grouped[f"weighted_mean_{column}"] = (
            species_grouped["total_importance"] / species_grouped["count"]
        )

        # Keep only the weighted mean importance column
        result_df[f"weighted_mean_{column}"] = species_grouped[
            f"weighted_mean_{column}"
        ]

    return result_df.reset_index()


# ! ------------------------------
def f_nuniques(values):
    return values.nunique()


# ! ------------------------------
def main_cat_index(values):
    return values.value_counts().index[0]


# ! ------------------------------
def main_cat_perc(values):
    return values.value_counts(normalize=True).values[0].round(2)


# ! ------------------------------
def draw_pie(
    ax,
    data,
    max_size,
    min_size=0,
    scale_by100perc=False,
    size_scale=1,
    cmap="viridis",
):
    sizes = [
        data["max"].values[0],
        data["min"].values[0],
        data["mean"].values[0],
    ]
    total_size = data["size"].values[0]

    if scale_by100perc:
        scaled_radius = np.sqrt(total_size / 100) * 0.8
    else:
        scaled_radius = np.sqrt(total_size / max_size) * 0.425
        if np.isnan(scaled_radius):
            scaled_radius = 0.0

    # If the size is too small, return empty pie chart
    if scaled_radius <= min_size:
        ax.pie([1], radius=0.001, colors=["w"])
        return
    # Merge sizes with colors
    df_sizes = pd.DataFrame(sizes, columns=["size"], index=["max", "min", "mean"])
    # Attach colors
    df_sizes["color"] = sns.color_palette(cmap, 3)
    # Sort by size
    df_sizes = df_sizes.sort_values("size", ascending=False)

    ax.pie(
        df_sizes["size"],
        radius=scaled_radius * size_scale,
        colors=df_sizes["color"],
        wedgeprops=dict(edgecolor="w", linewidth=0),
        startangle=90,
        counterclock=True,
    )
    ax.set_title("")
    ax.set_ylabel("")
    ax.set_xlabel("")

def response_classification(
    df_all_in,
    scale_shap,
    scale_response,
    roc_threshold,
    dir_save,
    make_plots,
    verbose=True,
):

    # TODO: DEBUG to overwrite species (and uncleaned variables)
    response_pval = 0.05  # Does not matter, pval classification is done later
    pval_threshold = 0.05  # Does not matter, pval classification is done later
    # For debugging purposes (see DEBUG lines below)
    overwrite_species = "Picea abies"
    overwrite_group = 2

    df_patterns = []
    df_models = []
    df_resp_class = []
    runs_dict = {}

    outer_count = 0

    for my_species in df_all_in.species.unique():

        # TODO: DEBUG to overwrite species
        # my_species = overwrite_species
        # if my_species != overwrite_species:
        # continue

        # Verbose
        outer_count += 1
        if verbose:
            print(f"{outer_count}/{len(df_all_in.species.unique())}\t{my_species}")

        # ! Filter for species
        df = df_all_in.query(f"species == '{my_species}'").copy()
        # ! Filter for valid models
        df = df.query("test_boot_mean >= @roc_threshold").copy()
        # ! Aggregate model runs of the same spei-temp combination
        df_agg, df_clean = aggregate_spei_temp_responses(my_species, df)
        df_agg_red = df_agg.copy()
        df_agg_red["group"] = 9999

        group_count = 0
        perc_count = 0

        for i in range(df_agg_red.shape[0]):
            iruns = df_agg_red.loc[i, "runs"].copy()

            # For first run of this species, create new entry
            if df_agg_red.loc[i, "species"] not in runs_dict:
                runs_dict[df_agg_red.loc[i, "species"]] = iruns
            # If species is already in, add up runs
            else:
                runs_dict[df_agg_red.loc[i, "species"]] += iruns

            # If group should be analyzed, continue:
            # Rerun interaction plots for this pattern group
            idf = (
                df.copy()
                .query(f"species == '{my_species}' and run in @iruns")
                .sort_values("run")
            )

            # Add group counter
            group_count += 1
            df_agg_red.loc[i, "group"] = group_count
            idf["group"] = group_count

            # # TODO: IMPORTANT: When removing this part, I need to fix the short-cutting I did above too!
            # # Pick a specific group of spei-temp combinations to check for direction of change
            # if group_count == overwrite_group:
            #     # display(idf)
            #     raise ValueError(
            #         "🟥 I am debugging the plot_shap_response and stop here to get the correct idf."
            #     )

            # Gather models
            idf_model = df_clean.query(f"subset_group in @iruns").copy()
            idf_model["group"] = group_count
            df_models.append(idf_model)

            # Display
            if verbose:
                print(f" ----- GROUP {group_count} of {my_species} -----")
            # display(df_agg_red.loc[i, :].to_frame().T)
            # display(idf)

            # ! Response assessment (check if variable is in the model)
            if idf["SPEI - Metrics"].dropna().__len__() == 0:
                pass
            else:
                tmp = classify_shap_response_per_species_and_feature_group(
                    df=idf.copy(),
                    species_in=my_species,
                    group_in=group_count,
                    response="SPEI",
                    scale_shap=scale_shap,
                    scale_response=scale_response,
                    pval_threshold=response_pval,
                    show=False,
                    remove_labels=False,
                    dir_analysis=f"{dir_save}/{my_species}/single_response_classification",
                    filesuffix=f"{group_count}",
                    make_plots=make_plots,
                    verbose=verbose,
                )
                df_resp_class.append(tmp)

            if idf["Temperature - Metrics"].dropna().__len__() == 0:
                pass
            else:
                tmp = classify_shap_response_per_species_and_feature_group(
                    df=idf.copy(),
                    species_in=my_species,
                    group_in=group_count,
                    response="Temperature",
                    scale_shap=scale_shap,
                    scale_response=scale_response,
                    pval_threshold=response_pval,
                    show=False,
                    remove_labels=False,
                    dir_analysis=f"{dir_save}/{my_species}/single_response_classification",
                    filesuffix=f"{group_count}",
                    make_plots=make_plots,
                    verbose=verbose,
                )
                df_resp_class.append(tmp)

            # Add counter
            perc_count += df_agg.loc[i, "perc_runs"]

        # Finalizing
        df_agg_red["perc_run_subset"] = (
            df_agg_red["group_size"] / df_agg_red["group_size"].sum() * 100
        ).astype(int)

        df_agg_red["spei_response_increasing_mortality"] = ""
        df_agg_red["temp_response_increasing_mortality"] = ""
        df_agg_red["consistent_response"] = ""
        df_agg_red["merge_with_group_x"] = ""

        df_agg_red = move_vars_to_front(
            df_agg_red,
            [
                "species",
                "group",
                "group_size",
                "spei_response_increasing_mortality",
                "temp_response_increasing_mortality",
                "consistent_response",
                "merge_with_group_x",
                # "Temperature - Season",
                # "SPEI - Duration",
                # "SPEI - Duration Agg.",
                "perc_runs",
                "perc_run_subset",
                # "SPEI - Season",
                "spei_metrics",
                # "temperature_metrics",
            ],
        )

        df_patterns.append(df_agg_red)

    # Concatenate all models
    df_models = pd.concat(df_models).reset_index(drop=True)
    df_patterns = pd.concat(df_patterns).reset_index(drop=True)
    df_resp_class_concat = pd.concat(df_resp_class).reset_index(drop=True)

    # ! CAREFUL: IF THIS IS RUN PER MP, THEN THE FINAL DF WILL BE AT THE SPECIES LEVEL!
    # Check if more than one species was analyzed
    if df_all_in.species.nunique() == 1:
        extra = f"{my_species}/"
        if verbose:
            print(
                "🚨 Careful, this function was run for a single species, not looped over multiple ones! --> Aggregation of csv files needed!"
            )
    else:
        extra = ""

    df_models.to_csv(
        f"{dir_save}/{extra}/runs_aggregated_by_spei_temp_pairs-filtered_by_roc-not_grouped.csv",
        index=False,
    )

    df_patterns.to_csv(
        f"{dir_save}/{extra}/runs_aggregated_by_spei_temp_pairs-filtered_by_roc-grouped.csv",
        index=False,
    )
    
    if verbose:
        print(f"🟢 Saved to disk: {dir_save}/{extra}/runs_aggregated_by_spei_temp_pairs-filtered_by_roc-grouped.csv")

    # This is not needed because the classification files are saved separately
    # df_resp_class_concat.to_csv(
    #     f"{dir_save}/{extra}/runs_aggregated_by_spei_temp_pairs-response_classification.csv",
    #     index=False,
    # )
    
    return []
    


def classify_shap_response_per_species_and_feature_group(
    df,
    species_in,
    group_in,
    response,
    scale_shap,
    scale_response,
    pval_threshold,
    show,
    remove_labels,
    dir_analysis,
    filesuffix,
    make_plots,
    verbose=True,
):
    # Make dirs
    os.makedirs(dir_analysis, exist_ok=True)
    os.makedirs(f"{dir_analysis}/all_runs", exist_ok=True)

    # Clean the dataset so that all rows where response is na are removed
    remove_these = []
    for run in df["subset_group"].values:

        check1 = get_relevant_feature(df, run, response)
        if check1 is None:
            remove_these.append(run)

    # print(
    #     f" - For {response}, the following runs are removed due to missing features: {remove_these}"
    # )

    df_plot = df.query("subset_group not in @remove_these")

    # Inputs
    unique_runs = df_plot["subset_group"].unique()
    n_runs = len(unique_runs)
    if n_runs > 20:
        n_cols = 6
        figx = 25
    else:
        n_cols = 4
        figx = 15
    n_rows = (n_runs + n_cols - 1) // n_cols  # Calculate the number of rows needed

    # Start figure skeleton
    if n_runs <= n_cols:
        fig, axs = plt.subplots(
            n_rows, n_cols, figsize=(figx, 4), sharey=False, sharex=False
        )
    elif remove_labels:
        fig, axs = plt.subplots(
            n_rows, n_cols, figsize=(figx, 2 * n_rows), sharey=False, sharex=False
        )
    else:
        fig, axs = plt.subplots(
            n_rows, n_cols, figsize=(figx, 3 * n_rows), sharey=False, sharex=False
        )

    df_trends = []

    for idx, irun in enumerate(unique_runs):

        # Get current species from filepath in the df
        ispecies = dir_analysis.split("/")[8]
        
        # Clean irun
        irun_clean = irun.split(" (")[0]

        # ! Get data -----------------------------------------------------
        # Get feature to plot
        feat_response = get_relevant_feature(df_plot, irun, response)
        # Get filepath
        filepath = df_plot.query("subset_group == @irun")["dir"].values[0]
        # Load feature data to access corresponding SHAP values
        X_shap = pd.read_csv(f"{filepath}/final_model/X_test.csv", index_col=0)
        feature_names = pd.Series(X_shap.columns)
        feat_response_values = X_shap[feat_response].values
        pos_response = feature_names[feature_names == feat_response].index[0]

        # Load SHAP data at position of feature (using new code in the backend)
        shap_values = load_shap(
            filepath + "/new_shap/approximated/shap_values_test.pkl"
        )
        shap_values = shap_values[:, pos_response, 1].values

        # Clean data for trend assessing
        # Flip values for SPEI to facilitate interpretation
        if response == "SPEI":
            feat_response_values = -feat_response_values

        # Scale feature and shap values
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()
        if scale_shap:
            shap_values = scaler.fit_transform(shap_values.reshape(-1, 1)).flatten()

        if scale_response:
            feat_response_values = scaler.fit_transform(
                feat_response_values.reshape(-1, 1)
            ).flatten()

        # Verbose
        if verbose:
            print(
                f" - Working on species {ispecies} | run {irun} | ({idx + 1}/{n_runs}) \t| Feature: {feat_response}"
            )

        # ! LM fitting -----------------------------------------------------

        # Get different derivatives for LMs
        x_lin = feat_response_values
        x_quad = feat_response_values**2
        x_cub = feat_response_values**3
        
        # Taking log sometimes causes error, suppress it. NAs will be removed later
        with np.errstate(invalid="ignore"):
            x_log = np.log1p(feat_response_values)
        
        # Merge into one df
        df_lm_data = (
            pd.DataFrame(
                {
                    "x_lin": x_lin,
                    "x_quad": x_quad,
                    "x_cub": x_cub,
                    "x_log": x_log,
                    "y": shap_values,
                }
            )
            .sort_values("x_lin")
            .reset_index(drop=True)
        )

        # Loop over all models
        results = []
        # for modelType in ["linear", "quadratic", "cubic", "log"]:
        for modelType in ["linear"]:

            # Define features
            if modelType == "linear":
                features = ["x_lin"]
                abbrev = "lin"
            elif modelType == "quadratic":
                features = ["x_lin", "x_quad"]
                abbrev = "quad"
            elif modelType == "cubic":
                features = ["x_lin", "x_quad", "x_cub"]
                abbrev = "cubic"
            elif modelType == "log":
                features = ["x_log"]
                abbrev = "log"
                df_lm_data = df_lm_data.dropna()

            # Define formula
            formula = "y ~ " + " + ".join(features)

            # Fit model
            # Ignore warnings
            with np.errstate(invalid="ignore"):
                model = Lm(
                    formula,
                    data=pd.concat([df_lm_data["y"], df_lm_data[features]], axis=1),
                    family="gaussian",
                )
            
            # Fit model
            if verbose:
                sry = model.fit(verbose=True).reset_index()
            else:
                # set a trap and redirect stdout (see https://codingdose.info/posts/supress-print-output-in-python/)
                from contextlib import redirect_stdout
                trap = io.StringIO()
                with redirect_stdout(trap):
                    sry = model.fit(verbose=False).reset_index()
            
            # Get and save predictions
            y_pred = model.predict(df_lm_data[features])
            df_lm_data[f"y_pred_{modelType}"] = y_pred.copy()

            # Get metrics
            r2 = r2_score(df_lm_data.y, y_pred)
            rmse = root_mean_squared_error(df_lm_data.y, y_pred)
            aic = model.AIC

            # Record results
            iresults = pd.DataFrame(
                {
                    "method": modelType,
                    f"r2": r2,
                    f"rmse": rmse,
                    f"aic": aic,
                },
                index=[0],
            )

            # iresults["formula"] = formula
            iresults["r2"] = f"{r2}"
            iresults["legend"] = f"{modelType}"

            for i, row in sry.iterrows():
                iestimate = row["Estimate"]
                ipval = row["P-val"]

                iresults[f"coef{i}"] = iestimate
                iresults[f"coef{i}_pval"] = ipval

                # Check for trend for the slope
                if i == 1 and ipval < pval_threshold:
                    if row["Estimate"] > 0:
                        iresults["trend"] = "increasing"
                    elif row["Estimate"] < 0:
                        iresults["trend"] = "decreasing"
                else:
                    iresults["trend"] = "no trend"

            results.append(iresults)

            # For plot, get predictions along x-axis
            y_pred = model.predict(df_lm_data[features])
            if modelType == "log":
                y_pred = np.exp(y_pred) - 1

            df_lm_data[f"y_pred_{modelType}"] = y_pred

        results = pd.concat(results, axis=0).reset_index(drop=True)
        # results.to_csv(
        #     f"{dir_analysis}/{response}-{feat_response}-classification_lm.csv",
        #     index=False,
        # )

        # ! Trend assessment -----------------------------------------------------
        trends = pd.DataFrame()
        # New: Only do MK tests
        # Raw
        tmp = get_mk_test_raw(df_lm_data["x_lin"], df_lm_data["y"])
        trends = pd.concat([trends, tmp])
        # Loess
        tmp = get_mk_test_loess(df_lm_data["x_lin"], df_lm_data["y"])
        trends = pd.concat([trends, tmp])

        # Merge lm and other results
        trends = trends.reset_index(drop=True)
        trends = pd.concat([trends, results], axis=0).reset_index(drop=True)

        # Add information
        trends["species"] = ispecies
        trends["run"] = irun
        trends["feature"] = feat_response
        trends["group"] = group_in
        trends["response"] = response
        trends = move_vars_to_front(
            trends, ["species", "run", "group", "response", "feature", "trend"]
        )

        # Save it
        
        trends_file = f"{dir_analysis}/all_runs/{response}-{feat_response}-group_{filesuffix}-{irun_clean}.csv"
        
        if verbose:
            print(f"🟢 Saved to disk: {trends_file}")
        
        trends.to_csv(trends_file, index=False)

        trends_concat = pd.concat(
            [trends[["method", "trend"]], results[["method", "trend"]]], axis=0
        ).reset_index(drop=True)

        # * Old code for multiple assessments
        # # Note: results is from lm fitting above
        # trends = get_trend_assessment(df_lm_data["x_lin"], df_lm_data["y"], results)
        # # Output needs some cleaning...
        # colnames = trends["method"].tolist()
        # rownames = trends["trend"].tolist()
        # trends = pd.DataFrame()
        # trends["trend"] = rownames
        # trends = trends.T.reset_index(drop=True)
        # trends.columns = colnames
        # trends["species"] = ispecies
        # trends["run"] = irun
        # trends["feature"] = feat_response
        # trends["group"] = group_in
        # trends["response"] = response
        # *

        # Append to list
        df_trends.append(trends)

        # ! START PLOT -----------------------------------------------------
        if make_plots:

            # Get subplot
            row = idx // n_cols
            col = idx % n_cols

            # Get plot layout
            if response == "SPEI":
                # color = "gold"
                color = "Greens"
            elif response == "Temperature":
                color = "Oranges"
            else:
                color = "grey"

            if n_runs > n_cols:
                ax = axs[row, col]
            else:
                ax = axs[col]

            # Create 2D density plot
            sns.kdeplot(
                x=df_lm_data["x_lin"],
                y=df_lm_data["y"],
                cmap=color,
                fill=True,
                ax=ax,
                warn_singular=False,
            )

            # Add trend lines
            for i, row in results.iterrows():
                ax.plot(
                    df_lm_data["x_lin"],
                    df_lm_data[f"y_pred_{row['method']}"],
                    label=row["legend"],
                    linestyle="--",
                    linewidth=1,
                )

            # Add LOESS smoother
            sns.regplot(
                x=df_lm_data["x_lin"],
                y=df_lm_data["y"],
                scatter=False,
                lowess=True,
                line_kws={"color": "black", "linewidth": 1},
                ax=ax,
                label="LOESS",
            )

            # Add text
            trend_lin = trends_concat.query("method == 'linear'")["trend"].values[0]
            trend_mk = trends_concat.query("method == 'mk_raw'")["trend"].values[0]
            ax.text(
                0.05,
                0.95,
                f"Linear: {trend_lin}\nMK: {trend_mk}",
                horizontalalignment="left",
                verticalalignment="top",
                transform=ax.transAxes,
                fontsize=8,
            )

            # Add layout
            ylabel = "SHAP Value"
            xlabel = feat_response

            if scale_shap:
                ylabel = "Scaled SHAP Value"

            if scale_response:
                xlabel = "Scaled " + xlabel
            else:
                xlabel = "Original " + xlabel
                if response == "SPEI":
                    xlabel = xlabel + (" (mirrored by taking negative)")

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{irun}", weight="bold")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Set ylim and xlim
            if scale_shap:
                ax.set_ylim(-0.05, 1.05)
            if scale_response:
                ax.set_xlim(-0.05, 1.05)

            if remove_labels:
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.set_xticks([])
                ax.set_yticks([])

            # END of Loop for plotting single variable done

    if make_plots:

        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="center",
            title="Model Type",
            # bbox_to_anchor=(0., 0),
            ncol=4,
        )

        # Remove empty subplots if n_runs is not a multiple of n_cols
        for i in range(n_runs, n_rows * n_cols):
            fig.delaxes(axs.flatten()[i])

        # Give plot a title
        fig.suptitle(
            f"SHAP Scatterplot for {response}\n\n",
            fontsize=20,
            weight="bold",
        )

        plt.tight_layout()
        plt.savefig(
            f"{dir_analysis}/{response}-group_{filesuffix}.png"
        )
        if show:
            plt.show()
        else:
            plt.close()

    else:
        plt.close()
        if verbose:
            print(
                "🚨 No plots were made because 'make_plots = False' in plot_shap_response()!"
            )

    # Concatenate all results
    df_trends = pd.concat(df_trends)

    return df_trends


# Helper function to detect trend direction
def detect_trend(slope):
    if slope > 0:
        return "increasing"
    elif slope < 0:
        return "decreasing"
    else:
        return "no trend"


# 1. MK Test on Raw Data
def get_mk_test_raw(x, y, pval_threshold=0.05):
    result = mk.original_test(y)

    if result.p < pval_threshold:
        trend = detect_trend(result.slope)
    else:
        trend = "no trend"

    # Turn results into a dictionary
    result = {
        "method": "mk_raw",
        "trend": result.trend,
        "h": result.h,
        "p": result.p,
        "z": result.z,
        "tau": result.Tau,
        "s": result.s,
        "var_s": result.var_s,
        "slope": result.slope,
        "intercept": result.intercept,
    }

    # Turn into a DataFrame
    result = pd.DataFrame(result, index=[0])

    return result


# 2. MK Test on Loess Smoothed Data
def get_mk_test_loess(x, y, frac=0.3, pval_threshold=0.05):
    # Ensure x and y are numpy arrays
    x, y = np.array(x), np.array(y)

    # Apply LOESS smoothing
    loess_result = lowess(y, x, frac=frac, return_sorted=False)

    # Conduct Mann-Kendall trend test on smoothed data
    result = mk.original_test(loess_result)

    if result.p < pval_threshold:
        trend = detect_trend(result.slope)
    else:
        trend = "no trend"

    # Compile results
    result = {
        "method": "mk_smooth",
        "trend": result.trend,
        "h": result.h,
        "p": result.p,
        "z": result.z,
        "tau": result.Tau,
        "s": result.s,
        "var_s": result.var_s,
        "slope": result.slope,
        "intercept": result.intercept,
        "smoothed_y": [loess_result],
    }

    # Turn into a DataFrame
    result = pd.DataFrame(result, index=[0])

    return result


# 6. Piecewise Monotonicity on Raw Data
def get_piecewise_monotonicity(x, y):
    differences = np.diff(y)
    if np.all(differences >= 0):
        trend = "increasing"
    elif np.all(differences <= 0):
        trend = "decreasing"
    else:
        trend = "no trend"
    return pd.DataFrame(
        {
            "method": "piecewise_monotonicity_raw",
            "trend": [trend],
            "direction_changes": [np.sum(np.diff(np.sign(differences)) != 0)],
        }
    )


# 7. Directional Sign Test on Raw Data
def get_directional_sign_test_raw(y):
    diffs = np.diff(y)
    positives = np.sum(diffs > 0)
    negatives = np.sum(diffs < 0)
    trend = (
        "increasing"
        if positives > negatives
        else "decreasing" if negatives > positives else "no trend"
    )
    return pd.DataFrame(
        {
            "method": "directional_sign",
            "trend": [trend],
            "positives": [positives],
            "negatives": [negatives],
        }
    )


# 8. Directional Sign Test on Loess Smoothed Data
def get_directional_sign_test_loess(x, y, frac=0.3):
    loess_result = lowess(y, x, frac=frac, return_sorted=False)
    loess_result = directional_sign_test_raw(loess_result)
    loess_result["method"] = "directional_sign_loess"
    return loess_result


# 9. Cumulative Sum of Differences on Raw Data
def get_cumsum_diff_raw(y):
    diffs = np.diff(y)
    cumsum_diff = np.cumsum(diffs)
    trend = detect_trend(np.sum(diffs))
    return pd.DataFrame(
        {"method": "cumsum", "trend": [trend], "cumsum_diff": [cumsum_diff[-1]]}
    )


# 10. Cumulative Sum of Differences on Loess Smoothed Data
def get_cumsum_diff_loess(x, y, frac=0.3):
    loess_result = lowess(y, x, frac=frac, return_sorted=False)
    loess_result = directional_sign_test_raw(loess_result)
    loess_result["method"] = "cumsum_loess"
    return loess_result


# ! Combine all into one function -----------------------------------------------------
def get_trend_assessment(feat_response_values, shap_values, results):
    trends = results[["method", "trend"]].copy()

    trends = pd.concat(
        [
            trends,
            get_mk_test_raw(feat_response_values, shap_values)[["method", "trend"]],
        ]
    )
    trends = pd.concat(
        [
            trends,
            get_mk_test_loess(feat_response_values, shap_values, frac=0.3)[
                ["method", "trend"]
            ],
        ]
    )
    # trends = pd.concat(
    #     [
    #         trends,
    #         get_piecewise_monotonicity(feat_response_values, shap_values)[
    #             ["method", "trend"]
    #         ],
    #     ]
    # )
    # trends = pd.concat(
    #     [trends, get_directional_sign_test_raw(shap_values)[["method", "trend"]]]
    # )
    # trends = pd.concat(
    #     [
    #         trends,
    #         get_directional_sign_test_loess(
    #             feat_response_values, shap_values, frac=0.3
    #         )[["method", "trend"]],
    #     ]
    # )
    # trends = pd.concat([trends, get_cumsum_diff_raw(shap_values)[["method", "trend"]]])
    # trends = pd.concat(
    #     [
    #         trends,
    #         get_cumsum_diff_loess(feat_response_values, shap_values, frac=0.3)[
    #             ["method", "trend"]
    #         ],
    #     ]
    # )

    return trends.reset_index(drop=True)


# ------------------------------
# classify_shap_response_per_species_and_feature_group(
#     df=idf.copy(),
#     species_in=my_species,
#     group_in=group_count,
#     response="Temperature",
#     scale_shap=scale_shap,
#     scale_response=True,
#     pval_threshold=0.05,
#     show=True,
#     remove_labels=False,
#     dir_analysis=f"{dir_save}/{my_species}/by_roc/most_common_patterns",
#     filesuffix=f"most_common_runs-group_{group_count}",
#     make_plots=True,
# )


# -----------------------------------------------------------------------------------------------

def ___GLMM_FUNCTIONS___():
    pass


def glmm_get_data(myset, base_dir, idir, drop_smote=True, verbose=False):
    
    xpath = f"{base_dir}/{idir}/final_model/X_{myset}.csv"
    if not os.path.exists(xpath):
        if verbose:
            print(f"🚨 X_{myset} for {idir} does not exist! Skipping...")
        return None
        
    
    x = pd.read_csv(xpath).drop(
        columns={"Unnamed: 0"}
    )

    xid = pd.read_csv(f"{base_dir}/{idir}/treeid/X_{myset}_treeid.csv").drop(
        columns={"Unnamed: 0"}
    )

    if x.shape[0] != xid.shape[0]:
        if verbose:
            print(" - 🚨 x and xid have different number of rows")

    x = pd.concat([x, xid], axis=1)

    # Attach siteid
    x["site_id"] = x["tree_id"].str.split("_").str[0]
    x = x.drop(columns=["tree_id"])
    y = pd.read_csv(f"{base_dir}/{idir}/final_model/y_{myset}.csv").drop(
        columns={"Unnamed: 0"}
    )
    xy = pd.concat([y, x], axis=1)
    xy.columns = xy.columns.str.replace("_", "_").str.replace("-", "_")

    # X_test holds also SMOTE data, so we need to attach a common site id for the SMOTE data (-9999)
    # Or actually remove it
    if myset == "train":
        if drop_smote:
            if verbose:
                print(" - Dropping SMOTE data")
            xy = xy.dropna()
        else:
            if verbose:
                print("Keeping SMOTE data")
            xy["site_id"] = xy["site_id"].fillna("-9999")
    return xy


def glmm_model_evaluation_classification(
    glmm_model,
    X_train,
    y_train,
    X_test,
    y_test,
    prob_threshold=0.5,
    save_directory=None,
    metric="f1-score",
    verbose=True,
):
    # GLMM model prediction (returns predicted probabilities in logistic regression)
    y_train_pred_proba = pd.Series(
        glmm_model.predict(
            X_train, verify_predictions=False, skip_data_checks=True, verbose=verbose
        )
    )
    y_test_pred_proba = pd.Series(
        glmm_model.predict(
            X_test, verify_predictions=False, skip_data_checks=True, verbose=verbose
        )
    )

    # Convert probabilities into binary predictions based on the threshold
    y_train_pred = (y_train_pred_proba >= prob_threshold).astype("int")
    y_test_pred = (y_test_pred_proba >= prob_threshold).astype("int")

    # Convert arrays into pandas series for ease of manipulation
    y_train_pred = pd.Series(y_train_pred)
    y_test_pred = pd.Series(y_test_pred)

    # Save predictions, probabilities, and actuals to files
    if save_directory is not None:
        os.makedirs(f"{save_directory}", exist_ok=True)

        # Save predictions - binary
        y_train_pred.to_csv(f"{save_directory}/y_train_pred.csv")
        y_test_pred.to_csv(f"{save_directory}/y_test_pred.csv")

        # Save predictions - probabilities
        pd.DataFrame(y_train_pred_proba, columns=["predicted_proba"]).to_csv(
            f"{save_directory}/y_train_proba.csv"
        )
        pd.DataFrame(y_test_pred_proba, columns=["predicted_proba"]).to_csv(
            f"{save_directory}/y_test_proba.csv"
        )

        # Save actuals
        y_train.to_csv(f"{save_directory}/y_train.csv")
        y_test.to_csv(f"{save_directory}/y_test.csv")

        # Save features
        X_train.to_csv(f"{save_directory}/X_train.csv")
        X_test.to_csv(f"{save_directory}/X_test.csv")

        # Save the model
        with open(f"{save_directory}/glmm_model.pkl", "wb") as file:
            pickle.dump(glmm_model, file)

    # Calculate confusion matrices
    unique_labels = np.unique(np.concatenate((y_train, y_test)))
    confusion_train = confusion_matrix(y_train, y_train_pred, labels=unique_labels)
    confusion_test = confusion_matrix(y_test, y_test_pred, labels=unique_labels)

    # Classification reports for both training and test data
    report_train_dict = classification_report(
        y_train, y_train_pred, digits=3, output_dict=True
    )
    report_test_dict = classification_report(
        y_test, y_test_pred, digits=3, output_dict=True
    )

    report_train_txt = classification_report(y_train, y_train_pred, digits=3)
    report_test_txt = classification_report(y_test, y_test_pred, digits=3)

    # # Print training and testing information
    # if verbose:
    #     print("--- model_evaluation_classification():")
    #     print(f"Number of training data points: {len(y_train)}")
    #     print(f"Number of testing data points: {len(y_test)}")
    #     print("\nTrain Classification Report:\n", report_train_txt)
    #     print("\nTest Classification Report:\n", report_test_txt)

    # Save reports
    if save_directory is not None:
        with open(f"{save_directory}/classification_report_train.txt", "w") as f:
            f.write(report_train_txt)
        with open(f"{save_directory}/classification_report_test.txt", "w") as f:
            f.write(report_test_txt)

    # Get max f1 for scaling bar plots
    class_scores_train = [
        report_train_dict[str(label)][metric] for label in unique_labels
    ]
    class_scores_test = [
        report_test_dict[str(label)][metric] for label in unique_labels
    ]

    metric_max = max([0] + class_scores_train + class_scores_test)

    # Baseline performance (predicting 0 for all samples)
    y_baseline = pd.Series([0] * len(y_test))

    scores_baseline = bootstrap_classification_metric(
        y_test,
        y_baseline,
        metrics=["accuracy", "precision", "recall", "roc_auc"],
        n_bootstraps=100,
    )

    scores_test = bootstrap_classification_metric(
        y_test,
        y_test_pred,
        metrics=["accuracy", "precision", "recall", "roc_auc"],
        n_bootstraps=100,
    )

    scores_train = bootstrap_classification_metric(
        y_train,
        y_train_pred,
        metrics=["accuracy", "precision", "recall", "roc_auc"],
        n_bootstraps=100,
    )

    # Save baseline and other results to files
    if save_directory is not None:
        scores_baseline.to_csv(
            f"{save_directory}/final_model_scores_baseline.csv", index=False
        )
        scores_train.to_csv(
            f"{save_directory}/final_model_scores_train.csv", index=False
        )
        scores_test.to_csv(f"{save_directory}/final_model_scores_test.csv", index=False)

    df_metrics = pd.DataFrame(
        {
            "metric": "roc_auc",
            "train_boot_mean": scores_train["roc_auc"].iloc[0],
            "train_boot_sd": scores_train["roc_auc"].iloc[1],
            "test_boot_mean": scores_test["roc_auc"].iloc[0],
            "test_boot_sd": scores_test["roc_auc"].iloc[1],
        },
        index=[0],
    )

    if save_directory is not None:
        df_metrics.to_csv(f"{save_directory}/classification_metrics.csv", index=False)

    # ! Plot
    # Set the figure size
    plt.figure(figsize=(8, 4))

    # Combine confusion matrices to find common color scale
    combined_confusion = np.maximum(confusion_train, confusion_test)

    # Plot the confusion matrix for train data
    # plot_confusion_matrix(
    #     confusion_train,
    #     class_scores_train,
    #     unique_labels,
    #     title="Train Data",
    #     bpmax=metric_max,
    #     save_directory=save_directory,
    #     test_or_train="train",
    #     show=verbose,
    # )

    # # Plot the confusion matrix for test data
    # plt.figure(figsize=(8, 4))
    # plot_confusion_matrix(
    #     confusion_test,
    #     class_scores_test,
    #     unique_labels,
    #     title="Test Data",
    #     bpmax=metric_max,
    #     save_directory=save_directory,
    #     test_or_train="test",
    #     show=verbose,
    # )

    # Plot ROC AUC curve
    plot_roc_auc_both(
        (y_train, y_train_pred_proba),
        (y_test, y_test_pred_proba),
        save_directory=save_directory,
        show=verbose,
    )
    
    plot_pr_auc_both(
        (y_train, y_train_pred_proba),
        (y_test, y_test_pred_proba),
        save_directory=save_directory,
        show=verbose,
    )
    
def glmm_rfe(
    xtrain,
    ytrain,
    rtrain,
    verbose=True,
):
    import pandas as pd
    import statsmodels.api as sm
    
    # Start loop
    list_summaries = []
    list_model_info = []
    all_features_tested = False

    # Initialize the loop
    all_features_tested = False

    # Remove features until all are significant
    while not all_features_tested:
        # Create a logistic regression model
        n_features = xtrain.columns.__len__()
        org_features = xtrain.columns.tolist()
        formula = "target ~ " + " + ".join(xtrain.columns) + f" + (1|{rtrain.name})"
        print(f" - Features: {n_features}\t | Formula: {formula}")

        # Fit model
        model = Lmer(
            formula, data=pd.concat([ytrain, xtrain, rtrain], axis=1), family="binomial"
        )
        result = (
            model.fit(verbose=False)
            .sort_values("P-val", ascending=False)
            .drop("(Intercept)", axis=0)
        )

        list_summaries.append(result)
        
        # Count how many significant variables are left
        n_significant = (result["P-val"] < 0.05).sum()
        # print(f"🚨 {n_significant} significant features left")
        # display(result_2)
        
        # Calculate ROC-AUC
        y_pred = model.predict(pd.concat([xtrain, rtrain], axis=1), skip_data_checks=True, verify_predictions=False)
        roc_auc = roc_auc_score(ytrain, y_pred)
        pr_auc = average_precision_score(ytrain, y_pred)

        # Drop least important feature
        least_important_feature = result.iloc[0].name

        # Sometimes the interaction has switched variables order, so we need to check both
        if ":" in least_important_feature:
            var1, var2 = least_important_feature.split(":")
            var1_var2 = f"{var1}:{var2}"
            var2_var1 = f"{var2}:{var1}"

            xtrain = xtrain.drop(var1_var2, axis=1, errors="ignore")
            result_2 = result.drop(var1_var2, axis=0, errors="ignore").copy()

            xtrain = xtrain.drop(var2_var1, axis=1, errors="ignore")
            result_2 = result.drop(var2_var1, axis=0, errors="ignore").copy()
        else:
            xtrain = xtrain.drop(least_important_feature, axis=1, errors="ignore")
            result_2 = result.drop(
                least_important_feature, axis=0, errors="ignore"
            ).copy()

        # If main effect is dropped, drop interaction too
        for col in xtrain.columns:
            if ":" in col and (
                col.split(":")[0] == least_important_feature
                or col.split(":")[1] == least_important_feature
            ):
                xtrain = xtrain.drop(col, axis=1, errors="ignore")
                result_2 = result_2.drop(col, axis=0, errors="ignore")

        # Record results
        list_model_info.append(
            pd.DataFrame(
                {
                    "formula": formula,
                    "AIC": model.AIC,
                    "BIC": model.BIC,
                    "PR": pr_auc,
                    "ROC": roc_auc,
                    "n_significant": n_significant,
                    "n_features": n_features,
                    "org_features": [org_features],
                    "drop_feature": least_important_feature,
                }
            )
        )
        
        # Check if all features are significant
        print(" -------------------------- ")
        if xtrain.columns.__len__() == 0:
            all_features_tested = True

        if not verbose:
            clear_output()

    return pd.concat(list_model_info)

def glmm_run_per_species_and_model(
    ispecies,
    imodel,
    do_rfe=False,
    rfe_with_interactions=False,
    add_spei_temp_interaction=False,
    add_spei_temp_derivatives=False,
    best_model_method="None",
    path_prefix=None,
    path_suffix=None,
    return_all=False,
    verbose=False,
    skip_if_exists=True,
):
    
    if do_rfe:
        if best_model_method not in ["AIC", "BIC", "PR", "ROC"]:
            chime.error()
            raise ValueError("best_model_method must be one of 'AIC', 'BIC', 'PR', 'ROC'")
    
    if path_prefix is None or path_suffix is None:
        raise ValueError(f"Path input is not correct, needs to be specified!")
    
    if verbose:
        print(f"Species: {ispecies}\t | Model: {imodel}")

    # Specify run
    base_dir = path_prefix
    idir = f"{imodel}/{ispecies}"
    modeldir = path_suffix
    if do_rfe:
        modeldir = f"{modeldir}/best_model_{best_model_method}"
        
    os.makedirs(
        f"{base_dir}/{idir}/{modeldir}",
        exist_ok=True,
    )

    # ! Check if file exists
    path_summary = f"{base_dir}/{idir}/{modeldir}/summary.csv"
    if os.path.isfile(path_summary) and skip_if_exists:
        if verbose:
            print(" - 📂 Summary file exists, skipping...")
        return []

    if verbose:
        print(
            f" - 💾 Saving to: {base_dir}/{idir}/{modeldir}/"
        )
        
    # Get data
    xy_train = glmm_get_data(
        "train", base_dir=base_dir, idir=idir, drop_smote=True, verbose=verbose
    )
    xy_test = glmm_get_data("test", base_dir=base_dir, idir=idir, verbose=verbose)

    # Check if data was loaded
    if xy_train is None or xy_test is None:
        print(f" - Test and train was missing for {ispecies}: {imodel}")
        return []

    rando = "site_id"
    target = "target"

    ytrain = xy_train["target"]
    ytest = xy_test["target"]

    # Get the random effects column
    rando = "site_id"
    rtrain = xy_train[rando]
    rtest = xy_test[rando]

    # Reduce the xtrain and xtest to only predictors
    xtrain = xy_train.drop(columns=[target, rando]).copy()
    xtest = xy_test.drop(columns=[target, rando]).copy()
    preds = xtrain.columns.tolist()

    # Get spei and temp columns
    spei_var = get_var_from_category(preds, "SPEI")
    temp_var = get_var_from_category(preds, "Temperature")

    # Invert SPEI for easier interpretation
    if spei_var is not None:
        xtrain[spei_var] = -xtrain[spei_var]
        xtest[spei_var] = -xtest[spei_var]
    
    # Add spei-temp interaction (only if main effects are present)
    if not rfe_with_interactions and add_spei_temp_interaction: 
        if spei_var is not None and temp_var is not None:
            xtrain[f"{spei_var}:{temp_var}"] = xtrain[spei_var].copy() * xtrain[temp_var].copy()
            xtest[f"{spei_var}:{temp_var}"] = xtest[spei_var].copy() * xtest[temp_var].copy()
            preds = preds + [f"{spei_var}:{temp_var}"]
            
    # Normalize the data
    scaler = MinMaxScaler()  # StandardScaler()
    xtrain[preds] = scaler.fit_transform(xtrain[preds])
    xtest[preds] = scaler.transform(xtest[preds])
    
    # Add quadratic and logarithmic derivatives (only if main effects are present)
    if add_spei_temp_derivatives: 
        if spei_var is not None:
            xtrain[f"{spei_var}_sq"] = xtrain[spei_var].copy() * xtrain[spei_var].copy()
            xtrain[f"{spei_var}_log"] = np.log1p(xtrain[spei_var].copy())
            
            xtest[f"{spei_var}_sq"] = xtest[spei_var].copy() * xtest[spei_var].copy()
            xtest[f"{spei_var}_log"] = np.log1p(xtest[spei_var].copy())
            
            preds = preds + [f"{spei_var}_sq", f"{spei_var}_log", f"{temp_var}_sq", f"{temp_var}_log"]
            
        if temp_var is not None:
            xtrain[f"{temp_var}_sq"] = xtrain[temp_var].copy() * xtrain[temp_var].copy()
            xtrain[f"{temp_var}_log"] = np.log1p(xtrain[temp_var].copy())
            
            xtest[f"{temp_var}_sq"] = xtest[temp_var].copy() * xtest[temp_var].copy()
            xtest[f"{temp_var}_log"] = np.log1p(xtest[temp_var].copy())
            
            preds = preds + [f"{spei_var}_sq", f"{spei_var}_log", f"{temp_var}_sq", f"{temp_var}_log"]
    
    # return xtrain, xtest # ! DEBUG
    
    # Calculate basic statistics
    xry_train = xtrain.describe().T  # Transpose for easier manipulation
    xry_train['percent_missing'] = xtrain.isna().mean() * 100
    xry_train = xry_train.reset_index().rename(columns={'index': 'variable'})
    xry_test = xtest.describe().T  # Transpose for easier manipulation
    xry_test['percent_missing'] = xtest.isna().mean() * 100
    xry_test = xry_test.reset_index().rename(columns={'index': 'variable'})
    if verbose:
        display(" --- Summary of train data --- ")
        display(xtrain.head())
        display(xry_train)
        display(" --- Summary of test data --- ")
        display(xtest.head())
        display(xry_test)
            
    # Check if feature elimination should be done
    if do_rfe:
        # Add all possible interactions to the train and test data
        if rfe_with_interactions:
            cols = xtrain.columns
            for i, col in enumerate(cols):
                for j in range(i + 1, len(cols)):
                    col2 = cols[j]
                    xtrain[f"{col}:{col2}"] = xtrain[col].copy() * xtrain[col2].copy()
                    xtest[f"{col}:{col2}"] = xtest[col].copy() * xtest[col2].copy()
                    
                    # Normalize data
                    xtrain[f"{col}:{col2}"] = scaler.fit_transform(xtrain[f"{col}:{col2}"])
                    xtest[f"{col}:{col2}"] = scaler.transform(xtest[f"{col}:{col2}"])
        
        
        if verbose:
            display(" --- Start of RFE --- ")
        rfe_results = glmm_rfe(
        # rfe_results = glmm_rfe(
            xtrain,
            ytrain,
            rtrain,
            verbose,
        )

        # Plot AIC ~ n_features
        rfe_results.plot(x="n_features", y=best_model_method)
        plt.savefig(f"{base_dir}/{idir}/{modeldir}/rfe_plot-best_{best_model_method}.png")
        if verbose:
            plt.show()
        plt.close()
        
        # Save the results
        rfe_results.to_csv(f"{base_dir}/{idir}/{modeldir}/rfe_results.csv")

        # Fit the best model to test and train again
        if best_model_method == "BIC" or best_model_method == "AIC":
            best_model = rfe_results[
                rfe_results[best_model_method] == rfe_results[best_model_method].min()
            ]
        elif best_model_method == "PR" or best_model_method == "ROC":
            best_model = rfe_results[
                rfe_results[best_model_method] == rfe_results[best_model_method].max()
            ]
        
        best_features = best_model["org_features"].values[0]
        
        # Check if a squared variable is present and if so add its main effect too
        for f in best_features:
            if "_sq" in f:
                org_f = f.replace("_sq", "") # Get original variable name
                if org_f not in best_features:
                    best_features.append(org_f)
        
    else:
        best_features = xtrain.columns.tolist()
        
    best_formula = (
        "target ~ " + " + ".join(best_features) + f" + (1 | {rtrain.name})"
    )
    
    # Save best formula to txt
    with open(f"{base_dir}/{idir}/{modeldir}/_best_formula.txt", "w") as f:
        f.write(best_formula)

    model = Lmer(
        best_formula,
        data=pd.concat([ytrain, xtrain[best_features], rtrain], axis=1),
        family="binomial",
    )
    model.fit(verbose=False)
    model.coefs.to_csv(path_summary)
    
    if verbose:
        print(f" ------- RESULTS ------- ")
        print(f" - Best formula: {best_formula}")
        print(f" - Best features: {best_features}")
        print(f" - Model: {model}")
        display(model.coefs)

    # Evaluate the model
    glmm_model_evaluation_classification(
        glmm_model=model,
        X_train=pd.concat([xtrain[best_features], rtrain], axis=1),
        X_test=pd.concat([xtest[best_features], rtest], axis=1),
        y_train=ytrain,
        y_test=ytest,
        prob_threshold=0.5,
        save_directory=f"{base_dir}/{idir}/{modeldir}/",
        verbose=verbose,
    )

    return []
    
def glmm_wrapper_loop(
    df_in, 
    do_rfe,
    rfe_with_interactions,
    add_spei_temp_interaction,
    add_spei_temp_derivatives,
    best_model_method,
    path_prefix,
    path_suffix,
    return_all,
    verbose,
    skip_if_exists,
):
        
    for i, row in df_in.reset_index(drop=True).iterrows():
        if verbose:
            print(f"🟡 Species: {row.species}\t | Model: {row.model}")
            
        glmm_run_per_species_and_model(
            ispecies=row.species,
            imodel=row.model,
            do_rfe=do_rfe,
            rfe_with_interactions=rfe_with_interactions,
            add_spei_temp_interaction=add_spei_temp_interaction,
            add_spei_temp_derivatives=add_spei_temp_derivatives,
            best_model_method=best_model_method,
            path_prefix=path_prefix,
            path_suffix=path_suffix,
            return_all=return_all,
            verbose=verbose,
            skip_if_exists=skip_if_exists,
        )
        clear_output()
        
    return []

def get_category_from_var(var_to_match):

    # Debug for mapping onto index of glmm output
    if var_to_match == "(Intercept)":
        return "Intercept"

    # Read json file for dictionry
    import json

    with open(
        "./model_analysis/dict_preds.json",
        "r",
    ) as f:
        tmp_dict = json.load(f)

    # Clean spei var, just in case
    var_to_match = var_to_match.replace("-", "_")

    # Scan through all items and return the category
    for category, items in tmp_dict.items():
        for item in items:
            item = item.replace("-", "_")
            if item in var_to_match:
                return category

    raise ValueError(f"Variable '{var_to_match}' not found in dictionary!")


def get_category_from_var_wrapper(var_to_match):
    # Debug for interaction terms, needs to be split and checked twice
    if ":" in var_to_match:
        var1, var2 = var_to_match.split(":")
        var_both = (
            f"Interaction_{get_category_from_var(var1)}_{get_category_from_var(var2)}"
        )
        return var_both
    else:
        return get_category_from_var(var_to_match)

def get_var_from_category(check_these_variables, category):
    # Read json file for dictionry
    import json

    with open(
        "./model_analysis/dict_preds.json",
        "r",
    ) as f:
        tmp_dict = json.load(f)

    if category not in tmp_dict.keys():
        raise ValueError(f"Category '{category}' not found in dictionary!")

    # Compare check
    if category == "Temperature":
        search_for = ["tmoy", "tmax", "tmin"]
    elif category == "SPEI":
        search_for = ["spei"]
    else:
        search_for = tmp_dict[category]

    for var in check_these_variables:
        for search in search_for:
            if search in var:
                return var

def glmm_get_spei_var(vars_in):
    for v in vars_in:
        if "spei" in v:
            return v
    return None


def glmm_get_temp_var(vars_in):
    v = None
    for iv in vars_in:
        if "tmoy" in iv:
            v = iv
        elif "tmax" in iv:
            v = iv
        elif "tmin" in iv:
            v = iv
    return v


# ----

def shap_run_new(
  ispecies,
  imodel,
  run_interaction,
  approximate,
  test_or_train,
  force_run,
  verbose,  
):
    
    idir = f"./model_runs/_fullruns/{imodel}/{ispecies}"
    iseed = imodel.split(" ")[0].split("_")[1]
    shapdir = f"{idir}/new_shap"

    if not os.path.exists(f"{idir}/final_model"):
        print(f"Model not found: {idir}/final_model")
        return []
    
    if approximate:
        shapdir = f"{shapdir}/approximated"
    else:
        shapdir = f"{shapdir}/precise"

    os.makedirs(shapdir, exist_ok=True)

    # Load data
    xtest = pd.read_csv(f"{idir}/final_model/X_test.csv", index_col=0)
    ytest = pd.read_csv(f"{idir}/final_model/y_test.csv", index_col=0)
    idtest = pd.read_csv(f"{idir}/treeid/X_test_treeid.csv", index_col=0)

    xtrain = pd.read_csv(f"{idir}/final_model/X_train.csv", index_col=0)
    ytrain = pd.read_csv(f"{idir}/final_model/y_train.csv", index_col=0)
    idtrain = pd.read_csv(f"{idir}/treeid/X_train_treeid.csv", index_col=0)

    # Load model
    # Check if model exists
    rf_path = f"./model_runs/_fullruns/{imodel}/{ispecies}/final_model/rf_model.pkl"
    rf = pd.read_pickle(rf_path)

    if verbose:
        print(shapdir)
        print("")
        print(f"{xtest.shape[0]} samples in test set X")
        print(f"{ytest.shape[0]} samples in test set y")
        print(f"{idtest.shape[0]} samples in test set id")
        print("")
        print(f"{xtrain.shape[0]} samples in train set X")
        print(f"{ytrain.shape[0]} samples in train set y")
        print(f"{idtrain.shape[0]} samples in train set id")

    # Set data
    if test_or_train == "train":
        X_shap = xtrain.copy()
    else:
        X_shap = xtest.copy()

    # Run SHAP
    # Single effect
    # Check if file exists
    shap_file = f"{shapdir}/shap_values_{test_or_train}.pkl"
    if os.path.isfile(shap_file) and not force_run:
        if verbose:
            print(" - Single Effect SHAP values already exist, skipping...")
    else:
        if verbose:
            print(" - Calculating SHAP individual values")
        explainer = shap.TreeExplainer(
            rf,
            X_shap,
            feature_names=X_shap.columns,
            approximate=approximate,
        )
        shap_values = explainer(
            X_shap,
            check_additivity=False,
        )

        # Save values
        with open(shap_file, "wb") as file:
            pickle.dump(shap_values, file)

    # Get interaction values
    if run_interaction:
        shap_file = f"{shapdir}/shap_values_interaction_{test_or_train}.pkl"
        # Check if file exists
        if os.path.isfile(shap_file) and not force_run:
            if verbose:
                print(" - Interaction SHAP values already exist, skipping...")
        else:
            if verbose:
                print(" - Calculating SHAP interaction values")
            shap_values_interaction = shap.TreeExplainer(
                rf,
                approximate=approximate,
                feature_names=X_shap.columns,
            ).shap_interaction_values(X_shap)
            
            # Save it
            with open(shap_file, "wb") as file:
                pickle.dump(shap_values_interaction, file)
    
    return []
                    
                    
                    
def shap_run_new_loop(
    df_in,
    verbose=False,  
    run_interaction=True,
    approximate=False,
    test_or_train="test",
    force_run=False,
    ):
    
    for i, row in df_in.reset_index(drop=True).iterrows():
        if verbose:
            print(f"🟡 Species: {row.species}\t | Model: {row.model}")
        
        shap_run_new(
            ispecies=row.species,
            imodel=row.model,
            run_interaction=run_interaction,
            approximate=approximate,
            test_or_train="test",
            force_run=force_run,
            verbose=verbose,
        )
        
        
def shap_run_new_loop_mp(
    df_in,
    verbose=False,  
    run_interaction=True,
    approximate=False,
    test_or_train="test",
    force_run=False,
    num_cores=10,
    ):
    
    # Split into list
    df_in = split_df_into_list_of_group_or_ns(df_in, "model")
    # Run mp
    print("Running in parallel...")
    df_none = run_mp(
        shap_run_new_loop,
        arg_list=df_in,
        num_cores=num_cores,
        progress_bar=True,
        verbose=verbose,
        run_interaction=run_interaction,
        approximate=approximate,
        test_or_train=test_or_train,
        force_run=force_run,
    )
    
    
def ax_dataset_boxplot(
    ax=None,
    all_dfs=None,
    imps=None,
    base_fontsize=12,
    color_spei="blue",
    color_temp="red",
    color_rest="gray",
    pos_spei=None,
    pos_temp=None,
    all_or_top9="all",
    return_dfimp=False,
):

    # Reduce data if needed
    if all_or_top9 == "top9":
        top9 = [
            "Fagus sylvatica",
            "Quercus robur",
            "Quercus petraea",
            "Carpinus betulus",
            "Castanea sativa",
            "Quercus pubescens",
            "Pinus sylvestris",
            "Abies alba",
            "Picea abies",
        ]
        all_dfs = all_dfs.query("species in @top9")

    df_imp = all_dfs[["species"] + imps].copy()
    df_imp = calculate_weighted_mean_importance(df_imp, imps)
    df_imp.columns = df_imp.columns.str.replace("weighted_mean_", "")
    df_imp.columns = df_imp.columns.str.replace("mean_", "")
    df_imp.columns = df_imp.columns.str.replace(" - Importance", "")
    df_imp = df_imp.set_index("species")

    order = df_imp.median().sort_values(ascending=False).index
    df_imp = df_imp[order]
    
    if return_dfimp:
        return df_imp

    # Get ticks
    order_imp = []
    mean_imp = df_imp.median()

    for i in range(mean_imp.shape[0]):
        order_imp.append(
            # f"{mean_imp.index[i]} ({mean_imp.values[i].round(1)}%)"
            f"{mean_imp.index[i]}"
        )

    # Dictionary for renaming
    rename_dict = {
        "Light Competition": "Light competition",
        "Temperature": "Temperature anomaly",
        "SPEI": "CWB anomaly",
        "Interaction_Temperature_SPEI": "Climate Change Inter.",
        "Tree Size": "Tree size",
        "Stand Structure": "Stand structure",
        "Species Competition": "Species competition",
        "Topography": "Topography",
        "Management": "Management",
        "Soil Water Conditions": "Soil water conditions",
        "Soil Fertility": "Soil fertility",
        "NDVI": "NDVI",
    }

    df_imp.columns = df_imp.columns.map(rename_dict)

    category_dict = {
        "Light competition": "Stand Structure",
        "Temperature anomaly": "Climate Change",
        "Interaction_Temperature_SPEI": "Climate Change",
        "CWB anomaly": "Climate Change",
        "Tree size": "Tree Structure",
        "Stand structure": "Stand Structure",
        "Species competition": "Stand Structure",
        "Topography": "Topography",
        "Management": "Management",
        "Soil water conditions": "Soil Conditions",
        "Soil fertility": "Soil Conditions",
        "NDVI": "Forest Health",
        "Climate Change Inter.": "Climate Change Inter.",
        np.nan: "Other", # ! New
    }

    # Define color palette for each category
    color_dict = {
        "Stand Structure": "#A13323",
        "Climate Change": "#FBA346",
        "Tree Structure": "#CCE8E6",
        "Topography": "#A8BDC5",
        "Management": "#6d7a7f",
        "Soil Conditions": "#43619d",
        "Forest Health": "#8BC34A",
        "Other": "#c5d3d8", # ! New
        "Climate Change Inter.": "#c5d3d8",  # ! New
    }

    color_dict = {
        "Climate Change": "#7B52AB",
        "Stand Structure": "#c5d3d8",
        "Tree Structure": "#c5d3d8",
        "Topography": "#c5d3d8",
        "Management": "#c5d3d8",
        "Soil Conditions": "#c5d3d8",
        "Forest Health": "#c5d3d8",
        "Other": "#c5d3d8", # ! New
        "Climate Change Inter.": "#c5d3d8",  # ! New
    }
    
    # Define order of the palette
    palette_order = [
            color_rest,
            color_rest,
            color_rest,
            color_rest,
            color_rest,
            color_rest,
            color_rest,
            color_rest,
            color_rest,
            color_rest,
            color_rest,
            ]
    
    print("🚨 if the coloring is incorrect, adjust by hand in function 'ax_dataset_boxplot()' ")
    if pos_spei is not None and pos_temp is not None:
        palette_order[pos_spei] = color_spei
        palette_order[pos_temp] = color_temp
    elif all_or_top9 == "all":
        palette_order[1] = color_temp
        palette_order[2] = color_spei
    elif all_or_top9 == "top9":
        palette_order[2] = color_temp
        palette_order[3] = color_spei
    elif all_or_top9 == "glmm_all":
        palette_order[4] = color_temp
        palette_order[5] = color_spei
    elif all_or_top9 == "glmm_top9":
        raise ValueError("Fix the palette order in the function!")
        

    # Map the category to the features in df_imp
    color_category = [color_dict[category_dict[col]] for col in df_imp.columns]

    # Create boxplot with colors based on categories
    ax_is_none = False
    if ax is None:
        ax_is_none = True
        fig, ax = plt.subplots(figsize=(8, 3.5))

    # Add jittered points (optional)
    # sns.swarmplot(data=df_imp, orient="h", color="black", alpha=0.25)

    # Outlier props
    flierprops = dict(
        marker="o",
        markerfacecolor="black",
        markersize=6,
        linestyle="none",
        alpha=0.5,
    )

    # Create boxplot
    sns.boxplot(
        data=df_imp,
        ax=ax,
        orient="h",
        # palette=color_category,  # Apply colors based on the categories
        palette=palette_order,
        linewidth=1.5,
        width=0.5,
        flierprops=flierprops,
    )

    # sns.violinplot(
    #     data=df_imp,
    #     ax=ax,
    #     orient="h",
    #     palette=color_category,  # Apply colors based on the categories
    #     inner="quartile",
    # )

    # Change y-axis labels (optional, you can re-enable the commented order_imp section if needed)
    # ax.set_yticklabels(order_imp)

    # Layout adjustments
    ax.set_xlabel(
        "Importance (%)",
        fontsize=base_fontsize * 1.2,
        fontweight="bold",
        labelpad=10,
    )
    ax.set_xlim(-0.5, 35)
    ax.tick_params(axis="x", labelsize=base_fontsize * 1.0)

    # Fix y-axis
    ax.tick_params(axis="y", labelsize=base_fontsize * 1.1)
    ax.set_ylabel(
        "Feature Category",
        fontsize=base_fontsize * 1.2,
        fontweight="bold",
        labelpad=10,
    )

    # Remove top and right axis
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if ax_is_none:
        plt.tight_layout()
        plt.show()
    else:
        return ax
    # plt.tight_layout()
    # plt.savefig(f"{dir_patterns}/03_dataset_importance.png")
    # plt.show()

def calculate_weighted_mean_importance(df, importance_columns):
    """
    Calculate the weighted mean importance for each species and each importance column.

    Args:
    df (pd.DataFrame): Input DataFrame containing 'species' and importance columns.
    importance_columns (list): List of importance column names.

    Returns:
    pd.DataFrame: DataFrame with weighted mean importance for each species and importance column.
    """
    # Initialize an empty DataFrame to store the results
    result_df = pd.DataFrame()

    # Iterate over each importance column to calculate the weighted mean importance
    for column in importance_columns:
        # Group by species and calculate the sum of importance and the count of occurrences
        species_grouped = df.groupby("species").agg(
            total_importance=(column, "sum"), count=(column, "count")
        )

        # Calculate the weighted mean importance for each species
        species_grouped[f"weighted_mean_{column}"] = (
            species_grouped["total_importance"] / species_grouped["count"]
        )

        # Keep only the weighted mean importance column
        result_df[f"weighted_mean_{column}"] = species_grouped[
            f"weighted_mean_{column}"
        ]

    return result_df.reset_index()


def get_species_with_models(return_list_or_dict):
    allRuns = glob.glob(
        "../../notebooks/20_tree_level_predictions/model_runs/_fullruns/*/*"
    )
    speciesWithModels = {}
    # Go through all runs
    for run in allRuns:
        # Get the species name
        species = run.split("/")[-1]
        # Check if final model for this species exists
        if os.path.isfile(f"{run}/final_model_performance.csv"):
            # If species is not in the dictionary add it and set the count to 1
            if species not in speciesWithModels:
                speciesWithModels[species] = 1
            else:
                speciesWithModels[species] += 1

    # Get most recent species data
    tmp = get_final_nfi_data_for_analysis(verbose=False).query(
        "tree_state_change in ['alive_alive', 'alive_dead']"
    )
    
    # Get normalized and non normalized counts
    species = tmp["species_lat2"].value_counts()
    species_norm = tmp["species_lat2"].value_counts(normalize=True)
    
    # Order dictionary based on occurrence
    speciesWithModels = {
        key: speciesWithModels[key]
        for key in species_norm.keys().tolist()
        if key in speciesWithModels
    }
    
    # Return the dictionary or list
    if return_list_or_dict == "list":
        return list(speciesWithModels.keys())
    elif return_list_or_dict == "dict":
        return speciesWithModels
    
    
def run_prauc_for_rfs(df_in):
    for i, row in df_in.iterrows():

        # path_rf = f"./model_runs/_fullruns/{row.model}/{row.species}/final_model_pr_auc_prob-based.csv"
        path_rf = f"./model_runs/_fullruns/{row.model}/{row.species}"
        save_dir = f"{path_rf}/roc_auc_curves"
        os.makedirs(save_dir, exist_ok=True)

        y_test = pd.read_csv(f"{path_rf}/final_model/y_test.csv", index_col=0)
        y_test_pred_proba = pd.read_csv(
            f"{path_rf}/final_model/y_test_proba.csv", index_col=0
        )

        y_train = pd.read_csv(f"{path_rf}/final_model/y_train.csv", index_col=0)
        y_train_pred_proba = pd.read_csv(
            f"{path_rf}/final_model/y_train_proba.csv", index_col=0
        )

        # QC
        if y_test.shape[0] != y_test_pred_proba.shape[0]:
            raise ValueError(
                f"Shapes for test do not match: {y_test.shape[0]} vs {y_test_pred_proba.shape[0]}"
            )
        if y_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                f"Shapes for train do not match: {y_train.shape[0]} vs {y_train_pred_proba.shape[0]}"
            )

        # Turn all into series
        y_test = y_test.iloc[:, 0]
        y_train = y_train.iloc[:, 0]
        y_test_pred_proba = y_test_pred_proba.iloc[:, 1]
        y_train_pred_proba = y_train_pred_proba.iloc[:, 1]

        # display(y_test.head())
        # display(y_test_pred_proba.head())
        # display(y_train.head())
        # display(y_train_pred_proba.head())

        plot_pr_auc_both(
            (y_train, y_train_pred_proba),
            (y_test, y_test_pred_proba),
            # save_directory=save_directory,
            save_directory=save_dir,
            show=False,
        )
        
        
def get_latin_to_common():
    return {
        "Fagus sylvatica": "European beech",
        "Quercus robur": "Pedunculate oak",
        "Quercus petraea": "Sessile oak",
        "Carpinus betulus": "European hornbeam",
        "Castanea sativa": "Sweet chestnut",
        "Quercus pubescens": "Downy oak",
        "Pinus sylvestris": "Scots pine",
        "Abies alba": "European silver fir",
        "Picea abies": "Norway spruce",
    }