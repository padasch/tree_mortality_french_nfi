# Data wrangling
from matplotlib.pylab import f
import pandas as pd
import numpy as np
import random

# Data visualisation
from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    roc_auc_score,
    roc_curve,
    balanced_accuracy_score,
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.inspection import PartialDependenceDisplay, permutation_importance

# Metrics
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
)
from sklearn.metrics import (
    f1_score,
    recall_score,
    precision_score,
    accuracy_score,
    make_scorer,
)

from scipy.stats import pearsonr, f_oneway, chi2_contingency

# My functions
import sys

sys.path.insert(0, "../../src")
from run_mp import *
from utilities import *
from random_forest_utils import *
import warnings as warnings

# Other
from os import error
import datetime
import re
import chime

# -----------------------------------------------------------------------------------------------


def ___GENERAL___():
    pass


# Create txt file with name
def write_txt(text):
    with open(text, "w") as file:
        pass


def create_new_run_folder(folder_suffix=None):
    # Get today's date
    today = datetime.date.today().strftime("%Y-%m-%d")

    # Create the subdirectory in "model_runs" with today's date as the name
    subdirectory = os.path.join("model_runs", today)
    os.makedirs(subdirectory, exist_ok=True)

    # Set folder pattern in daily folder
    folder_pattern = "run_"

    # Filter subdirectory to regex match the folder_pattern (omits other files and folders)
    all_folders = [
        folder
        for folder in os.listdir(subdirectory)
        if re.match(folder_pattern, folder)
    ]

    # Count the number of folders in the subdirectory
    num_folders = len(all_folders)

    # print(num_folders, all_folders)

    # Create a new folder with the name "run_n" where n is the number of folders + 1
    if num_folders < 9:
        folder_nr = f"0{num_folders + 1}"
    else:
        folder_nr = num_folders + 1

    new_folder = os.path.join(subdirectory, f"{folder_pattern}{folder_nr}")

    if folder_suffix:
        new_folder += f"_{folder_suffix}"

    os.makedirs(new_folder)
    print(f"New folder created: {new_folder}")

    return new_folder


def create_new_run_folder_treemort(species_name=None):

    if species_name is None:
        raise ValueError("species_name must be specified!")

    # Get today's date
    today = datetime.date.today().strftime("%Y-%m-%d")

    # Create the subdirectory in "model_runs" with today's date as the name
    subdirectory = os.path.join(f"model_runs/{species_name}/{today}")
    os.makedirs(subdirectory, exist_ok=True)

    # Set folder pattern
    folder_pattern = "run_"

    # Filter subdirectory to regex match the folder_pattern (omits other files and folders)
    all_folders = [
        folder
        for folder in os.listdir(subdirectory)
        if re.match(folder_pattern, folder)
    ]

    # If all folders are empty, set the number of folders to 0
    if all_folders == []:
        num_folders = 0
    else:
        # Get the number of the latest folder
        num_folders = max(
            [int(re.search(r"\d+", folder).group()) for folder in all_folders]
        )

    # Create a new folder with the name "run_n" where n is the number of folders + 1
    if num_folders < 9:
        folder_nr = f"0{num_folders + 1}"
    else:
        folder_nr = num_folders + 1

    new_folder = os.path.join(subdirectory, f"{folder_pattern}{folder_nr}")

    os.makedirs(new_folder)
    print(f"New folder created: {new_folder}")

    return new_folder

def create_new_run_folder_treemort_fullrun(folder_suffix=None):

    # Create the subdirectory in "model_runs" with today's date as the name
    subdirectory = os.path.join(f"model_runs/_fullruns/")
    os.makedirs(subdirectory, exist_ok=True)

    # Set folder pattern
    folder_pattern = "run_"

    # Filter subdirectory to regex match the folder_pattern (omits other files and folders)
    all_folders = [
        folder
        for folder in os.listdir(subdirectory)
        if re.match(folder_pattern, folder)
    ]

    # If all folders are empty, set the number of folders to 0
    if all_folders == []:
        num_folders = 0
    else:
        # Get the number of the latest folder
        num_folders = max(
            [int(re.search(r"\d+", folder).group()) for folder in all_folders]
        )

    # Create a new folder with the name "run_n" where n is the number of folders + 1
    if num_folders < 9:
        folder_nr = f"0{num_folders + 1}"
    else:
        folder_nr = num_folders + 1

    new_folder = os.path.join(subdirectory, f"{folder_pattern}{folder_nr}")

    if folder_suffix is not None:
        new_folder = new_folder + f" - {folder_suffix}/"

    os.makedirs(new_folder)
    print(f"New folder created: {new_folder}")

    return new_folder



# -----------------------------------------------------------------------------------------------
def get_current_folder():
    # Get today's date
    today = datetime.date.today().strftime("%Y-%m-%d")

    # Create the subdirectory in "model_runs" with today's date as the name
    subdirectory = os.path.join("model_runs", today)

    # Set folder pattern in daily folder
    folder_pattern = "run_"

    # Filter subdirectory to regex match the folder_pattern (omits other files and folders)
    all_folders = [
        folder
        for folder in os.listdir(subdirectory)
        if re.match(folder_pattern, folder)
    ]

    # Since folders are sorted by run number, the last in the list is the newest
    # print(sorted(all_folders))
    current_folder = sorted(all_folders)[-1]
    return current_folder


def run_rfecv_treemort(
    dict_categories=None,
    var_ohe_dict=None,
    Xy_train_for_rfe=None,
    user_input=None,
    rfecv_params=None,
    debug_stop=False,
    debug_stop_after_n_iterations=10,
    verbose=True,
):

    # ! --------------------------------------------------------
    # ! Set parameters
    # ! --------------------------------------------------------

    # Folds
    cv_folds = user_input["cv_folds"]

    # Number of features to remove
    features_to_remove_above200 =  5
    features_to_remove_above100 =  5
    features_to_remove_above50 =  3
    features_to_remove_above30 = 1
    features_to_remove_above15 = 1
    features_to_remove_below15 = 1

    min_features_per_category = 2

    # Empty lists
    vars_to_keep_as_min_per_category = []
    ohe_vars_to_remove = []
    non_ohe_vars_to_remove = []
    new_ohe_vars_to_remove = []
    new_non_ohe_vars_to_remove = []

    ohe_vars_to_remove_n = 0
    non_ohe_vars_to_remove_n = 0
    new_ohe_vars_to_remove_n = 0
    new_non_ohe_vars_to_remove_n = 0

    # ! --------------------------------------------------------
    # ! Prepare final metrics dataframe:
    # ! --------------------------------------------------------

    df_cvmetrics_per_nfeatures = pd.DataFrame({})

    # Original number of features (not-ohe)
    # Code below reduces the var_ohe_dict to keys where a value
    # matches a column in X_train_imputed.
    reduced_var_ohe_dict = {
        key: value
        for key, value in var_ohe_dict.items()
        if any(col in Xy_train_for_rfe.columns for col in value)
    }
    # Remove target and strata from the dictionary
    reduced_var_ohe_dict.pop("target", None)
    reduced_var_ohe_dict.pop("test_train_strata", None)

    original_number_of_features = len(reduced_var_ohe_dict)
    remaining_features_numbers = original_number_of_features

    # Set iteration counter
    iteration_count = 1

    # ! --------------------------------------------------------
    # ! START REF-CV
    # ! --------------------------------------------------------
    continue_loop = True
    while continue_loop:
        # Set vectors to track metrics
        v_accuracy = []
        v_f1 = []
        v_recall = []
        v_precision = []
        v_roc_auc = []

        # Remove variables of previous iteration
        Xy_refcv = (
            Xy_train_for_rfe.copy()
            .drop(columns=ohe_vars_to_remove)
            .reset_index(drop=True)
        )

        # Get current number of features
        remaining_features_numbers = (
            original_number_of_features - non_ohe_vars_to_remove_n
        )

        # if verbose:
        print(
            f"\n⭐⭐⭐ Iteration {iteration_count} ------- Current Number of features: {original_number_of_features} - {non_ohe_vars_to_remove_n} = {remaining_features_numbers}",
        )

        # If no more features to remove, break loop
        if (remaining_features_numbers) == 0:
            continue_loop = False
            continue

        if user_input["method_validation"] == "cv":
            rf, sco, imp = SMOTE_cv(
                Xy_all=Xy_refcv,
                var_ohe_dict=var_ohe_dict,
                rf_params=rfecv_params,
                method_importance=user_input["method_importance"],
                smote_on_test=user_input["do_smote_test_validation"],
                do_tuning=user_input["do_tuning"],
                rnd_seed=user_input["seed_nr"],
                verbose=verbose,
                save_directory=None,
            )
        elif user_input["method_validation"] == "oob":
            rf, sco, imp = SMOTE_oob(
                Xy_all=Xy_refcv,
                var_ohe_dict=var_ohe_dict,
                rf_params=rfecv_params,
                method_importance=user_input["method_importance"],
                smote_on_test=user_input["do_smote_test_validation"],
                do_tuning=user_input["do_tuning"],
                rnd_seed=user_input["seed_nr"],
                verbose=verbose,
                save_directory=None,
            )
        else:
            raise ValueError(f"Failed during RFE - Invalid method_validation! Got: {user_input['method_validation']}")
            
        # ! --------------------------------------------------------
        # ! VARIABLE REMOVAL
        # ! --------------------------------------------------------
        # Set number of variables to remove
        n_to_remove = features_to_remove_below15

        if remaining_features_numbers > 15:
            n_to_remove = features_to_remove_above15

        if remaining_features_numbers > 30:
            n_to_remove = features_to_remove_above30

        if remaining_features_numbers > 50:
            n_to_remove = features_to_remove_above50

        if remaining_features_numbers > 100:
            n_to_remove = features_to_remove_above100

        if remaining_features_numbers > 200:
            n_to_remove = features_to_remove_above200

        # If there are more features left than the minimum per category
        # make sure to equal the count for each category by removing the variables form the imp
        if remaining_features_numbers > min_features_per_category * len(
            dict_categories
        ):
            # Remove variables that are of low importance but kept to distill df to equal number of features per category
            imp = imp.query(
                "Feature not in @vars_to_keep_as_min_per_category"
            ).reset_index(drop=True)
            variables_to_remove, vars_to_keep = get_variables_to_remove(
                df_in=imp,
                dict_in=dict_categories,
                min_features_per_category=min_features_per_category,
                n_to_remove=n_to_remove,
            )
        else:
            # If min number is achieved, reduce to top 1 feature per category
            variables_to_remove, vars_to_keep = get_variables_to_remove(
                df_in=imp,
                dict_in=dict_categories,
                min_features_per_category=1,
                n_to_remove=n_to_remove,
            )
            # Bugfix: If there are no variables to remove, remove remaining variables at once.
            if len(variables_to_remove) == 0:
                variables_to_remove, vars_to_keep = get_variables_to_remove(
                    df_in=imp,
                    dict_in=dict_categories,
                    min_features_per_category=0,
                    n_to_remove=imp.shape[
                        0
                    ],  # = goes through all variables in imp and checks if they can be dropped
                )

                # If still no variables are to be removed, remove the least important variable
                if len(variables_to_remove) == 0:
                    variables_to_remove = [imp["Feature"].iloc[-1]]
                    vars_to_keep = []

        # Update the variables that should be ignored as long as there are unequal number of features per category
        vars_to_keep_as_min_per_category = (
            vars_to_keep_as_min_per_category + vars_to_keep
        )
        # Reduce imp to the variables that are allowed to be removed
        imp = imp.query("Feature in @variables_to_remove").reset_index(drop=True)
        # Get names and length of variables in their ohe form
        new_ohe_vars_to_remove = imp["Vars_in_key"].to_list()
        # Unlist nested lists
        new_ohe_vars_to_remove = [
            item for sublist in new_ohe_vars_to_remove for item in sublist
        ]
        new_ohe_vars_to_remove_n = len(new_ohe_vars_to_remove)

        # Add to list of variables to remove
        ohe_vars_to_remove = new_ohe_vars_to_remove + ohe_vars_to_remove
        ohe_vars_to_remove_n = len(ohe_vars_to_remove)

        # Do the same for the non-ohe variable names
        new_non_ohe_vars_to_remove = imp["Feature"].to_list()
        new_non_ohe_vars_to_remove_n = len(new_non_ohe_vars_to_remove)

        # No unlisting needed
        non_ohe_vars_to_remove = new_non_ohe_vars_to_remove + non_ohe_vars_to_remove
        non_ohe_vars_to_remove_n = len(non_ohe_vars_to_remove)

        # ! --------------------------------------------------------
        # ! VARIABLES KEPT IN MODEL
        # ! --------------------------------------------------------
        ohe_vars_in_model = Xy_refcv.drop(
            columns=["target", "test_train_strata"],
            errors="ignore",
        ).columns.to_list()
        ohe_vars_in_model_n = len(ohe_vars_in_model)

        # Cross-check ohe_vars_in_model with the values in var_ohe_dict and return key if there is a match
        non_ohe_vars_in_model = [
            key
            for key, value in var_ohe_dict.items()
            if any(col in ohe_vars_in_model for col in value)
        ]
        non_ohe_vars_in_model = list(set(non_ohe_vars_in_model))
        non_ohe_vars_in_model_n = len(non_ohe_vars_in_model)

        if verbose:

            print(
                # f"\n --- REMOVAL ---",
                # f"\n - For one-hot-encoded variable names:",
                # "\n    - Outcommented:",
                # f"\n   - New variables to remove (ohe-encoded-names): {new_ohe_vars_to_remove_n}\t|{new_ohe_vars_to_remove}",
                # f"\n   - All variables to remove (ohe-encoded-names): {ohe_vars_to_remove_n}\t|{ohe_vars_to_remove}",
                # f"\n\n - For original variable names:",
                # f"\n   - New variables to remove (original-names): {new_non_ohe_vars_to_remove_n}\t|{new_non_ohe_vars_to_remove}",
                # f"\n   - All variables to remove (original-names): {non_ohe_vars_to_remove_n}\t|{non_ohe_vars_to_remove}",
                # f"\n\n --- KEEPING ---",
                # f"\n\n - Variables in model",
                # f"\n   - ohe-encoded-names\t{ohe_vars_in_model_n}\t| {ohe_vars_in_model}",
                # f"\n   - original-names\t{non_ohe_vars_in_model_n}\t| {non_ohe_vars_in_model}",
                # f"\n\n --- METRICS ---",
            )

            print(
                f" - Acc: {round(sco['accuracy'].iloc[0], 2)} (+- {round(sco['accuracy'].iloc[1], 2)})",
                f" | F1: {round(sco['f1'].iloc[0], 2)} (+- {round(sco['f1'].iloc[1], 2)})",
                f" | Precision: {round(sco['precision'].iloc[0], 2)} (+- {round(sco['precision'].iloc[1], 2)})",
                f" | Recall: {round(sco['recall'].iloc[0], 2)} (+- {round(sco['recall'].iloc[1], 2)})",
                f" | ROC-AUC: {round(sco['roc_auc'].iloc[0], 2)} (+- {round(sco['roc_auc'].iloc[1], 2)})",
            )

        # Write to file
        # print(f"\n - Write info file...\n", end="")
        with open(f"{user_input['current_dir']}/refcv_log_removal.txt", "a") as f:
            f.write(
                f"\n - Iteration {iteration_count} | {remaining_features_numbers}/{original_number_of_features} features in the model | Features removed:"
            )
            for var in new_non_ohe_vars_to_remove:
                f.write(f"\n\t - {var}")

        with open(f"{user_input['current_dir']}/refcv_log_keeping.txt", "a") as f:
            f.write(
                f"\n - Iteration {iteration_count} | {remaining_features_numbers}/{original_number_of_features} features in the model | Features kept:"
            )
            for var in non_ohe_vars_in_model:
                f.write(f"\n\t - {var}")

        # ! --------------------------------------------------------
        # ! Save to metrics dataframe
        # ! --------------------------------------------------------
        # Save rf metrics for current number of features
        df_new_row_metrics = pd.DataFrame(
            {
                # Metrics
                "accuracy": sco["accuracy"].iloc[0],
                "accuracy_sd": sco["accuracy"].iloc[1],
                "f1": sco["f1"].iloc[0],
                "f1_sd": sco["f1"].iloc[1],
                "recall": sco["recall"].iloc[0],
                "recall_sd": sco["recall"].iloc[1],
                "precision": sco["precision"].iloc[0],
                "precision_sd": sco["precision"].iloc[1],
                "roc_auc": sco["roc_auc"].iloc[0],
                "roc_auc_sd": sco["roc_auc"].iloc[1],
            }, index=[0]
        )
        if user_input["method_validation"] == "oob":
            df_new_row_metrics["oob"] = sco["oob"].iloc[0]
            df_new_row_metrics["oob_sd"] = np.nan
        
        df_new_row_vars = pd.DataFrame(
            {
                "n_features": [remaining_features_numbers],
                # Numbers
                "new_ohe_vars_to_remove_n": new_ohe_vars_to_remove_n,
                "new_non_ohe_vars_to_remove_n": new_non_ohe_vars_to_remove_n,
                "ohe_vars_to_remove_n": ohe_vars_to_remove_n,
                "non_ohe_vars_to_remove_n": non_ohe_vars_to_remove_n,
                "ohe_vars_in_model_n": ohe_vars_in_model_n,
                "non_ohe_vars_in_model_n": non_ohe_vars_in_model_n,
                # Lists
                "new_ohe_vars_to_remove": [new_ohe_vars_to_remove],
                "new_non_ohe_vars_to_remove": [new_non_ohe_vars_to_remove],
                "ohe_vars_to_remove": [ohe_vars_to_remove],
                "non_ohe_vars_to_remove": [non_ohe_vars_to_remove],
                "ohe_vars_in_model": [Xy_refcv.columns],
                "ohe_vars_in_model": [ohe_vars_in_model],
                "non_ohe_vars_in_model": [non_ohe_vars_in_model],
            }
        )

        # Concatenate
        df_new_row = pd.concat([df_new_row_metrics, df_new_row_vars], axis=1)
        df_cvmetrics_per_nfeatures = pd.concat(
            [df_cvmetrics_per_nfeatures, df_new_row], axis=0
        )

        iteration_count = iteration_count + 1

        if debug_stop and iteration_count == debug_stop_after_n_iterations:
            chime.error()
            continue_loop = False

    df_cvmetrics_per_nfeatures = move_vars_to_front(df_cvmetrics_per_nfeatures, ["n_features"])

    df_cvmetrics_per_nfeatures.to_csv(
        f"{user_input['current_dir']}/refcv_metrics.csv", index=False
    )
    # chime.success()

    return df_cvmetrics_per_nfeatures
def SMOTE_cv(
    Xy_all=None,
    var_ohe_dict=None,
    rf_params=None,
    method_importance="impurity",
    cv_folds=5,
    smote_on_test=False,
    do_tuning=False,
    rnd_seed=42,
    verbose=True,
    save_directory=None,
):
    # ! INPUT CHECKS --------------------------------------------------------------
    if verbose:
        print("--- SMOTE_refcv():")
    if Xy_all.isna().sum().sum() > 0:
        raise ValueError("SMOTE_refcv(): There are NA values in the input data!")
    if rf_params is None:
        rf_params = {
            "n_estimators": 150,
            "max_features": "sqrt",
            "max_depth": None,
            "criterion": "gini",
        }
    
    if method_importance not in ["impurity", "permutation"]:
        raise ValueError(
            f"SMOTE_refcv(): 'method_importance' must be 'impurity' or 'permutation'. Given was: {method_importance}"
        )
        
    do_feat_aggregation = True if var_ohe_dict is not None else False
    
    # ! SETUP ---------------------------------------------------------------------
    # Build model
    sm = SMOTE(random_state=rnd_seed)
    
    rf = RandomForestClassifier(
        n_estimators=rf_params["n_estimators"],
        max_features=rf_params["max_features"],
        max_depth=rf_params["max_depth"],
        criterion=rf_params["criterion"],
        random_state=rnd_seed,
        n_jobs=-1,
        class_weight="balanced",
    )
    
    # ! CV ---------------------------------------------------------------------
    if verbose:
        print(" - Running CV...")
    
    # Split into X and y
    X = Xy_all.drop(columns=["target", "test_train_strata"], errors="ignore")
    y = Xy_all["target"]
        
    # Setup
    cvfold=0
    scoring=[]
    featimp=[]
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=rnd_seed)

    for train_index, val_index in skf.split(X, y):
        # Update fold counter
        cvfold += 1
        if verbose:
            print(f" - Fold {cvfold}/{cv_folds}")

        # Split dfs
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Apply SMOTE
        X_train, y_train = sm.fit_resample(X_train, y_train)
        if smote_on_test:
            X_val, y_val = sm.fit_resample(X_val, y_val)

        # Train model
        rf.fit(X_train, y_train)

        # Attach scores
        y_pred = rf.predict(X_val)
        scores = {
            # "fold": cvfold,
            "f1": f1_score(y_val, y_pred, average="weighted"),
            "recall": recall_score(y_val, y_pred, average="weighted"),
            "precision": precision_score(y_val, y_pred, average="weighted"),
            "accuracy": balanced_accuracy_score(y_val, y_pred),
            "roc_auc": roc_auc_score(y_val, y_pred, average="weighted"),
        }
        scoring.append(pd.DataFrame(scores, index=[0]))
        
        # todo Remove this part. I moved permutation to use the full dataset instead.
        # # Variable importance - Permutation
        # if method_importance == "permutation":
        #     i_featimp = assessing_top_predictors(
        #         vi_method = "permutation",
        #         rf_in=rf,
        #         ignore_these=["target", "test_train_strata"],
        #         X_train_in=X_train,
        #         X_test_in=X_val,
        #         y_test_in=y_val,
        #         dict_ohe_in=var_ohe_dict,
        #         with_aggregation=do_feat_aggregation,
        #         verbose=verbose,
        #         random_state=rnd_seed,
        #         save_directory=save_directory,
        #     )
            
        #     featimp.append(i_featimp)
            
    # ! End of CV loop    
    # Get mean and sd of scores
    scoring = pd.DataFrame(
        {
            "mean": pd.concat(scoring, axis=0).mean(),
            "sd": pd.concat(scoring, axis=0).std(),
        }
    ).T
        
    # Get final variable importance
    if method_importance == "permutation":
        
        _, _, featimp = SMOTE_oob(
            Xy_all=Xy_all,
            var_ohe_dict=var_ohe_dict,
            rf_params=rf_params,
            method_importance="permutation",
            val_train_split=0.2,
            smote_on_test=smote_on_test,
            do_tuning=do_tuning,
            rnd_seed=rnd_seed,
            verbose=verbose,
            save_directory=save_directory,
        )
        
        # todo: Below is the code for aggregating the feature importance over every split.
        # featimp = pd.DataFrame({
        #     "Feature": pd.concat(featimp, axis=0).groupby("Feature").mean().reset_index()["Feature"],
        #     "Importance": pd.concat(featimp, axis=0).groupby("Feature").mean().reset_index()["Importance"],
        #     "Std": pd.concat(featimp, axis=0).groupby("Feature").std().reset_index()["Importance"],
        # })
        
    elif method_importance == "impurity":
        if do_tuning:
            # Run small grid search to find best parameters
            rf = RandomForestClassifier(random_state=rnd_seed, n_jobs=-1, class_weight="balanced")
            param_grid = rfe_tuning_params()
            grid = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, verbose=0)
            grid.fit(X, y)
        
            # Get best parameters
            rf_params = grid.best_params_

            # Define model
            rf = RandomForestClassifier(
                n_estimators=rf_params["n_estimators"],
                max_features=rf_params["max_features"],
                max_depth=rf_params["max_depth"],
                criterion=rf_params["criterion"],
                random_state=rnd_seed,
                n_jobs=-1,
                class_weight="balanced",
                oob_score=True,
            )
        
            
        # Impurity needs retraining of model on all data
        rf.fit(X, y)
        
        featimp = assessing_top_predictors(
            vi_method="impurity",
            rf_in=rf,
            ignore_these=["target", "test_train_strata"],
            X_train_in=X,
            dict_ohe_in=var_ohe_dict,
            with_aggregation=do_feat_aggregation,
            n_predictors=None,
            verbose=verbose,
            random_state=rnd_seed,
            save_directory=save_directory,
        )
    else:
        raise ValueError(f"SMOTE_refcv(): Method '{method_importance}' not implemented.")

    featimp.sort_values("Importance", ascending=False, inplace=True)

    # Verbose
    if verbose:
        print(" - CV Results:")
        display(scoring)
        print(" - Variable Importance:")
        display(featimp)

    return rf, scoring, featimp

# -----------------------------------------------------------------------------------------------
def SMOTE_oob(
    Xy_all=None,
    var_ohe_dict=None,
    rf_params=None,
    method_importance="impurity",
    val_train_split=0.1,
    smote_on_test=False,
    do_tuning=False,
    rnd_seed=42,
    verbose=True,
    save_directory=None,
):
    # ! INPUT CHECKS --------------------------------------------------------------
    if verbose:
        print("--- SMOTE_refcv():")
    if Xy_all.isna().sum().sum() > 0:
        raise ValueError("SMOTE_refcv(): There are NA values in the input data!")
    if rf_params is None:
        rf_params = {
            "n_estimators": 150,
            "max_features": "sqrt",
            "max_depth": None,
            "criterion": "gini",
        }
    
    if method_importance not in ["impurity", "permutation"]:
        raise ValueError(
            f"SMOTE_refcv(): 'method_importance' must be 'impurity' or 'permutation'. Given was: {method_importance}"
        )
        
    do_feat_aggregation = True if var_ohe_dict is not None else False
    
    # ! Preparation ---------------------------------------------------------------------
    # Split into test and train
    X = Xy_all.drop(columns=["target", "test_train_strata"], errors="ignore")
    y = Xy_all["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=val_train_split,
        random_state=rnd_seed,
        stratify=y,
    )
    
    # Apply SMOTE
    sm = SMOTE(random_state=rnd_seed)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    if smote_on_test:
        X_test, y_test = sm.fit_resample(X_test, y_test)
    
    # ! DEBUG TO USE ALL DATA FOR OOB! ---------------------------------------------------------------------
    if True:
        print("🔴🔴🔴 SMOTE_oob: Running on ALL data without splitting 🔴🔴🔴")
        # Apply oversampling on all data
        X, y = sm.fit_resample(X, y)
        
        X_train, y_train = X, y
        X_test, y_test = X, y
    
    # ! Tuning ---------------------------------------------------------------------
    
    if do_tuning:
        if verbose:
            print(" - Tuning...")
        display("❌❌❌❌ DEBUG 2")

            
        # Run small grid search to find best parameters
        rf = RandomForestClassifier(random_state=rnd_seed, n_jobs=-1, class_weight="balanced")
        param_grid = rfe_tuning_params()
        grid = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, verbose=0)
        grid.fit(X_train, y_train)
    
        # Get best parameters
        rf_params = grid.best_params_

    # ! Training ---------------------------------------------------------------------
    # Define model
    rf = RandomForestClassifier(
        n_estimators=rf_params["n_estimators"],
        max_features=rf_params["max_features"],
        max_depth=rf_params["max_depth"],
        criterion=rf_params["criterion"],
        random_state=rnd_seed,
        n_jobs=-1,
        class_weight="balanced",
        oob_score=True,
    ) 
    
    # Train model
    rf.fit(X_train, y_train)
    
    # ! Variable Importance ---------------------------------------------------------------------
    if method_importance == "permutation":
        # Variable importance - Permutation
        featimp = assessing_top_predictors(
            vi_method = "permutation",
            rf_in=rf,
            ignore_these=["target", "test_train_strata"],
            # X_train_in=X_train, # Not needed for permutation!
            X_test_in=X_test,
            y_test_in=y_test,
            dict_ohe_in=var_ohe_dict,
            with_aggregation=do_feat_aggregation,
            verbose=verbose,
            random_state=rnd_seed,
            save_directory=save_directory,
        )
    
    elif method_importance == "impurity":
        # Variable importance - Impurity
        featimp = assessing_top_predictors(
            vi_method = "impurity",
            rf_in=rf,
            ignore_these=["target", "test_train_strata"],
            X_train_in=X_train,
            dict_ohe_in=var_ohe_dict,
            with_aggregation=do_feat_aggregation,
            verbose=verbose,
            random_state=rnd_seed,
            save_directory=save_directory,
        )
        
    featimp.sort_values("Importance", ascending=False, inplace=True)
        
    # ! Model Evaluation ---------------------------------------------------------------------
    y_pred = rf.predict(X_test)
    y_pred = pd.Series(y_pred, index=y_test.index)
    scoring = bootstrap_classification_metric(
        y_test, y_pred, ["accuracy", "precision", "recall", "f1", "roc_auc"]
    )
    
    scoring["oob"] = np.nan
    scoring.at["mean", "oob"] = rf.oob_score_
    
    # Verbose
    if verbose:
        print(" - OOB Score: ", rf.oob_score_)
        print(" - Variable Importance:")
        display(featimp)
        print(" - Model Evaluation:")
        display(scoring)

    return rf, scoring, featimp

# -----------------------------------------------------------------------------------------------
def bootstrap_classification_metric(y_true, y_pred, metrics, n_bootstraps=100):

    # Reset index for input data
    y_true = y_true.reset_index(drop=True)
    y_pred = y_pred.reset_index(drop=True)

    # Initialize a dictionary to store bootstrap results for each metric
    bootstrap_results = {metric: [] for metric in metrics}

    # Perform bootstrap iterations
    np.random.seed(42)
    
    for _ in range(n_bootstraps):
        indices = np.random.choice(range(len(y_true)), len(y_true), replace=True)
        
        # Skip if only one class has been selected
        if len(np.unique(y_true[indices])) == 1:
            continue
        
        for metric in metrics:
            if metric == "accuracy":
                value = accuracy_score(y_true[indices], y_pred[indices])
            elif metric == "precision":
                value = precision_score(
                    y_true[indices], y_pred[indices], average="weighted", zero_division=np.nan
                )
            elif metric == "recall":
                value = recall_score(
                    y_true[indices], y_pred[indices], average="weighted"
                )
            elif metric == "f1":
                value = f1_score(y_true[indices], y_pred[indices], average="weighted")
            elif metric == "roc_auc":
                value = roc_auc_score(
                    y_true[indices], y_pred[indices], average="weighted"
                )

            bootstrap_results[metric].append(value)

    # Calculate mean and std for each metric
    means = []
    stds = []
    for metric in metrics:
        means.append(np.mean(bootstrap_results[metric]))
        stds.append(np.std(bootstrap_results[metric]))

    # Create the final DataFrame
    df_final = pd.DataFrame([means, stds], index=["mean", "sd"], columns=metrics)

    return df_final

def rfe_tuning_params():
    return {
        "n_estimators": [100],
        "max_features": [0.1, "sqrt"],
        "max_depth": [1, 3, 12],
        "criterion": ["gini"],
        "bootstrap": [True],
    }

def move_vars_to_front(df, vars):

    if not isinstance(vars, list):
        vars = [vars]

    # Move vars to front
    df = df[vars + [col for col in df.columns if col not in vars]]

    # Return
    return df

# -----------------------------------------------------------------------------------------------
def ___TUNING___():
    pass


def set_best_rf_params(model_type):
    # Return best variables from previous search
    print(f"Returning best parameters for {model_type} model.")

    if model_type == "regression":
        # ! REGRESSION -------------------------------------------------------------------
        best_params = {
            "n_estimators": 500,
            "max_features": 0.1,
            "max_depth": 20,
            "bootstrap": True,
            "criterion": "squared_error",  # ["squared_error", "absolute_error", "friedman_mse", "poisson"]
        }

    elif model_type == "binary":
        # ! BINARY CLASSIFICATION -------------------------------------------------------------------
        best_params = {
            "n_estimators": 500,
            "max_features": 0.2,
            "max_depth": 5,
            "bootstrap": True,
            "criterion": "gini",
        }

    elif model_type == "multiclass":
        # ! MULTI - CLASSIFICATION -------------------------------------------------------------------
        best_params = {
            "n_estimators": 500,
            "max_features": 0.2,
            "max_depth": 5,
            "bootstrap": True,
            "criterion": "gini",
        }
    else:
        chime.warning()
        raise ValueError("Invalid Model Type!")

    return best_params


def plot_grid_search_results(grid, rnd_or_psc=None, save_directory=None, show=True):
    """
    Params:
        grid: A trained GridSearchCV object.
    """
    ## Results from grid search
    results = grid.cv_results_
    means_test = results["mean_test_score"]
    stds_test = results["std_test_score"]
    means_train = results["mean_train_score"]
    stds_train = results["std_train_score"]

    ## Getting indexes of values per hyper-parameter
    masks = []
    masks_names = list(grid.best_params_.keys())
    for p_k, p_v in grid.best_params_.items():
        masks.append(list(results["param_" + p_k].data == p_v))

    params = grid.param_grid

    ## Ploting results
    fig, ax = plt.subplots(1, len(params), sharex="none", sharey="all", figsize=(20, 5))
    fig.suptitle("Score per parameter")
    fig.text(0.04, 0.5, "MEAN SCORE", va="center", rotation="vertical")
    pram_preformace_in_best = {}
    for i, p in enumerate(masks_names):
        m = np.stack(masks[:i] + masks[i + 1 :])
        pram_preformace_in_best
        best_parms_mask = m.all(axis=0)
        best_index = np.where(best_parms_mask)[0]
        x = np.array(params[p])
        y_1 = np.array(means_test[best_index])
        e_1 = np.array(stds_test[best_index])
        y_2 = np.array(means_train[best_index])
        e_2 = np.array(stds_train[best_index])
        ax[i].errorbar(x, y_1, e_1, linestyle="--", marker="o", label="test")
        ax[i].errorbar(x, y_2, e_2, linestyle="-", marker="^", label="train")
        ax[i].set_xlabel(p.upper())

    plt.legend()
    
    if save_directory is not None:
        if rnd_or_psc is None:
            raise ValueError("rnd_or_psc must be specified")
        else:
            fig.savefig(f"{save_directory}/fig_grid_search_{rnd_or_psc}.png")
    if show:
        plt.show()
    else:
        plt.close()
    # return fig


# -----------------------------------------------------------------------------------------------
def show_top_predictors(
    X_train=None,
    vars_to_ohe=None,
    rf_model=None,
    with_aggregation=False,
    n_predictors=20,
    verbose=False,
    current_dir=None,
):
    # Plot the variable importance
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    df_featimp = pd.DataFrame(
        {
            "Feature": X_train.columns[indices],
            "Importance": importances[indices],
        }
    )

    if verbose:
        print("Original size of df_featimp: ", df_featimp.shape)

    if with_aggregation:
        # For features matching the string in vars_to_ohe_red, sum up their importances and set name to vars_to_ohe_red
        # Make sure aggregation procedure is saved to file for checking later on:

        if vars_to_ohe is None:
            raise ValueError("vars_to_ohe must be specified for aggregation!")

        if verbose:
            display("Aggregating variables...")

        rows_to_drop = []
        rows_to_append = []
        text_to_save = []
        agg_dict = {}

        # vars_to_ohe_red = [var for var in vars_to_ohe if var in X_train.columns]

        # for var in vars_to_ohe_red:
        #     n_vars = 0
        #     feat_sum = 0
        #     merged_vars = []
        #     for i in range(len(df_featimp)):
        #         if var in df_featimp.loc[i, "Feature"]:
        #             merged_vars.append(df_featimp.loc[i, "Feature"])
        #             feat_sum += df_featimp.loc[i, "Importance"]
        #             n_vars += 1
        #             rows_to_drop.append(i)

        for var in vars_to_ohe:
            n_vars = 0
            feat_sum = 0
            merged_vars = []
            pattern = r"^" + var + r"_.*"

            for i in range(len(df_featimp)):
                if re.match(pattern, df_featimp.loc[i, "Feature"]):
                    merged_vars.append(df_featimp.loc[i, "Feature"])
                    feat_sum += df_featimp.loc[i, "Importance"]
                    n_vars += 1
                    rows_to_drop.append(i)

            # Attach to rows_to_append
            rows_to_append.append({"Feature": var, "Importance": feat_sum})

            # Print aggregation information
            # print(f"Merged {n_vars} vars into {var} containing: {merged_vars}")

            # Save aggregation to a dictionary
            agg_dict[var] = merged_vars

            # Save information to file
            if verbose:
                text_to_save = text_to_save + [
                    f"Merged {n_vars} vars into {var} containing:\n {merged_vars} \n\n"
                ]

        # Drop the rows that were merged
        df_featimp = df_featimp.drop(rows_to_drop)
        df_featimp = pd.concat(
            [df_featimp, pd.DataFrame(rows_to_append)], ignore_index=True
        )
        df_featimp = df_featimp.sort_values(by="Importance", ascending=False)

        if verbose:
            print("df_featimp after merging: ", df_featimp.shape)

            # Write to file
            file_path = f"{current_dir}/vip_aggregation_of_ohe_into_their_originals.txt"
            with open(file_path, "w") as file:
                for item in text_to_save:
                    file.write(f"{item}\n\n")
    else:
        agg_dict = None

    top_n = df_featimp.head(n_predictors)

    if verbose:
        # Show top n predictors table
        display(top_n)

    # Plot the variable importance
    sns_plot = sns.barplot(x="Importance", y="Feature", data=top_n, color="r")
    plt.tight_layout()

    # Save the barplot as an image file
    sns_plot.figure.savefig(f"{current_dir}/vip_plot_aggregated-{with_aggregation}.png")

    # Save the dataframe as a tab-separated file
    df_featimp.to_csv(f"{current_dir}/vip_table_aggregated-{with_aggregation}.csv")

    if verbose:
        plt.show()
    plt.close()

    # Return aggregation dictionary if needed
    return df_featimp, agg_dict


# -----------------------------------------------------------------------------------------------
def assessing_top_predictors(
    vi_method=None,
    rf_in=None,
    ignore_these=[],
    X_train_in=None,
    X_test_in=None,
    y_test_in=None,
    dict_ohe_in=None,
    with_aggregation=False,
    n_predictors=20,
    random_state=42,
    verbose=True,
    save_directory=None,
):
    # ! Checks -------------------------------------------------------
    if rf_in is None:
        raise ValueError("rf_in must be provided!")
    
    # ! Get feature importances --------------------------------------
    # IMPURITY
    if vi_method == "impurity":
        # More checks
        if X_train_in is None:
            raise ValueError(
                "X_train_in must be provided when using impurity importance!"
            )
        # Update n_predictors based on number of variables in X_train_in
        if n_predictors is None:
            n_predictors = len(X_train_in.columns)
        else:
            n_predictors = min(n_predictors, len(X_train_in.columns))

        # Get importances
        importances = rf_in.feature_importances_
        std = np.std([tree.feature_importances_ for tree in rf_in.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]
        df_featimp_org = pd.DataFrame(
            {
                "Feature": X_train_in.columns[indices],
                "Importance": importances[indices],
                "Std": std[indices],
            }
        )
        
    # PERMUTATION
    elif vi_method == "permutation":
        # More checks
        if X_test_in is None or y_test_in is None:
            raise ValueError(
                "X_test_in must be provided when using permutation importance!"
            )
        X_train_in = X_test_in.copy() # Makes sure that remaining code works properly (todo should be cleaned)
            
        # Update n_predictors based on number of variables in X_train_in
        if n_predictors is None:
            n_predictors = len(X_test_in.columns)
        else:
            n_predictors = min(n_predictors, len(X_test_in.columns))

        result = permutation_importance(
            rf_in,
            X_test_in,
            y_test_in,
            n_repeats=5,
            random_state=random_state,
            n_jobs=-1,
        )

        # Permutation importance can be negative because it measures the change in score when a feature is permuted.
        # We take the absolute value to get the importance.
        result.importances_mean = np.abs(result.importances_mean)
        
        df_featimp_org = pd.DataFrame(
            {
                "Feature": X_test_in.columns,
                "Importance": result.importances_mean,
                "Std": result.importances_std,
            }
        )

    else:
        raise ValueError(
            f"vi_method must be either 'impurity' or 'permutation'! Input: {vi_method}"
        )

    df_featimp_agg = pd.DataFrame(
        {
            "Feature": [],
            "Importance": [],
            "Std": [],
        }
    )

    # Ugly quick fix but works...
    if not with_aggregation:
        reduced_var_ohe_dict = dict_ohe_in

    if with_aggregation:
        # Reduce dictionary to hold only keys for which there is a value that matches the inputed X_train_in
        reduced_var_ohe_dict = {
            key: value
            for key, value in dict_ohe_in.items()
            if any(col in X_train_in.columns for col in value)
        }

        # Remove variables to ignore from dictionary
        for var in ignore_these:
            if var in reduced_var_ohe_dict.keys():
                del reduced_var_ohe_dict[var]

        # Loop through all keys in the dictionary
        for key in reduced_var_ohe_dict.keys():
            importances_per_key = []
            stdevs_per_key = []

            # Loop through all variables in the key
            for var in reduced_var_ohe_dict[key]:
                # Loop through all variables in the original featimp df
                for i in range(len(df_featimp_org)):
                    # Check if the row in the featimp corresponds to value of the key
                    # If so, gather all variables for that key and aggregate them.
                    if df_featimp_org["Feature"].iloc[i] == var:
                        importances_per_key.append(df_featimp_org["Importance"].iloc[i])
                        stdevs_per_key.append(df_featimp_org["Std"].iloc[i])

                n_vars_per_key = len(reduced_var_ohe_dict[key])
                if n_vars_per_key > 1:
                    importance = np.sum(importances_per_key)
                    stdev = sum([x**2 for x in stdevs_per_key])
                else:
                    importance = importances_per_key[0]
                    stdev = stdevs_per_key[0]

            new_row = pd.DataFrame(
                {
                    "Feature": [key],
                    "Importance": [importance],
                    "Std": [stdev],
                    "Importance_per_key": [importances_per_key],
                    "Vars_per_key": [n_vars_per_key],
                    "Vars_in_key": [list(reduced_var_ohe_dict[key])],
                }
            )

            df_featimp_agg = pd.concat([df_featimp_agg, new_row], axis=0)
            df_featimp_final = df_featimp_agg.sort_values(
                by="Importance", ascending=False
            ).reset_index(drop=True)

    else:
        df_featimp_final = (
            df_featimp_org.copy()
            .sort_values(by="Importance", ascending=False)
            .reset_index(drop=True)
        )

    # ! PLOT -------------------------------------------------------
    # Show top n predictors table
    top_n = df_featimp_final.head(n_predictors)
    if with_aggregation:
        my_title = (
            f"Top {n_predictors} predictors with aggregation (via {vi_method})"
        )
    else:
        my_title = (
            f"Top {n_predictors} predictors without aggregation (via {vi_method})"
        )

    # Set the figure size
    plt.figure(figsize=(8, 5))
    # Plot the variable importance
    sns_plot = sns.barplot(x="Importance", y="Feature", data=top_n, color="r")
    # Add standard deviation as error bars
    sns_plot.errorbar(
        x=top_n["Importance"],
        y=top_n["Feature"],
        xerr=top_n["Std"],
        fmt=" ",
        color="black",
    )

    sns_plot.set_title(my_title)
    plt.tight_layout()

    if save_directory is not None:
        if with_aggregation:
            sns_plot.figure.savefig(f"{save_directory}/fig-vip-{vi_method}-aggregated.png")
            df_featimp_final.to_csv(f"{save_directory}/tab-vip-{vi_method}-aggregated.csv")
        else:
            sns_plot.figure.savefig(f"{save_directory}/fig-vip-{vi_method}-not_aggregated.png")
            df_featimp_final.to_csv(f"{save_directory}/tab-vip-{vi_method}-not_aggregated.csv")

    if verbose:
        plt.show()
    else:
        plt.close()

    # ! Report -------------------------------------------------------
    if verbose:
        # Bugfix if no aggregation was done
        if reduced_var_ohe_dict is None:
            reduced_var_ohe_dict = []

        print(
            f"--- assessing_top_predictors()...",
            f"\n - Number of columns in X_train_in is equal to number of rows in df_featimp_org: {n_predictors == df_featimp_org.shape[0]}",
            f"\n - Number of vars in training set is equal to number of rows in df_featimp_agg: {len(reduced_var_ohe_dict) == df_featimp_agg.shape[0]}",
            f"\n - Size of original df_featimp_org: {df_featimp_org.shape}",
            f"\n - Size of aggregated df_featimp_agg: {df_featimp_agg.shape}",
        )

    return df_featimp_final


# -----------------------------------------------------------------------------------------------
def model_evaluation_regression(
    rf_model,
    X_train,
    y_train,
    X_test,
    y_test,
    save_directory=None,
    verbose=False,
):
    # Predict on the train and test data
    y_train_pred = rf_model.predict(X_train)
    y_test_pred = rf_model.predict(X_test)

    # Calculate the evaluation metrics for train data
    r2_train = r2_score(y_train, y_train_pred)
    r_train = pearsonr(y_train, y_train_pred)[0]
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    mse_train = mean_squared_error(y_train, y_train_pred)
    mae_train = mean_absolute_error(
        y_train,
        y_train_pred,
    )

    # Calculate the evaluation metrics for test data
    r2_test = r2_score(y_test, y_test_pred)
    r_test = pearsonr(y_test, y_test_pred)[0]
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mse_test = mean_squared_error(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)

    # Print the evaluation metrics for train data
    if verbose:
        print("\n--- Results Train | Test ---")
        print(" - r:\t\t ", round(r_train, 2), " | ", round(r_test, 2))
        print(" - R2:\t\t ", round(r2_train, 2), " | ", round(r2_test, 2))
        print(" - MSE:\t\t ", round(mse_train, 2), " | ", round(mse_test, 2))
        print(" - RMSE:\t ", round(rmse_train, 2), " | ", round(rmse_test, 2))
        print(" - MAE:\t\t ", round(mae_train, 2), " | ", round(mae_test, 2))

    # Set the figure size
    plt.figure(figsize=(12, 6))

    # Plot the predicted versus observed values for train data
    plt.subplot(1, 2, 1)
    sns.regplot(
        x=y_train,
        y=y_train_pred,
        scatter_kws=dict(color="gray", s=10, alpha=0.8),
        line_kws=dict(color="blue"),
    )
    plt.plot(ls="--", c="red")
    plt.xlabel("Observations")
    plt.ylabel("Predictions")
    plt.title(f"Train Data")  # (target: {y_test.name})")

    # Set y and x axis limits based on the maximum value in y_train_pred or y_train
    max_value_train = max(max(y_train_pred), max(y_train))
    plt.ylim(0, max_value_train * 1.15)
    plt.xlim(0, max_value_train * 1.15)

    # Add a red dotted 1:1 line
    plt.plot([0, max_value_train], [0, max_value_train], ls="--", c="r")

    # Set equal scaling (i.e., 1:1 aspect ratio)
    plt.gca().set_aspect("equal", adjustable="box")

    # Add metrics reporting to the top right corner
    plt.text(
        0.2,
        0.95,
        f"r: {round(r_train, 2)}\nR2: {round(r2_train, 2)}\nRMSE: {round(rmse_train, 2)}\nMAE: {round(mae_train, 2)}",
        horizontalalignment="right",
        verticalalignment="top",
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round"),
    )

    # Plot the predicted versus observed values for test data
    plt.subplot(1, 2, 2)
    sns.regplot(
        x=y_test,
        y=y_test_pred,
        scatter_kws=dict(color="gray", s=10, alpha=0.8),
        line_kws=dict(color="blue"),
    )
    plt.plot(ls="--", c="red")
    plt.xlabel("Observations")
    plt.ylabel("Predictions")
    plt.title(f"Test Data")  # (target: {y_train.name})")

    # Set y and x axis limits based on the maximum value in y_test_pred or y_test
    max_value_test = max(max(y_test_pred), max(y_test))
    plt.ylim(0, max_value_test * 1.15)
    plt.xlim(0, max_value_test * 1.15)

    # Add a red dotted 1:1 line
    plt.plot([0, max_value_test], [0, max_value_test], ls="--", c="r")

    # Set equal scaling (i.e., 1:1 aspect ratio)
    plt.gca().set_aspect("equal", adjustable="box")

    # Add metrics reporting to the top right corner
    plt.text(
        0.2,
        0.95,
        f"r: {round(r_test, 2)}\nR2: {round(r2_test, 2)}\nRMSE: {round(rmse_test, 2)}\nMAE: {round(mae_test, 2)}",
        horizontalalignment="right",
        verticalalignment="top",
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round"),
    )

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Save the figure
    if save_directory is not None:
        os.makedirs(save_directory, exist_ok=True)
        plt.savefig(f"{save_directory}/fig_model_evaluation.png")

    # Show the figures
    if verbose:
        plt.show()
    plt.close()


# -----------------------------------------------------------------------------------------------
def model_evaluation_classification(
    rf_model,
    X_train,
    y_train,
    X_test,
    y_test,
    prob_threshold=0.5,
    save_directory=None,
    metric="f1-score",
    verbose=True,
):
    # Predict probabilities for train and test data
    y_train_proba = rf_model.predict_proba(X_train)
    y_test_proba = rf_model.predict_proba(X_test)

    # Turn probabilities into binary predictions
    y_train_pred = (y_train_proba[:, 1] >= prob_threshold).astype("int")
    y_test_pred = (y_test_proba[:, 1] >= prob_threshold).astype("int")
    
    # Turn arrays into pandas series
    y_train_pred = pd.Series(y_train_pred)
    y_test_pred = pd.Series(y_test_pred)
    
    # Save everything to files
    if save_directory is not None:
        # Create directories
        os.makedirs(f"{save_directory}/final_model", exist_ok=True)
        
        # Predictions - binary
        y_train_pred.to_csv(f"{save_directory}/final_model/y_train_pred.csv")
        y_test_pred.to_csv(f"{save_directory}/final_model/y_test_pred.csv")
        
        # Predictions - probabilities
        pd.DataFrame(y_train_proba, columns=["0", "1"]).to_csv(f"{save_directory}/final_model/y_train_proba.csv")
        pd.DataFrame(y_test_proba, columns=["0", "1"]).to_csv(f"{save_directory}/final_model/y_test_proba.csv")
        
        # Actuals
        y_train.to_csv(f"{save_directory}/final_model/y_train.csv")
        y_test.to_csv(f"{save_directory}/final_model/y_test.csv")
        
        # Features
        X_train.to_csv(f"{save_directory}/final_model/X_train.csv")
        X_test.to_csv(f"{save_directory}/final_model/X_test.csv")
        
        # Model
        with open(f"{save_directory}/final_model/rf_model.pkl", "wb") as file:
            pickle.dump(rf_model, file)

    # Get unique class labels from the target variable
    unique_labels = np.unique(np.concatenate((y_train, y_test)))

    # Calculate the confusion matrices
    confusion_train = confusion_matrix(y_train, y_train_pred, labels=unique_labels)
    confusion_test = confusion_matrix(y_test, y_test_pred, labels=unique_labels)
    
    # Get confusion reports (once as txt for reporting and once as dict.)
    report_train_txt = classification_report(y_train, y_train_pred, digits=3)
    report_test_txt = classification_report(y_test, y_test_pred, digits=3)

    report_train_dict = classification_report(
        y_train, y_train_pred, digits=3, output_dict=True
    )
    report_test_dict = classification_report(
        y_test, y_test_pred, digits=3, output_dict=True
    )

    # Get max f1 for scaling bar plots
    class_scores_train = []
    class_scores_test = []

    for label in unique_labels:
        class_scores_train.append(report_train_dict[f"{label}"][metric])
        class_scores_test.append(report_test_dict[f"{label}"][metric])

    metric_max = max([0] + class_scores_train + class_scores_test)

    # Print the confusion matrix for train data
    if verbose:
        print("--- model_evaluation_classification():")
        print("----- Dataset -----")
        print(f"Number of training datapoints: {len(y_train)}")
        print(f"Number of test datapoints: {len(y_test)}")

    # * Create Baseline Model
    # Print baseline model that always predicts majority class
    # if user_input["model_task"] == "binary":
    n_sites_without_mort = len(y_test[y_test == 0])
    n_sites_with_mort = len(y_test[y_test == 1])
    y_baseline = [0] * len(y_test)
    y_baseline = pd.Series(y_baseline)
    
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
    
    if verbose:
        display(" --- Baseline ---")
        display(scores_baseline)
        display(" --- Training  ---")
        display(scores_train)
        display(" --- Testing ---")
        display(scores_test)

    # Write the same information to a file
    if save_directory is not None:
        scores_baseline.to_csv(f"{save_directory}/final_model_scores_baseline.csv", index=False)
        scores_train.to_csv(f"{save_directory}/final_model_scores_train.csv", index=False)
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
    plot_confusion_matrix(
        confusion_train,
        class_scores_train,
        unique_labels,
        title="Train Data",
        bpmax=metric_max,
        save_directory=save_directory,
        test_or_train="train",
        show=verbose,
    )

    # Plot the confusion matrix for test data
    plt.figure(figsize=(8, 4))
    plot_confusion_matrix(
        confusion_test,
        class_scores_test,
        unique_labels,
        title="Test Data",
        bpmax=metric_max,
        save_directory=save_directory,
        test_or_train="test",
        show=verbose,
    )
    
    # Plot ROC AUC curve
    plot_roc_auc_both(
        (y_train, y_train_proba[:, 1]),
        (y_test, y_test_proba[:, 1]),
        save_directory=save_directory,
        show=verbose
    )


def plot_confusion_matrix(
    conf_matrix,
    class_scores,
    class_labels,
    title,
    bpmax=None,
    save_directory=None,
    test_or_train="none",
    metric="f1-score",
    show=True,
):
    """
    Plots the confusion matrix and class accuracy for a given set of class labels.

    Parameters:
    conf_matrix (numpy.ndarray): The confusion matrix.
    class_labels (list): The list of class labels.
    title (str): The title of the plot.
    vmax (int, optional): The maximum value for the colorbar. Defaults to None.
    bpmax (int, optional): The maximum value for the y-axis of the bar plot. Defaults to None.

    Returns:
    None
    """
    # Set up the plot
    plt.subplot(1, 2, 1)
    # Plot heatmap
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    plt.title(title)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")

    # Plot class accuracy
    plt.subplot(1, 2, 2)
    if bpmax is not None:
        plt.ylim(0, 1.1)  # bpmax*1.1) -> use this for dynamic scaling
    plt.axhline(y=0.5, color="grey", linestyle="dotted")
    plt.bar(class_labels, class_scores, color="steelblue")
    plt.title(f"Class {metric}")
    plt.xlabel("Class")
    plt.ylabel(metric)
    plt.xticks(class_labels)  # Show x-axis ticks only for class labels

    # Show class accuracy on the plot
    for i, acc in enumerate(class_scores):
        plt.text(i, acc, f"{acc:.3f}", ha="center", va="bottom")

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Save the figure
    if save_directory is not None:
        plt.savefig(f"{save_directory}/fig_model_evaluation_{test_or_train}.png")

    if show:
        # Show the figures
        plt.show()
    else:
        plt.close()

# def extract_highest_accuracy(y_true, y_pred):
#     # Calculate accuracy for each class
#     class_accuracy = np.diag(confusion_matrix(y_true, y_pred)) / np.sum(confusion_matrix(y_true, y_pred), axis=1)

#     # Find the highest accuracy
#     highest_accuracy = np.max(class_accuracy)

#     return highest_accuracy

def plot_roc_auc(y_true, y_pred_prob, title="ROC AUC Curve", test_or_train=None,save_directory=None, show=True):
    """
    Plot the ROC AUC curve.

    Parameters:
    y_true (array-like): True binary labels.
    y_pred_prob (array-like): Target scores, can either be probability estimates of the positive class,
                              confidence values, or binary decisions.
    title (str): Title of the plot.
    """
    # Checks
    if test_or_train is None:
        raise ValueError("test_or_train must be specified!")
    
    # Calculate the ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)

    # Calculate the AUC score
    auc_score = roc_auc_score(y_true, y_pred_prob)

    # Plot the ROC curve
    plt.figure(figsize=(4, 4))
    plt.plot(fpr, tpr, label=f"{test_or_train} AUC = {auc_score:.2f}", color="red")
    plt.plot([0, 1], [0, 1], linestyle="--", color="black")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid()
    
    if save_directory is not None:
        plt.savefig(f"{save_directory}/fig-fig_roc_auc_curve_{test_or_train}.png")
    
    if show:
        plt.show()
    else:
        plt.close()
        
def plot_roc_auc_both(train_data, test_data, title='ROC AUC Curves', save_directory=None, show=True):
    """
    Plot the ROC AUC curves for both training and testing data side by side.
    
    Parameters:
    train_data (tuple): A tuple (y_true_train, y_pred_prob_train) for training data.
    test_data (tuple): A tuple (y_true_test, y_pred_prob_test) for testing data.
    title (str): Title of the plot.
    save_directory (str): Directory to save the plot.
    show (bool): Whether to display the plot.
    """
    # User input
    n_bootstraps = 1000  # Number of bootstrap samples
    random_seed = 42     # Random seed for reproducibility
    
    def bootstrap_roc(y_true, y_pred_prob, n_bootstraps, random_seed):
        rng = np.random.RandomState(random_seed)
        bootstrapped_fpr = []
        bootstrapped_tpr = []
        aucs = []

        for i in range(n_bootstraps):
            # Generate bootstrap sample
            indices = rng.randint(0, len(y_true), len(y_true))
            if len(np.unique(y_true[indices])) < 2:
                # Skip iteration if the sample does not contain both classes
                continue

            y_true_boot = y_true[indices]
            y_pred_prob_boot = y_pred_prob[indices]
            
            # Compute ROC curve and AUC for the bootstrap sample
            fpr, tpr, _ = roc_curve(y_true_boot, y_pred_prob_boot)
            auc = roc_auc_score(y_true_boot, y_pred_prob_boot)
            
            bootstrapped_fpr.append(fpr)
            bootstrapped_tpr.append(tpr)
            aucs.append(auc)

        return bootstrapped_fpr, bootstrapped_tpr, aucs

    def plot_roc_with_uncertainty(ax, fpr, tpr, aucs, label, color):
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        for fpr_i, tpr_i in zip(fpr, tpr):
            tpr_interp = np.interp(mean_fpr, fpr_i, tpr_i)
            tprs.append(tpr_interp)
        tprs = np.array(tprs)

        mean_tpr = tprs.mean(axis=0)
        std_tpr = tprs.std(axis=0)

        ax.plot(mean_fpr, mean_tpr, label=f'{label} (AUC = {np.mean(aucs):.2f})', color=color)
        ax.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color=color, alpha=0.3)

    # Given train and test data
    y_true_train, y_pred_prob_train = train_data
    y_true_test, y_pred_prob_test = test_data
    
    # Reset index
    y_true_train = y_true_train.reset_index(drop=True)
    y_true_test = y_true_test.reset_index(drop=True)

    # Calculate bootstrapped ROC for training data
    fpr_train, tpr_train, aucs_train = bootstrap_roc(y_true_train, y_pred_prob_train, n_bootstraps, random_seed)

    # Calculate bootstrapped ROC for testing data
    fpr_test, tpr_test, aucs_test = bootstrap_roc(y_true_test, y_pred_prob_test, n_bootstraps, random_seed)

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the ROC curve for training data with uncertainty bands
    plot_roc_with_uncertainty(axs[0], fpr_train, tpr_train, aucs_train, 'Training Data', 'blue')
    axs[0].plot([0, 1], [0, 1], linestyle='--', color='black')
    axs[0].set_xlim([0.0, 1.0])
    axs[0].set_ylim([0.0, 1.05])
    axs[0].set_xlabel('False Positive Rate')
    axs[0].set_ylabel('True Positive Rate')
    axs[0].set_title('Training Data')
    axs[0].legend(loc='lower right')

    # Plot the ROC curve for testing data with uncertainty bands
    plot_roc_with_uncertainty(axs[1], fpr_test, tpr_test, aucs_test, 'Testing Data', 'red')
    axs[1].plot([0, 1], [0, 1], linestyle='--', color='black')
    axs[1].set_xlim([0.0, 1.0])
    axs[1].set_ylim([0.0, 1.05])
    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive Rate')
    axs[1].set_title('Testing Data')
    axs[1].legend(loc='lower right')

    # Set the overall title
    fig.suptitle('ROC Curve with Uncertainty Bands', fontsize=12, weight='bold')

    # Adjust layout
    plt.tight_layout()

    # display(" --- AUC Scores ---")
    # print(f"Mean AUC Train: {np.mean(aucs_train):.2f}")
    # print(f"Mean AUC Test: {np.mean(aucs_test):.2f}")
    
    # Save the plot if save_directory is provided
    if save_directory is not None:
        os.makedirs(save_directory, exist_ok=True)
        plt.savefig(f"{save_directory}/fig-roc_auc_curves.png")
        pd.DataFrame({
            "test_mean": np.mean(aucs_test),
            "test_sd": np.std(aucs_test),
            "train_mean": np.mean(aucs_train),
            "train_sd": np.std(aucs_train)
            
        }, index=[0]).to_csv(f"{save_directory}/final_model_roc_auc_prob-based.csv", index=False)

    # Show or close the plot based on the `show` parameter
    if show:
        plt.show()
    else:
        plt.close()

def plot_pr_auc_both(train_data, test_data, title='PR AUC Curves', save_directory=None, show=True):
    """
    Plot the PR AUC curves for both training and testing data side by side.
    
    Parameters:
    train_data (tuple): A tuple (y_true_train, y_pred_prob_train) for training data.
    test_data (tuple): A tuple (y_true_test, y_pred_prob_test) for testing data.
    title (str): Title of the plot.
    save_directory (str): Directory to save the plot.
    show (bool): Whether to display the plot.
    """
    # User input
    n_bootstraps = 1000  # Number of bootstrap samples
    random_seed = 42     # Random seed for reproducibility
    from sklearn.metrics import precision_recall_curve, average_precision_score

    
    def bootstrap_pr(y_true, y_pred_prob, n_bootstraps, random_seed):
        rng = np.random.RandomState(random_seed)
        bootstrapped_prec = []
        bootstrapped_recall = []
        pr_aucs = []

        for i in range(n_bootstraps):
            # Generate bootstrap sample
            indices = rng.randint(0, len(y_true), len(y_true))
            if len(np.unique(y_true[indices])) < 2:
                continue

            y_true_boot = y_true[indices]
            y_pred_prob_boot = y_pred_prob[indices]
            
            # Compute PR curve and AUC for the bootstrap sample
            precision, recall, _ = precision_recall_curve(y_true_boot, y_pred_prob_boot)
            pr_auc = average_precision_score(y_true_boot, y_pred_prob_boot)
            
            bootstrapped_prec.append(precision)
            bootstrapped_recall.append(recall)
            pr_aucs.append(pr_auc)

        return bootstrapped_prec, bootstrapped_recall, pr_aucs

    def plot_pr_with_uncertainty(ax, precision, recall, pr_aucs, label, color):
        mean_recall = np.linspace(0, 1, 100)
        precisions = []
        for precision_i, recall_i in zip(precision, recall):
            precision_interp = np.interp(mean_recall, recall_i[::-1], precision_i[::-1])
            precisions.append(precision_interp)
        precisions = np.array(precisions)

        mean_precision = precisions.mean(axis=0)
        std_precision = precisions.std(axis=0)

        ax.plot(mean_recall, mean_precision, label=f'{label} (PR-AUC = {np.mean(pr_aucs):.2f})', color=color)
        ax.fill_between(mean_recall, mean_precision - std_precision, mean_precision + std_precision, color=color, alpha=0.3)

    # Given train and test data
    y_true_train, y_pred_prob_train = train_data
    y_true_test, y_pred_prob_test = test_data
    
    # Reset index
    y_true_train = y_true_train.reset_index(drop=True)
    y_true_test = y_true_test.reset_index(drop=True)

    # Calculate bootstrapped PR-AUC for training data
    precision_train, recall_train, pr_aucs_train = bootstrap_pr(y_true_train, y_pred_prob_train, n_bootstraps, random_seed)

    # Calculate bootstrapped PR-AUC for testing data
    precision_test, recall_test, pr_aucs_test = bootstrap_pr(y_true_test, y_pred_prob_test, n_bootstraps, random_seed)

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the PR curve for training data with uncertainty bands
    plot_pr_with_uncertainty(axs[0], precision_train, recall_train, pr_aucs_train, 'Training Data', 'blue')
    axs[0].set_xlim([0.0, 1.0])
    axs[0].set_ylim([0.0, 1.05])
    axs[0].set_xlabel('Recall')
    axs[0].set_ylabel('Precision')
    axs[0].set_title('Training Data')
    axs[0].legend(loc='lower left')

    # Plot the PR curve for testing data with uncertainty bands
    plot_pr_with_uncertainty(axs[1], precision_test, recall_test, pr_aucs_test, 'Testing Data', 'red')
    axs[1].set_xlim([0.0, 1.0])
    axs[1].set_ylim([0.0, 1.05])
    axs[1].set_xlabel('Recall')
    axs[1].set_ylabel('Precision')
    axs[1].set_title('Testing Data')
    axs[1].legend(loc='lower left')

    # Set the overall title
    fig.suptitle('Precision-Recall Curve with Uncertainty Bands', fontsize=12, weight='bold')

    # Adjust layout
    plt.tight_layout()

    # Save the plot if save_directory is provided
    if save_directory is not None:
        os.makedirs(save_directory, exist_ok=True)
        plt.savefig(f"{save_directory}/fig-pr_auc_curves.png")
        pd.DataFrame({
            "test_mean": np.mean(pr_aucs_test),
            "test_sd": np.std(pr_aucs_test),
            "train_mean": np.mean(pr_aucs_train),
            "train_sd": np.std(pr_aucs_train)
            
        }, index=[0]).to_csv(f"{save_directory}/final_model_pr_auc_prob-based.csv", index=False)

    # Show or close the plot based on the `show` parameter
    if show:
        plt.show()
    else:
        plt.close()


# -----------------------------------------------------------------------------------------------
def do_ohe(Xy, variables_not_to_ohe=[], verbose=True):
    ohe_these = []

    for var in Xy:
        if Xy[var].dtype == "O":
            ohe_these = ohe_these + [var]

    # Remove variables that should not be one-hot encoded
    ohe_these = [var for var in ohe_these if var not in variables_not_to_ohe]

    # One-hot encoding
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    df_ohe = pd.DataFrame(
        ohe.fit_transform(Xy[ohe_these]),
        columns=ohe.get_feature_names_out(ohe_these),
    )

    # Attach the non-encoded variables
    df_out = pd.concat(
        [
            Xy.drop(columns=ohe_these).reset_index(drop=True),
            df_ohe.reset_index(drop=True),
        ],
        axis=1,
    )

    # Verbose output
    if verbose:
        print(
            f"do_ohe():",
            f"\n - Shape before OHE:\t\t {Xy.shape}",
            f"\n - Shape after OHE:\t\t {df_out.shape}",
            f"\n - Change in Nr. of columns:\t {df_out.shape[1] - Xy.shape[1]} (dropped: {len(ohe_these)}, added: {df_ohe.shape[1]})",
            f"\n - Variables that were ohe'd:\t {' | '.join(sorted(ohe_these))}",
            # f"\n - New variables:\t\t {' | '.join(sorted(df_ohe.columns.to_list()))}",
        )

    return df_out


# -----------------------------------------------------------------------------------------------
def impute_numerical_na(
    Xy_train_in,
    Xy_test_in,
    target_in,
    method="knn",
    n_neighbours=10,
    vars_not_to_impute=[],
    verbose=True,
):
    # Define X for imputation
    X_train_in = Xy_train_in.copy().drop(columns=target_in).reset_index(drop=True)
    X_test_in = Xy_test_in.copy().drop(columns=target_in).reset_index(drop=True)

    # Get numerical variables with NAs (need to check each dataset separately)
    all_numerics = X_train_in.columns[(X_train_in.dtypes != "O")].tolist()

    num_na_train = X_train_in.columns[
        (X_train_in.dtypes != "O") & (X_train_in.isna().any())
    ].tolist()
    num_na_test = X_test_in.columns[
        (X_test_in.dtypes != "O") & (X_test_in.isna().any())
    ].tolist()

    # Remove target from imputation if still in dataset)
    if target_in in num_na_train:
        num_na_train.remove(target_in, errors="ignore")
    if target_in in num_na_test:
        num_na_test.remove(target_in, errors="ignore")

    num_na_train = list(set(num_na_train + num_na_test))

    # Check if target had NA values and inform
    detected_vars = []
    for var in vars_not_to_impute:
        if var in num_na_train:
            detected_vars.append(var)
            num_na_train.remove(var)

    # Do imputation
    if len(num_na_train) > 0:
        if method == "knn":
            imputer = KNNImputer(n_neighbors=n_neighbours)
            X_train_in[num_na_train] = imputer.fit_transform(X_train_in[num_na_train])
            X_test_in[num_na_train] = imputer.transform(X_test_in[num_na_train])

        elif method == "mean":
            for var in num_na_train:
                test_mean = X_train_in[var].mean()
                X_train_in[var] = X_train_in[var].fillna(test_mean)
                X_test_in[var] = X_test_in[var].fillna(test_mean)

        elif method == "median":
            for var in num_na_train:
                test_median = X_train_in[var].median()
                X_train_in[var] = X_train_in[var].fillna(test_median)
                X_test_in[var] = X_test_in[var].fillna(test_median)

        elif method == "minus_9999":
            X_train_in[num_na_train] = X_train_in[num_na_train].fillna(-9999)
            X_test_in[num_na_train] = X_test_in[num_na_train].fillna(-9999)

    # Attach target to X_train_in
    Xy_train_out = X_train_in.copy()
    Xy_train_out[target_in] = Xy_train_in[target_in]

    Xy_test_out = X_test_in.copy()
    Xy_test_out[target_in] = Xy_test_in[target_in]

    if verbose:
        print(
            f"impute_numerical_na():",
            f"\n - Shape of Xy_train before imputation: {Xy_train_in.shape}",
            f"\n - Shape of Xy_train before imputation: {Xy_train_out.shape}",
            f"\n",
            f"\n - Shape of Xy_test before imputation: {Xy_test_in.shape}",
            f"\n - Shape of Xy_test before imputation: {Xy_test_out.shape}",
            f"\n",
            f"\n - Out of {len(all_numerics)}, {len(num_na_train)} had NA values and were imputed with method {method} (for KNN, n = {n_neighbours}).",
            # f"\n - Imputed variables: {' | '.join(sorted(num_na_train))}",
        )

    if len(detected_vars) > 0:
        print(
            f"\n - ❌❌❌ Variables {detected_vars} had NA values but are not meant to be imputed! They were not imputed but their values should be fixed! ❌❌❌"
        )

    return Xy_train_out, Xy_test_out, num_na_train


# -----------------------------------------------------------------------------------------------
def overlay_barplot(ax, df1, df1_name, df2, df2_name, variable, subtitle=None):
    # Number of levels in variable
    nvar = len(set([var for var in df1[variable]] + [var for var in df2[variable]]))

    # Calculate the relative frequencies for each level in df1
    df1_counts = df1[variable].value_counts(normalize=True).sort_values(ascending=False)
    df1_counts = df1_counts * 100

    # Calculate the relative frequencies for each level in df2
    df2_counts = df2[variable].value_counts(normalize=True).sort_values(ascending=False)
    df2_counts = df2_counts * 100

    # Combine the counts into a single dataframe
    df_counts = pd.concat([df1_counts, df2_counts], axis=1)
    df_counts.columns = [df1_name, df2_name]

    # Plotting the bar plot
    ax.barh(
        df_counts.index, df_counts[df1_name], align="center", label=df1_name, alpha=0.5
    )
    ax.barh(
        df_counts.index, df_counts[df2_name], align="center", label=df2_name, alpha=0.5
    )

    # Adding titles and labels
    ax.set_title("Overlaying Bar Plot of variable of total " + str(nvar) + " strata")
    ax.set_xlabel("Relative Frequency [%]")
    ax.set_ylabel("Level of variable")
    ax.legend()

    if subtitle is not None:
        ax.text(
            0.5,
            0.95,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="center",
        )


# -----------------------------------------------------------------------------------------------
def get_weights_from_y(y_in, method="none"):
    if method == "none":
        return np.arange(1, len(y_in) + 1)
    if method == "squared":
        return y_in**2
    if method == "cubic":
        return y_in**3
    if method == "quadratic":
        return y_in**4
    if method == "inverse":
        return 1 / y_in
    if method == "inverse_squared":
        return 1 / y_in**2


# -----------------------------------------------------------------------------------------------
def aggregate_strata_with_too_little_obs(
    Xy_in,
    strata_vars,
    do_fold_too,
    split_test,
    cv_fold,
    seed_nr,
    print_plots=True,
):
    # Start printing call
    print(
        f"wrangle_stratification_of_dataset():",
        end="",
    )

    # ----------------------------------------
    # Get df_all_strata for splitting
    df_all_strata = merge_small_strata_to_one(Xy_in, strata_vars, "split", cv_fold)

    # Pick arbitrary target that is not in strata_vars
    random_column = random.choice(
        df_all_strata.drop(columns=strata_vars + ["test_train_strata"]).columns
    )
    # Split dataset into train and test set
    xtr, xte, ytr, yte = train_test_split(
        df_all_strata.drop(random_column, axis=1),
        df_all_strata[random_column],
        test_size=split_test,
        random_state=seed_nr,
        stratify=df_all_strata["test_train_strata"],
    )

    # strata_groups = df_all_strata["test_train_strata"].unique()

    if not do_fold_too:
        fig, ax = plt.subplots(1, figsize=(12.5, 25))
        display()
        overlay_barplot(ax, xtr, "Train", xte, "Test", "test_train_strata", "All Data")

        return df_all_strata.reset_index(drop=True)
    else:
        pass

    # Do folding too
    # ----------------------------------------
    print(
        f"\n - Check strata aggregation needed for splitted X_train to successfully create {cv_fold} CV-folds...",
        end="",
    )

    # Take training split from above
    new_df = pd.concat([xtr.reset_index(drop=True), ytr.reset_index(drop=True)], axis=1)
    new_df = attach_test_train_strata(
        new_df, strata_vars
    )  # Attach test_train_strata variable
    new_df = merge_small_strata_to_one(
        new_df, strata_vars, "fold", cv_fold
    )  # Merge strata for folding

    # Define folding
    skf = StratifiedKFold(n_splits=cv_fold, shuffle=True)

    # Run folding to see if it works with the merged strata
    for fold, (train_index, test_index) in enumerate(
        skf.split(new_df, new_df["test_train_strata"])
    ):
        xtr, xte = (
            new_df.iloc[train_index],
            new_df.iloc[test_index],
        )

    # Get strata groups that works for split and fold
    final_strata_groups = new_df["test_train_strata"].unique()

    # FINAL MERGING OF STRATA AND DISPLAY
    # ----------------------------------------
    print(
        f"\n - ✅ Got the final groups for strata aggregation.",
        f"\n  - Running routine again to create final distribution plots of train/test splits and train/validation folds.",
        end="",
    )

    df_out = Xy_in.copy()  # Get copy of input data
    df_out = attach_test_train_strata(
        df_out, strata_vars
    )  # Attach test_train_strata variable
    org_number_of_strata = df_out[
        "test_train_strata"
    ].nunique()  # Check how many stratification there were originally

    # If strata in df_out are not in final_strata_groups, then replace with "others"
    for s in df_out["test_train_strata"].unique():
        if s not in final_strata_groups:
            df_out["test_train_strata"] = df_out["test_train_strata"].replace(
                s, "others"
            )

    # Print information
    print(
        f"\n  - For successful splitting and {cv_fold} CV-folds, {org_number_of_strata} strata were merged into {len(final_strata_groups)} strata.",
        f"\n  - Stratas with too little observations were merged into 'others', which makes up {round(df_out[df_out['test_train_strata'] == 'others'].shape[0]/df_out.shape[0]*100, 2)}% of the data.",
    )

    # Make figures (for this, we first need to do the splitting and folding again!)
    # ----------------------------------------
    # SPLIT
    # Pick arbitrary target that is not in test_train_strata
    random_column = random.choice(
        df_out.drop(columns=strata_vars + ["test_train_strata"]).columns
    )
    # Split dataset into train and test set
    xtr, xte, ytr, yte = train_test_split(
        df_out.drop(random_column, axis=1),
        df_out[random_column],
        test_size=split_test,
        random_state=seed_nr,
        stratify=df_all_strata["test_train_strata"],
    )
    if print_plots:
        # Make plot
        fig, ax = plt.subplots(1, figsize=(25, 25))
        display()
        overlay_barplot(ax, xtr, "Train", xte, "Test", "test_train_strata", "All Data")

        # FOLD
        # Take training split from above
        new_df = pd.concat(
            [xtr.reset_index(drop=True), ytr.reset_index(drop=True)], axis=1
        )
        # No need to re-attach test_train_strata variable as above, because it is already attached
        # from the definition of df_out.

        # Define folding (take same as above, so no need to redefine)
        # Initiate plot
        fig, axs = plt.subplots(1, cv_fold, figsize=(25, 25))
        # Do folds
        for fold, (train_index, test_index) in enumerate(
            skf.split(new_df, new_df["test_train_strata"])
        ):
            xtr, xte = (
                new_df.iloc[train_index],
                new_df.iloc[test_index],
            )

            # Make plot
            axs[fold] = overlay_barplot(
                axs[fold],
                xtr,
                "Train",
                xte,
                "Test",
                "test_train_strata",
                str("Fold " + str(fold + 1)),
            )

    # Return final dataframe with correctly merged strata
    return df_out.reset_index(drop=True)


# -----------------------------------------------------------------------------------------------


# ----------------------------------------
# DEFINE FUNCTIONS FOR STRATIFICATION DISTRIBUTION CHECKS
# ----------------------------------------
# Attach test_train_strata dummy
def attach_test_train_strata(df_in, var_list):
    df_in["test_train_strata"] = ""

    # Attach test_train_strata variables
    for var in var_list:
        df_in["test_train_strata"] = df_in["test_train_strata"] + (
            "_" + df_in[var].astype(str)
        )
    return df_in


# Define merger function
def merge_small_strata_to_one(Xy_in, strata_vars, split_or_fold, cv_fold):
    # Start printing
    print(
        f"\n - Merging strata for successful {split_or_fold}...",
        end="",
    )

    # Get min. number of observations per group
    if split_or_fold == "split":
        min_obs_per_group = 2
        print(
            f"\n  - For splitting, the min. number of observations per group is {min_obs_per_group}.",
            end="",
        )
    if split_or_fold == "fold":
        min_obs_per_group = cv_fold
        print(
            f"\n  - For cv-folds, the min. number of observations per group is {cv_fold}.",
            end="",
        )

    # Aggregate strata according to min_obs_per_group
    # ----------------------------------------
    df_all_strata = Xy_in.copy()  # Get temporary df
    df_all_strata = attach_test_train_strata(
        df_all_strata, strata_vars
    )  # Attach test_train_strata variable

    stratification_lvls = df_all_strata[
        "test_train_strata"
    ].nunique()  # Check how many stratification lvls there are

    print(
        f"\n  - Stratification levels: {stratification_lvls}, based on variables: {strata_vars}.",
        end="",
    )

    # Get df with observations per strata
    df_observations_per_strata = df_all_strata.groupby("test_train_strata").size()
    # Reduce df to only hold strata with less than min_obs_per_group observations
    df_strata_with_too_little_obs = df_observations_per_strata[
        df_observations_per_strata < min_obs_per_group
    ]

    # Get sum of observations within strata with too little obs
    sum_of_observations_within_strata_with_too_little_obs = (
        df_strata_with_too_little_obs.sum()
    )

    # Print results
    org_numb_strata = df_all_strata["test_train_strata"].nunique()

    print(
        f"\n  - Out of {df_all_strata.shape[0]} observations, there are: {sum_of_observations_within_strata_with_too_little_obs} ({round(sum_of_observations_within_strata_with_too_little_obs/df_all_strata.shape[0]*100, 2)}%) that are in strata with less than {min_obs_per_group} observations.",
        f"\n  - These {sum_of_observations_within_strata_with_too_little_obs} observations from {df_strata_with_too_little_obs.shape[0]} stratas will be put into strata 'others'",
        end="",
    )

    # Replace strata with too little obs with "others"
    for s in df_strata_with_too_little_obs.index:
        df_all_strata["test_train_strata"] = df_all_strata["test_train_strata"].replace(
            s, "others"
        )

    fin_numb_strata = df_all_strata["test_train_strata"].nunique()

    print(
        f"\n  - From {org_numb_strata} strata, {fin_numb_strata} strata remain after merging strata with less than {min_obs_per_group} observations.",
    )

    return df_all_strata


# -----------------------------------------------------------------------------------------------
# def split_into_quantiles(arr, n_groups):
# labels = [f"{i/n_groups*100:.0f}-{(i+1)/n_groups*100:.0f}%" for i in range(n_groups)]
# groups = pd.qcut(arr, n_groups, duplicates='drop')
# print(f"Created {n_groups} groups: {labels}")
# return groups


# -----------------------------------------------------------------------------------------------
def get_tune_grid_regression():
    param_grid = {
        "n_estimators": [
            10,
            100,
            250,
            # 500,
            # 1000
        ],  # [int(x) for x in np.linspace(start=10, stop=2500, num=5)],
        "max_features": [
            # "sqrt",
            # "log2",
            0.01,
            # 0.1,
            0.25,
            # 0.5,
            # 0.75,
            # 1,
        ],
        # + [x for x in np.linspace(start=0.1, stop=0.8, num=5)],
        "max_depth": [
            1,
            # 5,
            8,
            15,
            # 20,
            # 25,
            # 100,
        ],  # [int(x) for x in np.linspace(1, 50, num=5)],
        "criterion": [
            "squared_error",
            # "absolute_error",
            # "friedman_mse",
            # "poisson",
        ],
        # "min_samples_split": [2, 5, 10
        # "min_samples_leaf": [1, 2, 4]
        # "bootstrap": [True, False],
    }
    return param_grid


# -----------------------------------------------------------------------------------------------
def get_tune_grid_classification():
    param_grid = {
        "n_estimators": [100, 500, 1000],  # Has minor influence
        "max_depth": [3, 12, 18],  # [1, 5, 8, 12, 15, 18]
        # 'min_samples_split': [2, 5, 10],
        # 'min_samples_leaf': [1, 2, 4],
        "max_features": [0.1],  # Minor influence
        # 'bootstrap': [True],
        "criterion": ["entropy"],  # 'gini',  entropy worked better on test
    }

    return param_grid


# -----------------------------------------------------------------------------------------------
def plot_quantiles(data, num_quantiles, variable_name, directory=None):
    # Calculate quantiles
    quantiles = np.percentile(
        data[variable_name], np.linspace(0, 100, num_quantiles + 1)
    )

    # Create histogram
    sns.histplot(data, x=variable_name)
    # plt.hist(array, bins="auto", color="royalblue", alpha=0.7, rwidth=0.85)

    # Add vertical lines for quantiles
    for quantile in quantiles:
        plt.axvline(quantile, color="red", linestyle="--")

    # Set labels and title
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {variable_name} with {num_quantiles} Quantiles")

    # Save if input dir is given
    if directory is not None:
        plt.savefig(f"{directory}/fig_histogram_{variable_name}.png")

    # Show the plot
    plt.show()


# -----------------------------------------------------------------------------------------------
def pdp_mp_wrapper(target, **kwargs):
    return PartialDependenceDisplay.from_estimator(target=target, **kwargs)


# -----------------------------------------------------------------------------------------------
def plot_aggregated_pdp(my_pdp, top_vars_pdp_ohed, var_ohe_dict, ymax=None):
    # Get dictionary ------------------------------------------------------------------------
    # Reduce var_ohe_dict to hold only keys where a value in the key is in top_vars_pdp_ohed
    var_ohe_dict_reduced = {
        key: value
        for key, value in var_ohe_dict.items()
        if any(col in top_vars_pdp_ohed for col in value)
    }

    # Flip dict for easier handling
    dict_flipd = {
        ohe: orig for orig, ohe_list in var_ohe_dict_reduced.items() for ohe in ohe_list
    }

    # Extract data from PDP object ------------------------------------------------------------
    final_df = pd.DataFrame(
        {
            "ohed_var": [],
            "non_ohed_var": [],
            "y_values": [],
            "x_values": [],
        }
    )

    for i in range(len(top_vars_pdp_ohed)):
        new_row = pd.DataFrame(
            {
                "ohed_var": top_vars_pdp_ohed[i],
                "non_ohed_var": dict_flipd[top_vars_pdp_ohed[i]],
                "y_values": my_pdp.pd_results[i]["average"][0].tolist(),
                "x_values": my_pdp.pd_results[i]["values"][0].tolist(),
            }
        )
        # Append new row to final_df
        final_df = pd.concat([final_df, new_row], axis=0)

    max_effect = np.max(final_df["y_values"])
    if ymax is None:
        max_effect = None
    df_groups = final_df.groupby("non_ohed_var")

    # Plot PDPs --------------------------------------------------------------------------------
    # Create a figure with m rows x n columns of subplots
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))  # Adjust the figure size as needed
    axs = axs.flatten()  # Flatten the array to easily iterate over it

    # Iterate over groups and plot
    ax_idx = 0
    for name, group in df_groups:
        if len(var_ohe_dict_reduced[name]) > 1:
            # Make barplot
            make_pdp_subplots(
                ax=axs[ax_idx],
                df_in=group,
                var_name=name,
                var_type="categorical",
                ymax=max_effect,
            )
        else:
            # Make lineplot
            make_pdp_subplots(
                ax=axs[ax_idx],
                df_in=group,
                var_name=name,
                var_type="numerical",
                ymax=max_effect,
            )
        ax_idx += 1
        if ax_idx == 10:  # Stop plotting after 10 plots
            break

    # Turn off the last two axes
    for ax in axs[10:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def make_pdp_subplots(ax, df_in, var_name, var_type, ymax=None):

    if var_type == "categorical":
        df = df_in[df_in["non_ohed_var"] == var_name].copy()
        pivot_df = df.pivot(index="ohed_var", columns="x_values", values="y_values")
        pivot_df["difference"] = pivot_df[1] - pivot_df[0]

        # Plot on the provided axis
        ax.bar(pivot_df.index, pivot_df["difference"])

        ax.set_xlabel(f"Levels in {var_name}")
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=6)
        ax.set_ylabel("Difference in Y")
        if ymax is not None:
            ax.axhline(0, linestyle="dotted", color="gray")
            ax.set_ylim(-ymax * 1.2, ymax * 1.2)
        ax.set_title(f"{var_name}")

    else:
        df_tmp = df_in[df_in["non_ohed_var"] == var_name].copy()
        df_tmp["x_values"] = df_tmp["x_values"].astype(float)

        # Plot on the provided axis
        ax.plot(df_tmp["x_values"], df_tmp["y_values"])
        ax.set_xlabel(var_name)
        ax.set_ylabel("Partial Dependence")
        if ymax is not None:
            ax.axhline(0, linestyle="dotted", color="gray")
            ax.set_ylim(-ymax * 1.2, ymax * 1.2)
        ax.set_title(f"{var_name}")


# ------------------------------------------------------------------------------------
def run_smote_classification(
    Xy_train=None,
    Xy_test=None,
    user_input=None,
    var_ohe_dict=None,
    rf_params=None,
    do_cv="none",
    cv_folds=5,
    do_eval=True,
    do_vi=True,
    smote_test=True,
    verbose=True,
    verbose_eval=True,
    save_directory=None,
):
    '''
    Note on variable importance calculation:
    Imputation method requires splitting df 
    '''

    # ! Checks ----------------------------------------------------------------
    if verbose:
        print("--- run_smote_classification():")
    if Xy_train.isna().sum().sum() > 0:
        raise ValueError(
            "run_smote_classification(): There are NA values in the train set!"
        )
    if do_eval:
        if Xy_test is None:
            raise ValueError(
                "run_smote_classification(): Xy_test must be given if do_eval is True!"
            )
        if Xy_train.isna().sum().sum() > 0:
            raise ValueError(
                "run_smote_classification(): There are NA values in the test set!"
            )
    if do_cv not in ["none", "cv", "oob"]:
        raise ValueError(f"'do_cv' must be set to 'none', 'cv' or 'oob'. Given was: {do_cv}")

    if do_cv == "oob":
        get_oob_score = True
    else:
        get_oob_score = False
        
    if do_cv == "none":
        scoring = None
        
    if do_vi not in ["none", "impurity", "imputation"]:
        raise ValueError(f"'do_vi' must be one of 'none', 'impurity', 'imputation'. Given was: {do_vi}")        
        

    # ! Build model ----------------------------------------------------------------
    oversample = SMOTE(random_state=user_input["seed_nr"])

    model = RandomForestClassifier(
        n_estimators=rf_params["n_estimators"],
        max_features=rf_params["max_features"],
        max_depth=rf_params["max_depth"],
        criterion=rf_params["criterion"],
        random_state=user_input["seed_nr"],
        n_jobs=-1,
        class_weight="balanced",
        oob_score=get_oob_score,
    )

    # Split into subsets
    X_train = Xy_train.drop(columns=["target", "test_train_strata"], errors="ignore")
    y_train = Xy_train["target"]
    
    # ! Model Training --------------------------------------------------------------
    # Start CV
    if do_cv == "cv":
        
        if verbose:
            print("\n - Running CV...")

        # Setup
        cvfold = 0
        scoring = []
        skf = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=user_input["seed_nr"]
        )

        for train_index, val_index in skf.split(X_train, y_train):
            # Update fold counter
            cvfold += 1

            # Split dfs
            X_train_cv, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
            y_train_cv, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

            # Apply SMOTE only to the training data
            X_train_cv, y_train_cv = oversample.fit_resample(X_train_cv, y_train_cv)
            if smote_test:
                X_val, y_val = oversample.fit_resample(X_val, y_val)

            # Train model
            model.fit(X_train_cv, y_train_cv)

            # Attach scores
            y_pred = model.predict(X_val)
            scores = {
                # "fold": cvfold,
                "f1": f1_score(y_val, y_pred, average="weighted"),
                "recall": recall_score(y_val, y_pred, average="weighted"),
                "precision": precision_score(y_val, y_pred, average="weighted"),
                "accuracy": balanced_accuracy_score(y_val, y_pred),
                "roc_auc": roc_auc_score(y_val, y_pred, average="weighted"),
            }
            scoring.append(pd.DataFrame(scores, index=[0]))
        
        # Get mean and sd of scores
        scoring = pd.DataFrame(
            {
                "mean": pd.concat(scoring, axis=0).mean(),
                "sd": pd.concat(scoring, axis=0).std(),
            }
        ).T
    # End CV
    elif do_cv == "oob":
        # Apply oversampling to train data
        X_train, y_train = oversample.fit_resample(X_train, y_train)
        # Train model
        model.fit(X_train, y_train)
        # Get OOB
        scoring = pd.DataFrame(
            {
                "oob": model.oob_score_
            }, index=[0]
        )
        
        scoring = pd.DataFrame(
            {
                "oob": model.oob_score_
            }, index=[0]
        )
    else:
        scoring = None

    # ! Variable Importance --------------------------------------------------------------
    # Train model
    model.fit(X_train, y_train)
    
    if do_cv == "oob":
        scoring = pd.DataFrame(
            {
                "oob": model.oob_score_
            }, index=[0]
        )

    # Get Vi if needed
    if do_vi == "impurity":
        if verbose:
            print("\n - Calculating variable importance...")
        df_vi = assessing_top_predictors(
            rf_in=model,
            ignore_these=["target", "test_train_strata"],
            X_train_in=X_train,
            dict_ohe_in=var_ohe_dict,
            with_aggregation=True,
            n_predictors=None,
            verbose=False,
        )
    else:
        df_vi = None

    # Evaluate model
    if do_eval:
        if verbose:
            print("\n - Evaluating model...")

        if smote_test:
            X_test, y_test = oversample.fit_resample(X_test, y_test)

        model_evaluation_classification(
            rf_model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            save_directory=save_directory,
            metric="f1-score",
            verbose=verbose_eval,
        )

    return model, scoring, df_vi


# -----------------------------------------------------------------------------------------------
def decode_one_hot_encoding(X_train, var_ohe_dict):

    print("... decoding one-hot encoding")

    decoded_df = X_train.copy()

    # Reduce var_ohe_dict to key that have an item with more than one element (avoids messing with non-OHE'd variable names)
    var_ohe_dict = {key: value for key, value in var_ohe_dict.items() if len(value) > 1}

    for var, levels in var_ohe_dict.items():
        # Filter columns with the specified variable prefix
        var_columns = [col for col in X_train.columns if col.startswith(var + "_")]

        # Check if there are one-hot encoding columns for the variable
        if var_columns:

            # Create a new column by aggregating the one-hot-encoded columns
            decoded_df[var] = (
                X_train[var_columns]
                .idxmax(axis=1)
                .str.split("_")
                .apply(lambda x: x[-1])
            )

            # Drop the original one-hot-encoded columns
            decoded_df.drop(var_columns, axis=1, inplace=True)

    return decoded_df


# Example usage:
# Replace 'your_var_ohe_dict' with the actual variable one-hot-encoding dictionary
# For example: {'var': ['var_1', 'var_2']}
# decoded_df = decode_one_hot_encoding(X_train, your_var_ohe_dict)


def cramers_v(array1, array2):
    # Function to calculate Cramer's V
    # From: https://www.statology.org/cramers-v-in-python/

    data = confusion_matrix(array1, array2)
    # data = np.array([[6, 9], [8, 5], [12, 9]])
    try:
        # Chi-squared test statistic, sample size, and minimum of rows and columns
        X2 = chi2_contingency(data, correction=False)[0]
        n = np.sum(data)
        minDim = min(data.shape) - 1

        # calculate Cramer's V
        V = np.sqrt((X2 / n) / minDim)
        return V
    except ValueError as e:
        # print(f" - Error calculating Cramer's V: {e}")
        return 0


def plot_corr_heatmap(
    corr,
    title="No title",
    vmin=-1,
    vmax=1,
    center=0,
    cmap="coolwarm",
    show=True,
    save_directory=None,
):

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(corr.shape[0] / 1.5, corr.shape[0] / 1.5))

    # Draw the heatmap with the mask and correct aspect ratio
    ax = sns.heatmap(
        corr,
        cmap=cmap,
        annot=True,
        fmt=".2f",
        square=True,
        vmin=vmin,
        vmax=vmax,
        center=center,
    )
    ax.set(xlabel="", ylabel="")
    ax.xaxis.tick_top()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_title(title)

    # Save if input dir is given
    if save_directory is not None:
        plt.savefig(f"{save_directory}/fig_heatmap_{title}.png")

    if show:
        plt.show()
    else:
        plt.close()


def remove_correlation_based_on_vi(
    Xy,
    var_ohe_dict,
    rf_vi,
    threshold=0.75,
    return_only_top_n=None,
    make_heatmaps=False,
    save_directory=None,
):

    print(" ---- REMOVAL OF HIGHLY CORRELATED VARIABLES ----")

    # ! Arrange the dataframe in the same order as the variable importance dataframe
    # Get order of features (note that they are NOT ohe'd, so I have to first decode the dataframe, before selection. As done below.)
    order_of_features = rf_vi.Feature.to_list()

    # Decode one-hot encoding
    df_decoded = decode_one_hot_encoding(Xy, var_ohe_dict)

    # Arrange columns in order of feature importance
    # Reduce order_of_features to variables that are in df_decoded
    order_of_features = [var for var in order_of_features if var in df_decoded.columns]
    df_decoded = df_decoded[order_of_features]

    # Display
    # df_decoded

    # ! Create correlation matrix
    # Separate variables into numerical and nominal
    numerical_variables = df_decoded.select_dtypes(include=["number"]).columns
    nominal_variables = df_decoded.select_dtypes(include=["object"]).columns

    # Use LabelEncoder for nominal variables
    label_encoder = LabelEncoder()
    df_encoded = df_decoded.copy()
    df_encoded[nominal_variables] = df_encoded[nominal_variables].apply(
        lambda col: label_encoder.fit_transform(col)
    )

    # Initialize matrices
    corr_matrix_numerical_numerical = pd.DataFrame(
        index=numerical_variables, columns=numerical_variables
    )
    corr_matrix_numerical_nominal = pd.DataFrame(
        index=numerical_variables, columns=nominal_variables
    )
    corr_matrix_nominal_nominal = pd.DataFrame(
        index=nominal_variables, columns=nominal_variables
    )

    # Suppress all warning
    original_warning_filters = warnings.filters[:]
    warnings.filterwarnings("ignore")

    # Calculate correlations
    for var1 in df_decoded.columns:
        for var2 in df_decoded.columns:
            if var1 in numerical_variables and var2 in numerical_variables:
                # Numerical-Numerical
                corr, _ = pearsonr(df_decoded[var1], df_decoded[var2])
                corr_matrix_numerical_numerical.loc[var1, var2] = abs(corr)
            elif var1 in numerical_variables and var2 in nominal_variables:
                # Numerical-Nominal
                f_statistic, p_value = f_oneway(
                    df_decoded[var1][df_encoded[var2] == 0],
                    df_decoded[var1][df_encoded[var2] == 1],
                )
                corr_matrix_numerical_nominal.loc[var1, var2] = p_value
                # Nominal-Numerical (reverse order)
                corr_matrix_numerical_nominal.loc[var2, var1] = p_value
            elif var1 in nominal_variables and var2 in nominal_variables:
                # Nominal-Nominal
                # Cannot compare the same variable against itself
                if var1 == var2:
                    corr_matrix_nominal_nominal.loc[var1, var2] = 1
                else:
                    cramer_v = cramers_v(df_encoded[var1], df_encoded[var2])
                    corr_matrix_nominal_nominal.loc[var1, var2] = cramer_v

    # Reset warnings
    warnings.filters = original_warning_filters

    # Change all to float
    corr_matrix_numerical_numerical = corr_matrix_numerical_numerical.astype(float)
    corr_matrix_numerical_nominal = corr_matrix_numerical_nominal.astype(float)
    corr_matrix_nominal_nominal = corr_matrix_nominal_nominal.astype(float)

    # Comparison numerical-nominal is not diagonally perfect (NA for comparison to itself)
    # Thus, remove all columns where first cell is NA.
    corr_matrix_numerical_nominal = corr_matrix_numerical_nominal.loc[
        :, corr_matrix_numerical_nominal.iloc[0].notna()
    ]

    num_num_empty = True if corr_matrix_numerical_numerical.empty else False
    num_nom_empty = True if corr_matrix_numerical_nominal.empty else False
    nom_nom_empty = True if corr_matrix_nominal_nominal.empty else False

    print(f" - Numerical-Numerical empty: {num_num_empty}")
    print(f" - Numerical-Nominal empty: {num_nom_empty}")
    print(f" - Nominal-Nominal empty: {nom_nom_empty}")

    if not num_nom_empty:
        # Then remove all rows where first cell is NA.
        corr_matrix_numerical_nominal = corr_matrix_numerical_nominal.loc[
            corr_matrix_numerical_nominal.iloc[:, 0].notna(), :
        ]

    # ! Make plots if requested
    if not num_num_empty:
        plot_corr_heatmap(
            corr_matrix_numerical_nominal,
            title="Comparison Numerical vs. Nominal via ANOVA",
            vmin=0.001,
            vmax=0.1,
            center=0.05,
            cmap="YlGnBu",
            show=make_heatmaps,
            save_directory=save_directory,
        )
    if not num_nom_empty:
        plot_corr_heatmap(
            corr_matrix_numerical_numerical,
            title="Comparison Numerical vs. Numerical via Pearson-r",
            vmin=0,
            vmax=1,
            center=0,
            cmap="coolwarm",
            show=make_heatmaps,
            save_directory=save_directory,
        )
    if not nom_nom_empty:
        plot_corr_heatmap(
            corr_matrix_nominal_nominal,
            title="Comparison Nominal vs. Nominal via Cramer's V",
            vmin=0,
            vmax=1,
            center=0,
            cmap="coolwarm",
            show=make_heatmaps,
            save_directory=save_directory,
        )

    # ! Remove highly correlated variables
    # Loop through every variable and remove it if it is highly correlated with another variable
    # Note: This is a very simple approach to feature selection. More advanced methods exist.

    # Create empty list to store variables to remove
    vars_to_remove = []
    replacements = {}  # Dictionary to store variable replacements

    # Loop through every variable
    # For the numerical-numerical pairs

    for var1 in corr_matrix_numerical_numerical.columns:
        if var1 not in vars_to_remove:
            for var2 in corr_matrix_numerical_numerical.columns:
                if var1 != var2:
                    if abs(corr_matrix_numerical_numerical.loc[var1, var2]) > threshold:
                        if var1 not in vars_to_remove and var2 not in vars_to_remove:
                            # Record removal
                            vars_to_remove.append(var2)

                            # Record replacement
                            if var1 not in replacements:
                                replacements[var1] = []
                            replacements[var1].append(var2)

    # For the nominal-nominal pairs
    for var1 in corr_matrix_nominal_nominal.columns:
        if var1 not in vars_to_remove:
            for var2 in corr_matrix_nominal_nominal.columns:
                if var1 != var2:
                    if abs(corr_matrix_nominal_nominal.loc[var1, var2]) > threshold:
                        if var1 not in vars_to_remove and var2 not in vars_to_remove:
                            # Record removal
                            vars_to_remove.append(var2)

                            # Record replacement
                            if var1 not in replacements:
                                replacements[var1] = []
                            replacements[var1].append(var2)

    # Keep final variables in non-ohed form
    final_variables_not_ohed = df_decoded.drop(columns=vars_to_remove).columns.to_list()

    if return_only_top_n is not None:
        final_variables_not_ohed = final_variables_not_ohed[:return_only_top_n]

    # Transform back to ohe form
    final_variables = []
    for var in final_variables_not_ohed:
        if var in var_ohe_dict.keys():
            final_variables.extend(var_ohe_dict[var])

    # Print to console
    print(
        f" - Aggregated {len(vars_to_remove)} correlating variables into {len(replacements)}. "
    )
    print(
        f" - Keeping {len(final_variables_not_ohed)} of {len(df_decoded.columns)} variables."
    )
    print(f"\n - Variables removed based on threshold = {threshold}:")
    for key in replacements.keys():
        print(f"  - {key:<30} -> {replacements[key]}\n")

    # Save to file
    if save_directory is not None:
        with open(f"{save_directory}/correlation_removal.txt", "w") as file:
            file.write(
                f" - Aggregated {len(vars_to_remove)} correlating variables into {len(replacements)}. "
            )
            file.write(
                f" - Keeping {len(final_variables_not_ohed)} of {len(df_decoded.columns)} variables."
            )
            file.write(f"\n - Variables removed based on threshold = {threshold}:")
            for key in replacements.keys():
                file.write(f"  - {key:<30} -> {replacements[key]}\n")

    # ! Return the final variables
    return final_variables


# -----------------------------------------------------------------------------------------------
def ___REGRESSION___():
    pass


def run_rf_regression(
    Xy=None,
    user_input=None,
    var_ohe_dict=None,
    return_vi_from_all_data_model=False,
    cv_folds=5,
    # cv_repeats=1,
    nestimators=150,
    verbose=True,
    return_eval=True,
    Xy_train=None,
    Xy_test=None,
    best_params=None,
    save_directory=None,
):

    # ! Build model ----------------------------------------------------------------
    if best_params is None:
        model = RandomForestRegressor(
            n_estimators=nestimators,
            random_state=user_input["seed_nr"],
            n_jobs=-1,
        )
    else:
        model = RandomForestRegressor(
            n_estimators=best_params["n_estimators"],
            max_features=best_params["max_features"],
            max_depth=best_params["max_depth"],
            criterion=best_params["criterion"],
            random_state=user_input["seed_nr"],
            n_jobs=-1,
        )

    # ! Option to train model on all data, no CV (to get feature importance for REF-CV)
    # The difference is simply that there is no train-test split before oversampling.
    if return_vi_from_all_data_model:

        # Impute all NAs as -9999
        # (should already happen outside, just making sure here)
        Xy = Xy.fillna(-9999)

        # Get stratification variable
        stratify_by = Xy["test_train_strata"]

        # Split into response and predictors
        X = Xy.drop(columns=["target", "test_train_strata"], errors="ignore")
        y = Xy["target"]

        # Train
        model.fit(
            X, y, sample_weight=get_weights_from_y(y, user_input["weight_method"])
        )

        # Variable Importance
        df_vi = assessing_top_predictors(
            rf_in=model,
            ignore_these=["target", "test_train_strata"],
            X_train_in=X,
            dict_ohe_in=var_ohe_dict,
            with_aggregation=True,
            n_predictors=None,
            verbose=verbose,
        )

        return df_vi

    # ! Split data and run CV -------------------------------------------------------
    if Xy is not None:

        # Impute all NAs as -9999
        # (should already happen outside, just making sure here)
        Xy = Xy.fillna(-9999)

        # Get stratification variable
        stratify_by = Xy["test_train_strata"]

        # Split into response and predictors
        X = Xy.drop(columns=["target", "test_train_strata"], errors="ignore")
        y = Xy["target"]

        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            stratify=stratify_by,
            test_size=user_input["test_split"],
            random_state=user_input["seed_nr"],
        )

    else:
        if (Xy_train is None) or (Xy_test is None):
            raise ValueError(
                "If no Xy is given, then Xy_train and Xy_test must be given!"
            )

        # Impute all NAs as -9999
        Xy_train = Xy_train.fillna(-9999)
        Xy_test = Xy_test.fillna(-9999)

        # Get stratification variable
        stratify_by = Xy_train["test_train_strata"]

        # Split into response and predictors
        X_train = Xy_train.drop(
            columns=["target", "test_train_strata"], errors="ignore"
        )
        y_train = Xy_train["target"]

        X_test = Xy_test.drop(columns=["target", "test_train_strata"], errors="ignore")
        y_test = Xy_test["target"]

    # Create Stratified K-fold cross validation
    # ! Note:
    # - Stratified K-fold is implemented in sklearn only for classification tasks to ensure
    # equal distribution of the target levels. For regression, random sampling suffices to
    # ensure equal distribution, so there is no additional stratification needed.
    # To ensure stratification by a specific variable, one would need to implement a new
    # function that does this manually.
    # - It is kinda implemented above for merging stratification levels with too few observations.
    # - Also there is no repeated option for KFold in sklearn, so this is not implemented here either.

    cv = KFold(
        n_splits=cv_folds,
        random_state=user_input["seed_nr"],
        shuffle=True,
    )

    # Set Scores for Cross-Validation
    scoring = {
        "r2": "r2",
        "mae": "neg_mean_absolute_error",
        "mse": "neg_mean_squared_error",
        "rmse": "neg_root_mean_squared_error",
    }

    # Do CV
    scores = cross_validate(
        model,
        X_train,
        y_train,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        params={
            "sample_weight": get_weights_from_y(y_train, user_input["weight_method"])
        },
    )

    if verbose:
        print("\n--- CV Results ---")
        for key in scores.keys():
            mean = np.mean(scores[key])
            std = np.std(scores[key])

            # sklaern defines higher values as better. So, errors are taken as negatives to be as close to 0
            # as possible. Therefore, we need to flip the mean for the errors.
            if key in ["test_mae", "test_mse", "test_rmse"]:
                mean = -mean
            print(f" - {key}: \t\t{round(mean,2)} +/- {round(std,2)}")

    # Train
    model.fit(
        X_train,
        y_train,
        sample_weight=get_weights_from_y(y_train, user_input["weight_method"]),
    )

    # Test
    if return_eval:
        model_evaluation_regression(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            save_directory=save_directory,
            verbose=True,
        )

    # Variable Importance
    df_vi = assessing_top_predictors(
        rf_in=model,
        ignore_these=["target", "test_train_strata"],
        X_train_in=X_train,
        dict_ohe_in=var_ohe_dict,
        with_aggregation=True,
        # n_predictors=show_top_n,
        verbose=False,
        save_directory=save_directory,
    )

    # Produce PDPs
    # todo

    return model, scores, df_vi


def add_vars_to_dict(dataset, df, dict, verbose=False):
    cols = df.drop(
        ["idp", "tree_id", "target", "first_year", "yrs_before_second_visit"],
        axis=1,
        errors="ignore",
    ).columns.tolist()
    dict[dataset] = df.columns.tolist()
    if verbose:
        print(f"Added {len(cols)} variables to '{dataset}' dataset: {cols}")
    else:
        print(f"Added {len(cols)} variables to '{dataset}' dataset.")
    return dict


def get_variables_to_remove(
    df_in,
    dict_in,
    n_to_remove,
    min_features_per_category,
):
    """
    Function to remove variables in the variable importance table that can be removed because there are still enough variables of the same category in the dataframe.
    - df_in: Variable importance table with columns Feature and Importance
    - dict_in: Category-Variable Dictionary
    - n_to_remove: The total number of variables that should be removed
    - min_features_per_category: The minimum number of variables that are "enough" so that the variable in question can be removed.

    Returns:
        list: List of variables that can be savely removed.
    """

    df_in["cat"] = ""
    for i in np.arange(0, len(df_in)):
        df_in.at[i, "cat"] = find_var_in_dict(df_in.at[i, "Feature"], dict_in)

    # Get inverted table to loop from least to highest importance
    df_rev = df_in.sort_values("Importance", ascending=True).reset_index(drop=True)

    # Loop over variables to be potentially removed
    vars_with_enough_of_same_category = []
    vars_to_keep = []
    for i in np.arange(0, n_to_remove):
        # Get category of current variable
        i_fea = df_rev.at[i, "Feature"]
        i_cat = df_rev.at[i, "cat"]
        # Set counter for how many variables of that category are "above" current var
        cat_count = 1 if min_features_per_category > 0 else 0
        # Loop over evey variable "above" current variable
        for j in np.arange(i + 1, len(df_rev)):
            # If the variables have the same category add to the counter
            # print(
            #     f"Comparing {i_fea} - {i_cat} to {df_rev.at[j, 'Feature']} - {df_rev.at[j, 'cat']}"
            # )
            if i_cat == df_rev.at[j, "cat"]:
                cat_count = cat_count + 1

        # print(i_fea, cat_count)
        # If enough variables of the same category are already there, add to list
        if cat_count > min_features_per_category:
            vars_with_enough_of_same_category.append(i_fea)
        else:
            vars_to_keep.append(i_fea)

    return vars_with_enough_of_same_category, vars_to_keep


def find_var_in_dict(var, dict):
    """
    Finds the variable that is nested in the list, which is the value of a key in the inputted dicionary.
    """
    cat_found = False
    for k in dict.keys():
        if var in dict[k]:
            if cat_found:
                raise ValueError(
                    f"Variable '{var}' was found in two categories!: {cat} and {k}"
                )
            cat = k
            cat_found = True

    if not cat_found:
        raise ValueError(f"No category found for variable: '{var}'")
    return cat


# -----------------------------------------------------------------------------------------------
def ___BACKUPS___():
    pass


def BU_run_smote_classification(
    Xy=None,
    user_input=None,
    var_ohe_dict=None,
    return_vi_from_all_data_model=False,
    return_cv_from_all_data_model=False,
    cv_folds=5,
    cv_repeats=1,
    nestimators=150,
    verbose=True,
    return_eval=True,
    Xy_train=None,
    Xy_test=None,
    best_params=None,
    save_directory=None,
):

    # ! Build model ----------------------------------------------------------------
    # SMOTE oversampling
    oversample = SMOTE(random_state=user_input["seed_nr"])
    from sklearn.linear_model import LogisticRegression

    # Random Forest Classifier
    if best_params is None:
        # model = LogisticRegression(
        #     random_state=user_input["seed_nr"],
        #     n_jobs=-1,
        #     class_weight="balanced",
        # )
        model = RandomForestClassifier(
            n_estimators=nestimators,
            random_state=user_input["seed_nr"],
            n_jobs=-1,
            class_weight="balanced",
        )
    else:
        #  model = LogisticRegression(
        #     random_state=user_input["seed_nr"],
        #     n_jobs=-1,
        #     class_weight="balanced",
        #  )

        model = RandomForestClassifier(
            n_estimators=best_params["n_estimators"],
            max_features=best_params["max_features"],
            max_depth=best_params["max_depth"],
            criterion=best_params["criterion"],
            random_state=user_input["seed_nr"],
            n_jobs=-1,
            class_weight="balanced",
        )

    # ! Run VI or CV with ALL DATA -------------------------------------------------------
    # Option to train model on all input data without splitting before oversampling
    if return_vi_from_all_data_model or return_cv_from_all_data_model:

        # Impute all NAs as -9999
        Xy = Xy.fillna(-9999)

        # Split into response and predictors
        X = Xy.drop(columns=["target", "test_train_strata"], errors="ignore")
        y = Xy["target"]

        # Apply oversampling to train set
        X_train_over, y_train_over = oversample.fit_resample(X, y)

        if return_vi_from_all_data_model:
            # Train
            model.fit(X_train_over, y_train_over)

            # Variable Importance
            df_vi = assessing_top_predictors(
                rf_in=model,
                ignore_these=["target", "test_train_strata"],
                X_train_in=X_train_over,
                dict_ohe_in=var_ohe_dict,
                with_aggregation=True,
                n_predictors=None,
                verbose=verbose,
            )

            return df_vi

        elif return_cv_from_all_data_model:
            # Run CV
            scoring = {
                "f1": "f1_weighted",
                "recall": "recall_weighted",
                "precision": "precision_weighted",
                "accuracy": "balanced_accuracy",
                "roc_auc": "roc_auc_ovr_weighted",
            }
            scores = cross_validate(
                model, X_train_over, y_train_over, scoring=scoring, cv=cv, n_jobs=-1
            )
            return scores

    # ! Split data and run CV -------------------------------------------------------
    if Xy is not None:

        # Impute all NAs as -9999
        Xy = Xy.fillna(-9999)

        # Split into response and predictors
        X = Xy.drop(columns=["target", "test_train_strata"], errors="ignore")
        y = Xy["target"]

        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=user_input["test_split"],
            stratify=y,
            random_state=user_input["seed_nr"],
        )

    else:
        if (Xy_train is None) or (Xy_test is None):
            raise ValueError(
                "If no Xy is given, then Xy_train and Xy_test must be given!"
            )

        # Impute all NAs as -9999
        Xy_train = Xy_train.fillna(-9999)
        Xy_test = Xy_test.fillna(-9999)

        # Split into response and predictors
        X_train = Xy_train.drop(
            columns=["target", "test_train_strata"], errors="ignore"
        )
        y_train = Xy_train["target"]

        X_test = Xy_test.drop(columns=["target", "test_train_strata"], errors="ignore")
        y_test = Xy_test["target"]

    # Apply oversampling to train set
    X_train_over, y_train_over = oversample.fit_resample(X_train, y_train)
    X_test, y_test = oversample.fit_resample(X_test, y_test)

    # Create Stratified K-fold cross validation
    # cv = RepeatedStratifiedKFold(
    #     n_splits=cv_folds, n_repeats=cv_repeats, random_state=user_input["seed_nr"],
    # )
    cv = StratifiedKFold(
        n_splits=cv_folds, random_state=user_input["seed_nr"], shuffle=True
    )
    # Get y-labels
    ylabels = sorted(y_train_over.unique())

    # Set Scores for Cross-Validation
    if user_input["model_task"] == "binary":
        scoring = {
            "f1": "f1_weighted",
            "recall": "recall_weighted",
            "precision": "precision_weighted",
            "accuracy": "balanced_accuracy",
            "roc_auc": "roc_auc_ovr_weighted",
        }

    elif user_input["model_task"] == "multiclass":
        scoring = {
            "f1": "f1_weighted",
            "recall": "recall_weighted",
            "precision": "precision_weighted",
            "accuracy": "balanced_accuracy",
            "roc_auc": "roc_auc_ovr_weighted",
        }
    else:
        raise ValueError(
            "run_smote_classification(): Either 'binary' or 'multiclass' must be True!"
        )

    # Do CV
    scores = cross_validate(
        model, X_train_over, y_train_over, scoring=scoring, cv=cv, n_jobs=-1
    )

    # Train
    model.fit(X_train_over, y_train_over)

    # Test
    if return_eval:
        model_evaluation_classification(
            model,
            X_train_over,
            y_train_over,
            X_test,
            y_test,
            save_directory=save_directory,
        )

    # Variable Importance
    df_vi = assessing_top_predictors(
        rf_in=model,
        ignore_these=["target", "test_train_strata"],
        X_train_in=X_train_over,
        dict_ohe_in=var_ohe_dict,
        with_aggregation=True,
        # n_predictors=show_top_n,
        verbose=False,
        save_directory=save_directory,
    )

    # Reporting
    if verbose:
        print(" - CV Results:")
        for key in scores.keys():
            print(
                f"  - {key}: {round(np.mean(scores[key]),3)} +/- {round(np.std(scores[key]),3)}"
            )

    return model, scores, df_vi


# -----------------------------------------------------------------------------------------------
