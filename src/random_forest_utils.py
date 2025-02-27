# Standard library
import datetime
import random
import re
import sys
import warnings
from os import error

# Data wrangling
import numpy as np
import pandas as pd

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport

# Machine learning
from scipy.stats import chi2_contingency, f_oneway, pearsonr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.preprocessing import OneHotEncoder

# Custom utilities
sys.path.insert(0, "../../src")
import chime
from random_forest_utils import *
from run_mp import *
from utilities import *


# -----------------------------------------------------------------------------------------------


def ___GENERAL___():
    pass


# Create txt file with name
def write_txt(text):
    with open(text, "w") as file:
        pass

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
                smote_on_test=False,
                do_tuning=False,
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
                smote_on_test=False,
                do_tuning=False,
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
    # OOB approach does not need to split into train and test
    # Apply SMOTE and feature importance assessment on all data
    
    # Split into X and y
    X = Xy_all.drop(columns=["target", "test_train_strata"], errors="ignore")
    y = Xy_all["target"]
    
    # Apply SMOTE
    sm = SMOTE(random_state=rnd_seed)
    X, y = sm.fit_resample(X, y)
    
    # ! Tuning ---------------------------------------------------------------------
    if do_tuning:
        if verbose:
            print(" - Tuning...")
            
        # Run small grid search to find best parameters
        rf = RandomForestClassifier(random_state=rnd_seed, n_jobs=-1, class_weight="balanced")
        param_grid = rfe_tuning_params()
        grid = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, verbose=0)
        grid.fit(X, y)
    
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
    rf.fit(X, y)
    
    # ! Variable Importance ---------------------------------------------------------------------
    if method_importance == "permutation":
        # Variable importance - Permutation
        featimp = assessing_top_predictors(
            vi_method = "permutation",
            rf_in=rf,
            ignore_these=["target", "test_train_strata"],
            X_train_in=X,
            X_test_in=X,
            y_test_in=y,
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
            X_train_in=X,
            dict_ohe_in=var_ohe_dict,
            with_aggregation=do_feat_aggregation,
            verbose=verbose,
            random_state=rnd_seed,
            save_directory=save_directory,
        )
        
    featimp.sort_values("Importance", ascending=False, inplace=True)
        
    # ! Model Evaluation ---------------------------------------------------------------------
    y_pred = rf.predict(X)
    y_pred = pd.Series(y_pred, index=y.index)
    scoring = bootstrap_classification_metric(
        y, y_pred, ["accuracy", "precision", "recall", "f1", "roc_auc"]
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
    make_plots=False,
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
    if make_plots:
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

def plot_roc_auc_both(train_data, test_data, title='ROC AUC Curves', save_directory=None, show=True, make_plot=True):
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

    # Save the AUC scores
    pd.DataFrame({
            "test_mean": np.mean(aucs_test),
            "test_sd": np.std(aucs_test),
            "train_mean": np.mean(aucs_train),
            "train_sd": np.std(aucs_train)
        }, index=[0]).to_csv(f"{save_directory}/roc_auc.csv", index=False)

    if make_plot:
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

        # Show or close the plot based on the `show` parameter
        if show:
            plt.show()
        else:
            plt.close()

def plot_pr_auc_both(train_data, test_data, title='PR AUC Curves', save_directory=None, show=True, make_plot=True):
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
    
    # Save the AUC scores
    pd.DataFrame({
            "test_mean": np.mean(pr_aucs_test),
            "test_sd": np.std(pr_aucs_test),
            "train_mean": np.mean(pr_aucs_train),
            "train_sd": np.std(pr_aucs_train)
        }, index=[0]).to_csv(f"{save_directory}/pr_auc.csv", index=False)

    if make_plot:
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
    if make_heatmaps:
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
    display(pd.read_csv(f"{i_dir}/roc_auc.csv"))
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
        shap_values = load_shap(filepath + "shap/approximated/shap_values_test.pkl")
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
            filepath + "shap/approximated/shap_values_test.pkl"
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

    # Plot ROC AUC curve
    plot_roc_auc_both(
        (y_train, y_train_pred_proba),
        (y_test, y_test_pred_proba),
        save_directory=save_directory,
        show=verbose,
        make_plot=False,
    )
    
    plot_pr_auc_both(
        (y_train, y_train_pred_proba),
        (y_test, y_test_pred_proba),
        save_directory=save_directory,
        show=verbose,
        make_plot=False,
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
        "./model_runs/feature_category_dictionary.json",
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
        "./model_runs/feature_category_dictionary.json",
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
    
    idir = f"./model_runs/all_runs/{imodel}/{ispecies}"
    iseed = imodel.split(" ")[0].split("_")[1]
    shapdir = f"{idir}/shap"

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
    rf_path = f"./model_runs/all_runs/{imodel}/{ispecies}/final_model/rf_model.pkl"
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
        np.nan: "Other", 
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
        "Other": "#c5d3d8", 
        "Climate Change Inter.": "#c5d3d8",  
    }

    color_dict = {
        "Climate Change": "#7B52AB",
        "Stand Structure": "#c5d3d8",
        "Tree Structure": "#c5d3d8",
        "Topography": "#c5d3d8",
        "Management": "#c5d3d8",
        "Soil Conditions": "#c5d3d8",
        "Forest Health": "#c5d3d8",
        "Other": "#c5d3d8", 
        "Climate Change Inter.": "#c5d3d8",  
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
    
    # Adjust coloring if needed 
    print("🚨 if the coloring is incorrect, adjust by hand in function 'ax_dataset_boxplot()' ")
    if pos_spei is not None and pos_temp is not None:
        palette_order[pos_spei] = color_spei
        palette_order[pos_temp] = color_temp
    elif all_or_top9 == "all":
        palette_order[1] = color_temp
        palette_order[2] = color_spei
    elif all_or_top9 == "top9":
        palette_order[2] = color_temp
        palette_order[4] = color_spei
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
        "../../notebooks/03_model_fitting_and_analysis/model_runs/all_runs/*/*"
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
    
    
def calculate_rf_performance(df_in, base_dir, skip_if_csv_exists=True):
    for i, row in df_in.iterrows():
        
        # Get paths
        path_rf = f"{base_dir}/{row.model}/{row.species}"
        save_dir = f"{path_rf}/rf_performance"
        os.makedirs(save_dir, exist_ok=True)

        # Load actual target data and predicted probabilities
        y_test = pd.read_csv(f"{path_rf}/final_model/y_test.csv", index_col=0)
        y_test_pred_proba = pd.read_csv(
            f"{path_rf}/final_model/y_test_proba.csv", index_col=0
        )

        y_train = pd.read_csv(f"{path_rf}/final_model/y_train.csv", index_col=0)
        y_train_pred_proba = pd.read_csv(
            f"{path_rf}/final_model/y_train_proba.csv", index_col=0
        )

        # Quality Control
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

        # Define expected csv files
        expected_roc = f"{save_dir}/roc_auc.csv"
        expected_pr = f"{save_dir}/pr_auc.csv"
        
        if skip_if_csv_exists and os.path.isfile(expected_roc):
            pass
        else:
            plot_roc_auc_both(
                (y_train, y_train_pred_proba),
                (y_test, y_test_pred_proba),
                # save_directory=save_directory,
                save_directory=save_dir,
                show=False,
                make_plot=False
            )
        
        if skip_if_csv_exists and os.path.isfile(expected_pr):
            pass
        else:
            plot_pr_auc_both(
                (y_train, y_train_pred_proba),
                (y_test, y_test_pred_proba),
                # save_directory=save_directory,
                save_directory=save_dir,
                show=False,
                make_plot=False
            )