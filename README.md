# Analysis of Tree Mortality Drivers Across France

This repository contains the data processing, feature engineering, and machine learning analysis code for the study "Hidden Climate Stressors: A Complex Interplay of Climate Anomalies Shapes European Tree Mortality" We analyzed over 500,000 trees from the French National Forest Inventory (2015–2023) to investigate climate-driven mortality using explainable machine learning.

> [Citation will be added upon publication]

## Table of Content

* [Code and Software](#code-and-software)
    + [System Requirements](#system-requirements)
    + [Installation Guide](#installation-guide)
+ [Demo and Instructions for Use](#demo-and-instructions-for-use)
    - [Data Availability](#data-availability)
    - [Running the Analysis](#running-the-analysis)
    - [File Formats & Compatibility](#file-formats-compatibility)
* [Directory Structure](#directory-structure)


## Code and Software
### System Requirements

This code was developed and executed on the following system specifications:

- **Hardware**: MacBook Pro, Apple M2 Pro, Sequoia 15.0
- **Python**: Python 3.11.5 in Visual Studio Code (Version 1.97.2)
- **R**: R 4.3.1 in RStudio (Version 2023.09.0+463)
- **QGIS**: QGIS 3.34 (Prizren)


### Installation Guide

#### Python

1. Ensure Python is installed.
2. Navigate to the repository root directory (where `requirements.txt` is located).
3. Create a virtual environment:
   ```bash
   python -m venv venv  # Use 'python3' if needed
   ```
4. Activate the environment:
   ```bash
   source venv/bin/activate  # For macOS/Linux
   venv\Scripts\activate  # For Windows (if applicable)
   ```
5. Install required libraries (may take several minutes):
   ```bash
   pip install -r requirements.txt
   ```
6. When running notebooks, ensure the correct virtual environment kernel is selected.

#### R

- The R code is located in `notebooks/02_collect_features/R (for spei data only)`.
- R was used specifically for the `{spei}` package (Version 1.8.1) to calculate the Standardized Precipitation-Evapotranspiration Index (SPEI).
- To set up the R environment, restore dependencies using `{renv}`:
  ```r
  renv::restore("notebooks/02_collect_features/R (for spei data only)/renv.lock")
  ```

#### QGIS

- QGIS was used to calculate topographical features using default settings (Version 3.34, Prizren).


## Demo and Instructions for Use

### Data Availability

- Due to file size limitations, the raw data cannot be included in this repository.
- The processed data required for model fitting, analysis, and figure generation can be downloaded from [Zenodo](https://zenodo.org/records/14941688?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImJhZGZjNTFhLTVlNGUtNDM5Ni1hYWRlLTViZDNjY2JlMTc2MyIsImRhdGEiOnt9LCJyYW5kb20iOiJmMzc5N2E0YjIyYjliNmNmOTA3OTk1MTQxMjkzNzBiNiJ9.RZatmIF_oKJpU2pKO6hg9trpF6jnj2aE4Gux-GVrcxwYXAiPUXDZNhggsxcUqG6g0loJQUYJi6iIFBnkoYK3Uw).
- After downloading, place the `data` folder in the repository’s root directory.

### Running the Analysis

1. **Preprocessing National Forest Inventory (NFI) Data**
   - The first step in the analysis is processing the NFI data using the notebooks inside `notebooks/01_process_nfi_data/`:
     - `01_clean_raw_nfi_data.ipynb` cleans the raw NFI dataset.
     - `02_calculate_mortality.ipynb` calculates temporal and spatial mortality metrics.

2. **Feature Extraction**
   - Environmental and structural features are extracted using `notebooks/02_collect_features/`:
     - `01_get_forest_structure.ipynb` extracts forest structure features from the NFI data.
     - `02_get_management.ipynb` extracts management features from the NFI data.
     - `03_get_topography.ipynb` extracts topographical features from the digital elevation model.
     - `04_get_soil.ipynb` extracts soil-related features.
     - `10_get_climate.ipynb` and `11_get_spei.ipynb` wrangles and extracts temperature and SPEI features.
         - The SPEI data calculation from evapotranspiration and precipitation data is done in `notebooks/02_collect_features/R (for spei data only)`, where:
         - The scripts inside `functions/` perform preprocessing.
         - The main project is `ifn_analysis.Rproj`.
         - The `scripts/` folder contains execution scripts.
         - The `output/` folder stores intermediate results.
     - `20_get_land_cover.ipynb` and `21_get_landsat_ndvi.ipynb` extracts NDVI features.
     - `30_get_hansen2013highresolution_treecover.ipynb` extracts tree cover data.
     - `40_get_niinements_tolerance.ipynb` extracts species tolerance data.

3. **Model Fitting and Analysis**
   - Model fitting and analysis are done inside `notebooks/03_model_fitting_and_analysis/`:
     - `01_model_fitting.ipynb` fits all random forest, including the recursive feature selection, and calculates model performance and SHAP values.
         - *🚨 Important: On a "normal" computer, this can take **several days**. See comments in the notebook on how to parallelize across multiple notebooks.*
     - `02_randomforests_analysis.ipynb` analyzes random forest model results.
     - `03_glmm_analysis.ipynb` performs generalized logistic mixed model fitting, analysis, and comparison to random forest models.
     - `04_gather_output.ipynb` collects model outputs into `output/`.
     - `05_create_reprod_subset.ipynb` creates a reproducible data subset for the most common nine species.

4. **Precomputed Results for Faster Analysis**
   - The [Zenodo](https://zenodo.org/records/14941688?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImJhZGZjNTFhLTVlNGUtNDM5Ni1hYWRlLTViZDNjY2JlMTc2MyIsImRhdGEiOnt9LCJyYW5kb20iOiJmMzc5N2E0YjIyYjliNmNmOTA3OTk1MTQxMjkzNzBiNiJ9.RZatmIF_oKJpU2pKO6hg9trpF6jnj2aE4Gux-GVrcxwYXAiPUXDZNhggsxcUqG6g0loJQUYJi6iIFBnkoYK3Uw) repository also provides precomputed model fitting results for the **nine most common species**.
   - Using this dataset, you can run `02_randomforests_analysis.ipynb` and `03_glmm_analysis.ipynb ` without delay.
   - Due to file size constraints, full model fitting results are not included but can be reproduced as described above.
   - To use the precomputed subset, place the `all_runs` folder inside: `notebooks/03_model_fitting_and_analysis/model_runs/`
   - **Important:** Running the analysis on this subset **does not** recreate all study results that are based on analyzing all species together. The subset is intended for reproducibility and transparency purposes.

5. **Notebook Execution Order**
   - Run notebooks in numerical order, as outputs from earlier notebooks are used in later ones.
   - Expected results shown in the study are shown within each notebook in the repository. Re-running these notebooks locally will remove these displayed results but are always retrievable from this repository.

### File Formats & Compatibility

- Large data files and models are stored in the `.feather` or `.pkl` format for efficiency and models are stored in the `.pkl` format. They can beopened with (make sure the `requirements.txt` is installed for dependencies):
  ```python
  # Feather files:
  loaded_file = pd.read_feather(filepath)

  # Pickle files:
  with open(filepath, "rb") as file:
       loaded_file = pickle.load(file)
  ```


---

## Directory Structure

```
├── .here
├── LICENSE.md
├── README.md
├── data
│   └── final
│       └── README.md
├── notebooks
│   ├── 01_process_nfi_data
│   │   ├── 01_clean_raw_nfi_data.ipynb
│   │   ├── 02_calculate_mortality copy.ipynb
│   │   ├── 02_calculate_mortality.ipynb
│   │   └── README.md
│   ├── 02_collect_features
│   │   ├── 01_get_forest_structure.ipynb
│   │   ├── 02_get_management.ipynb
│   │   ├── 03_get_topography.ipynb
│   │   ├── 04_get_soil.ipynb
│   │   ├── 10_get_climate.ipynb
│   │   ├── 11_get_spei.ipynb
│   │   ├── 20_get_land_cover.ipynb
│   │   ├── 21_get_landsat_ndvi.ipynb
│   │   ├── 30_get_hansen2013highresolution_treecover.ipynb
│   │   ├── 40_get_niinements_tolerance.ipynb
│   │   ├── R (for spei data only)
│   │   │   ├── functions
│   │   │   │   └── _setup.R
│   │   │   ├── ifn_analysis.Rproj
│   │   │   ├── output
│   │   │   ├── renv.lock
│   │   │   └── scripts
│   │   │       └── calculate_spei.R
│   │   └── README.md
│   └── 03_model_fitting_and_analysis
│       ├── 01_model_fitting copy 0.ipynb
│       ├── 01_model_fitting copy 1.ipynb
│       ├── 01_model_fitting copy 2.ipynb
│       ├── 01_model_fitting copy 3.ipynb
│       ├── 01_model_fitting copy 4.ipynb
│       ├── 01_model_fitting.ipynb
│       ├── 02_randomforests_analysis.ipynb
│       ├── 03_glmm_analysis.ipynb
│       ├── 04_gather_output.ipynb
│       ├── 05_create_reprod_subset.ipynb
│       ├── all_seeds.csv
│       ├── model_analysis
│       │   └── README.md
│       └── model_runs
│           ├── README.md
│           └── all_runs
├── output
├── requirements.txt
└── src
    ├── curlyBrace.py
    ├── imports.py
    ├── random_forest_utils.py
    ├── run_mp.py
    └── utilities.py
```