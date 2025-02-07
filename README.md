# Analysis of Tree Mortality Drivers across France

This repository contains the data processing, feature engineering, and machine learning analysis code for our study:

> Flush to Crush: The Paradox of Favourable Springs Leading to Tree Mortality

We analyze over 600'000 trees from the French National Forest Inventory (2015–2023) to investigate climate-driven mortality trends using explainable machine learning.

## Repository Structure  
```
├── data/                   # Placeholder for processed datasets (not included due to size constraints)
├── notebooks/              # Jupyter notebooks for key analysis steps
│   ├── 01_process_nfi_data/       # NFI Data cleaning & preprocessing
│   ├── 02_add_feature_data/       # Feature extraction & engineering
│   ├── 03_fit_and_analyze_models/ # Model fitting & results analysis
├── src/                    # Python source code
│   ├── imports.py          
│   ├── random_forest_utils.py     # Utility functions for Random Forest modeling
│   ├── run_mp.py                  # Multiprocessing implementation
│   ├── utilities.py                # General helper functions
├── LICENSE                  
├── README.md                # Project documentation (this file)
```

## Data Availability  
Due to file size constraints, full raw data for NFI data cleaning, feature extraction, model fitting, and analysis is available upon request.
