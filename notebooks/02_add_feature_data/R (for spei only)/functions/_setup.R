# Packages

packages <- c(

  # Workspace management
  "here", "conflicted",

  # HTML Output
  "DT", # datatable for interactive tables

  # Parallel computing
  "foreach",
  "doParallel",
  "parallel",

  # Data Wrangling
  "tidyverse", "purrr", "SPEI", "feather",

  # Data Visualisation
  "ggplot2", "ggridges", "patchwork", "Cairo",
  "visdat",

  # Machine Learning
  "caret", "ranger",

  # Spatial data
  "terra", "leaflet", "sp", "sf", "raster", "stars", "rnaturalearth",
  # "rgee",

  # To facilitate coding:
  "tictoc", "beepr"
)

# R functions
sapply(
  list.files(here::here("functions"),
    pattern = "f_",
    full.names = TRUE
  ),
  source
)


# Install packages not yet installed
installed_packages <- packages %in% rownames(installed.packages())
if (any(installed_packages == FALSE)) {
  install.packages(packages[!installed_packages])
}

## Specifying parallel computing
# plan (multisession, workers = 9)

# library(multidplyr)
# library(dplyr, warn.conflicts = FALSE)
#
# cluster <- new_cluster(parallel::detectCores() - 2) # Use all but two cores
# cluster_library(cluster, "dplyr")

# Packages loading
invisible(lapply(packages, library, character.only = TRUE))
conflicts_prefer(dplyr::select)
conflicts_prefer(dplyr::filter)
