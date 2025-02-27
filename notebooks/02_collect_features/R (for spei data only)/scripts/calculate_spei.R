source("./functions/_setup.R")

# Load data ----
merged_or_direct <- "merged"
nsplits = 2

spei_data_raw <- read_delim(paste0("/Volumes/SAMSUNG 1TB/digitalis_v3/processed/1km/all/", merged_or_direct, "/data_to_calculate_spei.csv"))
spei_data <- dplyr::as_tibble(spei_data_raw)
spei_data_raw$date <- lubridate::as_date(spei_data_raw$date)
spei_data_raw$year <- year(spei_data_raw$date)
spei_data_raw$month <- month(spei_data_raw$date)
nrow_before <- nrow(spei_data_raw)

# Debug: The file 1969_9 for etp is missing online, so replacing that data with the 1968_9
spei_data <- spei_data_raw
dummy_etp <- spei_data |> filter(date == "1968-09-01") |> mutate(date = as_date("1969-09-01"))
spei_data_split <- spei_data |> filter(date != "1969-09-01")
spei_data <- bind_rows(spei_data_split, dummy_etp)
nrow_after <- nrow(spei_data)

if (nrow_before != nrow_after){stop("Fixing NA values did not work!")}

# Scale ETP and PREC (where scaled 10x)
spei_data$etp <- spei_data$etp/10
spei_data$prec <- spei_data$prec/10

# Calculate prec minus etp
spei_data$prec_minus_etp <- spei_data$prec - spei_data$etp

# Timeseries Calculation Only ----

# Inputs
spei_timescales <-  1:24

# Get list of df ----
sites <- spei_data$idp |> unique()
n_sites <- sites |> length()
n_sites_10s <- floor(n_sites/20)
df_list <- list()
total_rows_allocated = 0

for (n in 1:20) {
  # Get first and last site index
  first_site = ((n-1)*(n_sites_10s))
  last_site = (n*n_sites_10s)-1
  if (n == 20) last_site = n_sites
  
  # Get subset
  subset_sites = sites[first_site:last_site]
  
  # Pack all sites from subset into the same nested df
  df_list[[n]] <- spei_data |> filter(idp %in% subset_sites)
  
  # Count allocated rows
  total_rows_allocated = total_rows_allocated + nrow(df_list[[n]])
  
  print(paste0("n = ", n, " putting ", length(subset_sites), " sites into df. That is ", nrow(df_list[[n]]), " rows"))
}

# Check if allocated rows are equal to original rows
if (total_rows_allocated != nrow(spei_data)) {
  stop("Splitting into nested dfs did not work")
}

# Split data into two lists ----
# The whole split is not doable all at once!
split_1 <- df_list[1:10]
split_2 <- df_list[11:20]

split_to_run <- split_2
split_to_save <- "./output/R-spei_split_2.csv"

# MP All ----

# SELECT SPLIT BY HAND !!!! Also fix saving file path at the bottom!
# To run both splits, you need to restart R!

# Start MP
n.cores <- parallel::detectCores()-1 # Use all but one core
my.cluster <- parallel::makeCluster(n.cores, type = "FORK") # Use FORK backend
doParallel::registerDoParallel(cl = my.cluster) # Register it to be used by %dopar%

# Run MP
mp_out <- 
  get_spei_timeseries_loop_mp(
    spei_data=split_to_run, 
    spei_nrs=spei_timescales,
    merged_or_direct=merged_or_direct,
    debug = F, 
    debug_n = 20, 
    verbose = T,
    load_if_exists=F,
  )

# Close MP
parallel::stopCluster(cl = my.cluster)

# Save file
write_csv(mp_out, split_to_save)

# -> These files are further processed in python notebook
# OLD CODE BELOW -----------------------------------------------------------------------------------

## Single Run ----
# Fct in loop:
# single_run <- get_spei_timeseries(
#   data_input = df_list[[1]] |> dplyr::filter(idp == df_list[[1]] |> slice_sample(n=1) |> pull(idp)),
#   spei_nrs = spei_timescales
# )

# # Loop
# for (i in 1:length(df_list)) {
#   cat(paste0("\n - Running list Nr. ", i, " | Takes ca. until: ", Sys.time()+2200)) #2200 ~ 35 Mins
#   if (i %in% c(1, 2)) next
#   ifilename <- paste0("/Volumes/SAMSUNG 1TB/digitalis_v3/processed/1km/all/", merged_or_direct, "/spei/spei_timeseries_individual_loop_", i, ".csv")
#   loop_run <- get_spei_timeseries_loop(
#       # data_input = df_list[[1]] |> dplyr::filter(idp %in% (df_list[[1]]$idp |> unique() |> sample(5))),
#       data_input = df_list[[i]],
#       filename = ifilename,
#       spei_nrs = 1:24
#   )
# }
# 
# # MP Single:
# mp_out <- 
#   get_spei_timeseries_loop_mp(
#     spei_data=df_list, 
#     spei_nrs=spei_timescales,
#     debug = T, 
#     debug_n = 20, 
#     verbose = T,
#     load_if_exists=F,
#     )

## MP All ----
# Start MP
n.cores <- parallel::detectCores()-1 # Use all but one core
my.cluster <- parallel::makeCluster(n.cores, type = "FORK") # Use FORK backend
doParallel::registerDoParallel(cl = my.cluster) # Register it to be used by %dopar%

# Run MP
mp_out <- 
  get_spei_timeseries_loop_mp(
    spei_data=df_list, 
    spei_nrs=spei_timescales,
    merged_or_direct=merged_or_direct,
    debug = F, 
    debug_n = 20, 
    verbose = T,
    load_if_exists=F,
  )

# Close MP
parallel::stopCluster(cl = my.cluster)

# Save file
write_csv(mp_out, "./safety_dump_spei.csv")

# __________________________________________________________________________________________________
# Full Extraction with Derivatives ----

# SPEI-Timescales to calculate for
spei_nrs <- c(3, 6, 12, 18)
time_subsets <- c("annual")

## Single process test ----
# Fct in loop:
get_spei_metrics(
  spei_data |>  dplyr::filter(idp == spei_data |> slice_sample(n=1) |> pull(idp)),
  spei_nrs,
  years_before_1st_visit = 5,
  return_timeseries = TRUE,
  time_subsets = time_subsets
)

# MP Loop:
mp_out <- get_spei_metrics_loop(
  spei_data, 
  spei_nrs, 
  c("annual"), 
  debug = T,
  debug_n = 100,
  load_if_exists = FALSE
  )

## Multi process test ----
# Followed this tutorial: https://www.blasbenito.com/post/02_parallelizing_loops_with_r/
# Takes about 45 minutes to run!

# Setup MP
n.cores <- parallel::detectCores() - 1 # Use all but one core
my.cluster <- parallel::makeCluster(n.cores, type = "FORK") # Use FORK backend
doParallel::registerDoParallel(cl = my.cluster) #register it to be used by %dopar%

# Run MP
mp_out <- get_spei_metrics_loop(
  spei_data, 
  spei_nrs, 
  c("annual"), 
  debug = FALSE,
  debug_n = 100,
  load_if_exists = FALSE
)

# Close MP
parallel::stopCluster(cl = my.cluster)

# ________________________________________________________________________
# Plot full SPEI timeseries with trends
sites_subset <- get_spei_metrics_loop(spei_data, spei_nrs, return_timeseries = T, debug = T, debug_n = 10)
drought_series  <- sites_subset |> select(idp, drought_series) |> slice(1) |> unnest(drought_series)
time_series  <- sites_subset |> select(data_input) |> slice(1) |> unnest(data_input)

# Print different SPEI extractions
time_series |> 
  tidyr::pivot_longer(
    cols = starts_with("spei"),
    names_to = "spei",
    values_to = "spei_val"
  ) |> 
  mutate(
    spei = ifelse(spei == "spei_3", "spei_03", spei),
    spei = ifelse(spei == "spei_6", "spei_06", spei),
  ) |> 
  ggplot(aes(x=date, y=spei_val, colour=spei, group=spei)) +
  geom_hline(yintercept = 0, linetype=2)+
  geom_line() +
  geom_smooth(method="lm") +
  facet_wrap(~spei, ncol = 1) +
  scale_color_viridis_d(end = 0.7) 

# Plot drought timeseries
drought_series |> 
  ggplot() +
  geom_hline(yintercept = 0, linetype = 2) +
  geom_point(aes(x = date, y = value, color = day_is_drought)) +
  geom_line(aes(x = date, y = value))
