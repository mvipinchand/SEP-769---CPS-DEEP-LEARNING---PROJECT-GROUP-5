# SEP-769: CPS-DEEP-LEARNING PROJECT: GROUP-5
Anomaly Detection and Forecasting for Solar Power Plant Performance across Deep Learning Models

Project Title:
Anomaly Detection and Forecasting of Solar Power Plant Performance across Deep Learning Models

Authors:
Purva Singh, Suraj Ramesh, Chris Xavier Mathias, Vipin Chandran Muthirikkaparambil

Overview:
This repository contains code and the necessary instructions to reproduce the results presented in the final report. The project performs:
  - Short-term forecasting of AC_POWER using deep learning models (FFNN, CNN, LSTM, and CNN-LSTM hybrid).
  - Unsupervised anomaly detection on inverter daily profiles using an autoencoder and localization using Dynamic Time Warping (DTW).

Prerequisites
-------------
1. Python 3.8+ (recommend using a virtual environment)
2. Recommended libraries (can be installed via requirements.txt if provided):
   - numpy
   - pandas
   - scikit-learn
   - matplotlib
   - tensorflow (or keras, matching the notebook implementation)
   - seaborn (if used for any auxiliary plotting)
   - dtw-related utilities (if custom; otherwise standard implementations)
   - jupyter

(If a `requirements.txt` is not provided, install the above manually, e.g.:
   pip install numpy pandas scikit-learn matplotlib tensorflow
)

Dataset Download
----------------
The dataset used in this project was originally sourced from Kaggle and consists of real-world solar power generation and weather sensor data collected over 34 days from two solar plants in India.

You need to manually download the following four CSV files and place them in the expected `data/` directory (create if not present):

  1. `Plant_1_Generation_Data.csv`
  2. `Plant_1_Weather_Sensor_Data.csv`
  3. `Plant_2_Generation_Data.csv`
  4. `Plant_2_Weather_Sensor_Data.csv`

Steps to obtain the data:
  - Go to Kaggle (https://www.kaggle.com) and search for the solar power generation / inverter-level dataset referenced in the report.
    * (If the original dataset link is missing in this repo, you may contact the report authors or search for “solar power plant inverter-level generation weather data India Kaggle” to locate the matching dataset.)
  - Download the four CSV files listed above.
  - Place them in a folder named `data/` at the repository root so paths look like:
      data/Plant_1_Generation_Data.csv
      data/Plant_1_Weather_Sensor_Data.csv
      data/Plant_2_Generation_Data.csv
      data/Plant_2_Weather_Sensor_Data.csv

Expected CSV Columns (should match report description):
  - Generation files: `DATE_TIME`, `PLANT_ID`, `SOURCE_KEY`, `DC_POWER`, `AC_POWER`, `DAILY_YIELD`, `TOTAL_YIELD`
  - Weather files: `DATE_TIME`, `PLANT_ID`, `SOURCE_KEY`, `AMBIENT_TEMPERATURE`, `MODULE_TEMPERATURE`, `IRRADIATION`

Reproducing the Results
-----------------------

1. **Data Preprocessing**
   - Clean column names (trim spaces, unify casing).
   - Parse `DATE_TIME` into datetime objects.
   - Sort by time and merge generation & weather data per plant via inner join on `DATE_TIME`.
   - Handle missing values:
       * For forecasting: interpolate power readings per inverter; forward-fill cumulative yields; drop weather rows with missing critical features.
       * For anomaly detection: reshape inverter-day profiles into 96-slot vectors; drop rows with >10 consecutive missing slots; otherwise interpolate and fill edges.
   - Feature engineering for forecasting includes aggregating to 15-minute plant-level statistics, cyclic hour encoding, and use of recent AC_POWER as autoregressive input.
   - Normalization (Min-Max) is applied appropriately (per row for anomaly detection, across features for forecasting).

2. **Forecasting Models (FFNN, CNN, LSTM, CNN-LSTM)**
   - Windowing: Create 96-timestep input windows (24h at 15-min intervals). FFNN uses flattened windows; others preserve sequence shape.
   - Train/test split: 80/20.
   - Scaling: Apply MinMax scaling to inputs and targets.
   - Model-specific details:
       * FFNN: Dense layers with ReLU, dropout, Adam optimizer, early stopping.
       * CNN: 1D Conv layers, max-pooling, Huber loss (delta=1500), Adam optimizer.
       * LSTM: Stacked LSTM (64 then 32 units), Nadam optimizer (learning rate 0.005), MSE loss.
       * Hybrid: Conv1D front-end + stacked LSTM, Nadam + Huber loss.
   - Train with early stopping and validation split (10% of training data).
   - Evaluate on test set; compute RMSE and MAE; inverse-transform outputs for plotting.

3. **Anomaly Detection**
   - Autoencoder architecture:
       * Encoder: Dense layers [512, 256, 128, 64] with ReLU + dropout (20%).
       * Decoder: Mirror encoder [64, 128, 256, 512] ending in sigmoid output.
       * Loss: MSE; optimizer: Adam (lr=0.001).
   - Train on normalized inverter-day profiles.
   - Compute reconstruction error per sample; set anomaly threshold as mean + 3*std of error distribution (≈0.0235).
   - Flag samples exceeding threshold as anomalies.

4. **Localization via DTW**
   - Split each 96-slot profile into 24 one-hour segments (4 slots each).
   - Compute DTW distance between original and reconstructed profiles segment-wise.
   - For each anomalous profile, compute 95th percentile of segment-wise DTW errors as local threshold.
   - Segments exceeding this are localized anomalies.

Running the Code
----------------
Assuming the notebooks are named:
  - `CPS_DL_GROUP_5_Project_Solar_Power_Plant_Forecasting_Source_Code.ipynb`
  - `CPS_DL_GROUP_5_Project_Solar_Power_Plant_Anomaly_Detection_Source_Code.ipynb`
