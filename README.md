# Turbofan-RUL-Prediction
Remaining useful life (RUL) prediction of turbofan engines using LSTM (NASA C-MAPSS)
This project implements a Deep Learning model based on Bidirectional LSTM networks to predict the Remaining Useful Life (RUL) of simulated aircraft engines, using NASA's C-MAPSS dataset.

📂 Dataset
Dataset used:
Turbofan Engine Degradation Simulation Dataset (C-MAPSS)
https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

All four scenarios were used: FD001, FD002, FD003, and FD004.

Each engine has multiple sensor measurements recorded throughout its operational life.

🛠️ Technologies Used
Python 3.12

TensorFlow / Keras

Scikit-learn

Matplotlib

tqdm

Development environment: Visual Studio Code (VSCode).

🚀 Project Overview
Data loading and preprocessing of sensor readings.

Normalization of the sensor data.

Creation of sequential windows for training the LSTM model.

Automated grid search to optimize model architecture (window size, LSTM units, regularization, etc.).

Final training using the best set of hyperparameters.

Evaluation on real-world test sets (FD001–FD004).

Visualization of real vs predicted RUL.

📈 Results
Metrics obtained: MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error) for each dataset.

Charts: Visual comparison between real and predicted RUL for each test engine.

Sample prediction results:
Dataset | MAE (cycles) | RMSE (cycles)
FD001   |   19.29      | 25.24
FD002   |   21.75      | 30.68
FD003   |   30.15      | 41.93
FD004   |   29.18      | 37.86
