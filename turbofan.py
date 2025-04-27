# === PROYECTO PROFESIONAL TURBOFAN ENGINE DEGRADATION - LSTM ===

import pandas as pd
import numpy as np
import os
import time
import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dropout, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping

# CONFIGURACIONES
base_path = './'
graficos_path = './graficos/'
os.makedirs(graficos_path, exist_ok=True)

# SENSORES ÚTILES
sensores_utiles = [
    'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7',
    'sensor_8', 'sensor_11', 'sensor_15',
    'sensor_17', 'sensor_20', 'sensor_21'
]

columnas = [
    'unit_number', 'time_in_cycles',
    'operational_setting_1', 'operational_setting_2', 'operational_setting_3',
    'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5',
    'sensor_6', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10',
    'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15',
    'sensor_16', 'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20',
    'sensor_21'
]

# CARGA Y PREPROCESAMIENTO
print("\n📥 Cargando datasets de entrenamiento...")
df_all = pd.DataFrame()
for i in range(1, 5):
    path = os.path.join(base_path, f'train_FD00{i}.txt')
    df = pd.read_csv(path, sep=' ', header=None)
    df.dropna(axis=1, inplace=True)
    df.columns = columnas
    df['dataset_id'] = f'FD00{i}'
    ciclos_max = df.groupby('unit_number')['time_in_cycles'].transform('max')
    df['RUL'] = ciclos_max - df['time_in_cycles']
    df_all = pd.concat([df_all, df], ignore_index=True)

# Normalización
df_proc = df_all[sensores_utiles + ['unit_number', 'time_in_cycles', 'dataset_id', 'RUL']].copy()
df_proc['dataset_code'] = df_proc['dataset_id'].astype('category').cat.codes
scaler = MinMaxScaler()
df_proc[sensores_utiles] = scaler.fit_transform(df_proc[sensores_utiles])

# FUNCIONES

def crear_ventanas(df, ventana, features):
    X, y = [], []
    for motor_id in df['unit_number'].unique():
        for dataset in df['dataset_id'].unique():
            datos = df[(df['unit_number'] == motor_id) & (df['dataset_id'] == dataset)]
            datos = datos.sort_values('time_in_cycles')
            for i in range(len(datos) - ventana):
                X.append(datos.iloc[i:i+ventana][features].values)
                y.append(datos.iloc[i+ventana]['RUL'])
    return np.array(X), np.array(y)

def build_model(input_shape, units_lstm=64, dropout_rate=0.3, optimizer='adam', reg_l2=0.001):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Bidirectional(LSTM(units_lstm, return_sequences=True, kernel_regularizer=l2(reg_l2))))
    model.add(Dropout(dropout_rate))
    model.add(Bidirectional(LSTM(units_lstm // 2, kernel_regularizer=l2(reg_l2))))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units_lstm // 2, activation='relu'))
    model.add(Dense(1))
    opt = Adam() if optimizer == 'adam' else RMSprop()
    model.compile(loss='mean_squared_error', optimizer=opt)
    return model

# GRID SEARCH DE HIPERPARÁMETROS
print("\n⚙️ Ejecutando tuning de hiperparámetros...")
ventanas = [30, 50]
unidades = [64, 128]
dropouts = [0.2, 0.3]
reg_l2s = [0.0, 0.001]
optimizers = ['adam', 'rmsprop']

resultados = []
combinaciones = list(itertools.product(ventanas, unidades, dropouts, reg_l2s, optimizers))

for i, (ventana, units, dropout, reg_l2, opt) in enumerate(tqdm(combinaciones)):
    features_lstm = sensores_utiles + ['dataset_code']
    X, y = crear_ventanas(df_proc, ventana, features_lstm)
    model = build_model((X.shape[1], X.shape[2]), units, dropout, opt, reg_l2)
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(X, y, validation_split=0.1, epochs=20, batch_size=64, callbacks=[early_stop], verbose=0)
    val_loss = min(history.history['val_loss'])
    resultados.append({
        'ventana': ventana, 'units': units, 'dropout': dropout,
        'reg_l2': reg_l2, 'optimizer': opt, 'val_loss': val_loss
    })

# ENTRENAMIENTO FINAL
print("\n🏆 Entrenando modelo final con mejores hiperparámetros...")
mejor = pd.DataFrame(resultados).sort_values(by='val_loss').iloc[0]
X, y = crear_ventanas(df_proc, int(mejor['ventana']), sensores_utiles + ['dataset_code'])
final_model = build_model((X.shape[1], X.shape[2]), int(mejor['units']), mejor['dropout'], mejor['optimizer'], mejor['reg_l2'])
final_model.fit(X, y, validation_split=0.1, epochs=50, batch_size=64, callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])
final_model.save(os.path.join(base_path, 'modelo_lstm_final.h5'))
pd.DataFrame(resultados).to_csv(os.path.join(base_path, 'resultados_modelos.csv'), index=False)

# EVALUACIÓN Y GUARDADO DE MÉTRICAS
print("\n📊 Evaluando sobre datasets de test...")
metricas = []

for i in range(1, 5):
    dataset_id = f'FD00{i}'
    test_path = os.path.join(base_path, f'test_{dataset_id}.txt')
    rul_path = os.path.join(base_path, f'RUL_{dataset_id}.txt')

    df_test = pd.read_csv(test_path, sep=' ', header=None)
    df_test.dropna(axis=1, inplace=True)
    df_test.columns = columnas
    df_test['dataset_id'] = dataset_id
    df_test['dataset_code'] = i - 1
    df_test[sensores_utiles] = scaler.transform(df_test[sensores_utiles])

    motores_test = df_test['unit_number'].unique()
    X_test = []
    for motor_id in motores_test:
        motor_df = df_test[df_test['unit_number'] == motor_id].sort_values('time_in_cycles')
        ventana = int(mejor['ventana'])
        seq = motor_df.tail(ventana)
        if len(seq) < ventana:
            padding = np.zeros((ventana - len(seq), len(sensores_utiles)))
            padding_df = pd.DataFrame(padding, columns=sensores_utiles)
            padding_df['dataset_code'] = i - 1
            seq = pd.concat([padding_df, seq[sensores_utiles + ['dataset_code']]], ignore_index=True)
        else:
            seq = seq[sensores_utiles + ['dataset_code']]
        X_test.append(seq.values)

    X_test = np.array(X_test)
    rul_real = pd.read_csv(rul_path, header=None).values.flatten()
    rul_pred = final_model.predict(X_test).flatten()

    mae = mean_absolute_error(rul_real, rul_pred)
    rmse = np.sqrt(mean_squared_error(rul_real, rul_pred))

    metricas.append({
        'Dataset': dataset_id,
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2)
    })

    # Guardar gráfico
    plt.figure(figsize=(10, 4))
    plt.plot(rul_real, label='RUL Real', marker='o')
    plt.plot(rul_pred, label='RUL Predicho', marker='x')
    plt.title(f'Comparación RUL - {dataset_id}')
    plt.xlabel('Motor')
    plt.ylabel('RUL')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(graficos_path, f'{dataset_id}_grafico.png'))
    plt.close()

# Guardar métricas
pd.DataFrame(metricas).to_csv(os.path.join(base_path, 'metrics.csv'), index=False)
print("\n✅ Proyecto terminado. Modelos, resultados, métricas y gráficos guardados!")


