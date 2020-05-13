from Escalador.Escalador import escalador_padrao, escalador_minmax
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

def print_columns(x):
    for d in x.T:
      print(f'media = {d.mean():+f} std = {d.std()}')


data = np.array([[0, 1, -2], [2, 1, 4], [4, 1, -2], [1, 2, 1]])


print('********************** Padrao *********************************')


print('Escalador do sklearn')

scaler = StandardScaler(with_mean=True, with_std=True)
scaled_data = scaler.fit_transform(data)

print(f'Dados escalados\n{scaled_data}')

print()
print_columns(scaled_data)
print()


print('Escalador proprio')
dados_escalado = escalador_padrao(data, f_media=True, f_std=True)

print(f'Dados escalados\n{dados_escalado}')

print()
print_columns(dados_escalado)
print()
print('***************************************************************')
print()


print('********************** MinMax *********************************')

print('Escalador do sklearn')

scaler = MinMaxScaler(feature_range= (1, 3))
scaled_data = scaler.fit_transform(data)

print(f'Dados escalados\n{scaled_data}')

print()

print('Escalador proprio')
dados_escalado = escalador_minmax(data, f_range = (1, 3))

print(f'Dados escalados\n{dados_escalado}')
print('***************************************************************')
