import numpy as np
from sklearn.linear_model import LinearRegression

# Dados de exemplo
temperaturas = np.array([20, 22, 25, 30]).reshape(-1, 1)
vendas = np.array([120, 150, 200, 300])

# Criando e treinando o modelo
modelo = LinearRegression()
modelo.fit(temperaturas, vendas)

# Previsão de exemplo
temp_teste = np.array([[28]])
previsao = modelo.predict(temp_teste)

print(f"Previsão de vendas para 28°C: {int(previsao[0])} sorvetes")
