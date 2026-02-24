# predict_btc_from_eth_dict_api.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from chronos import BaseChronosPipeline, Chronos2Pipeline

# Загрузка 
model = Chronos2Pipeline.from_pretrained("./fm")
# Загрузка данных
btc_df = pd.read_csv('assets/BTC_USDT_5years_1h.csv')
eth_df = pd.read_csv('assets/ETH_USDT_5years_1h.csv')

btc_prices = btc_df['close'].values
eth_prices = eth_df['close'].values

# Параметры
context_days = 30
predict_days = 7
hours_in_day = 24
context_hours = context_days * hours_in_day
predict_hours = predict_days * hours_in_day

# Контекст для обоих рядов
eth_context = eth_prices[-context_hours:-predict_hours]
btc_context = btc_prices[-context_hours:-predict_hours]

# Формат для multivariate forecasting с list of dicts
inputs = [{
    "target": np.array([eth_context, btc_context])  # 2D array (2, context_hours)
}]

# Предсказываем
quantiles, mean = model.predict_quantiles(
    inputs,
    prediction_length=predict_hours,
    quantile_levels=[0.1, 0.5, 0.9]
)

# mean[0] - предсказания для первого (и единственного) элемента в списке
# mean[0][0] - предсказание ETH
# mean[0][1] - предсказание BTC  👈

btc_predicted = mean[0][1].numpy()
btc_actual = btc_prices[-predict_hours:]

# Результаты
results = pd.DataFrame({
    'hour': range(1, predict_hours + 1),
    'btc_predicted': btc_predicted,
    'btc_actual': btc_actual
})
results.to_csv('btc_from_eth_dict_api.csv', index=False)

mae = np.mean(np.abs(btc_predicted - btc_actual))
print(f"MAE: {mae:.2f}")

# Минимальный график
plt.figure(figsize=(12, 6))
plt.plot(results['hour'], results['btc_actual'], label='Реальность', color='blue')
plt.plot(results['hour'], results['btc_predicted'], label='Предсказание', color='red', linestyle='--')
plt.xlabel('Часы')
plt.ylabel('Цена BTC')
plt.title('BTC: предсказание vs реальность')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('btc_comparison.png')
plt.show()