import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
import vectorbt as vbt

from chronos import Chronos2Pipeline

# Загрузка модели
is_f = False
if is_f:
    model = Chronos2Pipeline.from_pretrained("./fm")
else:
    model = Chronos2Pipeline.from_pretrained("amazon/chronos-2")

# Загрузка данных
btc_df = pd.read_csv('assets/BTC_USDT_5years_1h.csv')
doge_df = pd.read_csv('assets/DOGE_USDT_5years_1h.csv')
eth_df =  pd.read_csv('assets/ETH_USDT_5years_1h.csv')
sol_df = pd.read_csv('assets/SOL_USDT_5years_1h.csv')
# Добавляем timestamp
btc_df['timestamp'] = pd.date_range(end=pd.Timestamp.now(), periods=len(btc_df), freq='h')
doge_df['timestamp'] = pd.date_range(end=pd.Timestamp.now(), periods=len(doge_df), freq='h')
eth_df['timestamp'] = pd.date_range(end=pd.Timestamp.now(), periods=len(eth_df), freq='h')
sol_df['timestamp'] = pd.date_range(end=pd.Timestamp.now(), periods=len(sol_df), freq='h')


# Параметры
context_days = 1200
predict_days = 10
hours_in_day = 24
obs = 100
data = np.zeros(obs)
for z in range(1, 100):
    shift = z * hours_in_day
    context_hours = context_days * hours_in_day
    predict_hours = predict_days * hours_in_day

    # Индексы
    context_end = -predict_hours-shift
    context_start = - (context_hours + predict_hours +shift)

    # Данные для контекста
    btc_context = btc_df['close'].values[context_start:context_end]
    doge_context = doge_df['close'].values[context_start:context_end]
    eth_context = eth_df['close'].values[context_start:context_end]
    sol_context = sol_df['close'].values[context_start:context_end]
    # Будущие значения doge
    doge_future = doge_df['close'].values[-predict_hours-shift:-shift]
    eth_future = eth_df['close'].values[-predict_hours-shift:-shift]
    sol_future = sol_df['close'].values[-predict_hours-shift:-shift]

    # Входные данные
    inputs = [{
    "target": torch.tensor(btc_context, dtype=torch.float32),
    "past_covariates": {
        "doge": torch.tensor(doge_context, dtype=torch.float32),
        "eth": torch.tensor(eth_context, dtype=torch.float32),
        "sol": torch.tensor(sol_context, dtype=torch.float32)
    },
    "future_covariates": {
           "doge": torch.tensor(doge_future, dtype=torch.float32),
            "eth": torch.tensor(eth_future, dtype=torch.float32),
            "sol": torch.tensor(sol_future, dtype=torch.float32)
    }
    }]

    # Предсказание (без квантилей)
    quantiles, mean = model.predict_quantiles(
    inputs,
    prediction_length=predict_hours,
    quantile_levels=[0.5]  # только медиана
    )

    # Предсказание BTC
    btc_predicted = mean[0].numpy().flatten()
    btc_actual = btc_df['close'].values[-predict_hours-shift:-shift]
    btc_timestamps = btc_df['timestamp'].values[-predict_hours-shift:-shift]

    # Сигналы (239 значений - предсказания изменений)
    signals = np.where(btc_predicted[:-1] > btc_predicted[1:], 1, -1)

    # Для портфеля нужно 240 значений (как цены)
    # Добавляем первый сигнал (или последний) чтобы сравнять длину
    signals_full = np.append(signals[0], signals)  # или np.append(signals, signals[-1])

    # Или используем другой подход - сигнал на основе сравнения с текущей ценой
    signals = np.where(btc_predicted > btc_actual, 1, -1)  # сразу 240 значений

    # Создаем портфель
    pf = vbt.Portfolio.from_signals(
        btc_actual,
        entries=signals == 1,
        exits=signals == -1,
        init_cash=10000,
        freq='h'
    )

    # print(f"Total Return: {pf.total_return():.2%}")
    # print(f"Sharpe Ratio: {pf.sharpe_ratio():.2f}")
    # print(f"Max Drawdown: {pf.max_drawdown():.2%}")
    data[z-1] = pf.total_return()
print(data.mean())
