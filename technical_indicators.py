import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from arch import arch_model
from sklearn.metrics import mean_squared_error

def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def calculate_rsi(data, window=14):
    # 计算价格变化
    delta = data['Close'].diff()
    
    # 计算涨跌
    gain = delta.copy()
    loss = delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    loss = abs(loss)
    
    # 计算平均涨跌
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    # 计算相对强度
    rs = avg_gain / avg_loss
    
    # 计算RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    # 计算EMA
    ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
    
    # 计算MACD线和信号线
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    
    # 计算MACD柱状图
    macd_histogram = macd_line - signal_line
    
    return macd_line, signal_line, macd_histogram

def calculate_bollinger_bands(data, window=20, num_std=2):
    # 计算移动平均线
    middle_band = data['Close'].rolling(window=window).mean()
    
    # 计算标准差
    std = data['Close'].rolling(window=window).std()
    
    # 计算上下轨
    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)
    
    return upper_band, middle_band, lower_band

def plot_indicators(data, ticker):
    plt.figure(figsize=(15, 12))
    
    # 绘制价格和布林带
    plt.subplot(3, 1, 1)
    plt.plot(data['Close'], label='Close Price')
    plt.plot(data['upper_band'], label='Upper Band', linestyle='--', alpha=0.7)
    plt.plot(data['middle_band'], label='Middle Band', linestyle='--', alpha=0.7)
    plt.plot(data['lower_band'], label='Lower Band', linestyle='--', alpha=0.7)
    plt.title(f'{ticker} Price with Bollinger Bands')
    plt.legend()
    
    # 绘制RSI
    plt.subplot(3, 1, 2)
    plt.plot(data['rsi'], label='RSI')
    plt.axhline(y=70, color='r', linestyle='-', alpha=0.5)
    plt.axhline(y=30, color='g', linestyle='-', alpha=0.5)
    plt.title('Relative Strength Index (RSI)')
    plt.ylabel('RSI')
    plt.legend()
    
    # 绘制MACD
    plt.subplot(3, 1, 3)
    plt.plot(data['macd_line'], label='MACD Line')
    plt.plot(data['signal_line'], label='Signal Line')
    plt.bar(data.index, data['macd_histogram'], label='MACD Histogram', alpha=0.5)
    plt.title('Moving Average Convergence Divergence (MACD)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{ticker}_technical_analysis.png')
    plt.show()

def validate_garch_model(returns, p=1, q=1, model_type='GARCH'):
    """验证不同GARCH模型的表现"""
    # 拟合模型
    if model_type == 'GARCH':
        model = arch_model(returns, vol='Garch', p=p, q=q)
    elif model_type == 'EGARCH':
        model = arch_model(returns, vol='EGARCH', p=p, q=q)
    elif model_type == 'GJR-GARCH':
        model = arch_model(returns, vol='GJRGARCH', p=p, q=q)
    
    result = model.fit(disp='off')
    
    # 预测波动率
    forecasts = result.forecast(horizon=10)
    forecast_vol = np.sqrt(forecasts.variance.values[-1, :])
    
    # 计算实际波动率(作为基准)
    actual_vol = returns.rolling(window=10).std().dropna().iloc[10:].values
    
    # 计算MSE
    mse = mean_squared_error(actual_vol[:len(forecast_vol)], forecast_vol)
    
    return {
        'model': model_type,
        'parameters': f'p={p}, q={q}',
        'mse': mse,
        'log_likelihood': result.loglikelihood,
        'aic': result.aic,
        'bic': result.bic
    }

if __name__ == "__main__":
    # 设置参数
    ticker = 'AAPL'
    start_date = '2022-01-01'
    end_date = '2023-01-01'
    
    # 获取数据
    data = get_stock_data(ticker, start_date, end_date)
    
    # 计算指标
    data['rsi'] = calculate_rsi(data)
    data['macd_line'], data['signal_line'], data['macd_histogram'] = calculate_macd(data)
    data['upper_band'], data['middle_band'], data['lower_band'] = calculate_bollinger_bands(data)
    
    # 去除NaN值
    data = data.dropna()
    
    # 绘制指标
    plot_indicators(data, ticker)
    
    # 输出最近的指标值
    last_date = data.index[-1].strftime('%Y-%m-%d')
    print(f"Technical Indicators for {ticker} as of {last_date}:")
    print(f"RSI: {data['rsi'].iloc[-1]:.2f}")
    print(f"MACD Line: {data['macd_line'].iloc[-1]:.2f}")
    print(f"Signal Line: {data['signal_line'].iloc[-1]:.2f}")
    print(f"Bollinger Bands:")
    print(f"  Upper: {data['upper_band'].iloc[-1]:.2f}")
    print(f"  Middle: {data['middle_band'].iloc[-1]:.2f}")
    print(f"  Lower: {data['lower_band'].iloc[-1]:.2f}") 