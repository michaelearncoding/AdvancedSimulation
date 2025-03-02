import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 获取公开数据
def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# 简单的移动平均线策略
def moving_average_strategy(data, short_window=20, long_window=50):
    # 计算移动平均线
    data['short_ma'] = data['Close'].rolling(window=short_window).mean()
    data['long_ma'] = data['Close'].rolling(window=long_window).mean()
    
    # 生成交易信号
    data['signal'] = 0
    data.loc[data['short_ma'] > data['long_ma'], 'signal'] = 1
    data.loc[data['short_ma'] < data['long_ma'], 'signal'] = -1
    
    # 计算每日回报
    data['returns'] = data['Close'].pct_change()
    data['strategy_returns'] = data['signal'].shift(1) * data['returns']
    
    return data

# 计算策略绩效
def calculate_performance(data):
    # 累积回报
    data['cumulative_returns'] = (1 + data['returns']).cumprod()
    data['strategy_cumulative_returns'] = (1 + data['strategy_returns']).cumprod()
    
    # 年化收益率
    days = (data.index[-1] - data.index[0]).days
    annual_return = (data['strategy_cumulative_returns'].iloc[-1] ** (365/days)) - 1
    
    # 最大回撤
    strategy_cumulative = data['strategy_cumulative_returns']
    running_max = strategy_cumulative.cummax()
    drawdown = (strategy_cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # 计算夏普比率 (假设无风险利率为0)
    sharpe_ratio = np.sqrt(252) * (data['strategy_returns'].mean() / data['strategy_returns'].std())
    
    return {
        'Annual Return': annual_return,
        'Max Drawdown': max_drawdown,
        'Sharpe Ratio': sharpe_ratio
    }

# 可视化结果
def visualize_results(data):
    plt.figure(figsize=(12, 8))
    
    # 价格和移动平均线
    plt.subplot(2, 1, 1)
    plt.plot(data['Close'], label='Price')
    plt.plot(data['short_ma'], label=f'Short MA ({short_window})')
    plt.plot(data['long_ma'], label=f'Long MA ({long_window})')
    plt.title('Price and Moving Averages')
    plt.legend()
    
    # 策略收益与基准收益对比
    plt.subplot(2, 1, 2)
    plt.plot(data['cumulative_returns'], label='Buy and Hold')
    plt.plot(data['strategy_cumulative_returns'], label='Strategy')
    plt.title('Strategy Performance')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('strategy_performance.png')
    plt.show()

if __name__ == "__main__":
    # 参数设置
    ticker = 'SPY'  # 股票代码
    start_date = '2020-01-01'
    end_date = '2023-01-01'
    short_window = 20
    long_window = 50
    
    # 回测流程
    data = get_stock_data(ticker, start_date, end_date)
    data = moving_average_strategy(data, short_window, long_window)
    performance = calculate_performance(data)
    
    # 输出结果
    print(f"Strategy Performance for {ticker}:")
    for metric, value in performance.items():
        print(f"{metric}: {value:.4f}")
    
    visualize_results(data) 