import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def get_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    returns = data.pct_change().dropna()
    return returns

def portfolio_annualized_performance(weights, returns):
    # 计算投资组合收益率
    portfolio_return = np.sum(returns.mean() * weights) * 252
    # 计算投资组合波动率
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return portfolio_return, portfolio_std

def negative_sharpe_ratio(weights, returns, risk_free_rate=0):
    p_return, p_std = portfolio_annualized_performance(weights, returns)
    return -(p_return - risk_free_rate) / p_std

def optimize_portfolio(returns):
    num_assets = len(returns.columns)
    args = (returns)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    initial_guess = num_assets * [1. / num_assets]
    
    result = minimize(negative_sharpe_ratio, initial_guess,
                     args=args, method='SLSQP',
                     bounds=bounds, constraints=constraints)
    
    return result

def plot_efficient_frontier(returns, optimized_weights):
    num_assets = len(returns.columns)
    num_portfolios = 10000
    
    # 生成随机权重
    all_weights = np.zeros((num_portfolios, num_assets))
    ret_arr = np.zeros(num_portfolios)
    vol_arr = np.zeros(num_portfolios)
    sharpe_arr = np.zeros(num_portfolios)
    
    for i in range(num_portfolios):
        # 生成随机权重
        weights = np.random.random(num_assets)
        weights = weights / np.sum(weights)
        all_weights[i,:] = weights
        
        # 计算收益率和波动率
        ret_arr[i] = np.sum(returns.mean() * weights) * 252
        vol_arr[i] = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        
        # 计算夏普比率
        sharpe_arr[i] = ret_arr[i] / vol_arr[i]
    
    # 获取最优投资组合结果
    max_sr_ret, max_sr_vol = portfolio_annualized_performance(optimized_weights, returns)
    
    # 绘图
    plt.figure(figsize=(12, 8))
    plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis', alpha=0.5)
    plt.colorbar(label='Sharpe Ratio')
    plt.scatter(max_sr_vol, max_sr_ret, c='red', s=100, marker='*', label='Optimal Portfolio')
    plt.title('Efficient Frontier')
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.legend()
    plt.savefig('efficient_frontier.png')
    plt.show()

if __name__ == "__main__":
    # 设置股票和日期
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA']
    start_date = '2018-01-01'
    end_date = '2023-01-01'
    
    # 获取数据
    returns = get_data(tickers, start_date, end_date)
    
    # 优化投资组合
    result = optimize_portfolio(returns)
    optimal_weights = result['x']
    
    # 输出最优权重
    print("Optimal Portfolio Weights:")
    for i, ticker in enumerate(tickers):
        print(f"{ticker}: {optimal_weights[i]:.4f}")
    
    # 计算最优投资组合性能
    optimal_return, optimal_std = portfolio_annualized_performance(optimal_weights, returns)
    optimal_sharpe = optimal_return / optimal_std
    
    print(f"\nExpected Annual Return: {optimal_return:.4f}")
    print(f"Expected Volatility: {optimal_std:.4f}")
    print(f"Sharpe Ratio: {optimal_sharpe:.4f}")
    
    # 绘制有效前沿
    plot_efficient_frontier(returns, optimal_weights) 