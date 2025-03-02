from data_fetcher import DataFetcher
from strategy import Strategy
from backtest_engine import BacktestEngine
from performance_analyzer import PerformanceAnalyzer
from visualizer import Visualizer

def main():
    # 初始化数据获取器
    data_fetcher = DataFetcher(source="binance", 
                              symbol="BTCUSDT",
                              timeframe="1h",
                              start_date="2022-01-01",
                              end_date="2023-01-01")
    
    # 获取历史数据
    historical_data = data_fetcher.fetch_data()
    
    # 初始化策略
    strategy = Strategy(name="MovingAverageCrossover", 
                       params={"short_window": 20, "long_window": 50})
    
    # 初始化回测引擎
    backtest = BacktestEngine(data=historical_data, 
                             strategy=strategy,
                             initial_capital=10000,
                             commission=0.001)
    
    # 运行回测
    results = backtest.run()
    
    # 分析性能
    analyzer = PerformanceAnalyzer(results)
    performance_metrics = analyzer.calculate_metrics()
    
    # 可视化结果
    visualizer = Visualizer(results, performance_metrics)
    visualizer.plot_equity_curve()
    visualizer.plot_drawdowns()
    visualizer.plot_trade_analysis()
    
    print(performance_metrics)

if __name__ == "__main__":
    main() 