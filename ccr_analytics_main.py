import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

# 导入各个模块
from ccr_exposure_analytics import MarketFactorSimulator, DerivativePricer, CounterpartyExposureAnalyzer
from xva_engine import CreditCurve, XVACalculator
from sa_cva_capital import SACVACalculator
from technical_indicators import validate_garch_model

def run_ccr_analysis():
    print("====== 交易对手信用风险分析系统 ======")
    
    # 1. 设置基本参数
    print("\n1. 数据准备与市场因子模拟")
    ticker = 'SPY'  # 标的资产
    start_date = '2018-01-01'
    end_date = '2023-01-01'
    
    # 2. 获取市场数据与GARCH模型拟合
    print("\n2. GARCH模型验证")
    data = yf.download(ticker, start=start_date, end=end_date)
    returns = data['Close'].pct_change().dropna() * 100  # 放大100倍以解决尺度问题
    
    garch_result = validate_garch_model(returns, p=1, q=1, model_type='GARCH')
    print(f"GARCH(1,1) 模型结果: AIC={garch_result['aic']:.2f}, BIC={garch_result['bic']:.2f}")
    
    # 3. 模拟市场因子路径
    print("\n3. 模拟衍生品定价路径")
    
    # 调试打印 garch_result 的所有键，看看有哪些可用
    print(f"GARCH模型结果包含的键: {list(garch_result.keys())}")
    
    # 可能的替代键名，基于常见GARCH模型结果
    # 尝试获取波动率，如果键不存在则使用标准替代方法
    try:
        volatility = max(garch_result.get('mse', returns.std()), 0.001)
        
        # 如果返回的是数组，取均值
        if isinstance(volatility, (list, np.ndarray)):
            volatility = np.mean(volatility)
            
        # 确保波动率足够大
        volatility = max(volatility, 0.001)
    except:
        # 如果上述都失败，使用returns的标准差作为波动率
        volatility = max(returns.std(), 0.001)
    
    print(f"使用的波动率: {volatility:.6f}")
    simulator = MarketFactorSimulator([0.02], [volatility], np.array([[1.0]]))
    paths = simulator.simulate_paths(1000, 252)
    
    # 打印路径形状和特征
    print(f"模拟路径形状: {paths.shape}")
    print(f"路径第一行的标准差: {np.std(paths[0]):.6f}")
    print(f"路径第一行的最小值: {np.min(paths[0]):.6f}")
    print(f"路径第一行的最大值: {np.max(paths[0]):.6f}")
    
    # 4. 定价衍生品并计算风险暴露
    print("\n4. 计算交易对手风险暴露")
    pricer = DerivativePricer(valuation_date=datetime.now().strftime("%Y-%m-%d"))
    
    # 关键修复：提取第一个市场因子的所有路径数据 (从三维变为二维)
    rate_paths = paths[0]  # 提取第一个因子(利率)的所有路径
    print(f"利率路径形状: {rate_paths.shape}")
    
    swap_values = pricer.price_interest_rate_swap(0.03, rate_paths, 10000000, 5)
    
    # 检查掉期价值
    print(f"掉期价值形状: {swap_values.shape}")
    print(f"掉期价值的最大值: {np.max(np.abs(swap_values)):.2f}")
    print(f"掉期价值的平均值: {np.mean(swap_values):.2f}")
    
    analyzer = CounterpartyExposureAnalyzer()
    exposures = analyzer.calculate_exposure(swap_values)
    
    # 检查暴露计算
    print(f"暴露形状: {exposures.shape}")  
    print(f"最大暴露值: {np.max(exposures):.2f}")
    print(f"平均暴露值: {np.mean(exposures):.2f}")
    
    metrics = analyzer.compute_exposure_metrics(exposures)
    
    # 5. 计算XVA
    print("\n5. 计算XVA调整")
    # 创建信用曲线
    tenors = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
    survival_probs = [1.0, 0.995, 0.99, 0.98, 0.97, 0.95]
    credit_curve = CreditCurve(tenors, survival_probs)
    
    # 构建暴露曲线
    time_points = np.linspace(0, 5, 252)
    exposure_profile = dict(zip(time_points, metrics['EE']))
    
    # 计算XVA
    xva_calculator = XVACalculator()
    cva = xva_calculator.calculate_cva(exposure_profile, credit_curve)
    
    own_survival_probs = [1.0, 0.99, 0.98, 0.96, 0.94, 0.90]
    own_credit_curve = CreditCurve(tenors, own_survival_probs)
    dva = xva_calculator.calculate_dva(exposure_profile, own_credit_curve)
    
    fva = xva_calculator.calculate_fva(exposure_profile, 50)
    
    print(f"CVA: ${cva:.2f}")
    print(f"DVA: ${dva:.2f}")
    print(f"FVA: ${fva:.2f}")
    print(f"Total XVA: ${cva - dva + fva:.2f}")
    
    # 6. 计算SA-CVA资本
    print("\n6. SA-CVA资本计算")
    # 敏感性数据
    sensitivities = {
        'interest_rate': [
            {'currency': 'USD', 'tenor': '1Y', 'sensitivity': 50000},
            {'currency': 'USD', 'tenor': '5Y', 'sensitivity': 80000},
        ],
        'credit_spread': [
            {'counterparty': 'CP1', 'rating': 'A', 'sensitivity': 40000},
        ],
        'fx': [
            {'pair': 'USD/EUR', 'sensitivity': 120000},
        ]
    }
    
    sa_cva_calculator = SACVACalculator()
    capital_results = sa_cva_calculator.calculate_sa_cva_capital(sensitivities)
    
    print(f"Interest Rate Capital: ${capital_results['interest_rate_capital']:,.2f}")
    print(f"Credit Spread Capital: ${capital_results['credit_spread_capital']:,.2f}")
    print(f"FX Capital: ${capital_results['fx_capital']:,.2f}")
    print(f"Total SA-CVA Capital: ${capital_results['total_sa_cva_capital']:,.2f}")
    
    # 7. 生成图表和报告
    print("\n7. 图表生成")
    # time_points = np.linspace(0, 5, 252)
    time_points = np.linspace(0, 5, len(metrics['EE']))
    analyzer.plot_exposure_profile(metrics, "Interest Rate Swap Exposure Profile", time_points)
    
    sensitivity_results = xva_calculator.run_sensitivity_analysis(
        exposure_profile, credit_curve, 
        spread_shifts=[-50, -20, 0, 20, 50, 100]
    )
    xva_calculator.plot_sensitivity_analysis(sensitivity_results, 'CVA Sensitivity to Credit Spread Changes')
    
    sa_cva_calculator.plot_capital_breakdown(capital_results)
    
    print("\n分析完成! 查看生成的图表了解更多详情。")

if __name__ == "__main__":
    run_ccr_analysis()
