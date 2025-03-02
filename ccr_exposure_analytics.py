import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

class MarketFactorSimulator:
    """市场风险因子蒙特卡洛模拟引擎"""
    def __init__(self, initial_values, volatilities, correlations, dt=1/252):
        self.initial_values = np.array(initial_values)
        self.volatilities = np.array(volatilities)
        self.correlation_matrix = np.array(correlations)
        self.dt = dt
        
        # 计算协方差矩阵的Cholesky分解
        self.cholesky = np.linalg.cholesky(self.correlation_matrix)
    
    def simulate_paths(self, n_paths, n_steps):
        """生成风险因子路径"""
        n_factors = len(self.initial_values)
        paths = np.zeros((n_factors, n_paths, n_steps+1))
        
        # 设置初始值
        for i in range(n_factors):
            paths[i, :, 0] = self.initial_values[i]
        
        # 生成相关的随机数
        for t in range(1, n_steps+1):
            # 生成独立的正态随机数
            z = np.random.normal(0, 1, size=(n_factors, n_paths))
            
            # 应用相关性结构
            correlated_z = np.dot(self.cholesky, z)
            
            # 计算下一时间步的值
            for i in range(n_factors):
                # 几何布朗运动
                drift = 0  # 可以根据需要添加漂移项
                diffusion = self.volatilities[i] * np.sqrt(self.dt) * correlated_z[i]
                paths[i, :, t] = paths[i, :, t-1] * np.exp(drift + diffusion)
        
        return paths

class DerivativePricer:
    """衍生品定价模块"""
    def __init__(self, valuation_date):
        self.valuation_date = valuation_date
    
    def price_interest_rate_swap(self, fixed_rate, floating_paths, notional, term_years):
        """
        计算利率互换的价值
        
        Parameters:
        -----------
        fixed_rate: float
            固定利率（年化）
        floating_paths: ndarray
            浮动利率路径模拟
        notional: float
            名义本金
        term_years: int
            合约期限（年）
            
        Returns:
        --------
        ndarray
            每个模拟路径的掉期价值
        """
        # 获取路径数量和时间步数
        n_paths, n_steps = floating_paths.shape
        
        # 初始化价值数组
        values = np.zeros((n_paths, n_steps))
        
        # 设置付款频率（假设每年两次）
        payments_per_year = 2
        payment_interval = int(n_steps / (term_years * payments_per_year))
        
        # 计算每个时间点的掉期价值
        for t in range(n_steps):
            # 剩余时间（年）
            remaining_time = term_years - (t / n_steps * term_years)
            
            # 如果已经到期，价值为0
            if remaining_time <= 0:
                continue
            
            # 计算剩余付款次数
            remaining_payments = max(int(remaining_time * payments_per_year), 0)
            
            # 对每次未来付款计算现值
            for i in range(remaining_payments):
                payment_time = (i + 1) / payments_per_year  # 以年为单位的付款时间
                payment_step = min(t + int(payment_time * n_steps / term_years), n_steps - 1)
                
                # 简单折现因子
                discount_factor = np.exp(-floating_paths[:, t] * payment_time)
                
                # 固定利率支付减去浮动利率支付
                # 使用向量化操作确保形状匹配
                fixed_payment = notional * fixed_rate / payments_per_year
                floating_payment = notional * floating_paths[:, payment_step] / payments_per_year
                
                # 累加到总价值中
                values[:, t] += (fixed_payment - floating_payment) * discount_factor
        
        return values

class CounterpartyExposureAnalyzer:
    """交易对手风险暴露分析"""
    def __init__(self):
        self.exposures = {}
    
    def calculate_exposure(self, contract_values):
        """
        计算交易对手信用风险暴露
        
        Parameters:
        -----------
        contract_values: ndarray
            合约价值数组，形状为(n_scenarios, n_timesteps)
            
        Returns:
        --------
        ndarray
            暴露值数组，形状与输入相同
        """
        # 检查输入数据
        print(f"计算暴露：输入合约价值类型: {type(contract_values)}")
        print(f"合约价值统计: 最大={np.max(contract_values):.2f}, 最小={np.min(contract_values):.2f}")
        
        # 信用风险暴露是合约价值的正部分 (max(V, 0))
        # 关键修复：确保我们取的是正暴露
        exposures = np.maximum(contract_values, 0)
        
        # 检查输出结果
        print(f"计算出的暴露值: 最大={np.max(exposures):.2f}, 非零比例={np.mean(exposures > 0)*100:.2f}%")
        
        return exposures
    
    def compute_exposure_metrics(self, exposure_paths, confidence_level=0.95):
        """计算风险暴露指标"""
        n_paths, n_steps = exposure_paths.shape
        
        # 计算每个时间步的预期暴露
        expected_exposure = np.mean(exposure_paths, axis=0)
        
        # 计算潜在未来暴露 (PFE)
        pfe = np.percentile(exposure_paths, confidence_level*100, axis=0)
        
        # 计算有效预期暴露 (EEE)
        effective_ee = np.maximum.accumulate(expected_exposure)
        
        # 计算有效预期正暴露 (EEPE)
        eepe = np.mean(effective_ee)
        
        return {
            'EE': expected_exposure,
            'PFE': pfe,
            'EEE': effective_ee,
            'EEPE': eepe
        }
    
    def plot_exposure_profile(self, metrics, title, time_points):
        """绘制风险暴露曲线"""
        plt.figure(figsize=(12, 6))
        
        plt.plot(time_points, metrics['EE'], label='Expected Exposure (EE)')
        plt.plot(time_points, metrics['PFE'], label='Potential Future Exposure (PFE)')
        plt.plot(time_points, metrics['EEE'], label='Effective Expected Exposure (EEE)')
        
        plt.axhline(y=metrics['EEPE'], color='r', linestyle='--', 
                    label=f'Effective Expected Positive Exposure (EEPE): {metrics["EEPE"]:.2f}')
        
        plt.title(title)
        plt.xlabel('Time (Years)')
        plt.ylabel('Exposure')
        plt.legend()
        plt.grid(True)
        plt.savefig('exposure_profile.png')
        plt.show()

# 使用示例
if __name__ == "__main__":
    # 设置市场风险因子
    initial_rates = [0.02]  # 初始利率
    volatilities = [0.2]    # 波动率
    correlations = np.array([[1.0]])  # 相关性矩阵
    
    # 创建模拟器
    simulator = MarketFactorSimulator(initial_rates, volatilities, correlations)
    
    # 模拟利率路径
    n_paths = 1000
    n_steps = 252  # 一年的交易日
    paths = simulator.simulate_paths(n_paths, n_steps)
    
    # 创建定价器
    pricer = DerivativePricer(valuation_date="2023-01-01")
    
    # 定价利率互换
    notional = 1000000  # 名义本金
    fixed_rate = 0.025  # 固定利率
    remaining_time = 5  # 剩余期限（年）
    
    # 提取利率路径
    rate_paths = paths[0]
    
    # 计算互换的MTM值
    swap_values = pricer.price_interest_rate_swap(fixed_rate, rate_paths, notional, remaining_time)
    
    # 创建暴露分析器
    exposure_analyzer = CounterpartyExposureAnalyzer()
    
    # 计算暴露（不考虑担保品）
    exposures = exposure_analyzer.calculate_exposure(swap_values)
    
    # 计算暴露指标
    metrics = exposure_analyzer.compute_exposure_metrics(exposures)
    
    # 绘制暴露曲线
    time_points = np.linspace(0, remaining_time, n_steps)
    exposure_analyzer.plot_exposure_profile(metrics, "Interest Rate Swap Exposure Profile", time_points)
    
    # 计算有担保品的暴露
    collateralized_exposures = exposure_analyzer.calculate_exposure(
        swap_values, threshold=100000, collateral_lag=5
    )
    
    # 计算有担保品暴露指标
    collateralized_metrics = exposure_analyzer.compute_exposure_metrics(collateralized_exposures)
    
    # 绘制有担保品暴露曲线
    exposure_analyzer.plot_exposure_profile(
        collateralized_metrics, 
        "Interest Rate Swap Exposure Profile (With Collateral)", 
        time_points
    ) 