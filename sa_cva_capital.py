import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

class SACVACalculator:
    """标准化CVA资本计算器"""
    
    def __init__(self):
        # 预定义的风险权重（根据巴塞尔III标准）
        self.interest_rate_weights = {
            'USD': {'0.5Y': 0.015, '1Y': 0.015, '3Y': 0.012, '5Y': 0.01, '10Y': 0.009},
            'EUR': {'0.5Y': 0.016, '1Y': 0.016, '3Y': 0.013, '5Y': 0.011, '10Y': 0.01},
            'JPY': {'0.5Y': 0.014, '1Y': 0.014, '3Y': 0.011, '5Y': 0.009, '10Y': 0.008}
        }
        
        self.credit_spread_weights = {
            'AAA': 0.0038,
            'AA': 0.0042,
            'A': 0.0057,
            'BBB': 0.0097,
            'BB': 0.0198,
            'B': 0.0318,
            'CCC': 0.0478
        }
        
        self.fx_weights = {
            'USD/EUR': 0.04,
            'USD/JPY': 0.04,
            'EUR/JPY': 0.04
        }
        
        # 相关性参数
        self.rho_between_tenors = 0.99  # 期限间相关性
        self.rho_between_currencies = 0.5  # 货币间相关性
        self.rho_between_risk_types = 0.5  # 风险类型间相关性
    
    def calculate_delta_sensitivity(self, derivative_value, risk_factor, shift=0.0001):
        """
        计算Delta敏感性
        
        参数:
            derivative_value: 衍生品定价函数
            risk_factor: 风险因子当前值
            shift: 风险因子移动量
        """
        base_value = derivative_value(risk_factor)
        shifted_value = derivative_value(risk_factor + shift)
        
        return (shifted_value - base_value) / shift
    
    def calculate_vega_sensitivity(self, derivative_value, volatility, shift=0.01):
        """计算Vega敏感性"""
        base_value = derivative_value(volatility)
        shifted_value = derivative_value(volatility + shift)
        
        return (shifted_value - base_value) / shift
    
    def calculate_risk_weighted_sensitivity(self, sensitivity, risk_weight):
        """计算风险加权敏感性"""
        return sensitivity * risk_weight
    
    def calculate_kb(self, weighted_sensitivities, correlations):
        """
        计算风险类别内的资本要求
        
        参数:
            weighted_sensitivities: 风险加权敏感性列表
            correlations: 相关性矩阵
        """
        n = len(weighted_sensitivities)
        kb = 0
        
        # 第一项：加权敏感性的平方和
        sum_squares = np.sum(np.square(weighted_sensitivities))
        
        # 第二项：加权相关敏感性的乘积
        cross_sum = 0
        for i in range(n):
            for j in range(i+1, n):
                cross_sum += weighted_sensitivities[i] * weighted_sensitivities[j] * correlations[i, j]
        
        kb = np.sqrt(sum_squares + 2 * cross_sum)
        return kb
    
    def calculate_sa_cva_capital(self, sensitivities_data):
        """
        计算SA-CVA资本总和
        
        参数:
            sensitivities_data: 包含各种风险敏感性的字典
            {
                'interest_rate': [{'currency': 'USD', 'tenor': '5Y', 'sensitivity': 10000}, ...],
                'credit_spread': [{'counterparty': 'CP1', 'rating': 'A', 'sensitivity': 5000}, ...],
                'fx': [{'pair': 'USD/EUR', 'sensitivity': 20000}, ...]
            }
        """
        # 计算各风险类别的加权敏感性
        ir_weighted_sens = []
        ir_buckets = []
        
        for ir_sens in sensitivities_data.get('interest_rate', []):
            currency = ir_sens['currency']
            tenor = ir_sens['tenor']
            sensitivity = ir_sens['sensitivity']
            
            if currency in self.interest_rate_weights and tenor in self.interest_rate_weights[currency]:
                weight = self.interest_rate_weights[currency][tenor]
                weighted = self.calculate_risk_weighted_sensitivity(sensitivity, weight)
                ir_weighted_sens.append(weighted)
                ir_buckets.append((currency, tenor))
        
        credit_weighted_sens = []
        for cs_sens in sensitivities_data.get('credit_spread', []):
            rating = cs_sens['rating']
            sensitivity = cs_sens['sensitivity']
            
            if rating in self.credit_spread_weights:
                weight = self.credit_spread_weights[rating]
                weighted = self.calculate_risk_weighted_sensitivity(sensitivity, weight)
                credit_weighted_sens.append(weighted)
        
        fx_weighted_sens = []
        for fx_sens in sensitivities_data.get('fx', []):
            pair = fx_sens['pair']
            sensitivity = fx_sens['sensitivity']
            
            if pair in self.fx_weights:
                weight = self.fx_weights[pair]
                weighted = self.calculate_risk_weighted_sensitivity(sensitivity, weight)
                fx_weighted_sens.append(weighted)
        
        # 构建相关性矩阵
        n_ir = len(ir_weighted_sens)
        ir_corr = np.ones((n_ir, n_ir))
        
        for i in range(n_ir):
            for j in range(i+1, n_ir):
                if ir_buckets[i][0] == ir_buckets[j][0]:  # 同一货币，不同期限
                    ir_corr[i, j] = ir_corr[j, i] = self.rho_between_tenors
                else:  # 不同货币
                    ir_corr[i, j] = ir_corr[j, i] = self.rho_between_currencies
        
        # 计算利率风险的资本
        ir_capital = self.calculate_kb(ir_weighted_sens, ir_corr) if ir_weighted_sens else 0
        
        # 简化处理信用利差和外汇风险（实际应用中需要更详细的相关性结构）
        n_credit = len(credit_weighted_sens)
        credit_corr = 0.5 * np.ones((n_credit, n_credit)) + 0.5 * np.eye(n_credit)
        credit_capital = self.calculate_kb(credit_weighted_sens, credit_corr) if credit_weighted_sens else 0
        
        n_fx = len(fx_weighted_sens)
        fx_corr = 0.6 * np.ones((n_fx, n_fx)) + 0.4 * np.eye(n_fx)
        fx_capital = self.calculate_kb(fx_weighted_sens, fx_corr) if fx_weighted_sens else 0
        
        # 计算风险类型间的相关性调整
        rho = self.rho_between_risk_types
        cross_term = 2 * rho * (ir_capital * credit_capital + ir_capital * fx_capital + credit_capital * fx_capital)
        
        # 总资本要求
        total_capital = np.sqrt(ir_capital**2 + credit_capital**2 + fx_capital**2 + cross_term)
        
        return {
            'interest_rate_capital': ir_capital,
            'credit_spread_capital': credit_capital,
            'fx_capital': fx_capital,
            'total_sa_cva_capital': total_capital
        }
    
    def plot_capital_breakdown(self, capital_results):
        """绘制资本构成明细"""
        categories = ['Interest Rate', 'Credit Spread', 'FX', 'Total SA-CVA']
        values = [
            capital_results['interest_rate_capital'],
            capital_results['credit_spread_capital'],
            capital_results['fx_capital'],
            capital_results['total_sa_cva_capital']
        ]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(categories[:-1], values[:-1], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        plt.axhline(y=values[-1], color='r', linestyle='-', label='Total SA-CVA Capital')
        
        # 添加数据标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'${int(height):,}', ha='center', va='bottom')
        
        plt.title('SA-CVA Capital Breakdown')
        plt.ylabel('Capital Amount ($)')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig('sa_cva_capital_breakdown.png')
        plt.show()
    
    def run_scenario_analysis(self, base_sensitivities, scenarios):
        """
        运行情景分析
        
        参数:
            base_sensitivities: 基准敏感性数据
            scenarios: 情景列表，每个情景是一个敏感性调整因子
        """
        results = []
        
        # 计算基准资本
        base_capital = self.calculate_sa_cva_capital(base_sensitivities)
        results.append({
            'scenario': 'Base Case',
            'total_capital': base_capital['total_sa_cva_capital'],
            'breakdown': base_capital
        })
        
        # 针对每个情景计算资本
        for scenario_name, adjustments in scenarios.items():
            # 复制基准敏感性数据
            scenario_sens = {k: v.copy() for k, v in base_sensitivities.items()}
            
            # 应用调整
            for risk_type, factor in adjustments.items():
                if risk_type in scenario_sens:
                    for i in range(len(scenario_sens[risk_type])):
                        scenario_sens[risk_type][i]['sensitivity'] *= factor
            
            # 计算情景资本
            scenario_capital = self.calculate_sa_cva_capital(scenario_sens)
            
            results.append({
                'scenario': scenario_name,
                'total_capital': scenario_capital['total_sa_cva_capital'],
                'breakdown': scenario_capital
            })
        
        return results
    
    def plot_scenario_comparison(self, scenario_results):
        """绘制情景比较结果"""
        scenarios = [result['scenario'] for result in scenario_results]
        capitals = [result['total_capital'] for result in scenario_results]
        
        plt.figure(figsize=(12, 6))
        plt.bar(scenarios, capitals, color='skyblue')
        
        # 添加数据标签
        for i, v in enumerate(capitals):
            plt.text(i, v + 10, f'${int(v):,}', ha='center')
        
        plt.title('SA-CVA Capital Under Different Scenarios')
        plt.ylabel('Capital Amount ($)')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('sa_cva_scenario_comparison.png')
        plt.show()

# 使用示例
if __name__ == "__main__":
    # 创建SA-CVA计算器
    calculator = SACVACalculator()
    
    # 示例敏感性数据
    sensitivities = {
        'interest_rate': [
            {'currency': 'USD', 'tenor': '1Y', 'sensitivity': 50000},
            {'currency': 'USD', 'tenor': '5Y', 'sensitivity': 80000},
            {'currency': 'EUR', 'tenor': '3Y', 'sensitivity': 60000}
        ],
        'credit_spread': [
            {'counterparty': 'CP1', 'rating': 'A', 'sensitivity': 40000},
            {'counterparty': 'CP2', 'rating': 'BBB', 'sensitivity': 70000},
            {'counterparty': 'CP3', 'rating': 'BB', 'sensitivity': 30000}
        ],
        'fx': [
            {'pair': 'USD/EUR', 'sensitivity': 120000},
            {'pair': 'USD/JPY', 'sensitivity': 90000}
        ]
    }
    
    # 计算SA-CVA资本
    capital_results = calculator.calculate_sa_cva_capital(sensitivities)
    
    print("SA-CVA Capital Results:")
    for key, value in capital_results.items():
        print(f"{key}: ${value:,.2f}")
    
    # 绘制资本构成
    calculator.plot_capital_breakdown(capital_results)
    
    # 定义情景
    scenarios = {
        'Increased IR Volatility': {'interest_rate': 1.5, 'credit_spread': 1.0, 'fx': 1.0},
        'Credit Stress': {'interest_rate': 1.0, 'credit_spread': 2.0, 'fx': 1.2},
        'FX Shock': {'interest_rate': 1.2, 'credit_spread': 1.3, 'fx': 1.8},
        'Global Stress': {'interest_rate': 1.5, 'credit_spread': 1.7, 'fx': 1.6}
    }
    
    # 运行情景分析
    scenario_results = calculator.run_scenario_analysis(sensitivities, scenarios)
    
    # 绘制情景比较
    calculator.plot_scenario_comparison(scenario_results) 