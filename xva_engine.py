import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class CreditCurve:
    """信用曲线构建与操作"""
    def __init__(self, tenors, survival_probs):
        """
        参数:
            tenors: 期限点列表（年）
            survival_probs: 对应的生存概率
        """
        self.tenors = np.array(tenors)
        self.survival_probs = np.array(survival_probs)
        self._build_curve()
    
    def _build_curve(self):
        """构建插值曲线"""
        self.curve = interp1d(
            self.tenors, 
            self.survival_probs, 
            kind='linear', 
            bounds_error=False, 
            fill_value=(self.survival_probs[0], self.survival_probs[-1])
        )
    
    def get_survival_prob(self, t):
        """获取t时刻的生存概率"""
        return self.curve(t)
    
    def get_default_prob(self, t1, t2):
        """获取t1到t2期间的违约概率"""
        return self.get_survival_prob(t1) - self.get_survival_prob(t2)
    
    def shift_curve(self, shift_bps):
        """平行移动信用曲线（用于敏感性分析）"""
        # 转换基点到小数
        shift_decimal = shift_bps / 10000
        
        # 计算调整后的生存概率（简化模型）
        adjusted_probs = np.maximum(0, np.minimum(1, self.survival_probs - shift_decimal * self.tenors))
        
        return CreditCurve(self.tenors, adjusted_probs)

class XVACalculator:
    """XVA计算引擎"""
    def __init__(self):
        self.results = {}
    
    def calculate_cva(self, exposure_profile, credit_curve, recovery_rate=0.4, discount_curve=None):
        """计算信用估值调整(CVA)
        
        参数:
            exposure_profile: 预期暴露时间序列字典 {time: exposure}
            credit_curve: CreditCurve实例
            recovery_rate: 回收率
            discount_curve: 折现曲线函数，默认为无风险折现
        """
        times = np.array(list(exposure_profile.keys()))
        exposures = np.array(list(exposure_profile.values()))
        
        # 违约概率
        sorted_idx = np.argsort(times)
        sorted_times = times[sorted_idx]
        sorted_exposures = exposures[sorted_idx]
        
        # 违约损失率
        lgd = 1 - recovery_rate
        
        # 计算CVA
        cva = 0
        for i in range(len(sorted_times)-1):
            t1 = sorted_times[i]
            t2 = sorted_times[i+1]
            
            # t1到t2期间的违约概率
            pd = credit_curve.get_default_prob(t1, t2)
            
            # 该期间的平均暴露
            avg_exposure = (sorted_exposures[i] + sorted_exposures[i+1]) / 2
            
            # 折现因子（如果提供了折现曲线）
            if discount_curve is not None:
                discount_factor = discount_curve((t1 + t2) / 2)
            else:
                discount_factor = 1  # 简化处理
            
            # 累加当前期间的CVA贡献
            cva += lgd * pd * avg_exposure * discount_factor
        
        return cva
    
    def calculate_dva(self, exposure_profile, own_credit_curve, recovery_rate=0.4, discount_curve=None):
        """计算负债估值调整(DVA)"""
        # DVA可视为负的CVA，使用相反的暴露
        negative_exposure = {t: -e for t, e in exposure_profile.items()}
        dva = self.calculate_cva(negative_exposure, own_credit_curve, recovery_rate, discount_curve)
        return dva
    
    def calculate_fva(self, exposure_profile, funding_spread, discount_curve=None):
        """计算融资估值调整(FVA)
        
        参数:
            exposure_profile: 预期暴露时间序列
            funding_spread: 融资利差（年化基点）
            discount_curve: 折现曲线
        """
        times = np.array(list(exposure_profile.keys()))
        exposures = np.array(list(exposure_profile.values()))
        
        # 转换融资利差为小数
        funding_spread_decimal = funding_spread / 10000
        
        # 计算FVA
        fva = 0
        sorted_idx = np.argsort(times)
        sorted_times = times[sorted_idx]
        sorted_exposures = exposures[sorted_idx]
        
        for i in range(len(sorted_times)-1):
            t1 = sorted_times[i]
            t2 = sorted_times[i+1]
            
            # 该期间的时长
            dt = t2 - t1
            
            # 该期间的平均暴露
            avg_exposure = (sorted_exposures[i] + sorted_exposures[i+1]) / 2
            
            # 折现因子
            if discount_curve is not None:
                discount_factor = discount_curve((t1 + t2) / 2)
            else:
                discount_factor = 1
            
            # 融资成本
            funding_cost = avg_exposure * funding_spread_decimal * dt
            
            # 累加当前期间的FVA贡献
            fva += funding_cost * discount_factor
        
        return fva
    
    def run_sensitivity_analysis(self, exposure_profile, credit_curve, 
                               recovery_rate=0.4, discount_curve=None, 
                               spread_shifts=[-50, -25, 0, 25, 50, 100]):
        """信用利差敏感性分析"""
        sensitivity_results = []
        
        for shift in spread_shifts:
            # 移动信用曲线
            shifted_curve = credit_curve.shift_curve(shift)
            
            # 计算新的CVA
            cva = self.calculate_cva(exposure_profile, shifted_curve, recovery_rate, discount_curve)
            
            sensitivity_results.append({
                'Spread Shift (bps)': shift,
                'CVA': cva
            })
        
        return pd.DataFrame(sensitivity_results)
    
    def plot_sensitivity_analysis(self, sensitivity_df, title):
        """绘制敏感性分析结果"""
        plt.figure(figsize=(10, 6))
        
        plt.plot(sensitivity_df['Spread Shift (bps)'], sensitivity_df['CVA'], marker='o')
        
        plt.title(title)
        plt.xlabel('Credit Spread Shift (bps)')
        plt.ylabel('CVA')
        plt.grid(True)
        plt.savefig('cva_sensitivity.png')
        plt.show()

# 使用示例
if __name__ == "__main__":
    # 创建信用曲线
    tenors = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
    survival_probs = [1.0, 0.99, 0.98, 0.96, 0.94, 0.90, 0.86, 0.80]
    credit_curve = CreditCurve(tenors, survival_probs)
    
    # 创建示例暴露曲线
    times = np.linspace(0, 10, 21)
    # 假设的暴露模式：先上升后下降
    exposures = 1000000 * np.sin(np.pi * times / 10)
    exposure_profile = dict(zip(times, exposures))
    
    # 创建XVA计算器
    xva_calculator = XVACalculator()
    
    # 计算CVA
    cva = xva_calculator.calculate_cva(exposure_profile, credit_curve)
    print(f"CVA: ${cva:.2f}")
    
    # 计算DVA (假设银行自身的信用曲线略差)
    own_survival_probs = [1.0, 0.985, 0.97, 0.94, 0.91, 0.85, 0.80, 0.72]
    own_credit_curve = CreditCurve(tenors, own_survival_probs)
    dva = xva_calculator.calculate_dva(exposure_profile, own_credit_curve)
    print(f"DVA: ${dva:.2f}")
    
    # 计算FVA
    funding_spread = 50  # 50 bps
    fva = xva_calculator.calculate_fva(exposure_profile, funding_spread)
    print(f"FVA: ${fva:.2f}")
    
    # 计算总XVA
    total_xva = cva - dva + fva
    print(f"Total XVA: ${total_xva:.2f}")
    
    # 运行敏感性分析
    sensitivity_results = xva_calculator.run_sensitivity_analysis(
        exposure_profile, credit_curve, 
        spread_shifts=[-100, -50, -20, 0, 20, 50, 100, 200]
    )
    
    # 绘制敏感性分析结果
    xva_calculator.plot_sensitivity_analysis(
        sensitivity_results, 
        'CVA Sensitivity to Credit Spread Changes'
    ) 