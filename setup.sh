#!/bin/bash
echo "创建交易对手风险分析环境..."

conda create -n ccr_analytics python=3.9 -y
source activate ccr_analytics

echo "安装依赖包..."
conda install numpy pandas matplotlib scipy -y
conda install -c conda-forge yfinance arch statsmodels -y
conda install -c conda-forge seaborn scikit-learn -y

echo "创建项目目录..."
mkdir -p data
mkdir -p results
mkdir -p figures

echo "环境设置完成!"
echo "使用 'source activate ccr_analytics' 激活环境"
echo "使用 'python ccr_analytics_main.py' 运行分析"