# AdvancedSimulation: Counterparty Credit Risk Analytics Framework

## Project Overview
A comprehensive Python-based counterparty credit risk analytics platform for derivative pricing, exposure calculation, and XVA quantification. This framework implements Monte Carlo simulation with GARCH volatility modeling for 1,000+ scenarios to accurately model market risk factors and calculate exposure profiles.

## Key Features
- **Advanced Market Risk Modeling**: GARCH(1,1) volatility forecasting with Monte Carlo simulation
- **Modular OOP Architecture**: Specialized components for different aspects of counterparty risk
- **Derivative Pricing**: Interest rate swap valuation with term structure modeling
- **Exposure Metrics**: Expected Exposure (EE), Potential Future Exposure (PFE), Effective Expected Positive Exposure (EEPE)
- **XVA Calculation**: Credit Valuation Adjustment (CVA), Debt Valuation Adjustment (DVA), Funding Valuation Adjustment (FVA)
- **Regulatory Capital**: Basel III SA-CVA methodology implementation across multiple risk factors
- **Sensitivity Analysis**: Credit spread sensitivity tools for capital optimization strategies
- **Interactive Visualization**: Exposure profiles, sensitivity analysis, and capital breakdown charts

## Code Structure

```text
AdvancedSimulation/
├── ccr_analytics_main.py # Main execution script
├── ccr_exposure_analytics.py # Core classes for risk calculation
├── market_factor_simulator.py # Market factor simulation implementation
├── derivatives_pricer.py # Financial derivatives pricing models
├── technical_indicators.py # GARCH and other technical analysis tools
├── xva_calculator.py # XVA calculation logic
├── regulatory_capital.py # SA-CVA methodology implementation
└── visualization_tools.py # Plotting and visualization utilities
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/michealearncoding/AdvancedSimulation.git
cd AdvancedSimulation
```


The framework will:
1. Download market data and fit GARCH(1,1) model
2. Simulate market factor paths
3. Price interest rate swaps across simulated paths
4. Calculate counterparty exposure metrics
5. Compute XVA adjustments
6. Calculate regulatory capital requirements
7. Generate visualization charts

## Example Outputs

### Exposure Profile
The exposure profile shows Expected Exposure (EE), Potential Future Exposure (PFE), and Effective Expected Positive Exposure (EEPE) over the life of the derivative contract.

### CVA Sensitivity Analysis
Sensitivity of Credit Valuation Adjustment to changes in counterparty credit spreads, ranging from -50 to +100 basis points.

### SA-CVA Capital Breakdown
Capital requirements breakdown by risk factor (Interest Rate, Credit Spread, FX) according to the Basel III SA-CVA methodology.

