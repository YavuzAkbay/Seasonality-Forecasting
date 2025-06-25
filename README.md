# 🧠 Financial Forecasting with Seasonality Detection

A Python-based financial forecasting system that intelligently detects seasonal patterns and generates professional forecasts using advanced statistical methods and interactive visualizations.

## 🌟 Features

🔍 Intelligent Seasonality Detection

- **Multiple Statistical Tests**: Autocorrelation, Kruskal-Wallis, ANOVA, and Regression F-tests
- **Adaptive Forecasting**: Automatically switches between seasonal and trend-only forecasting
- **Robust Decision Framework**: Requires majority of tests to confirm seasonality
- **Minimum Data Requirements**: Ensures at least 3 years of data for reliable analysis

📈 Advanced Trend Analysis

- **Multi-Method Changepoint Detection**: Rolling correlation, piecewise regression, and statistical methods
- **Configurable Segment Length**: Customizable minimum trend segment length (default: 5 quarters)
- **Conservative Trend Application**: High confidence thresholds for trend adjustments
- **Comprehensive Segment Analysis**: Detailed breakdown of growth, decline, and stability phases

🎨 Professional Visualizations

- **PyGWalker Integration**: Interactive, Tableau-like dashboards
- **Drag-and-Drop Interface**: Create professional charts without coding
- **Export Capabilities**: Generate presentation-ready visualizations
- **Business Intelligence Features**: Performance categorization and benchmarking

📊 Comprehensive Analytics

- **Seasonal Pattern Analysis**: QoQ transition patterns
- **Growth Rate Calculations**: QoQ and YoY performance metrics
- **Outlier Detection**: Statistical outlier removal for cleaner patterns
- **Confidence Metrics**: Statistical significance and sample size reporting

## 🚀 Quick Start

### Prerequisites

```bash
pip install pandas numpy pygwalker scikit-learn scipy statsmodels openpyxl
```

### Basic Usage

```python
from financial_forecaster import ProfessionalFinancialForecaster

# Initialize the forecaster
forecaster = ProfessionalFinancialForecaster(
    'your_data.xlsx', 
    min_segment_length=5,           # Minimum quarters for trend segments
    seasonality_threshold=0.05      # Statistical significance threshold
)

# Run complete analysis
walker, viz_data, results = forecaster.run_professional_analysis()
```

### Data Format Requirements

Your Excel file should contain these columns:
- time: Date column (YYYY-MM-DD format)
- Total Revenue: Revenue figures in millions
- Net Income: Net income figures in millions

Example:

| time  | Total Revenue | Net Income |
| ------------- | ------------- | ------------- |
| 2020-01-01  | 91819  | 22236 |
| 2020-04-01  | 58313  | 11249 |
| 2020-07-01  | 64698  | 12673 |

## 📋 Key Components

1. Seasonality Detection Engine
The system runs four statistical tests to determine if your data exhibits seasonal patterns:

```python
# Automatic seasonality detection
seasonality_detected = forecaster.detect_seasonality()

if seasonality_detected:
    print("✅ Seasonality confirmed - using seasonal forecast")
else:
    print("❌ No seasonality - using trend-only forecast")
```

Statistical Tests Used:

- Autocorrelation Function (ACF): Tests correlation at 4-quarter lags
- Kruskal-Wallis Test: Non-parametric comparison of quarterly distributions
- One-way ANOVA: Parametric test for quarterly differences
- Regression F-test: Significance of quarter dummy variables

2. Advanced Trend Segmentation

```python
# Configure minimum segment length for conservative analysis
forecaster = ProfessionalFinancialForecaster(
    'data.xlsx', 
    min_segment_length=6  # Require 6+ quarters per trend segment
)
```

Changepoint Detection Methods:

- Rolling correlation analysis
- Piecewise regression with K-means clustering
- Statistical change detection (CUSUM-like)

3. Interactive Dashboard Creation

```python
# Launch professional PyGWalker dashboard
walker, viz_data, results = forecaster.run_professional_analysis()

# The dashboard opens automatically with:
# - Drag-and-drop chart creation
# - Professional export options
# - Real-time filtering and analysis
```

## 📊 Output Examples

### Console Output

```bash
📋 SEASONALITY DETECTION RESULTS
----------------------------------------
Autocorrelation Test: ✅ SIGNIFICANT
                        p-value: 0.0000
Kruskal-Wallis Test : ✅ SIGNIFICANT
                        p-value: 0.0055
One-way ANOVA       : ✅ SIGNIFICANT
                        p-value: 0.0007
Regression F-test   : ✅ SIGNIFICANT
                        p-value: 0.0007

🎯 OVERALL SEASONALITY ASSESSMENT:
   Significant tests: 4/4
   Threshold required: 2
   ✅ SEASONALITY DETECTED
   📈 Seasonal range: 43.5% of mean
============================================================

✅ SEASONALITY CONFIRMED - PROCEEDING WITH SEASONAL FORECAST
Detected 3 trend segments with minimum length of 5 quarters

📊 DETECTED TREND SEGMENTS (Min Length: 5 quarters):
----------------------------------------------------------------------
Segment 1: 2017-07 to 2020-04 (12 quarters)
   Trend: SIDEWAYS (slope: 59, R²: 0.000)
Segment 2: 2020-04 to 2021-07 (6 quarters)
   Trend: UPWARD (slope: 4,245, R²: 0.180)
Segment 3: 2021-07 to 2025-01 (15 quarters)
   Trend: SIDEWAYS (slope: 311, R²: 0.008)

=== SEASONAL FORECASTING Q2 2025 ===
Last reported quarter: Q1 2025
Current trend: SIDEWAYS (based on 5+ quarter segments)
Sideways trend detected - using pure seasonal pattern
```
### Forecast Results

```bash
================================================================================
                    PROFESSIONAL FINANCIAL FORECAST REPORT
================================================================================

SEASONALITY ASSESSMENT
--------------------------------------------------
✅ SEASONALITY DETECTED
   Statistical tests passed: 4/4
   Analysis period: 9 years

FORECAST METHODOLOGY
--------------------------------------------------
Forecast Type: SEASONAL
Forecast Period: Q2 2025
Minimum Trend Length: 5 quarters

KEY FINANCIAL PROJECTIONS
--------------------------------------------------
Revenue Forecast:    $86,992M
Net Margin Forecast: 24.93%
Net Income Forecast: $21,684M

PERFORMANCE INDICATORS
--------------------------------------------------
Revenue Growth (QoQ): -8.8%
Income Growth (QoQ):  -12.5%
Current Trend:        SIDEWAYS
Trend Confidence:     0.8%

SEASONAL PATTERNS APPLIED
--------------------------------------------------
Seasonal Revenue Pattern: -8.8%
Seasonal Margin Pattern:  -4.1%
```

## 🎯 Advanced Configuration

### Custom Seasonality Thresholds

```python
# Stricter seasonality detection (1% significance level)
forecaster = ProfessionalFinancialForecaster(
    'data.xlsx',
    seasonality_threshold=0.01
)

# More lenient detection (10% significance level)  
forecaster = ProfessionalFinancialForecaster(
    'data.xlsx',
    seasonality_threshold=0.10
)
```

### Conservative Trend Analysis

```python
# Require longer segments for more stable trends
forecaster = ProfessionalFinancialForecaster(
    'data.xlsx',
    min_segment_length=8  # 2+ years per segment
)
```

## 🔧 Technical Details

### Dependencies

- pandas: Data manipulation and analysis
- numpy: Numerical computing
- pygwalker: Interactive visualization engine
- scikit-learn: Machine learning algorithms
- scipy: Statistical functions
- statsmodels: Time series analysis

### Performance Considerations

- Minimum Data: 12 quarters (3 years) required
- Optimal Data: 20+ quarters for robust seasonality detection
- Memory Usage: Efficient for datasets up to 1000+ quarters
- Processing Time: < 10 seconds for typical financial datasets

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the GPL v3 - see the (https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. **Do not use this for actual trading decisions without proper risk management and professional financial advice.** Past performance does not guarantee future results. Trading stocks involves substantial risk of loss.

## 🙏 Acknowledgments

- Built with PyGWalker for professional visualizations
- Statistical methods based on established econometric practices
- Inspired by modern business intelligence platforms

## 📧 Contact

Yavuz Akbay - akbay.yavuz@gmail.com

---

⭐️ If this project helped with your financial analysis, please consider giving it a star!

**Built with ❤️ for the intersection of mathematics, machine learning and finance**
