import pandas as pd
import numpy as np
import pygwalker as pyg
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from scipy import stats
from scipy.stats import chi2_contingency, kruskal
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ProfessionalFinancialForecaster:
    def __init__(self, data_path, min_segment_length=5, seasonality_threshold=0.05):
        self.data_path = data_path
        self.data = None
        self.clean_data = None
        self.trend_segments = []
        self.min_segment_length = min_segment_length
        self.forecast_results = None
        self.seasonality_threshold = seasonality_threshold
        self.seasonality_detected = False
        self.seasonality_results = {}
        
    def load_and_prepare_data(self):
        self.data = pd.read_excel(self.data_path)
        self.data['time'] = pd.to_datetime(self.data['time'])
        self.data['year'] = self.data['time'].dt.year
        self.data['quarter'] = self.data['time'].dt.quarter
        self.data = self.data.sort_values(by='time')
        self.clean_data = self.data.dropna(subset=['Total Revenue', 'Net Income']).copy()
        self.clean_data['Net Margin'] = self.clean_data['Net Income'] / self.clean_data['Total Revenue']
        print(f"Data loaded: {len(self.clean_data)} quarters from {self.clean_data['time'].min()} to {self.clean_data['time'].max()}")
        print(f"Minimum segment length set to: {self.min_segment_length} quarters")
    
    def test_seasonality_autocorrelation(self, data, seasonal_lag=4):
        from statsmodels.tsa.stattools import acf
        try:
            autocorr = acf(data, nlags=min(len(data)-1, seasonal_lag*3), fft=False)
            if len(autocorr) > seasonal_lag:
                seasonal_autocorr = autocorr[seasonal_lag]
                n = len(data)
                confidence_bound = 1.96 / np.sqrt(n)
                is_significant = abs(seasonal_autocorr) > confidence_bound
                return {
                    'test_name': 'Autocorrelation Test',
                    'seasonal_autocorr': seasonal_autocorr,
                    'confidence_bound': confidence_bound,
                    'is_significant': is_significant,
                    'p_value_approx': 2 * (1 - stats.norm.cdf(abs(seasonal_autocorr * np.sqrt(n))))
                }
            else:
                return {
                    'test_name': 'Autocorrelation Test',
                    'seasonal_autocorr': 0,
                    'confidence_bound': 0,
                    'is_significant': False,
                    'p_value_approx': 1.0,
                    'error': 'Insufficient data for seasonal lag'
                }
        except Exception as e:
            return {
                'test_name': 'Autocorrelation Test',
                'error': str(e),
                'is_significant': False,
                'p_value_approx': 1.0
            }
    
    def test_seasonality_kruskal_wallis(self, data, quarters):
        try:
            quarter_groups = []
            for q in [1, 2, 3, 4]:
                quarter_data = data[quarters == q]
                if len(quarter_data) > 0:
                    quarter_groups.append(quarter_data)
            if len(quarter_groups) >= 2:
                statistic, p_value = kruskal(*quarter_groups)
                return {
                    'test_name': 'Kruskal-Wallis Test',
                    'statistic': statistic,
                    'p_value': p_value,
                    'is_significant': p_value < self.seasonality_threshold,
                    'quarters_tested': len(quarter_groups)
                }
            else:
                return {
                    'test_name': 'Kruskal-Wallis Test',
                    'error': 'Insufficient quarters for comparison',
                    'is_significant': False,
                    'p_value': 1.0
                }
        except Exception as e:
            return {
                'test_name': 'Kruskal-Wallis Test',
                'error': str(e),
                'is_significant': False,
                'p_value': 1.0
            }
    
    def test_seasonality_anova(self, data, quarters):
        try:
            quarter_groups = []
            for q in [1, 2, 3, 4]:
                quarter_data = data[quarters == q]
                if len(quarter_data) > 1:
                    quarter_groups.append(quarter_data)
            if len(quarter_groups) >= 2:
                statistic, p_value = stats.f_oneway(*quarter_groups)
                return {
                    'test_name': 'One-way ANOVA',
                    'f_statistic': statistic,
                    'p_value': p_value,
                    'is_significant': p_value < self.seasonality_threshold,
                    'quarters_tested': len(quarter_groups)
                }
            else:
                return {
                    'test_name': 'One-way ANOVA',
                    'error': 'Insufficient data for ANOVA',
                    'is_significant': False,
                    'p_value': 1.0
                }
        except Exception as e:
            return {
                'test_name': 'One-way ANOVA',
                'error': str(e),
                'is_significant': False,
                'p_value': 1.0
            }
    
    def test_seasonality_regression(self, data, quarters):
        try:
            quarter_dummies = pd.get_dummies(quarters, prefix='Q')
            quarter_dummies = quarter_dummies.iloc[:, :-1]
            if len(quarter_dummies.columns) > 0:
                model = LinearRegression()
                model.fit(quarter_dummies, data)
                predictions = model.predict(quarter_dummies)
                ss_res = np.sum((data - predictions) ** 2)
                ss_tot = np.sum((data - np.mean(data)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                n = len(data)
                k = len(quarter_dummies.columns)
                if k > 0 and n > k + 1:
                    f_statistic = (r_squared / k) / ((1 - r_squared) / (n - k - 1))
                    p_value = 1 - stats.f.cdf(f_statistic, k, n - k - 1)
                else:
                    f_statistic = 0
                    p_value = 1.0
                return {
                    'test_name': 'Regression F-test',
                    'r_squared': r_squared,
                    'f_statistic': f_statistic,
                    'p_value': p_value,
                    'is_significant': p_value < self.seasonality_threshold,
                    'coefficients': model.coef_.tolist()
                }
            else:
                return {
                    'test_name': 'Regression F-test',
                    'error': 'No quarter variation in data',
                    'is_significant': False,
                    'p_value': 1.0
                }
        except Exception as e:
            return {
                'test_name': 'Regression F-test',
                'error': str(e),
                'is_significant': False,
                'p_value': 1.0
            }
    
    def calculate_seasonal_strength(self, data, quarters):
        try:
            quarter_stats = {}
            overall_cv = np.std(data) / np.mean(data) if np.mean(data) != 0 else 0
            for q in [1, 2, 3, 4]:
                quarter_data = data[quarters == q]
                if len(quarter_data) > 1:
                    quarter_mean = np.mean(quarter_data)
                    quarter_std = np.std(quarter_data)
                    quarter_cv = quarter_std / quarter_mean if quarter_mean != 0 else 0
                    quarter_stats[f'Q{q}'] = {
                        'mean': quarter_mean,
                        'std': quarter_std,
                        'cv': quarter_cv,
                        'count': len(quarter_data)
                    }
            if quarter_stats:
                quarterly_means = [stats['mean'] for stats in quarter_stats.values()]
                seasonal_range = max(quarterly_means) - min(quarterly_means)
                seasonal_range_pct = seasonal_range / np.mean(quarterly_means) if np.mean(quarterly_means) != 0 else 0
            else:
                seasonal_range = 0
                seasonal_range_pct = 0
            return {
                'overall_cv': overall_cv,
                'quarter_stats': quarter_stats,
                'seasonal_range': seasonal_range,
                'seasonal_range_pct': seasonal_range_pct
            }
        except Exception as e:
            return {
                'error': str(e),
                'overall_cv': 0,
                'seasonal_range_pct': 0
            }
    
    def detect_seasonality(self):
        print("\n🔍 COMPREHENSIVE SEASONALITY DETECTION")
        print("=" * 60)
        revenue_data = self.clean_data['Total Revenue'].values
        quarters = self.clean_data['quarter'].values
        min_years = 3
        min_quarters = min_years * 4
        if len(revenue_data) < min_quarters:
            print(f"❌ INSUFFICIENT DATA: Need at least {min_quarters} quarters ({min_years} years) for seasonality testing")
            print(f"   Current data: {len(revenue_data)} quarters")
            self.seasonality_detected = False
            self.seasonality_results = {
                'detected': False,
                'reason': f'Insufficient data: {len(revenue_data)} quarters < {min_quarters} required'
            }
            return False
        print(f"✅ Data requirements met: {len(revenue_data)} quarters available")
        print(f"   Testing period: {self.clean_data['time'].min().strftime('%Y-%m')} to {self.clean_data['time'].max().strftime('%Y-%m')}")
        tests_results = []
        print(f"\n📊 Running Autocorrelation Test...")
        autocorr_result = self.test_seasonality_autocorrelation(revenue_data)
        tests_results.append(autocorr_result)
        print(f"📊 Running Kruskal-Wallis Test...")
        kruskal_result = self.test_seasonality_kruskal_wallis(revenue_data, quarters)
        tests_results.append(kruskal_result)
        print(f"📊 Running One-way ANOVA...")
        anova_result = self.test_seasonality_anova(revenue_data, quarters)
        tests_results.append(anova_result)
        print(f"📊 Running Regression F-test...")
        regression_result = self.test_seasonality_regression(revenue_data, quarters)
        tests_results.append(regression_result)
        print(f"📊 Calculating Seasonal Strength...")
        strength_metrics = self.calculate_seasonal_strength(revenue_data, quarters)
        significant_tests = [test for test in tests_results if test.get('is_significant', False)]
        total_valid_tests = len([test for test in tests_results if 'error' not in test])
        seasonality_threshold_tests = max(1, total_valid_tests // 2)
        self.seasonality_detected = len(significant_tests) >= seasonality_threshold_tests
        self.seasonality_results = {
            'detected': self.seasonality_detected,
            'tests_results': tests_results,
            'strength_metrics': strength_metrics,
            'significant_tests': len(significant_tests),
            'total_valid_tests': total_valid_tests,
            'threshold_tests': seasonality_threshold_tests,
            'data_points': len(revenue_data),
            'years_analyzed': len(self.clean_data['year'].unique())
        }
        print(f"\n📋 SEASONALITY DETECTION RESULTS")
        print("-" * 40)
        for test in tests_results:
            if 'error' not in test:
                test_name = test['test_name']
                is_sig = test.get('is_significant', False)
                p_val = test.get('p_value', test.get('p_value_approx', 'N/A'))
                status = "✅ SIGNIFICANT" if is_sig else "❌ NOT SIGNIFICANT"
                print(f"{test_name:20s}: {status}")
                if isinstance(p_val, (int, float)):
                    print(f"{'':22s}  p-value: {p_val:.4f}")
            else:
                print(f"{test['test_name']:20s}: ⚠️  ERROR - {test['error']}")
        print(f"\n🎯 OVERALL SEASONALITY ASSESSMENT:")
        print(f"   Significant tests: {len(significant_tests)}/{total_valid_tests}")
        print(f"   Threshold required: {seasonality_threshold_tests}")
        if self.seasonality_detected:
            print(f"   ✅ SEASONALITY DETECTED")
            print(f"   📈 Seasonal range: {strength_metrics.get('seasonal_range_pct', 0):.1%} of mean")
        else:
            print(f"   ❌ NO SIGNIFICANT SEASONALITY")
            print(f"   📊 Data appears to follow random/trend patterns")
        print("=" * 60)
        return self.seasonality_detected
    
    def detect_trend_changepoints(self):
        revenue = self.clean_data['Total Revenue'].values
        time_index = np.arange(len(revenue))
        changepoints_correlation = self.detect_changepoints_correlation(revenue, window=max(6, self.min_segment_length))
        changepoints_regression = self.detect_changepoints_regression(time_index, revenue)
        changepoints_statistical = self.detect_changepoints_statistical(revenue)
        all_changepoints = sorted(set(changepoints_correlation + changepoints_regression + changepoints_statistical))
        filtered_changepoints = [0]
        for cp in all_changepoints:
            if cp - filtered_changepoints[-1] >= self.min_segment_length:
                filtered_changepoints.append(cp)
        if len(filtered_changepoints) > 1 and len(revenue) - filtered_changepoints[-1] < self.min_segment_length:
            if len(filtered_changepoints) > 2:
                filtered_changepoints.pop(-1)
        if filtered_changepoints[-1] != len(revenue) - 1:
            filtered_changepoints.append(len(revenue) - 1)
        print(f"Detected {len(filtered_changepoints)-1} trend segments with minimum length of {self.min_segment_length} quarters")
        return filtered_changepoints
    
    def detect_changepoints_correlation(self, data, window=6):
        changepoints = []
        correlations = []
        window = max(window, self.min_segment_length)
        for i in range(window, len(data) - window):
            before = data[max(0, i-window):i]
            after = data[i:min(len(data), i+window)]
            if len(before) >= self.min_segment_length and len(after) >= self.min_segment_length:
                before_trend = np.corrcoef(np.arange(len(before)), before)[0, 1]
                after_trend = np.corrcoef(np.arange(len(after)), after)[0, 1]
                correlation_change = abs(before_trend - after_trend)
                correlations.append((i, correlation_change))
        if correlations:
            correlations.sort(key=lambda x: x[1], reverse=True)
            n_changepoints = max(1, len(correlations) // 5)
            changepoints = [cp[0] for cp in correlations[:n_changepoints]]
        return sorted(changepoints)
    
    def detect_changepoints_regression(self, x, y):
        changepoints = []
        best_score = float('inf')
        max_segments = min(5, len(y) // self.min_segment_length)
        for n_segments in range(2, max_segments + 1):
            kmeans = KMeans(n_clusters=n_segments, random_state=42, n_init=10)
            segments = kmeans.fit_predict(x.reshape(-1, 1))
            segment_changes = []
            for i in range(1, len(segments)):
                if segments[i] != segments[i-1]:
                    segment_changes.append(i)
            valid_segmentation = True
            prev_idx = 0
            for change_idx in segment_changes + [len(y)]:
                if change_idx - prev_idx < self.min_segment_length:
                    valid_segmentation = False
                    break
                prev_idx = change_idx
            if not valid_segmentation:
                continue
            total_error = 0
            prev_idx = 0
            for change_idx in segment_changes + [len(y)]:
                segment_x = x[prev_idx:change_idx].reshape(-1, 1)
                segment_y = y[prev_idx:change_idx]
                model = LinearRegression()
                model.fit(segment_x, segment_y)
                predictions = model.predict(segment_x)
                error = np.mean((segment_y - predictions) ** 2)
                total_error += error
                prev_idx = change_idx
            if total_error < best_score:
                best_score = total_error
                changepoints = segment_changes
        return changepoints
    
    def detect_changepoints_statistical(self, data):
        changepoints = []
        short_window = max(3, self.min_segment_length // 2)
        long_window = max(6, self.min_segment_length)
        short_ma = pd.Series(data).rolling(window=short_window).mean()
        long_ma = pd.Series(data).rolling(window=long_window).mean()
        ma_diff = short_ma - long_ma
        ma_diff_change = ma_diff.diff().abs()
        threshold = ma_diff_change.quantile(0.8)
        significant_changes = ma_diff_change > threshold
        for i in range(len(significant_changes)):
            if significant_changes.iloc[i] and i > long_window:
                changepoints.append(i)
        return changepoints
    
    def analyze_trend_segments(self):
        changepoints = self.detect_trend_changepoints()
        revenue = self.clean_data['Total Revenue'].values
        time_index = np.arange(len(revenue))
        self.trend_segments = []
        print(f"\n📊 DETECTED TREND SEGMENTS (Min Length: {self.min_segment_length} quarters):")
        print("-" * 70)
        for i in range(len(changepoints) - 1):
            start_idx = changepoints[i]
            end_idx = changepoints[i + 1]
            segment_x = time_index[start_idx:end_idx + 1].reshape(-1, 1)
            segment_y = revenue[start_idx:end_idx + 1]
            segment_dates = self.clean_data.iloc[start_idx:end_idx + 1]['time']
            model = LinearRegression()
            model.fit(segment_x, segment_y)
            slope = model.coef_[0]
            r_squared = model.score(segment_x, segment_y)
            segment_length = end_idx - start_idx + 1
            slope_threshold = 1000
            if abs(slope) < slope_threshold:
                trend_direction = "SIDEWAYS"
            elif slope > 0:
                trend_direction = "UPWARD"
            else:
                trend_direction = "DOWNWARD"
            segment_info = {
                'start_date': segment_dates.iloc[0],
                'end_date': segment_dates.iloc[-1],
                'start_idx': start_idx,
                'end_idx': end_idx,
                'slope': slope,
                'r_squared': r_squared,
                'trend_direction': trend_direction,
                'length': segment_length,
                'model': model
            }
            self.trend_segments.append(segment_info)
            print(f"Segment {i+1}: {segment_dates.iloc[0].strftime('%Y-%m')} to {segment_dates.iloc[-1].strftime('%Y-%m')} ({segment_length} quarters)")
            print(f"   Trend: {trend_direction} (slope: {slope:,.0f}, R²: {r_squared:.3f})")
        return self.trend_segments
    
    def get_current_trend(self):
        if not self.trend_segments:
            self.analyze_trend_segments()
        if self.trend_segments:
            latest_segment = self.trend_segments[-1]
            return latest_segment['trend_direction'], latest_segment['slope'], latest_segment['r_squared']
        return "UNKNOWN", 0, 0
    
    def remove_outliers(self, series, threshold=2):
        if len(series) == 0:
            return series
        mean = series.mean()
        std = series.std()
        if std == 0:
            return series
        z_scores = np.abs((series - mean) / std)
        return series[z_scores <= threshold]
    
    def calculate_seasonal_patterns(self):
        seasonal_changes = {}
        for from_q in range(1, 5):
            to_q = from_q + 1 if from_q < 4 else 1
            changes_revenue = []
            changes_margin = []
            for year in self.clean_data['year'].unique():
                from_data = self.clean_data[(self.clean_data['year'] == year) & (self.clean_data['quarter'] == from_q)]
                to_year = year + 1 if from_q == 4 else year
                to_data = self.clean_data[(self.clean_data['year'] == to_year) & (self.clean_data['quarter'] == to_q)]
                if not from_data.empty and not to_data.empty:
                    rev_change = (to_data['Total Revenue'].values[0] - from_data['Total Revenue'].values[0]) / from_data['Total Revenue'].values[0]
                    changes_revenue.append(rev_change)
                    margin_change = (to_data['Net Margin'].values[0] - from_data['Net Margin'].values[0]) / from_data['Net Margin'].values[0]
                    changes_margin.append(margin_change)
            if changes_revenue:
                clean_rev_changes = self.remove_outliers(pd.Series(changes_revenue))
                clean_margin_changes = self.remove_outliers(pd.Series(changes_margin))
                seasonal_changes[f'Q{from_q}_to_Q{to_q}'] = {
                    'revenue_change': clean_rev_changes.mean() if len(clean_rev_changes) > 0 else 0,
                    'margin_change': clean_margin_changes.mean() if len(clean_margin_changes) > 0 else 0,
                    'sample_size': len(clean_rev_changes)
                }
        return seasonal_changes
    
    def determine_next_quarter(self):
        last_date = self.clean_data['time'].max()
        last_quarter = self.clean_data.loc[self.clean_data['time'] == last_date, 'quarter'].values[0]
        last_year = self.clean_data.loc[self.clean_data['time'] == last_date, 'year'].values[0]
        next_quarter = last_quarter + 1 if last_quarter < 4 else 1
        next_year = last_year if last_quarter < 4 else last_year + 1
        return int(last_quarter), int(last_year), int(next_quarter), int(next_year)
    
    def seasonal_forecast_with_advanced_trend(self):
        seasonality_exists = self.detect_seasonality()
        if not seasonality_exists:
            print(f"\n⚠️  NO SEASONALITY DETECTED - SWITCHING TO TREND-ONLY FORECAST")
            return self.trend_only_forecast()
        print(f"\n✅ SEASONALITY CONFIRMED - PROCEEDING WITH SEASONAL FORECAST")
        self.analyze_trend_segments()
        seasonal_patterns = self.calculate_seasonal_patterns()
        last_quarter, last_year, next_quarter, next_year = self.determine_next_quarter()
        current_trend, trend_slope, trend_r2 = self.get_current_trend()
        print(f"\n=== SEASONAL FORECASTING Q{next_quarter} {next_year} ===")
        print(f"Last reported quarter: Q{last_quarter} {last_year}")
        print(f"Current trend: {current_trend} (based on {self.min_segment_length}+ quarter segments)")
        last_data = self.clean_data[self.clean_data['time'] == self.clean_data['time'].max()].iloc[0]
        last_revenue = last_data['Total Revenue']
        last_margin = last_data['Net Margin']
        transition_key = f'Q{last_quarter}_to_Q{next_quarter}'
        if transition_key in seasonal_patterns:
            seasonal_rev_change = seasonal_patterns[transition_key]['revenue_change']
            seasonal_margin_change = seasonal_patterns[transition_key]['margin_change']
        else:
            seasonal_rev_change = 0
            seasonal_margin_change = 0
        forecast_revenue = last_revenue * (1 + seasonal_rev_change)
        forecast_margin = last_margin * (1 + seasonal_margin_change)
        trend_confidence_threshold = 0.7
        if current_trend == "UPWARD" and trend_r2 > trend_confidence_threshold:
            trend_adjustment = min(0.02, abs(trend_slope) / last_revenue)
            forecast_revenue *= (1 + trend_adjustment)
            print(f"Applying conservative upward trend adjustment: +{trend_adjustment:.1%}")
        elif current_trend == "DOWNWARD" and trend_r2 > trend_confidence_threshold:
            trend_adjustment = min(0.02, abs(trend_slope) / last_revenue)
            forecast_revenue *= (1 - trend_adjustment)
            print(f"Applying conservative downward trend adjustment: -{trend_adjustment:.1%}")
        elif current_trend == "SIDEWAYS":
            print("Sideways trend detected - using pure seasonal pattern")
        else:
            print(f"Trend confidence too low (R²={trend_r2:.3f}) - using pure seasonal pattern")
        forecast_income = forecast_revenue * forecast_margin
        revenue_change = (forecast_revenue - last_revenue) / last_revenue
        income_change = (forecast_income - last_data['Net Income']) / last_data['Net Income']
        return {
            'quarter': f'Q{next_quarter} {next_year}',
            'forecast_type': 'SEASONAL',
            'seasonality_detected': True,
            'seasonality_results': self.seasonality_results,
            'forecast': {
                'revenue': forecast_revenue,
                'net_margin': forecast_margin,
                'net_income': forecast_income
            },
            'trend_analysis': {
                'current_trend': current_trend,
                'trend_slope': trend_slope,
                'trend_r_squared': trend_r2,
                'segments': self.trend_segments,
                'min_segment_length': self.min_segment_length
            },
            'changes': {
                'revenue_change': revenue_change,
                'income_change': income_change,
                'seasonal_revenue_pattern': seasonal_rev_change,
                'seasonal_margin_pattern': seasonal_margin_change
            },
            'confidence': {
                'pattern_sample_size': seasonal_patterns.get(transition_key, {}).get('sample_size', 0),
                'trend_r_squared': trend_r2
            },
            'seasonal_patterns': seasonal_patterns
        }
    
    def trend_only_forecast(self):
        self.analyze_trend_segments()
        last_quarter, last_year, next_quarter, next_year = self.determine_next_quarter()
        current_trend, trend_slope, trend_r2 = self.get_current_trend()
        print(f"\n=== TREND-ONLY FORECASTING Q{next_quarter} {next_year} ===")
        print(f"Last reported quarter: Q{last_quarter} {last_year}")
        print(f"Current trend: {current_trend} (R²: {trend_r2:.3f})")
        last_data = self.clean_data[self.clean_data['time'] == self.clean_data['time'].max()].iloc[0]
        last_revenue = last_data['Total Revenue']
        last_margin = last_data['Net Margin']
        if current_trend == "UPWARD" and trend_r2 > 0.5:
            trend_adjustment = min(0.03, abs(trend_slope) / last_revenue)
            forecast_revenue = last_revenue * (1 + trend_adjustment)
            print(f"Applying upward trend adjustment: +{trend_adjustment:.1%}")
        elif current_trend == "DOWNWARD" and trend_r2 > 0.5:
            trend_adjustment = min(0.03, abs(trend_slope) / last_revenue)
            forecast_revenue = last_revenue * (1 - trend_adjustment)
            print(f"Applying downward trend adjustment: -{trend_adjustment:.1%}")
        else:
            forecast_revenue = last_revenue
            print("No clear trend - using last quarter's value")
        forecast_margin = last_margin
        forecast_income = forecast_revenue * forecast_margin
        revenue_change = (forecast_revenue - last_revenue) / last_revenue
        income_change = (forecast_income - last_data['Net Income']) / last_data['Net Income']
        return {
            'quarter': f'Q{next_quarter} {next_year}',
            'forecast_type': 'TREND_ONLY',
            'seasonality_detected': False,
            'seasonality_results': self.seasonality_results,
            'forecast': {
                'revenue': forecast_revenue,
                'net_margin': forecast_margin,
                'net_income': forecast_income
            },
            'trend_analysis': {
                'current_trend': current_trend,
                'trend_slope': trend_slope,
                'trend_r_squared': trend_r2,
                'segments': self.trend_segments,
                'min_segment_length': self.min_segment_length
            },
            'changes': {
                'revenue_change': revenue_change,
                'income_change': income_change,
                'seasonal_revenue_pattern': 0,
                'seasonal_margin_pattern': 0
            },
            'confidence': {
                'pattern_sample_size': 0,
                'trend_r_squared': trend_r2
            },
            'seasonal_patterns': {}
        }
    
    def prepare_visualization_data(self, forecast_results):
        viz_data = self.clean_data.copy()
        viz_data['Net Income'] = viz_data['Total Revenue'] * viz_data['Net Margin']
        viz_data['Quarter_Year'] = viz_data['time'].dt.to_period('Q').astype(str)
        viz_data['Year'] = viz_data['time'].dt.year
        viz_data['Quarter'] = viz_data['time'].dt.quarter
        viz_data['Data_Type'] = 'Historical'
        viz_data['Seasonality_Detected'] = forecast_results['seasonality_detected']
        viz_data['Forecast_Type'] = forecast_results['forecast_type']
        viz_data['Revenue_QoQ_Growth'] = viz_data['Total Revenue'].pct_change() * 100
        viz_data['Income_QoQ_Growth'] = viz_data['Net Income'].pct_change() * 100
        viz_data['Margin_Change'] = viz_data['Net Margin'].diff()
        viz_data['Revenue_YoY_Growth'] = viz_data['Total Revenue'].pct_change(periods=4) * 100
        viz_data['Income_YoY_Growth'] = viz_data['Net Income'].pct_change(periods=4) * 100
        viz_data['Trend_Segment'] = 'Unknown'
        viz_data['Trend_Direction'] = 'Unknown'
        viz_data['Segment_Length'] = 0
        viz_data['Trend_R_Squared'] = 0.0
        for i, segment in enumerate(forecast_results['trend_analysis']['segments']):
            mask = (viz_data.index >= segment['start_idx']) & (viz_data.index <= segment['end_idx'])
            viz_data.loc[mask, 'Trend_Segment'] = f'Segment_{i+1}'
            viz_data.loc[mask, 'Trend_Direction'] = segment['trend_direction']
            viz_data.loc[mask, 'Segment_Length'] = segment['length']
            viz_data.loc[mask, 'Trend_R_Squared'] = segment['r_squared']
        last_date = viz_data['time'].iloc[-1]
        next_date = last_date + pd.DateOffset(months=3)
        forecast_row = {
            'time': next_date,
            'Total Revenue': forecast_results['forecast']['revenue'],
            'Net Margin': forecast_results['forecast']['net_margin'],
            'Net Income': forecast_results['forecast']['net_income'],
            'Quarter_Year': next_date.to_period('Q'),
            'Year': next_date.year,
            'Quarter': next_date.quarter,
            'Data_Type': 'Forecast',
            'Seasonality_Detected': forecast_results['seasonality_detected'],
            'Forecast_Type': forecast_results['forecast_type'],
            'Revenue_QoQ_Growth': forecast_results['changes']['revenue_change'] * 100,
            'Income_QoQ_Growth': forecast_results['changes']['income_change'] * 100,
            'Margin_Change': forecast_results['forecast']['net_margin'] - viz_data['Net Margin'].iloc[-1],
            'Revenue_YoY_Growth': np.nan,
            'Income_YoY_Growth': np.nan,
            'Trend_Segment': 'Forecast',
            'Trend_Direction': forecast_results['trend_analysis']['current_trend'],
            'Segment_Length': 1,
            'Trend_R_Squared': forecast_results['trend_analysis']['trend_r_squared']
        }
        viz_data['Seasonal_Revenue_Pattern'] = 0.0
        viz_data['Seasonal_Margin_Pattern'] = 0.0
        if forecast_results['seasonality_detected']:
            for pattern_key, pattern_data in forecast_results['seasonal_patterns'].items():
                from_q, to_q = pattern_key.split('_to_')
                from_quarter = int(from_q[1:])
                to_quarter = int(to_q[1:])
                mask = viz_data['Quarter'] == to_quarter
                viz_data.loc[mask, 'Seasonal_Revenue_Pattern'] = pattern_data['revenue_change'] * 100
                viz_data.loc[mask, 'Seasonal_Margin_Pattern'] = pattern_data['margin_change'] * 100
        forecast_df = pd.DataFrame([forecast_row])
        viz_data = pd.concat([viz_data, forecast_df], ignore_index=True)
        viz_data['Revenue_Billions'] = viz_data['Total Revenue'] / 1000
        viz_data['Income_Billions'] = viz_data['Net Income'] / 1000
        viz_data['Performance_Category'] = pd.cut(
            viz_data['Revenue_QoQ_Growth'].fillna(0), 
            bins=[-np.inf, -5, 0, 5, 10, np.inf], 
            labels=['Declining', 'Weak', 'Stable', 'Growing', 'Strong']
        )
        return viz_data
    
    def create_professional_dashboard(self, forecast_results):
        print("\n📊 Creating Professional Interactive Dashboard with PyGWalker...")
        viz_data = self.prepare_visualization_data(forecast_results)
        print(f"\n=== DASHBOARD DATA SUMMARY ===")
        print(f"Total data points: {len(viz_data)}")
        print(f"Historical quarters: {len(viz_data[viz_data['Data_Type'] == 'Historical'])}")
        print(f"Forecast quarters: {len(viz_data[viz_data['Data_Type'] == 'Forecast'])}")
        print(f"Seasonality detected: {forecast_results['seasonality_detected']}")
        print(f"Forecast type: {forecast_results['forecast_type']}")
        walker = pyg.walk(
            viz_data,
            spec="./financial_dashboard_config.json",
            kernel_computation=True,
            use_kernel_calc=True
        )
        print("\n🎯 PROFESSIONAL DASHBOARD FEATURES:")
        print("• Seasonality Detection Results")
        print("• Interactive Revenue & Net Income Analysis")
        print("• Trend Segment Visualization")
        print("• Seasonal Pattern Analysis (if detected)")
        print("• Growth Rate Comparisons")
        print("• Forecast vs Historical Performance")
        if forecast_results['seasonality_detected']:
            print("\n💡 SEASONAL ANALYSIS VISUALIZATIONS:")
            print("1. Seasonal patterns: Seasonal_Revenue_Pattern by Quarter")
            print("2. Seasonality strength: Revenue by Quarter (colored by Seasonality_Detected)")
        else:
            print("\n💡 TREND-ONLY ANALYSIS VISUALIZATIONS:")
            print("1. Trend analysis: Revenue by Trend_Direction")
            print("2. Performance tracking: Revenue_QoQ_Growth over time")
        return walker, viz_data
    
    def generate_professional_report(self, forecast_results):
        print("\n" + "="*80)
        print("                    PROFESSIONAL FINANCIAL FORECAST REPORT")
        print("="*80)
        print(f"\nSEASONALITY ASSESSMENT")
        print("-" * 50)
        if forecast_results['seasonality_detected']:
            print("✅ SEASONALITY DETECTED")
            print(f"   Statistical tests passed: {forecast_results['seasonality_results']['significant_tests']}/{forecast_results['seasonality_results']['total_valid_tests']}")
            print(f"   Analysis period: {forecast_results['seasonality_results']['years_analyzed']} years")
        else:
            print("❌ NO SIGNIFICANT SEASONALITY")
            print("   Data follows trend/random patterns")
            print(f"   Statistical tests passed: {forecast_results['seasonality_results']['significant_tests']}/{forecast_results['seasonality_results']['total_valid_tests']}")
        print(f"\nFORECAST METHODOLOGY")
        print("-" * 50)
        print(f"Forecast Type: {forecast_results['forecast_type']}")
        print(f"Forecast Period: {forecast_results['quarter']}")
        print(f"Minimum Trend Length: {forecast_results['trend_analysis']['min_segment_length']} quarters")
        print(f"\nKEY FINANCIAL PROJECTIONS")
        print("-" * 50)
        print(f"Revenue Forecast:    ${forecast_results['forecast']['revenue']:,.0f}M")
        print(f"Net Margin Forecast: {forecast_results['forecast']['net_margin']:.2%}")
        print(f"Net Income Forecast: ${forecast_results['forecast']['net_income']:,.0f}M")
        print(f"\nPERFORMANCE INDICATORS")
        print("-" * 50)
        print(f"Revenue Growth (QoQ): {forecast_results['changes']['revenue_change']:+.1%}")
        print(f"Income Growth (QoQ):  {forecast_results['changes']['income_change']:+.1%}")
        print(f"Current Trend:        {forecast_results['trend_analysis']['current_trend']}")
        print(f"Trend Confidence:     {forecast_results['trend_analysis']['trend_r_squared']:.1%}")
        if forecast_results['seasonality_detected']:
            print(f"\nSEASONAL PATTERNS APPLIED")
            print("-" * 50)
            print(f"Seasonal Revenue Pattern: {forecast_results['changes']['seasonal_revenue_pattern']:+.1%}")
            print(f"Seasonal Margin Pattern:  {forecast_results['changes']['seasonal_margin_pattern']:+.1%}")
        print("\n" + "="*80)
    
    def run_professional_analysis(self):
        print("🚀 Starting Professional Financial Forecasting Analysis...")
        self.load_and_prepare_data()
        print("\n📊 Generating Advanced Forecast with Seasonality Detection...")
        self.forecast_results = self.seasonal_forecast_with_advanced_trend()
        self.generate_professional_report(self.forecast_results)
        print("\n📈 Launching Professional Interactive Dashboard...")
        walker, viz_data = self.create_professional_dashboard(self.forecast_results)
        return walker, viz_data, self.forecast_results

# Usage Example
if __name__ == "__main__":
    # Initialize with seasonality detection
    forecaster = ProfessionalFinancialForecaster(
        'Apple.xlsx', 
        min_segment_length=5,
        seasonality_threshold=0.05  # 5% significance level
    )
    
    # Run complete analysis with seasonality detection
    walker, visualization_data, results = forecaster.run_professional_analysis()
