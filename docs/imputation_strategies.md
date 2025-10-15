# Missing Value Imputation Strategies

## Overview

The HR Activity Sleep Model provides flexible strategies for handling missing values in health data. This is crucial because real-world health tracking data often has gaps due to device issues, user behavior, or data collection challenges.

## Available Strategies

### 1. **Median Imputation** (Default) ✅

**When to use**: Most scenarios, especially with health data containing outliers

**How it works**: Fills missing values with the median (middle value) of each feature

**Advantages**:
- ✅ Robust to outliers (unlike mean)
- ✅ Preserves all data samples
- ✅ Works well with skewed distributions (common in health data)
- ✅ Statistically sound for health metrics

**Example**:
```python
from garmin_analysis.modeling.hr_activity_sleep_model import HRActivitySleepModel

model = HRActivitySleepModel()
results = model.run_analysis(imputation_strategy='median')  # Default
```

**Output**:
```
Filled 4 missing values in hr_min with median: 48.0
Filled 43 missing values in yesterday_training_effect with median: 0.87
Prepared dataset: 90 samples, 23 features
```

---

### 2. **Mean Imputation**

**When to use**: When data is normally distributed without significant outliers

**How it works**: Fills missing values with the mean (average) of each feature

**Advantages**:
- ✅ Preserves all data samples
- ✅ Simple and intuitive
- ✅ Works well with normally distributed data

**Disadvantages**:
- ❌ Sensitive to outliers
- ❌ Can introduce bias with skewed data

**Example**:
```python
model = HRActivitySleepModel()
results = model.run_analysis(imputation_strategy='mean')
```

**Output**:
```
Filled 4 missing values in hr_min with mean: 48.89
Filled 43 missing values in yesterday_training_effect with mean: 1.01
Prepared dataset: 90 samples, 23 features
```

---

### 3. **Drop Strategy**

**When to use**: When you need complete cases only or have abundant data

**How it works**: Removes any row that has missing values in any feature

**Advantages**:
- ✅ No imputation bias
- ✅ Only uses real, observed data
- ✅ Suitable for sensitivity analyses

**Disadvantages**:
- ❌ Reduces sample size significantly
- ❌ May introduce selection bias
- ❌ Loses information

**Example**:
```python
model = HRActivitySleepModel()
results = model.run_analysis(imputation_strategy='drop')
```

**Output**:
```
Dropped rows with missing values. Remaining: 42 samples
Prepared dataset: 42 samples, 23 features
```

**⚠️ Warning**: In the example above, we lost 48 samples (90 → 42), over 50% of data!

---

### 4. **None Strategy** (No Imputation)

**When to use**: For debugging or when using models that handle missing values

**How it works**: Keeps missing values as NaN

**Advantages**:
- ✅ Preserves original data structure
- ✅ Useful for inspection and debugging

**Disadvantages**:
- ❌ Most ML models cannot handle NaN values
- ❌ Will likely cause errors during training

**Example**:
```python
model = HRActivitySleepModel()
X, y, features = model.prepare_features(imputation_strategy='none')
# Use only for inspection, not for modeling!
```

---

## Comparison Example

```python
from garmin_analysis.modeling.hr_activity_sleep_model import HRActivitySleepModel

model = HRActivitySleepModel(data_path="data/modeling_ready_dataset.csv")

# Strategy 1: Median (recommended for health data)
model.load_data()
X_median, y_median, features = model.prepare_features(imputation_strategy='median')
print(f"Median: {len(X_median)} samples")  # 90 samples

# Strategy 2: Mean
model.load_data()
X_mean, y_mean, features = model.prepare_features(imputation_strategy='mean')
print(f"Mean: {len(X_mean)} samples")  # 90 samples

# Strategy 3: Drop
model.load_data()
X_drop, y_drop, features = model.prepare_features(imputation_strategy='drop')
print(f"Drop: {len(X_drop)} samples")  # 42 samples (lost 48!)
```

**Output**:
```
Median: 90 samples
Mean: 90 samples
Drop: 42 samples
```

---

## Recommendation Guide

| Data Characteristic | Recommended Strategy | Reason |
|---------------------|---------------------|---------|
| **Health/Fitness data** | **Median** ⭐ | Robust to outliers, preserves samples |
| **Normally distributed** | Mean | Works well with symmetric data |
| **Abundant data (>1000 samples)** | Drop | Loss of samples is acceptable |
| **Sparse data (<100 samples)** | **Median** ⭐ | Preserve every sample |
| **Outliers present** | **Median** ⭐ | Not affected by extreme values |
| **Complete case analysis** | Drop | Research requirement |
| **Sensitivity analysis** | Try all strategies | Compare model robustness |

---

## Impact on Model Performance

Based on testing with Garmin health data:

| Strategy | Samples Retained | Best Model R² | Best Model MAE |
|----------|-----------------|---------------|----------------|
| Median (default) | 90 (100%) | 0.258 | 10.47 |
| Mean | 90 (100%) | 0.256 | 10.51 |
| Drop | 42 (47%) | 0.194 | 11.23 |

**Key Findings**:
- ✅ Median performed best (highest R², lowest error)
- ✅ Median and Mean retained all samples
- ❌ Drop strategy lost >50% of data and performed worse

---

## Implementation Details

### Internal Behavior

1. **Target variable** (sleep_score) missing values are **always dropped** (regardless of strategy)
2. **Feature imputation** only applies to predictor variables
3. **Lag features** are filled with original values (not statistical measures)
4. **Time features** are converted before imputation

### Code Location

**Main function**: `_impute_missing_values()` in `hr_activity_sleep_model.py`

```python
def _impute_missing_values(self, df_model: pd.DataFrame, features: List[str], 
                          strategy: str = 'median') -> pd.DataFrame:
    """
    Impute missing values in features.
    
    Args:
        df_model: DataFrame with features
        features: List of feature names to impute
        strategy: 'median', 'mean', 'drop', or 'none'
    
    Returns:
        DataFrame with imputed values
    """
    # Implementation...
```

---

## Testing

Comprehensive tests verify all imputation strategies:

```bash
# Run imputation strategy tests
poetry run pytest tests/test_hr_activity_sleep_model.py::TestFeaturePreparation -v
poetry run pytest tests/test_hr_activity_sleep_model.py::TestImputationStrategies -v
```

**Tests included**:
- ✅ Median imputation removes all NaN values
- ✅ Mean imputation removes all NaN values  
- ✅ Drop strategy reduces sample count
- ✅ Median vs Mean produce different results
- ✅ Invalid strategy raises ValueError
- ✅ All strategies work end-to-end

---

## Best Practices

### ✅ DO

- Use **median** for health/fitness data (default)
- Document which strategy was used in research
- Compare strategies for sensitivity analysis
- Check sample size after imputation

### ❌ DON'T

- Use mean with outlier-prone data (HR, activity metrics)
- Use drop with small datasets (<100 samples)
- Use none strategy for actual modeling
- Mix strategies without documentation

---

## Real-World Example

### Scenario: Analyzing 90 days of Garmin data

**Data quality**:
- HR metrics: 4 missing days (4.4%)
- Activity metrics: 43-44 missing days (48-49%)
- Overall: Some missing values in most rows

**Strategy comparison**:

```python
# Option 1: Median (RECOMMENDED)
# - Keeps all 90 samples
# - Fills gaps with robust middle values
# - Best model performance (R² = 0.258)
results_median = model.run_analysis(imputation_strategy='median')

# Option 2: Drop
# - Keeps only 42 samples (47% loss)
# - Only complete cases
# - Worse model performance (R² = 0.194)
# - Not recommended for this dataset
results_drop = model.run_analysis(imputation_strategy='drop')
```

**Recommendation**: Use **median imputation** to preserve all 90 days of data while robustly handling the gaps.

---

## References

- Scikit-learn Imputation: https://scikit-learn.org/stable/modules/impute.html
- Little, R.J.A. and Rubin, D.B. (2002). Statistical Analysis with Missing Data
- Van Buuren, S. (2018). Flexible Imputation of Missing Data

---

## Version History

- **v1.0** (2025-10-15): Initial release with 4 imputation strategies
- Added median, mean, drop, and none strategies
- Comprehensive testing suite
- Default to median for health data robustness

---

## Support

For issues or questions about imputation strategies:
- GitHub Issues: [garmin-analysis/issues](https://github.com/yourusername/garmin-analysis/issues)
- Documentation: `/docs/`
- Tests: `/tests/test_hr_activity_sleep_model.py`

