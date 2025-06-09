# Avazu Click-Through Rate (CTR) Prediction

A machine learning project for predicting whether a user will click on an online advertisement using the Avazu CTR prediction dataset from Kaggle.

## Overview

This project implements a binary classification solution to predict click-through rates on online advertisements. The dataset contains 40+ million samples with user behavior and ad characteristics.

## Dataset

- **Source**: Kaggle Avazu CTR Prediction Competition
- **Size**: ~40 million training samples, 24 features
- **Target**: Binary classification (click/no-click)
- **Features**: Anonymous categorical variables (C1, C14-C21), device info, site info, app info, timestamps

## Key Features

### Data Analysis & Preprocessing
- Exploratory data analysis with visualizations
- Handling class imbalance (83% no-click, 17% click)
- Feature engineering from timestamp data (hour, day, month, day_of_week)
- Outlier detection and treatment using IQR method
- Label encoding for categorical variables
- Feature selection based on correlation analysis

### Models Implemented
1. **Dummy Classifier** (Baseline)
2. **CatBoost Classifier** 
3. **XGBoost Classifier**
4. **Hyperparameter-tuned CatBoost** (using Optuna)

### Data Balancing
- Random undersampling to address class imbalance
- Comparison of model performance on balanced vs imbalanced data

## Results

### Complete Model Performance Summary

| Model | Dataset | Test Accuracy | Precision | Recall | F1 Score | AUC Score | Log Loss |
|-------|---------|---------------|-----------|--------|----------|-----------|----------|
| Dummy Classifier | Imbalanced | 83.0% | 0.000 | - | - | 0.500 | 6.121 |
| CatBoost | Imbalanced | 85.1% | 0.154 | 0.828 | 0.259 | **0.884** | 0.322 |
| XGBoost | Imbalanced | 83.6% | 0.080 | 0.648 | 0.142 | 0.765 | 0.390 |
| Dummy Classifier | Balanced | 50.0% | 1.000 | 0.500 | 0.667 | 0.500 | 18.011 |
| CatBoost | Balanced | 82.4% | 0.889 | 0.786 | 0.834 | **0.908** | 0.434 |
| XGBoost | Balanced | 64.8% | 0.744 | 0.625 | 0.679 | 0.715 | 0.633 |
| **Optimized CatBoost** | Balanced | **76.8%** | **0.835** | **0.736** | **0.782** | **0.851** | **0.489** |

### Hyperparameter Optimization Results
- **Optuna optimization**: Achieved best AUC score of 0.760 after hyperparameter tuning
- **Final optimized model**: CatBoost with Bayesian bootstrap, depth=9, learning_rate=0.254

## Key Insights

- **Best overall model**: CatBoost on balanced data achieved highest AUC (0.908) and excellent precision-recall balance
- **Hyperparameter tuning**: Optimized CatBoost provides the best balance of metrics with AUC=0.851
- **Data balancing impact**: Significantly improved model generalization and reduced class bias
- **Imbalanced vs Balanced**: Balanced data models show better generalization despite sometimes lower raw accuracy
- **Important features**: Device characteristics, site information, and time-based features dominate predictions
- **Feature engineering**: Removed redundant features (C15, C16, month, C17) and created temporal features

## Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn catboost xgboost imbalanced-learn optuna kaggle
```

## Usage

1. Download the Avazu dataset from Kaggle
2. Place your `kaggle.json` API credentials in the project directory
3. Run the notebook cells sequentially
4. Models will be trained and evaluated automatically
5. Final predictions will be saved to `submission.csv`

## Files

- `avazu_ctr_pred.ipynb` - Main notebook with complete analysis and modeling
- `submission.csv` - Final predictions for test set (generated)

## Notes

- The notebook uses a sample of 500K-2M records for faster processing
- Full dataset processing is demonstrated for production deployment
- Hyperparameter tuning is implemented for model optimization
- Feature importance analysis included for model interpretability

This implementation demonstrates a complete machine learning pipeline from data exploration to model deployment for a real-world advertising prediction problem. 