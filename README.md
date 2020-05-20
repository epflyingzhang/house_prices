# Predicting Housing Price


## Project Summary
In the Kaggle competition [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/). We are asked to build a model that to predict the house prices based on characteristics using the provided data set.

### Round 1 - Baseline Modeling

In this round of analysis, I looked at the house prices data, performed basic data cleaning and EDA, fit the data with a list of common regression models and compared their performances. I chose *XGBoostRegression*, without hyperparameter tuning, as the final model and achieved a PL score of 0.13921 (root mean square log error) on the test dataset, ranking at around 50% on Kaggle (as of May 2020).

**Data cleaning**  
- Drop 4 columns with over 80% of missing value
- Separated categorical and numeric variables according to datetype (except for 'MSSubClass')
- Replaced missing values with a new category 'ZZZ' for categorical variables
- Replaced missing values with median for numeric variables


**EDA**:   
- Visualized categorical variables and their relationships with target variable using countplot and barplot
- Visualized numeric variables and their relationships with target variable using histplot, box plot, and scatter plot with target variable
- Correlation matrix (heat map) for numeric variables
- Correlation and p-value between numeric features and target variable


**Feature engineering / selection**   
- Applied StandardScaler to all numeric variables
- Created new features from categorical variables using OneHotEncoding (with drop_first=True)
- All features are used for all models


**Modeling**  
- Used 5-fold cross-validation to compare a list of regression models (default parameters)
- XGBOOST and GradientBoost have the best performance in terms of 'neg_mean_absolute_error'
- Looked at the feature significance from XGBOOST but no further feature selection is performed


**Future improvements**  

- The modeling results from SVR and Multi-variate Linear Regression seem to be completely off. It is worth to investigate why. Multicolinearity could be a problem but I am not yet convinced this is the only reason.
- Some numerical variables can be transformed to categorical (e.g. Year, Month)
- We can spend more time on Linear Models (Ridge, LASSO). This would require more work on feature transformation (normality check + log / box-cox transformation), and feature extraction/selection. Would be good to practice these technics.
- Use GridSearch to tune hyperparameters.

### Round 2 - Feature Engineering

In this round of analysis, I focused on creating a workflow for data cleaning and feature engineering. More specifically, I have used the following technics:

- Ordinal encoding for features with ordinality
- Imputing missing values using correlated features
- Creating new features such as NeighborhoodRating
- Checking normality for target variable and important features, and applying log-transform
- Visualizing potential outliers and making careful decision for deletion
- Creating customized One Hot Encoding method to avoid information leakage from test data
- Implementing one subclass of sklearn BaseEstimator per data transformation step, and stacking them
into a sklearn Pipeline to allow for clean code and easy manipulation.

After feature engineering, I also performed PCA analysis due to the high dimensionality.

In the modeling part, I focused on LASSO regression model and used GridSearchCV to tune regularization parameter. We achieved a PL score of 0.12352 with the final model, which was a significant improvement from round 1 (0.13921 XGBOOST).



## Tools
**Python Version:** 3.7  
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn

## Resources
In order to improve the performance of the machine learning model, I borrowed ideas from some articles online.

- [House Prices Solution [top 1%]](https://www.kaggle.com/jesucristo/1-house-prices-solution-top-1)
- [House prices: Lasso, XGBoost, and a detailed EDA](https://www.kaggle.com/erikbruin/house-prices-lasso-xgboost-and-a-detailed-eda)
