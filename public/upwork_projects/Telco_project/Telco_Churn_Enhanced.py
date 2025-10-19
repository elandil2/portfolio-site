# %% Cell 1: Import Libraries and Setup
##############################
# Telco Customer Churn Prediction - Enhanced Version
##############################

# Problem: Develop a machine learning model to predict customers likely to churn from a telecom company.
# This enhanced version includes advanced EDA visualizations, Optuna for hyperparameter tuning, and additional models.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import optuna
from optuna import Trial, study
import warnings
warnings.simplefilter(action="ignore")

# Set plot style for fancy visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 300)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# %% Cell 2: Load and Initial Data Inspection
df = pd.read_csv("Telco-Customer-Churn.csv")

# Convert TotalCharges to numeric, handle errors
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

# Convert Churn to binary
df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

print("Dataset Shape:", df.shape)
print("Data Types:\n", df.dtypes)
print("First 5 rows:\n", df.head())
print("Missing Values:\n", df.isnull().sum())

# %% Cell 3: Enhanced EDA - General Overview
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.select_dtypes(include=[np.number]).quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

# %% Cell 4: Variable Types and Analysis Functions
def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# %% Cell 5: Enhanced Categorical Variable Analysis
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        dataframe[col_name].value_counts().plot(kind='bar', ax=ax[0], color=sns.color_palette("husl", len(dataframe[col_name].unique())))
        ax[0].set_title(f'{col_name} Distribution')
        ax[0].set_ylabel('Count')
        dataframe[col_name].value_counts().plot(kind='pie', ax=ax[1], autopct='%1.1f%%', colors=sns.color_palette("husl", len(dataframe[col_name].unique())))
        ax[1].set_title(f'{col_name} Ratio')
        plt.tight_layout()
        plt.show()

for col in cat_cols:
    cat_summary(df, col, plot=True)

# %% Cell 6: Enhanced Numeric Variable Analysis
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        fig, ax = plt.subplots(2, 2, figsize=(14, 10))

        # Histogram
        sns.histplot(dataframe[numerical_col], ax=ax[0,0], kde=True, color='skyblue')
        ax[0,0].set_title(f'{numerical_col} Histogram')

        # Box plot
        sns.boxplot(y=dataframe[numerical_col], ax=ax[0,1], color='lightgreen')
        ax[0,1].set_title(f'{numerical_col} Box Plot')

        # Violin plot
        sns.violinplot(y=dataframe[numerical_col], ax=ax[1,0], color='salmon')
        ax[1,0].set_title(f'{numerical_col} Violin Plot')

        # Cumulative distribution
        sns.ecdfplot(dataframe[numerical_col], ax=ax[1,1], color='purple')
        ax[1,1].set_title(f'{numerical_col} ECDF')

        plt.tight_layout()
        plt.show()

for col in num_cols:
    num_summary(df, col, plot=True)

# %% Cell 7: Target Variable Analysis
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Churn", col)

for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)

# %% Cell 8: Advanced Visualizations - Correlation and Relationships
# Correlation Matrix with enhanced heatmap
f, ax = plt.subplots(figsize=[18, 13])
mask = np.triu(np.ones_like(df[num_cols].corr(), dtype=bool))
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="coolwarm", mask=mask, linewidths=.5)
ax.set_title("Advanced Correlation Matrix", fontsize=20, fontweight='bold')
plt.show()

# Pair plot for numeric variables colored by Churn
sns.pairplot(df[num_cols + ['Churn']], hue='Churn', palette='Set1', diag_kind='kde')
plt.suptitle('Pair Plot of Numeric Variables by Churn', y=1.02, fontsize=16)
plt.show()

# Violin plots for numeric variables by Churn
fig, axes = plt.subplots(1, len(num_cols), figsize=(20, 6))
for i, col in enumerate(num_cols):
    sns.violinplot(x='Churn', y=col, data=df, ax=axes[i], palette='muted')
    axes[i].set_title(f'{col} by Churn')
plt.tight_layout()
plt.show()

# %% Cell 9: Missing Values and Outlier Handling
# Fill missing TotalCharges with MonthlyCharges
df.iloc[df[df["TotalCharges"].isnull()].index, df.columns.get_loc('TotalCharges')] = df[df["TotalCharges"].isnull()]["MonthlyCharges"]

# Tenure adjustment
df["tenure"] = df["tenure"] + 1

# Outlier handling
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

# %% Cell 10: Feature Engineering
# Tenure year categories
df.loc[(df["tenure"]>=0) & (df["tenure"]<=12),"NEW_TENURE_YEAR"] = "0-1 Year"
df.loc[(df["tenure"]>12) & (df["tenure"]<=24),"NEW_TENURE_YEAR"] = "1-2 Year"
df.loc[(df["tenure"]>24) & (df["tenure"]<=36),"NEW_TENURE_YEAR"] = "2-3 Year"
df.loc[(df["tenure"]>36) & (df["tenure"]<=48),"NEW_TENURE_YEAR"] = "3-4 Year"
df.loc[(df["tenure"]>48) & (df["tenure"]<=60),"NEW_TENURE_YEAR"] = "4-5 Year"
df.loc[(df["tenure"]>60) & (df["tenure"]<=72),"NEW_TENURE_YEAR"] = "5-6 Year"

# Engaged customers
df["NEW_Engaged"] = df["Contract"].apply(lambda x: 1 if x in ["One year","Two year"] else 0)

# No protection
df["NEW_noProt"] = df.apply(lambda x: 1 if (x["OnlineBackup"] != "Yes") or (x["DeviceProtection"] != "Yes") or (x["TechSupport"] != "Yes") else 0, axis=1)

# Young not engaged
df["NEW_Young_Not_Engaged"] = df.apply(lambda x: 1 if (x["NEW_Engaged"] == 0) and (x["SeniorCitizen"] == 0) else 0, axis=1)

# Total services
df['NEW_TotalServices'] = (df[['PhoneService', 'InternetService', 'OnlineSecurity',
                               'OnlineBackup', 'DeviceProtection', 'TechSupport',
                               'StreamingTV', 'StreamingMovies']]== 'Yes').sum(axis=1)

# Any streaming
df["NEW_FLAG_ANY_STREAMING"] = df.apply(lambda x: 1 if (x["StreamingTV"] == "Yes") or (x["StreamingMovies"] == "Yes") else 0, axis=1)

# Auto payment
df["NEW_FLAG_AutoPayment"] = df["PaymentMethod"].apply(lambda x: 1 if x in ["Bank transfer (automatic)","Credit card (automatic)"] else 0)

# Average charges
df["NEW_AVG_Charges"] = df["TotalCharges"] / df["tenure"]

# Increase
df["NEW_Increase"] = df["NEW_AVG_Charges"] / df["MonthlyCharges"]

# Service fee per service
df["NEW_AVG_Service_Fee"] = df["MonthlyCharges"] / (df['NEW_TotalServices'] + 1)

# %% Cell 11: Encoding
cat_cols, num_cols, cat_but_car = grab_col_names(df)

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]

for col in binary_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["Churn", "NEW_TotalServices"]]

df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# %% Cell 12: Scaling
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# %% Cell 13: Base Model Training
y = df["Churn"]
X = df.drop(["Churn","customerID"], axis=1)

models = [('LR', LogisticRegression(random_state=12345)),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier(random_state=12345)),
          ('RF', RandomForestClassifier(random_state=12345)),
          ('XGB', XGBClassifier(random_state=12345)),
          ("LightGBM", LGBMClassifier(random_state=12345)),
          ("CatBoost", CatBoostClassifier(verbose=False, random_state=12345))]

for name, model in models:
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")

# %% Cell 14: Optuna Hyperparameter Tuning
# Random Forest
def rf_objective(trial):
    max_depth = trial.suggest_int("max_depth", 5, 50)
    max_features = trial.suggest_categorical("max_features", [3, 5, 7, "sqrt", None])
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    n_estimators = trial.suggest_int("n_estimators", 100, 1000)
    model = RandomForestClassifier(max_depth=max_depth, max_features=max_features,
                                   min_samples_split=min_samples_split, n_estimators=n_estimators, random_state=17)
    cv_results = cross_validate(model, X, y, cv=5, scoring="roc_auc")
    return cv_results['test_score'].mean()

rf_study = optuna.create_study(direction="maximize")
rf_study.optimize(rf_objective, n_trials=50)
rf_best_params = rf_study.best_params
rf_final = RandomForestClassifier(**rf_best_params, random_state=17).fit(X, y)

# XGBoost
def xgb_objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
    max_depth = trial.suggest_int("max_depth", 3, 12)
    n_estimators = trial.suggest_int("n_estimators", 100, 1000)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1)
    model = XGBClassifier(learning_rate=learning_rate, max_depth=max_depth,
                          n_estimators=n_estimators, colsample_bytree=colsample_bytree, random_state=17)
    cv_results = cross_validate(model, X, y, cv=5, scoring="roc_auc")
    return cv_results['test_score'].mean()

xgb_study = optuna.create_study(direction="maximize")
xgb_study.optimize(xgb_objective, n_trials=50)
xgb_best_params = xgb_study.best_params
xgb_final = XGBClassifier(**xgb_best_params, random_state=17).fit(X, y)

# LightGBM
def lgbm_objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
    n_estimators = trial.suggest_int("n_estimators", 100, 1000)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1)
    num_leaves = trial.suggest_int("num_leaves", 20, 100)
    model = LGBMClassifier(learning_rate=learning_rate, n_estimators=n_estimators,
                           colsample_bytree=colsample_bytree, num_leaves=num_leaves, random_state=17)
    cv_results = cross_validate(model, X, y, cv=5, scoring="roc_auc")
    return cv_results['test_score'].mean()

lgbm_study = optuna.create_study(direction="maximize")
lgbm_study.optimize(lgbm_objective, n_trials=50)
lgbm_best_params = lgbm_study.best_params
lgbm_final = LGBMClassifier(**lgbm_best_params, random_state=17).fit(X, y)

# CatBoost
def catboost_objective(trial):
    iterations = trial.suggest_int("iterations", 200, 1000)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
    depth = trial.suggest_int("depth", 3, 10)
    model = CatBoostClassifier(iterations=iterations, learning_rate=learning_rate,
                               depth=depth, verbose=False, random_state=17)
    cv_results = cross_validate(model, X, y, cv=5, scoring="roc_auc")
    return cv_results['test_score'].mean()

catboost_study = optuna.create_study(direction="maximize")
catboost_study.optimize(catboost_objective, n_trials=50)
catboost_best_params = catboost_study.best_params
catboost_final = CatBoostClassifier(**catboost_best_params, verbose=False, random_state=17).fit(X, y)

# %% Cell 15: Ensemble Model
voting_clf = VotingClassifier(estimators=[('rf', rf_final), ('xgb', xgb_final), ('lgbm', lgbm_final), ('cat', catboost_final)], voting='soft')
voting_clf.fit(X, y)

# %% Cell 16: Final Model Evaluation
final_models = [('RF_Optuna', rf_final), ('XGB_Optuna', xgb_final), ('LGBM_Optuna', lgbm_final), ('CatBoost_Optuna', catboost_final), ('Voting', voting_clf)]

for name, model in final_models:
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")

# %% Cell 17: Feature Importance
def plot_importance(model, features, num=20, save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_final, X)
plot_importance(xgb_final, X)
plot_importance(lgbm_final, X)
plot_importance(catboost_final, X)

# %% Cell 18: Summary Report
print("### Telco Churn Prediction - Enhanced Project Summary ###")
print("1. Enhanced EDA with advanced visualizations including pair plots, violin plots, and custom-styled heatmaps.")
print("2. Implemented Optuna for hyperparameter tuning, replacing GridSearchCV.")
print("3. Added ensemble model (Voting Classifier) for improved performance.")
print("4. Structured code with cell-like sections for easy Jupyter notebook conversion.")
print("5. Best performing model based on AUC:", max(final_models, key=lambda x: cross_validate(x[1], X, y, cv=5, scoring="roc_auc")['test_score'].mean())[0])