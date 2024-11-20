# %% [markdown]
# Library 
# 
# 

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
df = pd.read_csv('loan.csv')

df



# %%

print('\n Train Dataset summary : \n')

pd.concat([
            pd.DataFrame(df.count()).T.rename(index={0: 'count'}),
           pd.DataFrame(df.nunique()).T.rename(index={0: 'number of unique'}),
           pd.DataFrame(df.dtypes).T.rename(index={0: 'dtype'}),
           pd.DataFrame(df.isna().sum()).T.rename(index={0: 'null count'}),
           df.describe().drop('count')
          ]).T

# %%
df.drop(columns = ['id'], inplace = True)
df.head()

# %% [markdown]
# Converting categorical values to numerical

# %%
print('Categorical values of person_home_ownership : ', df.person_home_ownership.unique())
print('Categorical values of loan_intent : ', df.loan_intent.unique())
print('Categorical values of loan_grade : ', df.loan_grade.unique())
print('Categorical values of cb_person_default_on_file : ', df.cb_person_default_on_file.unique())


# %%
df_tensor = df.copy()

cb_person_default_on_file_mapping = {'Y' : 1, 'N' : 0}
loan_grade_mapping = {'A' : 7, 'B' : 6, 'C' : 5, 'D' : 4, 'E' : 3, 'F' : 2, 'G' : 1}
person_home_ownership_mapping = {'OWN' : 4, 'MORTGAGE' : 3, 'RENT' : 2, 'OTHER' : 1}

df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map(cb_person_default_on_file_mapping)
df['loan_grade'] = df['loan_grade'].map(loan_grade_mapping)
df['person_home_ownership'] = df['person_home_ownership'].map(person_home_ownership_mapping)

# %%
# Initialize the Label Encoder
label_encoder = LabelEncoder()

# Fit and transform the loan_intent column
df['loan_intent'] = label_encoder.fit_transform(df['loan_intent'])

# %%
df.duplicated().sum()

# %%
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

sns.set_style('darkgrid')

list_of_columns = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 
                   'loan_percent_income', 'cb_person_cred_hist_length']

fig, axs = plt.subplots(len(list_of_columns), 2, figsize=(15, 30))

for i, columns_name in enumerate(list_of_columns):
    sns.histplot(data=df, x=columns_name, hue='loan_status',
                 multiple='stack', palette='magma', ax=axs[i, 0])

    axs[i, 0].set_title(f'Histogram of {columns_name} by loan status')

    sns.boxplot(data=df, x=columns_name, palette='magma', ax=axs[i, 1])
  
    axs[i, 1].set_title(f'Boxplot of {columns_name}')

plt.tight_layout()

plt.show()

warnings.simplefilter(action='default', category=FutureWarning)

# %%

df = df[df.person_age < 100]
df = df[df.person_emp_length < 100]

# %%
warnings.simplefilter(action='ignore', category=FutureWarning)

plt.figure(figsize=(8, 5))

sns.histplot(data=df, x='loan_status', bins=2, color='purple', edgecolor='black')

plt.title('Distribution of Loan Status', fontsize=16)
plt.xlabel('Loan Status', fontsize=12)
plt.ylabel('Count', fontsize=12)

plt.show()

warnings.simplefilter(action='default', category=FutureWarning)


# %%
from sklearn.preprocessing import PolynomialFeatures

def create_features(train):
    df['income_to_age'] = df['person_income'] / df['person_age']
    df['loan_to_income'] = df['loan_amnt'] / df['person_income']
    df['rate_to_loan'] = df['loan_int_rate'] / df['loan_amnt']
    df['age_squared'] = df['person_age'] ** 2
    df['log_income'] = np.log1p(df['person_income'])
    df['age_credit_history_interaction'] = df['person_age'] * df['cb_person_cred_hist_length']
    df['high_loan_to_income'] = (df['loan_percent_income'] > 0.5).astype(int)
    df['loan_to_employment'] = df['loan_amnt'] / (df['person_emp_length'] + 1)
    df['is_new_credit_user'] = (df['cb_person_cred_hist_length'] < 2).astype(int)
    df['rate_to_grade'] = df.groupby('loan_grade')['loan_int_rate'].transform('mean')
    df['high_interest_rate'] = (df['loan_int_rate'] > df['loan_int_rate'].mean()).astype(int)
    df['age_to_credit_history'] = df['person_age'] / (df['cb_person_cred_hist_length'] + 1)
    df['income_home_mismatch'] = ((df['person_income'] > df['person_income'].quantile(0.8)) & (df['person_home_ownership'] == 'RENT')).astype(int)
    df['normalized_loan_amount'] = df.groupby('loan_intent')['loan_amnt'].transform(lambda x: (x - x.mean()) / x.std())
    df['income_to_loan'] = df['person_income'] / df['loan_amnt']
    df['age_cubed'] = df['person_age'] ** 3
    df['log_loan_amnt'] = np.log1p(df['loan_amnt'])
    df['age_interest_interaction'] = df['person_age'] * df['loan_int_rate']
    df['credit_history_to_age'] = df['cb_person_cred_hist_length'] / df['person_age']
    df['high_loan_amount'] = (df['loan_amnt'] > df['loan_amnt'].quantile(0.75)).astype(int)
    df['rate_to_credit_history'] = df['loan_int_rate'] / (df['cb_person_cred_hist_length'] + 1)
    df['intent_home_match'] = ((df['loan_intent'] == 'HOMEIMPROVEMENT') & (df['person_home_ownership'] == 'OWN')).astype(int)
    df['creditworthiness_score'] = (df['person_income'] / (df['loan_amnt'] * df['loan_int_rate'])) * (df['cb_person_cred_hist_length'] + 1)
    df['age_to_employment'] = df['person_age'] / (df['person_emp_length'] + 1)
    df['age_income_mismatch'] = ((df['person_age'] < 30) & (df['person_income'] > df['person_income'].quantile(0.9))).astype(int)
    df['rate_to_age'] = df['loan_int_rate'] / df['person_age']
    df['high_risk_flag'] = ((df['loan_percent_income'] > 0.4) &
                            (df['loan_int_rate'] > df['loan_int_rate'].mean()) &
                            (df['cb_person_default_on_file'] == 'Y')).astype(int)

    df['age_sin'] = np.sin(2 * np.pi * df['person_age'] / 100)
    df['age_cos'] = np.cos(2 * np.pi * df['person_age'] / 100)
    df['stability_score'] = (df['person_emp_length'] * df['person_income']) / (df['loan_amnt'] * (df['cb_person_cred_hist_length'] + 1))

    return train

df= create_features(df)

# %%
df.shape


# %%
plt.figure(figsize=(25,15))
sns.heatmap(df.corr(method='spearman'), annot=True, fmt=".1")
plt.show()

# %%
df.dtypes




# %%
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt


target = 'loan_status'
X = df.drop(columns=target)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

models = {
    "LightGBM": LGBMClassifier(n_jobs=-1, verbose=-1),
    "XGBoost": XGBClassifier(
        eval_metric='logloss',
        tree_method='hist',  # 使用 CPU 版本
        verbosity=0
    )
}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"{model_name} ROC AUC Score: {roc_auc:.4f}")
    
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(4, 3))
    plt.plot(fpr, tpr, label=f'{model_name} (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.show()

# %%
from sklearn.model_selection import cross_val_score, StratifiedKFold

for m_name, model in models.items():
    stratified_cv = StratifiedKFold(n_splits=5)
    cv_scores = cross_val_score(model, X, y, cv=stratified_cv, scoring='roc_auc')
    print(f'------------{m_name}')
    print("Cross-validation scores:", cv_scores)
    print("Mean cross-validation score:", cv_scores.mean())

# %%
from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(
    estimators=[(name, model) for name, model in models.items()],
    voting='soft'
)

stratified_cv = StratifiedKFold(n_splits=10)
cv_scores = cross_val_score(voting_clf, X, y, cv=stratified_cv, scoring='roc_auc')
print(f'------------Voting Classifier')
print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", cv_scores.mean())


