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
from joblib import dump
import os

# 保存路径为当前文件夹
model_path = './'  # 或者直接使用空字符串 model_path = ''

# 假设您已经训练好了 LightGBM 和 XGBoost 模型
lgbm_model = models['LightGBM']
xgb_model = models['XGBoost']

# 保存模型到当前文件夹
dump(lgbm_model, os.path.join(model_path, 'light_gbm_model.joblib'))
print("LightGBM 模型已保存至:", os.path.join(model_path, 'light_gbm_model.joblib'))

dump(xgb_model, os.path.join(model_path, 'xgb_model.joblib'))
print("XGBoost 模型已保存至:", os.path.join(model_path, 'xgb_model.joblib'))


# %% [markdown]
# Opportunity 1: 客户分层分析
# 描述：通过收入和风险分数对客户进行分层，识别不同客户群体的特征，为后续策略（如定价、营销、风险管理）提供支持。
# 

# %%
def analyze_loan_management():
    # 1. 客户分层分析
    def segment_customers():
        # 使用模型预测的违约概率
        default_proba = models['LightGBM'].predict_proba(X)[:, 1]
        
        # 基于收入和风险分层
        segments = pd.DataFrame({
            'income': X['person_income'],
            'risk_score': default_proba
        })
        
        #客户分层逻辑： action1:
        #使用 LightGBM 模型预测违约概率（default_proba）。
        #根据收入水平（income）和风险分数（risk_score），使用 pd.qcut 将客户分为低、中、高三个层级。
        #使用组合逻辑生成客户分层标签（如 Low_Low, Medium_High）。
        # 定义分层标准
        income_labels = ['Low', 'Medium', 'High']
        risk_labels = ['Low', 'Medium', 'High']
        
        segments['income_level'] = pd.qcut(segments['income'], 
                                         q=3, 
                                         labels=income_labels)
        segments['risk_level'] = pd.qcut(segments['risk_score'], 
                                       q=3, 
                                       labels=risk_labels)
        
        # 组合分层
        segments['segment'] = segments.apply(
            lambda x: f"{x['income_level']}_{x['risk_level']}", 
            axis=1
        )
        
        return segments
    
    # action2. 计算每个分层的关键指标
    def calculate_segment_metrics(segments):
        metrics = {}
        for segment in segments['segment'].unique():
            segment_data = segments[segments['segment'] == segment]
            metrics[segment] = {
                'count': len(segment_data),
                'avg_income': segment_data['income'].mean(),
                'avg_risk': segment_data['risk_score'].mean(),
                'std_income': segment_data['income'].std(),
                'std_risk': segment_data['risk_score'].std()
            }
        return metrics
    
    # 执行分析
    segments = segment_customers()
    metrics = calculate_segment_metrics(segments)
    
    return {
        'segments': segments,
        'metrics': metrics
    }

# 运行分析
results = analyze_loan_management()

# 显示分层结果
print("Customer Segment Analysis:")
for segment, data in results['metrics'].items():
    print(f"\n{segment}:")
    print(f"Count: {data['count']:,}")
    print(f"Average Income: ${data['avg_income']:,.2f} (±${data['std_income']:,.2f})")
    print(f"Average Risk Score: {data['avg_risk']:.3f} (±{data['std_risk']:.3f})")



# %%
#action3:
# 可视化
plt.figure(figsize=(15, 5))

# 1. 分层客户数量分布
plt.subplot(1, 3, 1)
segment_counts = pd.Series({k: v['count'] for k, v in results['metrics'].items()})
segment_counts.plot(kind='bar')
plt.title('Customer Segments Distribution')
plt.xticks(rotation=45)
plt.ylabel('Count')

# 2. 各分层平均风险
plt.subplot(1, 3, 2)
segment_risks = pd.Series({k: v['avg_risk'] for k, v in results['metrics'].items()})
segment_risks.plot(kind='bar', color='orange')
plt.title('Average Risk by Segment')
plt.xticks(rotation=45)
plt.ylabel('Risk Score')

# 3. 收入与风险关系
plt.subplot(1, 3, 3)
plt.scatter(results['segments']['income'], 
           results['segments']['risk_score'],
           alpha=0.3,
           c=results['segments']['risk_score'],
           cmap='YlOrRd')
plt.colorbar(label='Risk Score')
plt.xlabel('Income')
plt.ylabel('Risk Score')
plt.title('Risk vs Income Distribution')

plt.tight_layout()
plt.show()

# %% [markdown]
# Opportunity 2: 定价策略优化
# 描述：通过基于收入和风险水平的定价矩阵优化贷款利率，提升盈利能力，同时控制风险。

# %%
def optimize_pricing_strategy(customer_data):
    # action1 :构建风险定价矩阵,定义了以收入和风险为基础的利率矩阵（base_rates），为不同组合的客户提供基础利率
    
    def create_pricing_matrix():
        # 基础利率矩阵 (示例数值)
        base_rates = {
            'Low': {
                'Low': 0.05,    # 低风险低收入
                'Medium': 0.045, # 低风险中等收入
                'High': 0.04    # 低风险高收入
            },
            'Medium': {
                'Low': 0.07,
                'Medium': 0.065,
                'High': 0.06
            },
            'High': {
                'Low': 0.09,
                'Medium': 0.085,
                'High': 0.08
            }
        }
        return base_rates
    
    # action2 :计算风险调整系数:根据客户分层的平均风险和收入稳定性（收入标准差/均值）计算风险调整系数。
    def calculate_risk_adjustments(segment_metrics):
        adjustments = {}
        for segment, metrics in segment_metrics.items():
            # 基于风险分数和收入稳定性的调整
            risk_factor = metrics['avg_risk']
            income_stability = metrics['std_income'] / metrics['avg_income']
            
            # 综合调整系数
            adjustment = (1 + risk_factor) * (1 + income_stability * 0.1)
            adjustments[segment] = adjustment
        return adjustments
    
    # action3 :生成最终定价建议 :将基础利率与风险调整系数结合，生成最终贷款利率及建议利率范围。
    def generate_pricing_recommendations(base_rates, adjustments):
        recommendations = {}
        for segment, adjustment in adjustments.items():
            income_level, risk_level = segment.split('_')
            base_rate = base_rates[risk_level][income_level]
            final_rate = base_rate * adjustment
            
            recommendations[segment] = {
                'base_rate': base_rate,
                'risk_adjustment': adjustment,
                'final_rate': final_rate,
                'rate_range': (final_rate * 0.95, final_rate * 1.05)
            }
        return recommendations
    
    # 执行定价策略优化
    base_rates = create_pricing_matrix()
    risk_adjustments = calculate_risk_adjustments(customer_data['metrics'])
    pricing_recommendations = generate_pricing_recommendations(base_rates, risk_adjustments)
    
    return pricing_recommendations

# 应用定价策略
pricing_strategy = optimize_pricing_strategy(results)

# 显示定价建议
print("\nPricing Strategy Recommendations:")
for segment, rates in pricing_strategy.items():
    print(f"\n{segment}:")
    print(f"Base Rate: {rates['base_rate']:.2%}")
    print(f"Risk Adjustment: {rates['risk_adjustment']:.3f}x")
    print(f"Final Rate: {rates['final_rate']:.2%}")
    print(f"Suggested Range: {rates['rate_range'][0]:.2%} - {rates['rate_range'][1]:.2%}")

# %% [markdown]
# Opportunity 3: 交叉销售机会识别
# 描述：识别中高收入、低风险的客户群体，作为交叉销售目标客户。

# %%
segments = results['segments']
#action 1 :筛选交叉销售目标客户：筛选中高收入（Medium 或 High）且低风险（Low）的客户。排除已识别的高价值客户（如高收入低风险的客户）。
cross_sell_segments = segments[
    (segments['income_level'].isin(['Medium', 'High'])) &  # 中高收入
    (segments['risk_level'] == 'Low') &                    # 低风险
    ~((segments['income_level'] == 'High') &               # 排除已识别的高价值客户
      (segments['risk_level'] == 'Low'))
]

# action2 :定义交叉销售行动建议：明确目标客户清单的生成标准。推荐适合该群体的产品（如信用卡升级、理财产品等）。提出营销活动的设计渠道（如推送、客户经理沟通等）
def define_cross_sell_actions(cross_sell_segments):
    actions = {
        'target_list': {
            'priority': 'High',
            'description': '生成目标客户清单',
            'criteria': {
                'income_range': '中高收入段',
                'risk_threshold': '低风险分数',
                'product_holding': '现有产品持有情况'
            }
        },
        'product_recommendations': {
            'priority': 'Medium',
            'description': '制定产品推荐策略',
            'products': [
                '信用卡升级',
                '理财产品',
                '保险服务'
            ]
        },
        'campaign_design': {
            'priority': 'Medium',
            'description': '设计营销活动',
            'channels': [
                '手机银行推送',
                '客户经理一对一沟通',
                '精准营销邮件'
            ]
        },
        'success_metrics': {
            'priority': 'Low',
            'description': '设定成功指标',
            'kpis': [
                '交叉销售转化率',
                '客均产品持有数增长',
                '客户价值提升度'
            ]
        }
    }
    return actions

# 执行actions定义
cross_sell_actions = define_cross_sell_actions(cross_sell_segments)

# %%
#Opportunity 1: Enhanced Credit Risk Assessment



# %% [markdown]
# Opportunity 4: Enhanced Credit Risk Assessment
# Description: Improve the accuracy of predicting which loan applicants might default, thus reducing financial losses.
# 
# Actions:
# 
# Model Refinement: Utilize advanced machine learning techniques such as ensemble models (Random Forest, Gradient Boosting) or deep learning models to improve prediction accuracy.
# Feature Engineering: Experiment with additional features that could influence loan repayment, such as economic indicators or more detailed employment history.

# %%
import pandas as pd
import numpy as np
from joblib import load
import os

def load_models():
    # 当前文件夹路径
    model_path = './'  # 或者直接使用空字符串 model_path = ''
    
    # 加载保存的模型
    light_gbm_model = load(os.path.join(model_path, 'light_gbm_model.joblib'))
    xgb_model = load(os.path.join(model_path, 'xgb_model.joblib'))
    
    # 返回加载的模型
    return {
        'LightGBM': light_gbm_model,
        'XGBoost': xgb_model
    }

# Load dataset - Assuming the DataFrame is named df
df = pd.read_csv('loan.csv')
models = load_models()  # Assuming function to load pre-trained models

# %% [markdown]
# Step 2: Identifying High-Value Customers
# High-value customers in this scenario might be defined as those with high income and low risk levels.

# %%
def identify_high_value_segments(df):
    df_high_value = df[(df['person_income'] > df['person_income'].quantile(0.8)) & (df['loan_status'] == 0)]
    return df_high_value

high_value_segments = identify_high_value_segments(df)
print("High Value Segments Identified:\n", high_value_segments.shape)

# %% [markdown]
# Step 3: Define Actions
# Here, we can recommend actions to enhance service and products offered to these segments.

# %%
def define_high_value_actions(high_value_segments):
    actions = {
        'service_actions': {
            'vip_service': 'Provide dedicated account manager',
            'priority': 'High',
            'target_customers': high_value_segments.index.tolist()
        },
        'product_actions': {
            'premium_products': 'Recommend high-end finance products',
            'custom_loans': 'Customized financing solutions',
            'priority': 'Medium'
        },
        'risk_actions': {
            'monitoring': 'Establish alert monitoring system',
            'priority': 'High'
        }
    }
    return actions

action_plan = define_high_value_actions(high_value_segments)
print("Action Plan for High Value Customers:")
print(action_plan)

# %% [markdown]
# Step 4: Implementation Plan (Mock-up)
# This illustrates limitations owing to data-sensitive operations once model assumptions and strategies have been simulated.

# %%
def implement_actions(action_plan):
    print(f"Implementing {action_plan['service_actions']['vip_service']}")
    # Code directly affecting banking systems with API calls or database updates would be implemented here and not run in a typical environment for security.

implementation_details = implement_actions(action_plan)
print("Implementation Completed.")

# %% [markdown]
# Step 5: Monitoring & Adjustments
# The performance of actions taken can be simulated or captured using metrics like customer satisfaction and engagement rates. In a real-world application, you'd periodically review these analytics to fine-tune operations.

# %%
def monitor_performance(high_value_segments):
    # Implementation of analytics dashboards or reports would directly pull new data for those segments
    print(f"Monitoring high-value customers, current segment count: {len(high_value_segments)}")

monitor_performance(high_value_segments)

# %% [markdown]
# Summary
# This would provide you a strategic implementation of activities to maximize business with high-value customers based on sample loan data. Each step would further need detailed specifications depending on tool integrations and compliance requirements within your infrastructure. This simulation helps in understanding the flow but requires real-time data and systems to execute.

# %% [markdown]
# Opportunity 5: 高风险客户的风险缓解 的实现代码，涵盖了 贷后跟踪与提醒服务 和 提高贷款审批门槛或要求更高担保 的功能。

# %%

df['id'] = df.index

# %%
from joblib import load
# 加载模型
def load_models():
    model_path = './'
    light_gbm_model = load(os.path.join(model_path, 'light_gbm_model.joblib'))
    xgb_model = load(os.path.join(model_path, 'xgb_model.joblib'))
    return {
        'LightGBM': light_gbm_model,
        'XGBoost': xgb_model
    }

models = load_models()

# 使用 LightGBM 预测违约概率
def add_default_proba(df, model, model_name):
    """
    使用模型预测违约概率，并添加到数据框中。
    Args:
        df (pd.DataFrame): 输入数据框
        model: 已加载的模型
        model_name (str): 模型名称，用于区分列名
    Returns:
        pd.DataFrame: 添加预测概率后的数据框
    """
    # 确保特征与训练时一致
    feature_columns = [
        'person_age', 'person_income', 'person_emp_length', 'loan_amnt',
        'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length'
    ]
    
    # 检查所需列是否存在
    for col in feature_columns:
        if col not in df.columns:
            raise ValueError(f"Missing feature column: {col}")
    
    # 预测违约概率
    df[f'default_proba_{model_name}'] = model.predict_proba(df[feature_columns])[:, 1]
    return df

# 添加 LightGBM 模型预测的违约概率
df = add_default_proba(df, models['LightGBM'], 'LightGBM')

# 添加 XGBoost 模型预测的违约概率（可选）
df = add_default_proba(df, models['XGBoost'], 'XGBoost')

# 打印数据框以确认新列
print(df[['default_proba_LightGBM', 'default_proba_XGBoost']].head())

# %%
#action1:1. 识别高风险客户高风险客户通常满足以下条件：收入较低：低于收入的 20 分位数。违约概率高：风险分数（如 risk_score 或模型预测的违约概率 default_proba）高于 80 分位数。
import pandas as pd
import numpy as np
import joblib

# 加载已保存的模型
lightgbm_model = joblib.load('./light_gbm_model.joblib')
xgb_model = joblib.load('./xgb_model.joblib')

# 确保数据中包含模型需要的特征
feature_columns = lightgbm_model.feature_name_
for col in feature_columns:
    if col not in df.columns:
        df[col] = 0  # 如果缺少特征，填充为 0

# 使用模型进行违约概率预测
df['lightgbm_proba'] = lightgbm_model.predict_proba(df[feature_columns])[:, 1]
df['xgb_proba'] = xgb_model.predict_proba(df[feature_columns])[:, 1]

# 计算两模型的平均违约概率
df['default_proba'] = (df['lightgbm_proba'] + df['xgb_proba']) / 2

# 查看预测结果
print(df[['id', 'default_proba']].head())

# %% [markdown]
# 风险分类
# 根据违约概率对客户进行风险分类，并为高风险客户提供贷后跟踪计划。
# 风险分类规则：
# 高风险客户：default_proba > 0.8
# 中风险客户：0.5 < default_proba <= 0.8
# 低风险客户：default_proba <= 0.5

# %%
# 定义风险分类函数
def assign_risk_category(df):
    conditions = [
        (df['default_proba'] > 0.8),
        (df['default_proba'] > 0.5) & (df['default_proba'] <= 0.8),
        (df['default_proba'] <= 0.5)
    ]
    categories = ['High Risk', 'Medium Risk', 'Low Risk']
    df['risk_category'] = np.select(conditions, categories, default='Unknown')
    return df

# 应用风险分类
df = assign_risk_category(df)

# 查看风险分类结果
print(df[['id', 'default_proba', 'risk_category']].head())

# %% [markdown]
# 步骤 3: 提供贷后跟踪与提醒服务
# 为高风险客户和中风险客户生成贷后跟踪计划：
# 高风险客户：每月跟踪。
# 中风险客户：每季度跟踪。
# 低风险客户：无需额外跟踪。

# %%
# 定义贷后跟踪计划
def create_follow_up_plan(df):
    follow_up_plan = []
    for _, row in df.iterrows():
        if row['risk_category'] == 'High Risk':
            follow_up_plan.append(f"Customer {row['id']}: Monthly follow-up required.")
        elif row['risk_category'] == 'Medium Risk':
            follow_up_plan.append(f"Customer {row['id']}: Quarterly follow-up suggested.")
        else:
            follow_up_plan.append(f"Customer {row['id']}: No follow-up needed.")
    df['follow_up_plan'] = follow_up_plan
    return df

# 应用贷后跟踪计划
df = create_follow_up_plan(df)

# 查看贷后跟踪计划
print(df[['id', 'risk_category', 'follow_up_plan']].head())

# %% [markdown]
# 步骤 4: 提高贷款审批门槛或要求更高担保
# 根据客户的风险等级调整贷款政策：
# 高风险客户：
# 提高贷款审批门槛，直接拒绝贷款或要求全额担保。
# 中风险客户：
# 降低贷款额度（例如降低为70%），并要求部分担保。
# 低风险客户：
# 正常批准贷款，无需额外担保。

# %%
# 定义贷款政策调整函数
def adjust_loan_policy(df):
    loan_policy = []
    for _, row in df.iterrows():
        if row['risk_category'] == 'High Risk':
            loan_policy.append(f"Customer {row['id']}: Loan rejected or full collateral required.")
        elif row['risk_category'] == 'Medium Risk':
            reduced_loan = row['loan_amnt'] * 0.7  # 降低贷款额度为70%
            loan_policy.append(f"Customer {row['id']}: Loan reduced to {reduced_loan:.2f}, partial collateral required.")
        else:
            loan_policy.append(f"Customer {row['id']}: Loan approved without collateral.")
    df['loan_policy'] = loan_policy
    return df

# 应用贷款政策调整
df = adjust_loan_policy(df)

# 查看贷款政策调整结果
print(df[['id', 'risk_category', 'loan_policy']].head())

# %% [markdown]
# 步骤 5: 保存结果
# 将生成的风险分类、贷后跟踪计划和贷款政策调整建议保存为新的 CSV 文件。

# %%
# 保存结果到 CSV
output_path = './high_risk_mitigation_results.csv'
df.to_csv(output_path, index=False)

print(f"结果已保存到 {output_path}")

# %% [markdown]
# 
# 
# 为了更直观地展示客户的分布情况，您可以绘制违约概率分布和风险等级分布图。

# %%
import matplotlib.pyplot as plt

# 绘制违约概率分布
plt.figure(figsize=(8, 5))
plt.hist(df['default_proba'], bins=30, color='blue', alpha=0.7)
plt.title('Distribution of Default Probability')
plt.xlabel('Default Probability')
plt.ylabel('Number of Customers')
plt.show()

# 绘制风险等级分布
risk_counts = df['risk_category'].value_counts()

plt.figure(figsize=(6, 4))
risk_counts.plot(kind='bar', color=['red', 'orange', 'green'])
plt.title('Customer Risk Category Distribution')
plt.ylabel('Number of Customers')
plt.xlabel('Risk Category')
plt.xticks(rotation=0)
plt.show()

# %%



