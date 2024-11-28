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
def identify_default_patterns():
    # 1. 计算特征重要性
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': models['LightGBM'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    # 2. 计算违约概率
    default_proba = models['LightGBM'].predict_proba(X)[:, 1]
    
    # 3. 创建风险评分
    risk_levels = pd.cut(default_proba, 
                        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    # 4. 创建分析数据框
    analysis_df = pd.DataFrame({
        'risk_score': default_proba,
        'risk_level': risk_levels
    })
    
    # 5. 分析每个风险等级的特征统计
    risk_stats = {}
    for feature in X.columns:
        feature_stats = pd.DataFrame({
            'feature_value': X[feature],
            'risk_level': risk_levels
        }).groupby('risk_level')['feature_value'].agg(['mean', 'std', 'count'])
        risk_stats[feature] = feature_stats
    
    return {
        'feature_importance': feature_importance,
        'risk_scores': default_proba,
        'risk_stats': risk_stats
    }

# 运行分析
results = identify_default_patterns()

# 显示结果
print("Top 5 Risk Indicators:")
print(results['feature_importance'].head())

print("\nRisk Statistics for Top Feature:")
top_feature = results['feature_importance'].iloc[0]['feature']
print(f"\n{top_feature}:")
print(results['risk_stats'][top_feature])

# 可视化
plt.figure(figsize=(15, 5))

# 1. 特征重要性
plt.subplot(1, 3, 1)
top_5 = results['feature_importance'].head()
plt.bar(range(5), top_5['importance'])
plt.xticks(range(5), [f[:10] + '...' if len(f) > 10 else f for f in top_5['feature']], rotation=45)
plt.title('Top 5 Risk Indicators')

# 2. 风险分布
plt.subplot(1, 3, 2)
plt.hist(results['risk_scores'], bins=50)
plt.title('Risk Score Distribution')
plt.xlabel('Risk Score')
plt.ylabel('Count')

# 3. 最重要特征与风险关系
plt.subplot(1, 3, 3)
plt.scatter(X[top_feature], results['risk_scores'], alpha=0.5)
plt.xlabel(top_feature)
plt.ylabel('Risk Score')
plt.title(f'Risk Score vs {top_feature}')

plt.tight_layout()
plt.show()

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
    
    # 2. 计算每个分层的关键指标
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

# %%
def optimize_pricing_strategy(customer_data):
    # 构建风险定价矩阵
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
    
    # 计算风险调整系数
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
    
    # 生成最终定价建议
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

# %%
def optimize_pricing_strategy(customer_data):
    def optimize_pricing(segment):
        pricing_rules = {
            'low_risk': {
                'base_rate': 0.045,  # 4.5%
                'max_adjustment': 1.03,  # 最大上浮3%
                'volume_discount': 0.002  # 0.2%
            },
            'medium_risk': {
                'base_rate': 0.065,  # 6.5%
                'max_adjustment': 1.05,  # 最大上浮5%
                'volume_discount': 0.001  # 0.1%
            },
            'high_risk': {
                'base_rate': 0.080,  # 8.0%
                'max_adjustment': 1.40,  # 最大上浮40%
                'volume_discount': 0.000  # 无折扣
            }
        }
        return pricing_rules[segment]

    def assess_risk_level(metrics):
        if metrics['avg_risk'] < 0.01:
            return 'low_risk'
        elif metrics['avg_risk'] < 0.05:
            return 'medium_risk'
        else:
            return 'high_risk'

    def calculate_final_rate(customer_segment, loan_amount):
        risk_level = assess_risk_level(customer_data['metrics'][customer_segment])
        pricing_rule = optimize_pricing(risk_level)
        
        # 基础利率
        final_rate = pricing_rule['base_rate']
        
        # 风险调整 - 修正计算方式
        risk_adjustment = min(1 + customer_data['metrics'][customer_segment]['avg_risk'], 
                            pricing_rule['max_adjustment'])
        final_rate *= risk_adjustment
        
        # 批量折扣
        if loan_amount > 50000:
            final_rate -= pricing_rule['volume_discount']
            
        return {
            'base_rate': pricing_rule['base_rate'],
            'risk_adjustment': risk_adjustment,
            'final_rate': final_rate,
            'volume_discount': pricing_rule['volume_discount'],
            'risk_level': risk_level
        }

    pricing_recommendations = {}
    for segment in customer_data['metrics'].keys():
        avg_loan = customer_data['metrics'][segment]['avg_loan_amount']
        rates = calculate_final_rate(segment, avg_loan)
        
        pricing_recommendations[segment] = {
            'risk_level': rates['risk_level'],
            'base_rate': rates['base_rate'],
            'risk_adjustment': rates['risk_adjustment'],
            'final_rate': rates['final_rate'],
            'volume_discount': rates['volume_discount'],
            'rate_range': (rates['final_rate'] * 0.95, rates['final_rate'] * 1.05)
        }
    
    return pricing_recommendations

# 测试数据
customer_segments = {
    'metrics': {
        'segment_1': {
            'avg_risk': 0.008,
            'avg_loan_amount': 30000
        },
        'segment_2': {
            'avg_risk': 0.03,
            'avg_loan_amount': 60000
        },
        'segment_3': {
            'avg_risk': 0.07,
            'avg_loan_amount': 40000
        }
    }
}

# 运行优化定价策略
pricing_strategy = optimize_pricing_strategy(customer_segments)

# 显示定价建议
for segment, rates in pricing_strategy.items():
    print(f"\n{segment}:")
    print(f"Risk Level: {rates['risk_level']}")
    print(f"Base Rate: {rates['base_rate']:.2%}")
    print(f"Risk Adjustment: {rates['risk_adjustment']:.3f}x")
    print(f"Volume Discount: {rates['volume_discount']:.2%}")
    print(f"Final Rate: {rates['final_rate']:.2%}")
    print(f"Suggested Range: {rates['rate_range'][0]:.2%} - {rates['rate_range'][1]:.2%}")

# %%
        },
        'product_actions': {
            'premium_products': '推荐高端理财产品',
            'custom_loans': '定制化融资方案',
            'priority': 'Medium'
        },
        'risk_actions': {
            'monitoring': '建立预警监控机制',
            'priority': 'High'
        }
    }

# 3. 生成行动计划
action_plan = define_high_value_actions(high_value_segments)

# 4. 获取优先行动项
high_priority_actions = {
    key: actions for key, actions in action_plan.items()
    if actions['priority'] == 'High'
    
}


# %%

loan_performance = pd.DataFrame({
    'loan_amount': X['loan_amnt'],
    'risk_score': models['LightGBM'].predict_proba(X)[:, 1]
}).groupby(pd.qcut(X['loan_amnt'], 5))['risk_score'].mean()

def analyze_risk_pricing():
    # 获取预测概率
    default_proba = models['LightGBM'].predict_proba(X)[:, 1]
    
    # 创建分析框架
    analysis_df = pd.DataFrame({
        'loan_amount': X['loan_amnt'],
        'income': X['person_income'],
        'risk_score': default_proba
    })
    
    return analysis_df

pricing_analysis = analyze_risk_pricing()

pricing_analysis

# %%
def calculate_risk_based_pricing(df):
    # 基础利率设置
    base_rate = 0.10  # 10% 基准利率
    
    # 计算风险调整系数
    risk_scores = models['LightGBM'].predict_proba(df)[:, 1]
    
    # 创建定价框架
    pricing_df = pd.DataFrame({
        'loan_amount': df['loan_amnt'],
        'income': df['person_income'],
        'risk_score': risk_scores
    })
    
    # 计算建议利率
    pricing_df['suggested_rate'] = base_rate + (risk_scores * 0.05)
    
    return pricing_df

pricing_analysis = calculate_risk_based_pricing(X)

pricing_analysis

# %%
# 分析客户行为模式
customer_behavior = analyze_loan_management()
retention_metrics = customer_behavior['metrics']

customer_behavior,retention_metrics 

# %%
# 从风险模式识别中发现
risk_patterns = identify_default_patterns()
high_risk_indicators = risk_patterns['feature_importance'].head()
high_risk_indicators

# %%
# 客户群体分析
segment_analysis = results['metrics']
for segment, metrics in segment_analysis.items():
    segment_performance = metrics['avg_risk']
    segment_size = metrics['count']

# %%
# Opportunity 1: 高价值客户群识别

# 从客户分层分析中发现
segments = results['segments']
high_value_segments = segments[
    (segments['income_level'] == 'High') & 
    (segments['risk_level'] == 'Low')
]


high_value_segments

# %%
#actions
def define_high_value_actions(high_value_segments):
    """
    定义高价值客户的关键行动方案
    """
    return {
        'service_actions': {
            'vip_service': '提供专属客户经理',
            'priority': 'High',
            'target_customers':  high_value_segments.index.tolist()
        },
        'product_actions': {
            'premium_products': '推荐高端理财产品',
            'custom_loans': '定制化融资方案',
            'priority': 'Medium'
        },
        'risk_actions': {
            'monitoring': '建立预警监控机制',
            'priority': 'High'
        }
    }

# 3. 生成行动计划
action_plan = define_high_value_actions(high_value_segments)

# 4. 获取优先行动项
high_priority_actions = {
    key: actions for key, actions in action_plan.items()
    if actions['priority'] == 'High'
    
}


# %%
# Opportunity 2: 交叉销售机会客户群识别
segments = results['segments']
cross_sell_segments = segments[
    (segments['income_level'].isin(['Medium', 'High'])) &  # 中高收入
    (segments['risk_level'] == 'Low') &                    # 低风险
    ~((segments['income_level'] == 'High') &               # 排除已识别的高价值客户
      (segments['risk_level'] == 'Low'))
]

# 相应的Actions定义
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
# Opportunity 3: 定价优化机会
pricing_analysis = calculate_risk_based_pricing(X)

pricing_optimization_segments = pricing_analysis[
    (pricing_analysis['risk_score'] < 0.3) &                # 风险可控
    (pricing_analysis['loan_amount'] >= 50000) &           # 大额贷款
    (pricing_analysis['suggested_rate'] > 0.12)            # 当前定价偏高
]

# 相应的Actions定义
def define_pricing_optimization_actions(optimization_segments):
    actions = {
        'rate_adjustment': {
            'priority': 'High',
            'description': '利率优化方案',
            'components': {
                'base_rate_review': '重新评估基准利率',
                'risk_premium': '优化风险溢价计算',
                'volume_discount': '引入规模折扣'
            }
        },
        'competitive_analysis': {
            'priority': 'Medium',
            'description': '竞争对手定价分析',
            'focus_areas': [
                '市场利率水平跟踪',
                '竞品定价策略分析',
                '客户流失风险评估'
            ]
        },
        'implementation_plan': {
            'priority': 'High',
            'description': '实施计划',
            'steps': {
                'pilot_testing': '试点测试新定价',
                'customer_communication': '客户沟通策略',
                'system_updates': '系统更新计划'
            }
        },
        'monitoring_framework': {
            'priority': 'Medium',
            'description': '效果监控框架',
            'metrics': [
                '利率调整后的转化率',
                '收益影响分析',
                '客户满意度跟踪',
                '市场份额变化'
            ]
        }
    }
    return actions

# 执行actions定义
pricing_optimization_actions = define_pricing_optimization_actions(pricing_optimization_segments)

# %%
# Opportunity 4: 风险预警与管理优化
risk_patterns = identify_default_patterns()

# 1. 首先查看字典结构
print("Available keys:", risk_patterns.keys())

# 2. 修改为处理字典格式的代码
def process_risk_patterns(risk_data):
    high_risk_cases = {
        'risk_metrics': risk_data.get('risk_metrics', {}),
        'warning_indicators': {
            'high_risk': [],
            'medium_risk': [],
            'low_risk': []
        }
    }
    
    # 处理风险指标
    for customer_id, metrics in risk_data.get('customer_metrics', {}).items():
        risk_level = metrics.get('risk_score', 0)
        payment_status = metrics.get('payment_status', '')
        utilization = metrics.get('utilization', 0)
        
        if (risk_level > 0.7 or 
            payment_status == 'overdue' or 
            utilization > 0.8):
            high_risk_cases['warning_indicators']['high_risk'].append(customer_id)
            
    return high_risk_cases

# 3. 执行风险分析
early_warning_segments = process_risk_patterns(risk_patterns)

# 4. 定义相应的actions
def define_risk_management_actions(warning_segments):
    actions = {
        'early_warning_system': {
            'priority': 'High',
            'description': '建立预警机制',
            'components': {
                'risk_indicators': [
                    '风险分数',
                    '支付状态',
                    '额度使用率'
                ],
                'alert_thresholds': {
                    'risk_score': '>0.7',
                    'payment_status': 'overdue',
                    'utilization': '>80%'
                }
            }
        }
    }
    return actions

# 5. 执行
risk_management_actions = define_risk_management_actions(early_warning_segments)

# %%
# Opportunity 5: 客户忠诚度提升计划
def identify_loyalty_opportunity():
    # 从现有分析中获取客户行为数据
    customer_behavior = analyze_loan_management()
    retention_metrics = customer_behavior['metrics']
    
    loyalty_segments = {
        'stable_customers': [],
        'at_risk_customers': [],
        'growth_potential': []
    }
    
    # 定义忠诚度计划的Actions
    actions = {
        'reward_program': {
            'priority': 'High',
            'description': '忠诚度奖励体系',
            'components': {
                'points_system': {
                    'payment_on_time': '100点',
                    'product_adoption': '200点',
                    'referral': '300点'
                },
                'rewards': [
                    '利率优惠',
                    '费用减免',
                    '额度提升'
                ]
            }
        },
        'engagement_strategy': {
            'priority': 'Medium',
            'description': '客户互动策略',
            'activities': {
                'regular_communication': [
                    '个性化内容推送',
                    '生日/节日关怀',
                    '产品使用建议'
                ],
                'feedback_collection': [
                    '满意度调查',
                    '产品建议收集',
                    '服务体验反馈'
                ]
            }
        },
        'retention_program': {
            'priority': 'High',
            'description': '客户保留计划',
            'measures': {
                'early_warning': '流失风险预警',
                'personalized_offers': '个性化挽留方案',
                'service_upgrade': '服务等级提升'
            }
        },
        'value_added_services': {
            'priority': 'Medium',
            'description': '增值服务',
            'services': [
                '财务咨询',
                '专属客户经理',
                '优先服务通道'
            ]
        }
    }
    
    return {
        'segments': loyalty_segments,
        'actions': actions
    }

# 应用方法
def implement_loyalty_program():
    # 1. 初始化忠诚度计划
    loyalty_opportunity = identify_loyalty_opportunity()
    
    # 2. 执行忠诚度提升行动
    def execute_loyalty_actions():
        results = {
            'rewards_issued': [],
            'engagement_metrics': {},
            'retention_rate': 0.0,
            'customer_satisfaction': 0.0
        }
        
        # 实施奖励计划
        def implement_rewards():
            return {
                'points_awarded': 0,
                'rewards_redeemed': 0
            }
        
        # 执行客户互动
        def run_engagement():
            return {
                'communication_rate': 0.0,
                'response_rate': 0.0
            }
        
        # 跟踪保留效果
        def track_retention():
            return {
                'retention_rate': 0.0,
                'churn_reduction': 0.0
            }
        
        return results
    
    # 3. 监控效果
    def monitor_loyalty_performance():
        return {
            'loyalty_score': 0.0,
            'customer_lifetime_value': 0.0,
            'satisfaction_index': 0.0
        }
    
    return {
        'program_setup': loyalty_opportunity,
        'execution_results': execute_loyalty_actions(),
        'performance_metrics': monitor_loyalty_performance()
    }


