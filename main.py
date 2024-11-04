import streamlit as st
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
import re

# 设置页面配置
st.set_page_config(page_title="Loan Application Form", layout="wide")

# 日志配置
log_filename = f'loan_app_{datetime.now().strftime("%Y%m%d")}.log'
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 定义表单字段配置
FORM_FIELDS = {
    'person_age': {'type': 'number', 'label': 'Age', 'validation': 'int'},
    'person_income': {'type': 'number', 'label': 'Annual Income', 'validation': 'float'},
    'person_home_ownership': {'type': 'radio', 'label': 'Home Ownership',
                            'values': ['Rent', 'Own', 'Mortgage']},
    'person_emp_length': {'type': 'number', 'label': 'Employment Length (years)', 'validation': 'float'},
    'loan_intent': {'type': 'radio_with_other', 'label': 'Loan Purpose',
                   'values': ['Education', 'Medical', 'Personal', 'Business', 'Debt Consolidation']},
    'loan_grade': {'type': 'radio', 'label': 'Loan Grade',
                  'values': ['A', 'B', 'C', 'D']},
    'loan_amnt': {'type': 'number', 'label': 'Loan Amount', 'validation': 'float'},
    'loan_int_rate': {'type': 'number', 'label': 'Interest Rate', 'validation': 'float'},
    'cb_person_default_on_file': {'type': 'radio', 'label': 'Default History',
                                 'values': ['Yes', 'No']},
    'cb_person_cred_hist_length': {'type': 'number', 'label': 'Credit History Length', 'validation': 'int'},
    'loan_status': {'type': 'radio', 'label': 'Loan Status',
                   'values': ['0', '1']}
}

def main():
    st.title("Loan Application Form")
    
    # 侧边栏导航
    page = st.sidebar.radio("Navigation", ["Submit Application", "View All Data", "View New Data"])
    
    if page == "Submit Application":
        show_application_form()
    elif page == "View All Data":
        show_all_data()
    else:
        show_new_data()

def show_application_form():
    with st.form("loan_application"):
        form_data = {}
        
        # 创建两列布局
        col1, col2 = st.columns(2)
        
        fields_per_column = len(FORM_FIELDS) // 2
        current_col = col1
        field_count = 0
        
        for field, props in FORM_FIELDS.items():
            if field_count == fields_per_column:
                current_col = col2
            
            if props['type'] == 'number':
                form_data[field] = current_col.number_input(
                    props['label'],
                    min_value=0.0,
                    step=1.0 if props['validation'] == 'int' else 0.01
                )
            elif props['type'] == 'radio':
                form_data[field] = current_col.radio(
                    props['label'],
                    props['values']
                )
            elif props['type'] == 'radio_with_other':
                options = props['values'] + ['Other']
                selected = current_col.radio(props['label'], options)
                if selected == 'Other':
                    form_data[field] = current_col.text_input(f"Specify other {props['label'].lower()}")
                else:
                    form_data[field] = selected
                    
            field_count += 1
        
        submitted = st.form_submit_button("Submit Application")
        if submitted:
            save_application(form_data)
            st.success("Application submitted successfully!")

def show_all_data():
    try:
        df = pd.read_csv('loan.csv')
        st.dataframe(df)
        st.write(f"Total records: {len(df)}")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

def show_new_data():
    # 实现查看新数据的逻辑
    pass

def save_application(data):
    # 实现保存申请数据的逻辑
    pass

if __name__ == "__main__":
    main()