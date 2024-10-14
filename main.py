import tkinter as tk
from tkinter import messagebox, scrolledtext
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

class LoanApplicationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("贷款违约风险预测应用")
        self.create_main_page()

    def create_main_page(self):
        # 创建主页面
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(padx=10, pady=10)

        # 项目介绍
        self.project_info = (
            "项目背景：\n"
            "随着贷款需求的增加，贷款违约风险成为金融机构关注的核心问题。有效评估和预测贷款违约风险\n"
            "不仅可以帮助银行减少损失，还能提高贷款审批的效率。本项目旨在通过机器学习模型，\n"
            "结合用户输入的数据，预测贷款申请者的违约风险。\n\n"
            "功能概述：\n"
            "- 用户输入：用户可以通过表单输入个人信息，包括姓名、收入、贷款金额和信用分数。\n"
            "- 数据存储：所有用户提交的数据将被保存到 CSV 文件中，便于后续分析。\n"
            "- 数据分析：应用将分析历史数据，生成可视化图表，展示信用分数与贷款金额之间的关系。\n"
            "- 风险预测：基于机器学习模型，预测用户的贷款违约风险。\n\n"
            "技术栈：\n"
            "- 编程语言：Python\n"
            "- 图形用户界面：Tkinter\n"
            "- 数据分析与可视化：Pandas, Matplotlib\n"
            "- 机器学习：Scikit-learn（可选，若后续添加模型）\n\n"
            "使用说明：\n"
            "1. 安装依赖：确保安装了所需的 Python 库，例如 Tkinter、Pandas 和 Matplotlib。\n"
            "2. 运行应用：将代码保存为 loan_application_app.py，在命令行中运行：\n"
            "   python loan_application_app.py\n"
            "3. 输入数据：在应用界面中填写贷款申请表单并提交。\n"
            "4. 查看分析结果：点击“分析数据”按钮，查看生成的图表。\n\n"
            "未来工作：\n"
            "- 模型优化：集成更多机器学习模型以提高预测准确性。\n"
            "- 用户界面改进：优化界面设计，提升用户体验。\n"
            "- 功能扩展：增加更多数据分析功能，如违约率分析和收入分布等。\n"
        )

        # 创建滚动文本框
        self.text_area = scrolledtext.ScrolledText(self.main_frame, wrap=tk.WORD, width=60, height=30)
        self.text_area.insert(tk.INSERT, self.project_info)
        self.text_area.config(state=tk.DISABLED)  # 设置为只读
        self.text_area.pack()

        # 创建按钮
        tk.Button(self.main_frame, text="开始申请", command=self.open_application_form).pack(pady=10)

    def open_application_form(self):
        self.main_frame.destroy()  # 关闭主页面
        self.create_application_form()  # 创建申请表单

    def create_application_form(self):
        # 创建申请表单
        self.form_frame = tk.Frame(self.root)
        self.form_frame.pack(padx=10, pady=10)

        tk.Label(self.form_frame, text="姓名").grid(row=0, column=0)
        self.name_entry = tk.Entry(self.form_frame)
        self.name_entry.grid(row=0, column=1)

        tk.Label(self.form_frame, text="收入").grid(row=1, column=0)
        self.income_entry = tk.Entry(self.form_frame)
        self.income_entry.grid(row=1, column=1)

        tk.Label(self.form_frame, text="贷款金额").grid(row=2, column=0)
        self.loan_amount_entry = tk.Entry(self.form_frame)
        self.loan_amount_entry.grid(row=2, column=1)

        tk.Label(self.form_frame, text="信用分数").grid(row=3, column=0)
        self.credit_score_entry = tk.Entry(self.form_frame)
        self.credit_score_entry.grid(row=3, column=1)

        tk.Button(self.form_frame, text="提交", command=self.submit).grid(row=4, column=0, columnspan=2)
        tk.Button(self.form_frame, text="分析数据", command=self.analyze_data).grid(row=5, column=0, columnspan=2)
        tk.Button(self.form_frame, text="返回", command=self.return_to_main).grid(row=6, column=0, columnspan=2)

    def submit(self):
        # 收集数据
        name = self.name_entry.get()
        income = self.income_entry.get()
        loan_amount = self.loan_amount_entry.get()
        credit_score = self.credit_score_entry.get()

        # 数据验证
        if not all([name, income, loan_amount, credit_score]):
            messagebox.showerror("错误", "所有字段都是必填的！")
            return

        # 保存数据到 DataFrame
        data = {
            "Name": [name],
            "Income": [float(income)],
            "Loan Amount": [float(loan_amount)],
            "Credit Score": [int(credit_score)]
        }
        df = pd.DataFrame(data)

        # 将数据保存到 CSV 文件
        df.to_csv("loan_applications.csv", mode='a', header=False, index=False)
        messagebox.showinfo("成功", "申请已成功提交！")

        # 清空输入框
        self.clear_form()

        # 进行贷款状态预测
        self.predict_loan_status()

    def clear_form(self):
        self.name_entry.delete(0, tk.END)
        self.income_entry.delete(0, tk.END)
        self.loan_amount_entry.delete(0, tk.END)
        self.credit_score_entry.delete(0, tk.END)

    def predict_loan_status(self):
        # 简单的贷款状态预测（示例）
        try:
            df = pd.read_csv("loan_applications.csv", names=["Name", "Income", "Loan Amount", "Credit Score"])
            X = df[["Income", "Loan Amount", "Credit Score"]]
            y = (df["Credit Score"] < 600).astype(int)  # 假设信用分数低于600为违约

            # 训练模型
            model = LogisticRegression()
            model.fit(X, y)

            # 进行预测
            new_data = [[float(self.income_entry.get()), float(self.loan_amount_entry.get()), int(self.credit_score_entry.get())]]
            prediction = model.predict(new_data)

            status = "违约风险" if prediction[0] == 1 else "低违约风险"
            messagebox.showinfo("预测结果", f"贷款状态预测：{status}")

        except Exception as e:
            messagebox.showerror("错误", f"预测失败：{str(e)}")

    def return_to_main(self):
        self.form_frame.destroy()  # 关闭申请表单
        self.create_main_page()  # 返回主页面

    def analyze_data(self):
        try:
            df = pd.read_csv("loan_applications.csv", names=["Name", "Income", "Loan Amount", "Credit Score"])
            plt.figure(figsize=(10, 6))
            plt.scatter(df["Credit Score"], df["Loan Amount"], alpha=0.5)
            plt.title("信用分数与贷款金额的关系")
            plt.xlabel("信用分数")
            plt.ylabel("贷款金额")
            plt.grid()
            plt.show()
        except FileNotFoundError:
            messagebox.showerror("错误", "没有找到数据文件，请先提交申请。")

if __name__ == "__main__":
    root = tk.Tk()
    app = LoanApplicationApp(root)
    root.mainloop()