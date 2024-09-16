import tkinter as tk
from tkinter import messagebox

def save_data():
    messagebox.showinfo("Info", "Data input saved!\n"
)

def clear_inputs():
    entry_project_code.delete(0, tk.END)
    entry_project_name.delete(0, tk.END)
    entry_project_manager.delete(0, tk.END)
    entry_budget.delete(0, tk.END)
    entry_employee_no.delete(0, tk.END)
    messagebox.showinfo("Info", "Input cleared!")

# Create the main window
root = tk.Tk()
root.title("Project Information")

# Create labels and entry fields
tk.Label(root, text="Project Code:").grid(row=0, column=0, padx=10, pady=10)
entry_project_code = tk.Entry(root)
entry_project_code.grid(row=0, column=1, padx=10, pady=10)

tk.Label(root, text="Project Name:").grid(row=1, column=0, padx=10, pady=10)
entry_project_name = tk.Entry(root)
entry_project_name.grid(row=1, column=1, padx=10, pady=10)

tk.Label(root, text="Project Manager:").grid(row=2, column=0, padx=10, pady=10)
entry_project_manager = tk.Entry(root)
entry_project_manager.grid(row=2, column=1, padx=10, pady=10)

tk.Label(root, text="Budget:").grid(row=3, column=0, padx=10, pady=10)
entry_budget = tk.Entry(root)
entry_budget.grid(row=3, column=1, padx=10, pady=10)

tk.Label(root, text="Employee No:").grid(row=4, column=0, padx=10, pady=10)
entry_employee_no = tk.Entry(root)
entry_employee_no.grid(row=4, column=1, padx=10, pady=10)

# Create Buttons
button_save = tk.Button(root, text="Save", command=save_data)
button_save.grid(row=5, column=0, padx=10, pady=10)

button_clear = tk.Button(root, text="Clear", command=clear_inputs)
button_clear.grid(row=5, column=1, padx=10, pady=10)

root.mainloop()

