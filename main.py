import tkinter as tk
from tkinter import messagebox
from tkinter import IntVar, Radiobutton
from tkinter import filedialog  

def savefile():
     # Check if any required fields are empty
    if not entry_project_code.get():
        messagebox.showerror("Error", "Project Code cannot be empty.")
        return
    if not entry_project_name.get():
        messagebox.showerror("Error", "Project Name cannot be empty.")
        return
    if x.get() == -1:  # No manager selected
        messagebox.showerror("Error", "Please select a Project Manager.")
        return
    if not entry_budget.get():
        messagebox.showerror("Error", "Budget cannot be empty.")
        return
    if not entry_employee_no.get():
        messagebox.showerror("Error", "Employee No cannot be empty.")
        return
    
    file = filedialog.asksaveasfile(initialdir="C:\\Users\\psion\\OneDrive - HKUST\\DASC2110 Object-Oriented Programming for Data Analytics in Science\\Teaching Materials\\19. GUI Part 3\\guiPart3",
                                    defaultextension='.txt',
                                    filetypes=[
                                        ("Text file", ".txt"),
                                        ("HTML file", ".html"),
                                        ("All files", ".*")
                                    ])

    if file is None:        # handle the exception here
        return

    filetext = f"Project Code: {entry_project_code.get()}\n"
    filetext += f"Project Name: {entry_project_name.get()}\n"
    filetext += f"Project Manager: {manager[x.get()]}\n"
    filetext += f"Budget: {entry_budget.get()}\n"
    filetext += f"Employee No: {entry_employee_no.get()}\n"

    file.write(filetext)  # write it to our file
    file.close()  # close it
    messagebox.showinfo("Info", "Data input saved!\n")
    clear_inputs()

def clear_inputs():
    entry_project_code.delete(0, tk.END)
    entry_project_name.delete(0, tk.END)
    entry_budget.delete(0, tk.END)
    entry_employee_no.delete(0, tk.END)
    x.set(-1)  # Reset radio button selection
    messagebox.showinfo("Info", "Input cleared!")

manager = ["Aiay", "Charu"]

# Create the main window
root = tk.Tk()
root.title("Project Information")

x = IntVar()

# Create labels and entry fields
tk.Label(root, text="Project Code:").grid(row=0, column=0, padx=10, pady=10, sticky="e")
entry_project_code = tk.Entry(root)
entry_project_code.grid(row=0, column=1, padx=10, pady=10)

tk.Label(root, text="Project Name:").grid(row=1, column=0, padx=10, pady=10, sticky="e")
entry_project_name = tk.Entry(root)
entry_project_name.grid(row=1, column=1, padx=10, pady=10)

tk.Label(root, text="Project Manager:").grid(row=2, column=0, padx=10, pady=10, sticky="ne")
manager_frame = tk.Frame(root)
manager_frame.grid(row=2, column=1, padx=10, pady=10, sticky="w")

for index, item in enumerate(manager):
    radiobutton = Radiobutton(manager_frame,
                              text=item,
                              variable=x,
                              value=index,
                              font=("Impact", 20))
    radiobutton.grid(row=1, column=index, sticky="w", pady=5)

tk.Label(root, text="Budget:").grid(row=3, column=0, padx=10, pady=10, sticky="e")
entry_budget = tk.Entry(root)
entry_budget.grid(row=3, column=1, padx=10, pady=10)

tk.Label(root, text="Employee No:").grid(row=4, column=0, padx=10, pady=10, sticky="e")
entry_employee_no = tk.Entry(root)
entry_employee_no.grid(row=4, column=1, padx=10, pady=10)

# Create Buttons
button_save = tk.Button(root, text="Save", command=savefile)
button_save.grid(row=5, column=0, padx=10, pady=10)

button_clear = tk.Button(root, text="Clear", command=clear_inputs)
button_clear.grid(row=5, column=1, padx=10, pady=10)

root.mainloop()