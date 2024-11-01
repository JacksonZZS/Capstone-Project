import tkinter as tk
from tkinter import ttk, messagebox,filedialog
import csv
import pandas as pd
from pathlib import Path
import re
from filelock import FileLock, Timeout
import logging
from datetime import datetime


log_filename = f'loan_app_{datetime.now().strftime("%Y%m%d")}.log'
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class DataPreviewWindow:
    def __init__(self, parent):
        self.window = tk.Toplevel(parent)
        self.window.title("Data Preview")
        self.window.geometry("1200x800")
        
        # Create main frame
        self.main_frame = ttk.Frame(self.window)
        self.main_frame.pack(fill='both', expand=True)


        # Create treeview frame
        self.tree_frame = ttk.Frame(self.main_frame)
        self.tree_frame.pack(fill='both', expand=True, padx=5, pady=5)

        # Create treeview
        self.tree = ttk.Treeview(self.tree_frame)
        
        # Create scrollbars
        self.vsb = ttk.Scrollbar(self.tree_frame, orient="vertical", command=self.tree.yview)
        self.hsb = ttk.Scrollbar(self.tree_frame, orient="horizontal", command=self.tree.xview)
        
        # Configure treeview
        self.tree.configure(yscrollcommand=self.vsb.set, xscrollcommand=self.hsb.set)

        # Grid layout
        self.tree.grid(row=0, column=0, sticky='nsew')
        self.vsb.grid(row=0, column=1, sticky='ns')
        self.hsb.grid(row=1, column=0, sticky='ew')

        # Configure grid weights
        self.tree_frame.grid_rowconfigure(0, weight=1)
        self.tree_frame.grid_columnconfigure(0, weight=1)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(self.main_frame, textvariable=self.status_var)
        self.status_bar.pack(fill='x', padx=5, pady=2)

        # Load data
        self.load_data()

    def load_data(self):
        try:
            # Clear existing items
            for item in self.tree.get_children():
                self.tree.delete(item)

            # Read all data
            df = pd.read_csv('loan.csv')
            
            # Configure columns
            self.tree["columns"] = list(df.columns)
            self.tree["show"] = "headings"
            
            # Set column headings
            for column in df.columns:
                self.tree.heading(column, text=column)
                self.tree.column(column, width=100)

            # Load all data
            for idx, row in df.iterrows():
                self.tree.insert("", 'end', values=list(row))

            # Update status bar
            self.status_var.set(f"Total records: {len(df)}")

        except Exception as e:
            messagebox.showerror("Error", f"Error loading data: {str(e)}")

class NewDataWindow:
    def __init__(self, parent, new_data):
        self.window = tk.Toplevel(parent)
        self.window.title("Newly Added Data")
        self.window.geometry("1200x600")

        # Create main frame
        self.main_frame = ttk.Frame(self.window)
        self.main_frame.pack(fill='both', expand=True)

        # Add label showing number of new records
        ttk.Label(
            self.main_frame, 
            text=f"New Records Added: {len(new_data)}",
            font=('Arial', 10, 'bold')
        ).pack(pady=5)

        # Create treeview frame
        self.tree_frame = ttk.Frame(self.main_frame)
        self.tree_frame.pack(fill='both', expand=True, padx=5, pady=5)

        # Create treeview
        self.tree = ttk.Treeview(self.tree_frame)
        
        # Create scrollbars
        self.vsb = ttk.Scrollbar(self.tree_frame, orient="vertical", command=self.tree.yview)
        self.hsb = ttk.Scrollbar(self.tree_frame, orient="horizontal", command=self.tree.xview)
        
        # Configure treeview
        self.tree.configure(yscrollcommand=self.vsb.set, xscrollcommand=self.hsb.set)

        # Grid layout
        self.tree.grid(row=0, column=0, sticky='nsew')
        self.vsb.grid(row=0, column=1, sticky='ns')
        self.hsb.grid(row=1, column=0, sticky='ew')

        # Configure grid weights
        self.tree_frame.grid_rowconfigure(0, weight=1)
        self.tree_frame.grid_columnconfigure(0, weight=1)

        # Load the new data
        self.load_data(new_data)

    def load_data(self, df):
        # Configure columns
        self.tree["columns"] = list(df.columns)
        self.tree["show"] = "headings"
        
        # Set column headings
        for column in df.columns:
            self.tree.heading(column, text=column)
            self.tree.column(column, width=100)

        # Add data
        for idx, row in df.iterrows():
            self.tree.insert("", 'end', values=list(row))



class LoanApplicationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Loan Application Form")
        self.root.geometry("1200x600")

        self.radio_vars = {}
        self.other_entries = {}
        
        # Store the initial data count when GUI is opened
        self.initial_data_count = self.get_current_data_length()
        self.previous_data_length = self.initial_data_count
        
        # Create menu bar
        self.create_menu()
        
        # Create main frame with scrollbar
        self.canvas = tk.Canvas(root)
        self.scrollbar = ttk.Scrollbar(root, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        
        self.fields = {
            'person_age': {'type': 'entry', 'label': 'Age', 'validation': 'int'},
            'person_income': {'type': 'entry', 'label': 'Annual Income', 'validation': 'float'},
            'person_home_ownership': {'type': 'radio', 'label': 'Home Ownership',
                                    'values': ['Rent', 'Own', 'Mortgage']},
            'person_emp_length': {'type': 'entry', 'label': 'Employment Length (years)', 'validation': 'float'},
            # Changed to radio_with_other
            'loan_intent': {'type': 'radio_with_other', 'label': 'Loan Purpose',
                           'values': ['Education', 'Medical', 'Personal', 'Business', 'Debt Consolidation']},
            # Changed to radio
            'loan_grade': {'type': 'radio', 'label': 'Loan Grade',
                          'values': ['A', 'B', 'C', 'D']},
            'loan_amnt': {'type': 'entry', 'label': 'Loan Amount', 'validation': 'float'},
            'loan_int_rate': {'type': 'entry', 'label': 'Interest Rate', 'validation': 'float'},
            'loan_percent_income': {'type': 'calculated', 'label': 'Income Percentage'},
            # Changed to radio
            'cb_person_default_on_file': {'type': 'radio', 'label': 'Default History',
                                         'values': ['Yes', 'No']},
            'cb_person_cred_hist_length': {'type': 'entry', 'label': 'Credit History Length', 'validation': 'int'},
            'loan_status': {'type': 'radio', 'label': 'Loan Status',
                           'values': ['0', '1']}
        }
        
        
        # Create widget variables
        self.widgets = {}
        self.checkbox_vars = {}  # Keep for other checkboxes
        self.radio_vars = {}     # New, for radio buttons
        self.other_var = None
        self.other_entry = None
        current_row = 0
        
        for field, props in self.fields.items():
            label_text = f"{props['label']} *" if props['type'] != 'calculated' else props['label']
            ttk.Label(self.scrollable_frame, text=label_text).grid(row=current_row, column=0, padx=5, pady=5, sticky=tk.W)
            
            if props['type'] == 'entry':
                # Entry widget code remains unchanged
                self.widgets[field] = ttk.Entry(self.scrollable_frame)
                self.widgets[field].grid(row=current_row, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
                
                if field in ['loan_amnt', 'person_income']:
                    self.widgets[field].bind('<KeyRelease>', self.calculate_loan_percent_income)
                
                current_row += 1
            elif props['type'] == 'calculated':
                # Calculated type code remains unchanged
                self.widgets[field] = ttk.Entry(self.scrollable_frame, state='readonly')
                self.widgets[field].grid(row=current_row, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
                current_row += 1
            elif props['type'] == 'checkbox':
                # Original checkbox code remains unchanged
                checkbox_frame = ttk.Frame(self.scrollable_frame)
                checkbox_frame.grid(row=current_row, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
                
                self.checkbox_vars[field] = {}
                for i, value in enumerate(props['values']):
                    var = tk.BooleanVar()
                    self.checkbox_vars[field][value] = var
                    ttk.Checkbutton(checkbox_frame, text=value, variable=var).grid(row=0, column=i, padx=5)
                
                current_row += 1
            elif props['type'] == 'radio':
                # New radio button code
                radio_frame = ttk.Frame(self.scrollable_frame)
                radio_frame.grid(row=current_row, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
                
                self.radio_vars[field] = tk.StringVar()
                for i, value in enumerate(props['values']):
                    ttk.Radiobutton(radio_frame, text=value, 
                                  variable=self.radio_vars[field], 
                                  value=value).grid(row=0, column=i, padx=5)
                
                current_row += 1
            elif props['type'] == 'radio_with_other':
                radio_frame = ttk.Frame(self.scrollable_frame)
                radio_frame.grid(row=current_row, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
                
                self.radio_vars[field] = tk.StringVar()
                
                # Create radio buttons for predefined values
                for i, value in enumerate(props['values']):
                    ttk.Radiobutton(
                        radio_frame, 
                        text=value, 
                        variable=self.radio_vars[field], 
                        value=value,
                        command=lambda f=field: self.radio_clicked(f)  
                    ).grid(row=0, column=i, padx=5)

                # Create "Other" radio button
                ttk.Radiobutton(
                    radio_frame, 
                    text="Other", 
                    variable=self.radio_vars[field],
                    value="OTHER",
                    command=lambda f=field: self.other_clicked(f) 
                ).grid(row=0, column=len(props['values']), padx=5)
                
                # Create entry for "Other" option
                self.other_entries[field] = ttk.Entry(radio_frame, state='disabled')
                self.other_entries[field].grid(row=0, column=len(props['values'])+1, padx=5)
                
                current_row += 1
                            
        
        # Create all buttons in a single frame at the bottom
        button_frame = ttk.Frame(self.scrollable_frame)
        button_frame.grid(row=current_row, column=0, columnspan=3, pady=20)

        # Left group of buttons
        left_buttons_frame = ttk.Frame(button_frame)
        left_buttons_frame.pack(side='left', padx=10)
        
        ttk.Button(
            left_buttons_frame,
            text="Reset",
            command=self.reset_form
        ).pack(side='left', padx=5)
        
        ttk.Button(
            left_buttons_frame,
            text="Submit",
            command=self.submit_data
        ).pack(side='left', padx=5)

        # Right group of buttons
        right_buttons_frame = ttk.Frame(button_frame)
        right_buttons_frame.pack(side='left', padx=10)
        
        ttk.Button(
            right_buttons_frame,
            text="Preview All Data",
            command=self.show_data_preview
        ).pack(side='left', padx=5)
        
        ttk.Button(
            right_buttons_frame,
            text="View New Data",
            command=self.show_new_data
        ).pack(side='left', padx=5)
        # Create tooltips
        self.create_tooltips()
        self.previous_data_length = self.get_current_data_length()


    def other_clicked(self, field):
        """Handle when 'Other' radio button is clicked"""
        
        if field in self.radio_vars and field in self.other_entries:
            if self.radio_vars[field].get() == "OTHER":
                self.other_entries[field].config(state='normal')
            else:
                self.other_entries[field].config(state='disabled')
                self.other_entries[field].delete(0, tk.END)

    def radio_clicked(self, field):
        """Handle when any radio button is clicked"""
        
        if field in self.radio_vars and field in self.other_entries:
            if self.radio_vars[field].get() != "OTHER":
                self.other_entries[field].config(state='disabled')
                self.other_entries[field].delete(0, tk.END)

    def get_form_data(self):
        """Get form data"""
        data = {}
        for field, props in self.fields.items():
            if props['type'] == 'entry':
                data[field] = self.widgets[field].get()
            elif props['type'] == 'checkbox':
                selected = []
                for value, var in self.checkbox_vars[field].items():
                    if var.get():
                        selected.append(value)
                data[field] = selected
            elif props['type'] in ['radio', 'radio_with_other']:
                value = self.radio_vars[field].get()
                if value == "OTHER" and field == 'loan_intent':
                    value = self.other_entry.get()
                data[field] = value
        return data

    def get_current_data_length(self):
        try:
            lock = FileLock("loan.csv.lock", timeout=10)  
            with lock:
                try:
                    df = pd.read_csv('loan.csv')
                    return len(df)
                except FileNotFoundError:
                    df = pd.DataFrame()
                    df.to_csv('loan.csv', index=False)
                    return 0
                except Exception as e:
                    messagebox.showerror("Error", f"Error accessing data: {str(e)}")
                    return 0
        except Timeout:
            messagebox.showerror("Error", "Could not acquire file lock. Please try again later.")
            return 0
        except Exception as e:
            messagebox.showerror("Error", f"Unexpected error: {str(e)}")
            return 0

    def get_new_data(self):
        try:
            current_df = pd.read_csv('loan.csv')
            if len(current_df) > self.initial_data_count:
                # Get all rows added since GUI was opened
                new_data = current_df.iloc[self.initial_data_count:]
                return new_data
            return None
        except Exception as e:
            messagebox.showerror("Error", f"Error accessing data: {str(e)}")
            return None

    def show_data_preview(self):
        DataPreviewWindow(self.root)

    def show_new_data(self):
        new_data = self.get_new_data()
        if new_data is not None and not new_data.empty:
            NewDataWindow(self.root, new_data)
            # Don't update previous_data_length here anymore
        else:
            messagebox.showinfo("Info", "No new data available since GUI was opened")

    def create_tooltips(self):
        """Create field information tooltips"""
        from tkinter import messagebox
        
        tooltips = {
            'person_emp_length': """
            • Must be positive
            • Cannot exceed (age - 16) years
            """,
            'cb_person_cred_hist_length': """
            • Must be positive
            • Cannot exceed (age - 18) years
            """
        }

        def show_tooltip(field):
            messagebox.showinfo("Field Information", tooltips[field])
        
        # Add help buttons for each field that has a tooltip
        for field in tooltips:
            if field in self.widgets:
                help_button = ttk.Button(
                    self.scrollable_frame, 
                    text="?", 
                    width=2,
                    command=lambda f=field: show_tooltip(f)
                )
                help_button.grid(
                    row=self.widgets[field].grid_info()['row'], 
                    column=2, 
                    padx=5
                )

    def create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Import CSV", command=self.import_csv)
        file_menu.add_command(label="Export CSV", command=self.export_csv)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Preview Data", command=self.show_preview)

    def show_preview(self):
        """Display data preview window"""
        DataPreviewWindow(self.root)

    def import_csv(self):
        """Import CSV file"""
        try:
            filename = filedialog.askopenfilename(
                filetypes=[('CSV Files', '*.csv')],
                title="Choose a CSV file to import"
            )
            if filename:
                # Read the selected CSV file
                df = pd.read_csv(filename)
                
                # Verify columns match
                required_columns = set(self.fields.keys())
                file_columns = set(df.columns) - {'id'}
                
                if not required_columns == file_columns:
                    messagebox.showerror("Error", 
                        "Column names in the selected file do not match the required format.")
                    return
                
                # Backup existing file if it exists
                if Path('loan.csv').exists():
                    backup_path = 'loan_backup.csv'
                    Path('loan.csv').rename(backup_path)
                
                # Save imported data
                df.to_csv('loan.csv', index=False)
                messagebox.showinfo("Success", "Data imported successfully!")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error importing data: {str(e)}")

    def export_csv(self):
        """export CSV file"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension='.csv',
                filetypes=[('CSV Files', '*.csv')],
                title="Save CSV file as"
            )
            if filename:
                if Path('loan.csv').exists():
                    df = pd.read_csv('loan.csv')
                    df.to_csv(filename, index=False)
                    messagebox.showinfo("Success", "Data exported successfully!")
                else:
                    messagebox.showwarning("Warning", "No data to export.")
                    
        except Exception as e:
            messagebox.showerror("Error", f"Error exporting data: {str(e)}")



    def calculate_loan_percent_income(self, event=None):
        try:
            loan_amount = float(self.widgets['loan_amnt'].get() or 0)
            income = float(self.widgets['person_income'].get() or 0)
            
            if income <= 0:
                messagebox.showerror("Error", "Income must be greater than 0")
                return
                
            percent = (loan_amount / income)
            if percent > 10:
                messagebox.showwarning("Warning", 
                    "Loan amount exceeds 10 times annual income. Please consider carefully.")
            
            self.widgets['loan_percent_income'].config(state='normal')
            self.widgets['loan_percent_income'].delete(0, tk.END)
            self.widgets['loan_percent_income'].insert(0, f"{percent:.2f}")
            self.widgets['loan_percent_income'].config(state='readonly')
                
        except ValueError:
            self.widgets['loan_percent_income'].config(state='normal')
            self.widgets['loan_percent_income'].delete(0, tk.END)
            self.widgets['loan_percent_income'].config(state='readonly')

    def checkbox_clicked(self, field):
        # If any checkbox is selected, disable the "Other" option
        if any(var.get() for var in self.checkbox_vars[field].values()):
            self.other_var.set(False)
            self.other_entry.config(state='disabled')
            self.other_entry.delete(0, tk.END)


    def validate_entry(self, value, validation_type, field):
        try:
            if validation_type == 'int':
                val = int(value)
                
                if field == 'person_age':
                    if val < 18:
                        return False, "Age must be at least 18 years old"
                
                elif field == 'cb_person_cred_hist_length':
                    if val < 0:
                        return False, "Credit history length must be positive"
                    
                    # Get age value
                    try:
                        age = int(self.widgets['person_age'].get())
                        # Credit history length cannot exceed (age-18) years
                        max_credit_length = age - 18
                        if val > max_credit_length:
                            return False, f"Credit history length cannot exceed {max_credit_length} years (age minus 18)"
                    except ValueError:
                        # If age field is empty or invalid, do basic validation only
                        if val > 100:  # Set a reasonable maximum
                            return False, "Credit history length seems too long"
                
            elif validation_type == 'float':
                val = float(value)
                
                if field == 'person_income':
                    if val <= 0:
                        return False, "Income must be positive"
                
                elif field == 'loan_amnt':
                    if val <= 0:
                        return False, "Loan amount must be positive"
                
                elif field == 'person_emp_length':
                    if val < 0:
                        return False, "Employment length must be positive"
                    
                    # Get age value
                    try:
                        age = int(self.widgets['person_age'].get())
                        # Employment length cannot exceed (age-18) years
                        max_emp_length = age - 16
                        if val > max_emp_length:
                            return False, f"Employment length cannot exceed {max_emp_length} years (age minus 16)"
                    except ValueError:
                        # If age field is empty or invalid, do basic validation only
                        if val > 100:  # Set a reasonable maximum
                            return False, "Employment length seems too long"
            
                
                elif field == 'loan_int_rate':
                    if val < 0:
                        return False, "Interest rate must be positive"
                
            return True, None
        except ValueError:
            return False, f"Please enter a valid {validation_type}"
        
    def validate_other_entry(self, value):
        if not value:
            return False, "Other option cannot be empty"
        if not re.match("^[a-zA-Z0-9]+$", value):
            return False, "Other option can only contain letters and numbers"
        return True, None

    def get_checkbox_value(self, field):
        if field == 'loan_intent' and self.other_var.get():
            return self.other_entry.get().upper()
        selected = [value for value, var in self.checkbox_vars[field].items() if var.get()]
        return selected[0] if selected else None

    def validate_inputs(self):
        errors = []
        
        # Validate entry fields
        for field, props in self.fields.items():
            if props['type'] == 'entry':
                value = self.widgets[field].get()
                if not value:
                    errors.append(f"{props['label']} is required")
                else:
                    valid, error_msg = self.validate_entry(value, props['validation'], field)
                    if not valid:
                        errors.append(f"{props['label']}: {error_msg}")

        # Validate radio_with_other fields
        for field, props in self.fields.items():
            if props['type'] == 'radio_with_other':
                value = self.radio_vars[field].get()
                if not value:
                    errors.append(f"{props['label']}: Please select one option")
                elif value == "OTHER":
                    other_value = self.other_entries[field].get().strip()
                    valid, error_msg = self.validate_other_entry(other_value)
                    if not valid:
                        errors.append(f"{props['label']}: {error_msg}")

        # Validate checkbox fields
        for field, props in self.fields.items():
            if props['type'] in ['checkbox', 'checkbox_with_other']:
                if field == 'loan_intent':
                    if not any(var.get() for var in self.checkbox_vars[field].values()) and not self.other_var.get():
                        errors.append(f"{props['label']}: Please select one option")
                    elif self.other_var.get():
                        valid, error_msg = self.validate_other_entry(self.other_entry.get())
                        if not valid:
                            errors.append(f"{props['label']}: {error_msg}")
                else:
                    if not any(var.get() for var in self.checkbox_vars[field].values()):
                        errors.append(f"{props['label']}: Please select one option")

        # Additional validation for loan amount vs income
        try:
            loan_amount = float(self.widgets['loan_amnt'].get() or 0)
            income = float(self.widgets['person_income'].get() or 0)
            if loan_amount > 0 and income > 0:
                if loan_amount / income > 10:
                    errors.append("Loan amount cannot exceed 10 times annual income")
        except ValueError:
            pass

        return errors
    
    def reset_form(self):
        """Reset form"""
        # Reset entry widgets
        for widget in self.widgets.values():
            if isinstance(widget, ttk.Entry):
                widget.config(state='normal')
                widget.delete(0, tk.END)
                if 'loan_percent_income' in str(widget):
                    widget.config(state='readonly')

        # Reset checkboxes
        for field_vars in self.checkbox_vars.values():
            for var in field_vars.values():
                var.set(False)

        # Reset radio buttons
        for var in self.radio_vars.values():
            var.set('')

        # Reset other entry
        if self.other_entry:
            self.other_entry.config(state='disabled')
            self.other_entry.delete(0, tk.END)
    def save_data(self):
        # Collect all input values
        data = {}
        for field, props in self.fields.items():
            if props['type'] == 'entry' or props['type'] == 'calculated':
                data[field] = self.widgets[field].get()
            elif props['type'] == 'checkbox' or props['type'] == 'checkbox_with_other':
                data[field] = self.get_checkbox_value(field)
        
        # Validate all fields are filled
        if not all(data.values()):
            messagebox.showerror("Error", "Please fill in all required fields")
            return

        # Convert data types
        try:
            numeric_fields = {
                'person_age': int,
                'person_income': float,
                'person_emp_length': float,
                'loan_amnt': float,
                'loan_int_rate': float,
                'loan_percent_income': float,
                'cb_person_cred_hist_length': int
            }
            
            for field, convert_func in numeric_fields.items():
                if field in data:
                    data[field] = convert_func(data[field])
                    
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers")
            return

        # Save data to CSV file
        try:
            df = pd.DataFrame([data])
            if Path('loan.csv').exists():
                df.to_csv('loan.csv', mode='a', header=False, index=False)
            else:
                df.to_csv('loan.csv', index=False)
            
            messagebox.showinfo("Success", "Data saved successfully")
            self.reset_form()

        except Exception as e:
            messagebox.showerror("Error", f"Error saving data: {str(e)}")        

    def submit_data(self):
        logging.info("Starting data submission process")
        
        errors = self.validate_inputs()
        if errors:
            error_message = "\n".join(errors)
            logging.warning(f"Validation errors: {error_message}")
            messagebox.showerror("Validation Error", error_message)
            return
        
        try:
            file_path = Path('loan.csv')
            lock = FileLock("loan.csv.lock", timeout=10)
            
            with lock:
                logging.info("File lock acquired")
                
                # Check if file exists
                if not file_path.exists():
                    logging.info("Creating new CSV file")
                    df = pd.DataFrame(columns=['id'] + list(self.fields.keys()))
                    next_id = 1
                else:
                    logging.info("Reading existing CSV file")
                    df = pd.read_csv(file_path)
                    next_id = df['id'].max() + 1 if not df.empty else 1
                
                # Prepare new data
                new_data = {'id': next_id}
                logging.info(f"Preparing new data with ID: {next_id}")
                
                # Get all field values
                for field, props in self.fields.items():
                    if props['type'] in ['entry', 'calculated']:
                        value = self.widgets[field].get()
                        if 'validation' in props:
                            if props['validation'] == 'int':
                                value = int(value) if value else 0
                            elif props['validation'] == 'float':
                                value = float(value) if value else 0.0
                        new_data[field] = value
                    elif props['type'] in ['radio', 'radio_with_other']:
                        value = self.radio_vars[field].get()
                        if value == "OTHER":
                            value = self.other_entries[field].get()
                            if not value.strip(): 
                                raise ValueError(f"Please enter a value for Other in {props['label']}")
                        new_data[field] = value if value else 'None'
                    elif props['type'] in ['checkbox', 'checkbox_with_other']:
                        new_data[field] = self.get_checkbox_value(field)
                
                # Create new row DataFrame
                new_df = pd.DataFrame([new_data])
                
                # Append new row to existing DataFrame
                df = pd.concat([df, new_df], ignore_index=True)
                
                # Save entire DataFrame to CSV
                df.to_csv(file_path, index=False)
                logging.info(f"Successfully saved data with ID: {next_id}")
                
                # Create backup after successful save
                try:
                    self.backup_data()
                    logging.info("Backup created successfully")
                except Exception as e:
                    logging.error(f"Backup failed: {str(e)}")
                    # Continue execution even if backup fails
                
                messagebox.showinfo("Success", "Data has been added successfully")
                
                # Reset form
                self.reset_form()
                logging.info("Form reset completed")
                
                # Refresh preview windows
                self.refresh_preview_windows()
                logging.info("Preview windows refreshed")
                        
        except Timeout:
            error_msg = "Could not acquire file lock. Please try again later."
            logging.error(error_msg)
            messagebox.showerror("Error", error_msg)
        except ValueError as ve:
            error_msg = f"Invalid input: {str(ve)}"
            logging.error(error_msg)
            messagebox.showerror("Validation Error", error_msg)
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            logging.error(f"Unexpected error: {error_msg}", exc_info=True)
            messagebox.showerror("Error", error_msg)

    def backup_data(self):
        try:
            from datetime import datetime
            import shutil
            import os
            
            backup_filename = f'loan_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            print(f"Creating backup: {backup_filename}")
            logging.info(f"Attempting to create backup: {backup_filename}")
            
            # 使用专门的备份锁文件
            with FileLock("backup.lock", timeout=10):
                shutil.copy2('loan.csv', backup_filename)
                
                # 验证备份文件是否创建成功
                if os.path.exists(backup_filename):
                    print(f"Backup successfully created at: {backup_filename}")
                    logging.info(f"Backup successfully created: {backup_filename}")
                else:
                    print("Backup file was not created")
                    logging.error("Backup file was not created")
                
        except Exception as e:
            error_msg = f"Backup failed: {str(e)}"
            print(error_msg)
            logging.error(error_msg)
            raise
        

    def refresh_preview_windows(self):
        """refresh all preview windows"""
        try:
            df = pd.read_csv('loan.csv')
            for widget in self.root.winfo_children():
                if isinstance(widget, tk.Toplevel):
                    for child in widget.winfo_children():
                        if isinstance(child, ttk.Frame):
                            for grandchild in child.winfo_children():
                                if isinstance(grandchild, ttk.Treeview):
                                    grandchild.delete(*grandchild.get_children())
                                    for _, row in df.iterrows():
                                        grandchild.insert("", 'end', values=list(row))
        except Exception as e:
            print(f"Error refreshing preview windows: {str(e)}")

    
    

def main():
    root = tk.Tk()
    app = LoanApplicationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()