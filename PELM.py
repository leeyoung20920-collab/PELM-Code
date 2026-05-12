import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import threading
import warnings

UI_FONT_FAMILY = "Times New Roman"
UI_FONT_SIZE = 15
try:
    import tkinter.font as tkfont
except:
    tkfont = None

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = [UI_FONT_FAMILY]
plt.rcParams['font.size'] = UI_FONT_SIZE
plt.rcParams['axes.unicode_minus'] = False

class TideMLOptimizerGUI:

    def __init__(self, root):
        self.root = root
        self.root.title("Physics-Constrained ML Correction")
        self.root.geometry("1600x1000")

        for fname in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont", "TkFixedFont"):
            tkfont.nametofont(fname).configure(family=UI_FONT_FAMILY, size=UI_FONT_SIZE)

        style = ttk.Style(self.root)
        try:
            style.theme_use('clam')
        except:
            pass

        style.configure("Treeview.Heading", font=(UI_FONT_FAMILY, UI_FONT_SIZE, "bold"))
        style.configure("Treeview", font=(UI_FONT_FAMILY, UI_FONT_SIZE), rowheight=28)
        style.configure("TButton", font=(UI_FONT_FAMILY, UI_FONT_SIZE))
        style.configure("TLabel", font=(UI_FONT_FAMILY, UI_FONT_SIZE))
        style.configure("TEntry", font=(UI_FONT_FAMILY, UI_FONT_SIZE))

        # Files
        self.training_var = tk.StringVar()
        self.observed_var = tk.StringVar()
        self.physical_var = tk.StringVar()
        self.training_file = None
        self.observed_file = None
        self.physical_file = None

        # Data storage
        self.training_data = None
        self.evaluation_data = None
        self.full_model_data = None

        # Model results
        self.models = {}
        self.corrected_data = {}
        self.results = {}
        self.setup_ui()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill='both', expand=True)

        self.setup_data_tab(notebook)
        self.setup_training_tab(notebook)
        self.setup_results_tab(notebook)

        self.status_var = tk.StringVar()
        self.status_var.set("Ready")

        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill='x', pady=(8, 0))
        ttk.Label(status_frame, textvariable=self.status_var).pack(side='left')

        self.progress_var = tk.IntVar(value=0)
        self.progress_bar = ttk.Progressbar(status_frame, mode='determinate', maximum=100, variable=self.progress_var)
        self.progress_bar.pack(side='right', fill='x', expand=True, padx=(10, 0))

    def setup_data_tab(self, notebook):
        data_frame = ttk.Frame(notebook)
        notebook.add(data_frame, text="Data")

        file_frame = ttk.LabelFrame(data_frame, text="Files", padding=10)
        file_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(file_frame, text="Training tide data :").grid(row=0, column=0, sticky='e', pady=5)
        ttk.Entry(file_frame, textvariable=self.training_var, width=80).grid(row=0, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.select_training_file).grid(row=0, column=2, padx=5)

        ttk.Label(file_frame, text="Observed tide data :").grid(row=1, column=0, sticky='e', pady=5)
        ttk.Entry(file_frame, textvariable=self.observed_var, width=80).grid(row=1, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.select_observed_file).grid(row=1, column=2, padx=5)

        ttk.Label(file_frame, text="Physical tide data :").grid(row=2, column=0, sticky='e', pady=5)
        ttk.Entry(file_frame, textvariable=self.physical_var, width=80).grid(row=2, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.select_physical_file).grid(row=2, column=2, padx=5)

        ttk.Button(file_frame, text="Load  data", command=self.load_data).grid(row=3, column=1, pady=20)

        preview_frame = ttk.LabelFrame(data_frame, text="Data preview", padding=10)
        preview_frame.pack(fill='both', expand=True, padx=10, pady=5)

        self.data_info_text = scrolledtext.ScrolledText(
            preview_frame, height=15, width=80, font=tkfont.nametofont("TkTextFont")
        )
        self.data_info_text.pack(fill='both', expand=True)

    def setup_training_tab(self, notebook):
        training_frame = ttk.Frame(notebook)
        notebook.add(training_frame, text="Training")

        param_frame = ttk.LabelFrame(training_frame, text="Training parameters", padding=10)
        param_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(param_frame, text="Validation split (0.05–0.80):").grid(row=0, column=0, sticky='w', pady=5)
        self.validation_size_str = tk.StringVar(value="0.15")
        ttk.Entry(param_frame, textvariable=self.validation_size_str, width=10).grid(
            row=0, column=1, sticky='w', padx=5
        )

        model_frame = ttk.LabelFrame(training_frame, text="Tree models", padding=10)
        model_frame.pack(fill='x', padx=10, pady=5)

        self.model_vars = {}
        self.model_vars['Random Forest'] = tk.BooleanVar(value=True)
        ttk.Checkbutton(model_frame, text="Random Forest", variable=self.model_vars['Random Forest']).grid(
            row=0, column=0, sticky='w', padx=10, pady=5)

        self.model_vars['Extra Trees'] = tk.BooleanVar(value=True)
        ttk.Checkbutton(model_frame, text="Extra Trees", variable=self.model_vars['Extra Trees']).grid(
            row=0, column=1, sticky='w', padx=10, pady=5)

        self.model_vars['Gradient Boosting'] = tk.BooleanVar(value=True)
        ttk.Checkbutton(model_frame, text="Gradient Boosting", variable=self.model_vars['Gradient Boosting']).grid(
            row=0, column=2, sticky='w', padx=10, pady=5)

        run_frame = ttk.Frame(training_frame)
        run_frame.pack(fill='x', padx=10, pady=5)
        ttk.Button(run_frame, text="Start training", command=self.start_training).pack(side='left')
        ttk.Label(run_frame, text=" ").pack(side='left', padx=5)

        log_frame = ttk.LabelFrame(training_frame, text="Logs", padding=10)
        log_frame.pack(fill='both', expand=True, padx=10, pady=5)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=20, width=100, font=tkfont.nametofont("TkTextFont"))
        self.log_text.pack(fill='both', expand=True)

    def setup_results_tab(self, notebook):
        results_frame = ttk.Frame(notebook)
        notebook.add(results_frame, text="Results")

        table_frame = ttk.LabelFrame(results_frame, text="Model performance comparison", padding=10)
        table_frame.pack(fill='x', padx=10, pady=5)

        columns = ('Model Type', 'Correction Method', 'RMSE (m)', 'MAE (m)', 'R²', 'Improvement over Physical %')
        self.results_tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=10)

        for col in columns:
            self.results_tree.heading(col, text=col)
            if col == 'Correction Method':
                self.results_tree.column(col, width=150)
            else:
                self.results_tree.column(col, width=120)

        scrollbar = ttk.Scrollbar(table_frame, orient='vertical', command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar.set)
        self.results_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        export_frame = ttk.Frame(results_frame)
        export_frame.pack(fill='x', padx=10, pady=5)
        ttk.Button(export_frame, text="Export corrected tides", command=self.export_results).pack(side='right', padx=5)

    def select_training_file(self):
        filename = filedialog.askopenfilename(title="Select training data", filetypes=[("CSV files", "*.csv")])
        if filename:
            self.training_file = filename
            self.training_var.set(filename)

    def select_observed_file(self):
        filename = filedialog.askopenfilename(title="Select observed tide data", filetypes=[("CSV files", "*.csv")])
        if filename:
            self.observed_file = filename
            self.observed_var.set(filename)

    def select_physical_file(self):
        filename = filedialog.askopenfilename(title="Select physical tide data", filetypes=[("CSV files", "*.csv")])
        if filename:
            self.physical_file = filename
            self.physical_var.set(filename)

    def load_data(self):
        try:
            if not all([self.training_file, self.observed_file, self.physical_file]):
                messagebox.showerror("Error", "Please select all three files before loading.")
                return

            # 1. Read files
            training_data = pd.read_csv(self.training_file)
            observed_data = pd.read_csv(self.observed_file)
            physical_data = pd.read_csv(self.physical_file)

            # 2. Normalize column names
            def normalize_columns(df):
                df = df.rename(columns=lambda x: x.strip())
                if 'Time' not in df.columns:
                    time_candidates = [c for c in df.columns if 'time' in c.lower()]
                    if time_candidates:
                        df = df.rename(columns={time_candidates[0]: 'Time'})
                return df

            training_data = normalize_columns(training_data)
            observed_data = normalize_columns(observed_data)
            physical_data = normalize_columns(physical_data)

            # 3. Ensure required columns and merge
            if 'Time' not in physical_data.columns or 'Physical' not in physical_data.columns:
                if len(physical_data.columns) >= 2:
                    physical_data = physical_data.rename(columns={physical_data.columns[1]: 'Physical'})
                else:
                    raise ValueError("Physical tide data must contain 'Time' and 'Physical' columns")

            self.full_model_data = pd.merge(
                physical_data[['Time', 'Physical']],
                observed_data.rename(columns={observed_data.columns[1]: 'Observed'})[['Time', 'Observed']],
                on='Time', how='left'
            )

            training_data = training_data.rename(columns={training_data.columns[1]: 'Observed'})
            self.training_data = pd.merge(
                training_data,
                self.full_model_data[['Time', 'Physical']],
                on='Time', how='inner'
            )
            self.training_data = self.training_data[
                self.training_data['Observed'].notna() & self.training_data['Physical'].notna()
            ].copy()

            self.evaluation_data = pd.merge(observed_data, self.full_model_data, on='Time', how='inner')

            if len(self.training_data) == 0:
                raise ValueError("Failed to build training data. Check time alignment.")
            if len(self.evaluation_data) == 0:
                raise ValueError("Failed to build evaluation data. Check time alignment.")

            rmse_orig = np.sqrt(np.mean((self.evaluation_data['Observed'] - self.evaluation_data['Physical']) ** 2))

            info_text = f"""Data loading completed.

 1.  Counts 
* Training data size: {len(self.training_data)}
* Evaluation data size: {len(self.evaluation_data)}
* Full model size: {len(self.full_model_data)}

 2.  Columns
* Training: {list(self.training_data.columns)}
* Evaluation: {list(self.evaluation_data.columns)}

 3.  Physical performance
* RMSE on evaluation data: {rmse_orig:.4f} m
"""
            self.data_info_text.delete("1.0", tk.END)
            self.data_info_text.insert(tk.END, info_text)
            self.status_var.set("Data loaded")

        except Exception as e:
            messagebox.showerror("Error", f"Data loading failed: {str(e)}")
            self.status_var.set("Ready")

    def build_features(self, df):
        features = pd.DataFrame(index=df.index)
        if 'Physical' not in df.columns:
            raise ValueError("Missing 'Physical' column for features")

        Physical = df['Physical'].values

        if 'Time' in df.columns:
            t = pd.to_datetime(df['Time'], errors='coerce')
            if t.notna().any():
                t = t.fillna(method='ffill')
                t0 = t.iloc[0]
                t_hours = (t - t0).dt.total_seconds() / 3600.0
            else:
                t_hours = pd.Series(np.arange(len(df), dtype=float), index=df.index)
        else:
            t_hours = pd.Series(np.arange(len(df), dtype=float), index=df.index)

        # Periodic prior features
        periods = {
            'M2': 12.4206,
            'K1': 23.9345,
            'spring–neap': 14.765 * 24.0,
            'Sa': 365.2422 * 24.0
        }
        th = t_hours.values.astype(float)
        for name, P in periods.items():
            omega = 2.0 * np.pi * th / float(P)
            features[f'per_{name}_sin'] = np.sin(omega)
            features[f'per_{name}_cos'] = np.cos(omega)

        # Basic lag features
        for lag in [1, 2, 3, 4, 6, 12, 24]:
            features[f'lag_{lag}'] = pd.Series(Physical).shift(lag).fillna(method='bfill').values

        # Rolling statistics
        s = pd.Series(Physical)
        for win in [3, 6, 12, 24]:
            features[f'roll_mean_{win}'] = s.rolling(win, min_periods=1).mean().values
            features[f'roll_std_{win}'] = s.rolling(win, min_periods=1).std().fillna(0).values

        # Differences
        for d in [1, 2, 3, 6, 12, 24]:
            features[f'diff_{d}'] = s.diff(d).fillna(0).values

        # Index-based phase indicators
        idx = np.arange(len(df))
        features['idx_mod_12'] = (idx % 12) / 12.0
        features['idx_mod_24'] = (idx % 24) / 24.0

        # Include Physical itself
        features['Physical'] = Physical

        return features

    def build_training_arrays(self):
        features = self.build_features(self.training_data)
        X = features.values
        y = self.training_data['Observed'].values
        return X, y

    def build_full_features(self):
        features_full = self.build_features(self.full_model_data)
        return features_full.values

    def start_training(self):
        if self.training_data is None or len(self.training_data) == 0:
            messagebox.showerror("Error", "Please load the data first or provide at least one non-NaN training label.")
            return
        thread = threading.Thread(target=self.train_models)
        thread.daemon = True
        thread.start()

    def log_message(self, message):
        try:
            self.log_text.insert(tk.END, f"{message}\n")
            self.log_text.see(tk.END)
            self.root.update_idletasks()
        except:
            pass

    def train_models(self):
        try:
            self.log_message("=======\n 1. Start training the tree-based correction system ")
            progress = 5
            self.progress_var.set(progress)

            self.log_message(f"\n Data usage: ")
            self.log_message(f"* Training data : {len(self.training_data)} samples.")
            self.log_message(f"* Observed data: {len(self.evaluation_data)} samples.")

            X, y = self.build_training_arrays()
            full_features = self.build_full_features()

            try:
                val_split = float(self.validation_size_str.get())
            except Exception:
                messagebox.showerror("Error", "Validation split must be a number like 0.15")
                self.status_var.set("Ready")
                return
            test_size = max(0.05, min(0.80, val_split))

            if not (0.05 <= val_split <= 0.80):
                self.log_message(f"[info] Validation split {val_split:.3f} adjusted to {test_size:.2f} within [0.05, 0.80].")

            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            rmse_orig = np.sqrt(np.mean((self.evaluation_data['Observed'] - self.evaluation_data['Physical']) ** 2))

            tree_models_to_try = self._get_selected_tree_models()
            self.results.clear()

            for name, model in tree_models_to_try.items():
                self.log_message(f"\n Now Training {name} ")
                try:
                    model.fit(X_train, y_train)
                    corrected = model.predict(full_features)

                    self.models[name] = model
                    self.corrected_data[name] = corrected

                    eval_df = self.evaluation_data.copy()
                    eval_df = eval_df.rename(columns={'Observed': 'Obs', 'Physical': 'Phys'})
                    eval_df['ML'] = corrected[:len(eval_df)]

                    rmse = np.sqrt(np.mean((eval_df['Obs'] - eval_df['ML']) ** 2))
                    mae = np.mean(np.abs(eval_df['Obs'] - eval_df['ML']))
                    r2 = r2_score(eval_df['Obs'], eval_df['ML'])

                    self.results[name] = {
                        'RMSE': rmse,
                        'MAE': mae,
                        'R2': r2,
                        'eval_pred': eval_df['ML'].values
                    }

                    improvement = ((rmse_orig - rmse) / rmse_orig) * 100
                    self.log_message(f"  RMSE: {rmse:.4f} m, MAE: {mae:.4f} m, R²: {r2:.4f}, "
                                     f"Improvement: {improvement:.1f}%")
                    self.log_message(f"  Validation split: {test_size:.2f}")

                except Exception as e:
                    self.log_message(f"! {name} failed: {str(e)}")
                    continue

                progress = min(95, progress + max(2, int(80 / max(1, len(tree_models_to_try)))))
                self.progress_var.set(progress)

            self.log_message(f"\n 2. Evaluation on Observed Data")
            self.log_message(f"Physical RMSE: {rmse_orig:.4f} m")

            if self.results:
                best = min(self.results.keys(), key=lambda x: self.results[x]['RMSE'])
                best_rmse = self.results[best]['RMSE']
                self.log_message(f"\nBest improvement: {((rmse_orig - best_rmse) / rmse_orig * 100):.1f}%")
                self.log_message(f"\nBest model: corrected training model ({best})")

            progress = 100
            self.progress_var.set(progress)
            self.log_message("\n 3. Training finished \n=======\n")
            self.log_message(" ")
            self.status_var.set("Training completed")

            self.update_results_table()

        except Exception as e:
            self.log_message(f"Training failed: {str(e)}")
            self.status_var.set("Training failed")
            self.progress_var.set(0)

    def _get_selected_tree_models(self):
        tree_models_to_try = {}
        if self.model_vars.get('Random Forest', tk.BooleanVar()).get():
            tree_models_to_try['Random Forest'] = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        if self.model_vars.get('Extra Trees', tk.BooleanVar()).get():
            tree_models_to_try['Extra Trees'] = ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        if self.model_vars.get('Gradient Boosting', tk.BooleanVar()).get():
            tree_models_to_try['Gradient Boosting'] = GradientBoostingRegressor(random_state=42)
        return tree_models_to_try

    def update_results_table(self):
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        if not self.results:
            return

        eval_df = self.evaluation_data.copy()
        rmse_orig = np.sqrt(np.mean((eval_df['Observed'] - eval_df['Physical']) ** 2))
        mae_orig = np.mean(np.abs(eval_df['Observed'] - eval_df['Physical']))
        r2_orig = r2_score(eval_df['Observed'], eval_df['Physical'])

        self.results_tree.insert('', 'end', values=(
            "Physical model", "Physical baseline", f"{rmse_orig:.4f}", f"{mae_orig:.4f}", f"{r2_orig:.4f}", "–"
        ))

        best_rmse = float('inf')
        best_method = None

        for name, metrics in self.results.items():
            rmse = metrics['RMSE']
            mae = metrics['MAE']
            r2 = metrics['R2']
            improvement = ((rmse_orig - rmse) / rmse_orig) * 100
            if rmse < best_rmse:
                best_rmse = rmse
                best_method = name
            display_name = f"Tree-{name}"
            self.results_tree.insert('', 'end', values=(
                "Training model", display_name, f"{rmse:.4f}", f"{mae:.4f}", f"{r2:.4f}", f"{improvement:.1f}%"
            ))

        for item in self.results_tree.get_children():
            values = self.results_tree.item(item, 'values')
            if values[0] == "Training model" and best_method and f"Tree-{best_method}" in values[1]:
                self.results_tree.set(item, 'Correction Method', f"★ Tree-{best_method}")

    def export_results(self):
        try:
            if not self.results:
                messagebox.showerror("Error", "No trained results to export.")
                return

            filename = filedialog.asksaveasfilename(
                title="Save corrected results",
                defaultextension=".xlsx",
                filetypes=[("Excel", "*.xlsx")]
            )
            if not filename:
                return

            writer = pd.ExcelWriter(filename, engine='xlsxwriter')

            full_df = self.full_model_data.copy()
            best = min(self.results.keys(), key=lambda x: self.results[x]['RMSE'])
            full_df['Best_Corrected'] = np.nan
            full_len = len(self.results[best]['eval_pred'])
            full_df.loc[:full_len - 1, 'Best_Corrected'] = self.results[best]['eval_pred']
            full_df.to_excel(writer, index=False, sheet_name="FullSeries")

            for name, metrics in self.results.items():
                df = pd.DataFrame({
                    'Time': self.evaluation_data['Time'],
                    'Observed': self.evaluation_data['Observed'],
                    'Physical': self.evaluation_data['Physical'],
                    f'{name}_Corrected': metrics['eval_pred'],
                })
                df.to_excel(writer, index=False, sheet_name=f"{name[:25]}")

            # Summary
            summary_rows = []
            eval_df = self.evaluation_data.copy()
            rmse_orig = np.sqrt(np.mean((eval_df['Observed'] - eval_df['Physical']) ** 2))
            mae_orig = np.mean(np.abs(eval_df['Observed'] - eval_df['Physical']))
            r2_orig = r2_score(eval_df['Observed'], eval_df['Physical'])
            summary_rows.append(["Physical baseline", rmse_orig, mae_orig, r2_orig, 0.0])
            for name, metrics in self.results.items():
                rmse = metrics['RMSE']; mae = metrics['MAE']; r2 = metrics['R2']
                improvement = ((rmse_orig - rmse) / rmse_orig) * 100
                summary_rows.append([name, rmse, mae, r2, improvement])
            summary_df = pd.DataFrame(summary_rows, columns=['Method', 'RMSE', 'MAE', 'R2', 'Improvement_%'])
            summary_df.to_excel(writer, index=False, sheet_name="Summary")

            workbook  = writer.book
            center_fmt = workbook.add_format({
                'font_name': 'Times New Roman',
                'align': 'center',
                'valign': 'vcenter'
            })

            def _format_sheet(sheet_name, ncols):
                ws = writer.sheets[sheet_name]
                ws.set_column(0, max(0, ncols - 1), 20, center_fmt)

            _format_sheet("FullSeries", len(full_df.columns))
            for name, metrics in self.results.items():
                sheet = f"{name[:25]}"
                ncols = 4
                _format_sheet(sheet, ncols)
            _format_sheet("Summary", len(summary_df.columns))

            writer.close()

            info_msg = f"""Export completed.

"Best tree correction"
Best correction: {best}
Corrected RMSE: {self.results[best]['RMSE']:.4f} m

"Export formatting"
- Column width: 20
- Alignment: center (horizontal & vertical)
- Font: Times New Roman

Saved to: {filename}"""
            messagebox.showinfo("Export success", info_msg)

        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {str(e)}")

def main():
    root = tk.Tk()
    app = TideMLOptimizerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()