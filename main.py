import tkinter as tk
from tkinter import ttk
import customtkinter as ctk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

url = 'https://raw.githubusercontent.com/YBIFoundation/Dataset/main/Cancer.csv'
df = pd.read_csv(url)

class CustomApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Cancer Analysis and Prediction")
        self.geometry("1920x1080")
        self.state('zoomed')

        # Create a notebook 
        self.notebook = ctk.CTkNotebook(self)
        self.notebook.pack(fill='both', expand=True)

        # Create tabs
        self.intro_tab = ctk.CTkFrame(self.notebook)
        self.view_tab = ctk.CTkFrame(self.notebook)
        self.analysis_tab = ctk.CTkFrame(self.notebook)
        self.predict_tab = ctk.CTkFrame(self.notebook)

        self.notebook.add(self.intro_tab, text="Intro")
        self.notebook.add(self.view_tab, text="View")
        self.notebook.add(self.analysis_tab, text="Analysis")
        self.notebook.add(self.predict_tab, text="Predict")

        # Intro tab
        self.create_intro_tab()

        # View tab
        self.create_view_tab()

        # Analysis tab
        self.create_analysis_tab()

        # Predict tab
        self.create_predict_tab()

    def create_intro_tab(self):
        title = ctk.CTkLabel(self.intro_tab, text="Cancer Analysis and Prediction Project", font=('Arial', 24))
        title.pack(pady=20)
        subtitle = ctk.CTkLabel(self.intro_tab, text="Built by [Your Name]", font=('Arial', 18))
        subtitle.pack(pady=10)

    def create_view_tab(self):
        tree = ttk.Treeview(self.view_tab, columns=list(df.columns), show='headings')
        for col in df.columns:
            tree.heading(col, text=col)
            tree.column(col, width=100)
        tree.pack(fill='both', expand=True)

        for index, row in df.iterrows():
            tree.insert("", tk.END, values=list(row))

    def create_analysis_tab(self):
        figure = plt.Figure(figsize=(12, 8), dpi=100)
        ax = figure.add_subplot(111)
        df['diagnosis'].value_counts().plot(kind='bar', ax=ax)
        ax.set_title('Diagnosis Distribution')
        ax.set_xlabel('Diagnosis')
        ax.set_ylabel('Count')

        canvas = FigureCanvasTkAgg(figure, master=self.analysis_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def create_predict_tab(self):
        self.entries = {}
        self.labels = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']
        for i, label in enumerate(self.labels):
            l = ctk.CTkLabel(self.predict_tab, text=label)
            l.grid(row=i, column=0, padx=10, pady=5)
            e = ctk.CTkEntry(self.predict_tab)
            e.grid(row=i, column=1, padx=10, pady=5)
            self.entries[label] = e

        self.predict_button = ctk.CTkButton(self.predict_tab, text="Predict", command=self.predict)
        self.predict_button.grid(row=len(self.labels), column=0, columnspan=2, pady=20)

        self.result_label = ctk.CTkLabel(self.predict_tab, text="")
        self.result_label.grid(row=len(self.labels)+1, column=0, columnspan=2, pady=10)

    def predict(self):
        data = {label: float(entry.get()) for label, entry in self.entries.items()}
        features = pd.DataFrame([data])

        X = df[self.labels]
        y = df['diagnosis']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression()
        model.fit(X_train, y_train)

        prediction = model.predict(features)
        result = "Positive" if prediction[0] == 1 else "Negative"
        self.result_label.configure(text=f"Prediction: {result}")

if __name__ == "__main__":
    app = CustomApp()
    app.mainloop()
