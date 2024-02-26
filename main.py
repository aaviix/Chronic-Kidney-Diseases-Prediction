import sys
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont, QIntValidator, QDoubleValidator, QPalette, QColor
from PyQt6.QtWidgets import QApplication, QTabWidget, QWidget, QLineEdit, QLabel, QPushButton, QComboBox
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# Read the dataset
data = pd.read_csv('Chronic-kidney-disease-final-edited.csv')

# Data preprocessing2
X = data.drop(columns='classification', axis=1)
Y = data['classification'].map({'ckd': 1, 'notckd': 0})

# Convert 'classification' to a numeric column
data['classification_numeric'] = data['classification'].map({'ckd': 1, 'notckd': 0})

# Removing rows where target variable 'Y' is NaN
valid_indices = Y.dropna().index
X = X.loc[valid_indices]
Y = Y.loc[valid_indices]

# One-hot encode categorical variables and handle missing values
X = pd.get_dummies(X, drop_first=True)
X.fillna(X.mean(), inplace=True)

# Scaling features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.25, random_state=2, stratify=Y)

# Train the model with increased max_iter
model = LogisticRegression(max_iter=7823)    #Don
model.fit(x_train, y_train)

# Evaluate the model
predictions = model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))
print("Recall:", recall_score(y_test, predictions))
print("F1 Score:", f1_score(y_test, predictions))

warnings.filterwarnings(action='ignore', category=UserWarning)


class Window(FigureCanvas):
    def __init__(self, parent, data):
        fig, self.ax = plt.subplots(2, figsize=(8.9, 10))
        super().__init__(fig)
        self.setParent(parent)
        self.data = data

        # Initial plot setup
        self.update_plots(None, None)

    def update_plots(self, age, bp):
        # Clear existing plots
        self.ax[0].clear()
        self.ax[1].clear()

        # Update scatter plot (Age vs Blood Pressure)
        # Filter data based on the user's input for age and bp if provided
        if age is not None:
            filtered_data = self.data[self.data['age'] == age]
        else:
            filtered_data = self.data

        if bp is not None:
            filtered_data = filtered_data[filtered_data['bp'] == bp]

        self.ax[0].scatter(filtered_data['age'], filtered_data['bp'])
        self.ax[0].set_xlabel('Age')
        self.ax[0].set_ylabel('Blood Pressure')
        self.ax[0].set_title('Scatter Plot of Age vs Blood Pressure')

        # Update line plot (Age vs Average Disease Classification)
        if age is not None:
            line_plot_data = self.data.groupby('age')['classification_numeric'].mean().loc[:age]
        else:
            line_plot_data = self.data.groupby('age')['classification_numeric'].mean()

        line_plot_data.plot(kind='line', ax=self.ax[1])
        self.ax[1].set_title('Age vs Average Disease Classification')
        self.ax[1].set_xlabel('Age')
        self.ax[1].set_ylabel('Average Disease Classification')

        # Refresh the plot
        self.draw()

    def save_current_plot(self, filename='plot_image.png'):
        self.figure.savefig(filename)

class AppMain(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.resize(900, 900)
        self.plot_window = Window(self, data)


def get_input_value(input_field):
    # Utility method to safely extract integer values from input fields
    try:
        return int(input_field.text())
    except ValueError:
        return None


class MainWindow(QTabWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chronic Kidney Disease Prediction")
        self.setFixedSize(QSize(1500, 1000))
        self.font = QFont("Arial", 10)
        self.setStyleSheet("QWidget { background-color: #2b2b2b; color: #d3d3d3; }")

        self.head = QLabel("Please Enter Patient's details.", self)
        self.head.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.head.move(900, 100)
        self.head.setAlignment(Qt.AlignmentFlag.AlignCenter)  # Changed for PyQt6

        self.label_age = QLabel("Age:", self)
        self.label_age.move(900, 220)
        self.label_age.setFont(self.font)
        self.value_age = QLineEdit(self)
        self.value_age.setValidator(QIntValidator(0, 120, self))  # Validator for age range 0-120
        self.value_age.move(1100, 215)
        self.value_age.resize(150, 30)

        self.label_bp = QLabel("Blood Pressure:", self)
        self.label_bp.move(900, 280)
        self.label_bp.setFont(self.font)
        self.value_bp = QLineEdit(self)
        self.value_bp.setValidator(QIntValidator(50, 80, self))
        self.value_bp.move(1100, 275)
        self.value_bp.resize(150, 30)

        self.label_sg = QLabel("Specific Gravity:", self)
        self.label_sg.move(900, 340)
        self.label_sg.setFont(self.font)
        self.value_sg = QLineEdit(self)
        self.value_sg.setValidator(QDoubleValidator(1, 1.02, 2, self))
        self.value_sg.move(1100, 340)
        self.value_sg.resize(150, 30)

        self.label_al = QLabel("Albumin:", self)
        self.label_al.move(900, 400)
        self.label_al.setFont(self.font)
        self.value_al = QLineEdit(self)
        self.value_al.setValidator(QIntValidator(0, 5, self))
        self.value_al.move(1100, 395)
        self.value_al.resize(150, 30)

        self.label_su = QLabel("Sugar:", self)
        self.label_su.move(900, 460)
        self.label_su.setFont(self.font)
        self.value_su = QLineEdit(self)
        self.value_su.setValidator(QIntValidator(0, 5, self))
        self.value_su.move(1100, 460)
        self.value_su.resize(150, 30)

        self.label_rbc = QLabel("Red Blood Count:", self)
        self.label_rbc.move(900, 520)
        self.label_rbc.setFont(self.font)
        self.value_rbc = QComboBox(self)
        self.value_rbc.addItem("Normal")
        self.value_rbc.addItem("Abnormal")
        self.value_rbc.move(1100, 520)
        self.value_rbc.resize(150, 30)

        self.label_pc = QLabel("Pus Cell:", self)
        self.label_pc.move(900, 580)
        self.label_pc.setFont(self.font)
        self.value_pc = QComboBox(self)
        self.value_pc.addItem("Normal")
        self.value_pc.addItem("Abnornal")
        self.value_pc.move(1100, 580)
        self.value_pc.resize(150, 30)

        self.label_pcc = QLabel("Pus Cell Clumps:", self)
        self.label_pcc.move(900, 640)
        self.label_pcc.setFont(self.font)
        self.value_pcc = QComboBox(self)
        self.value_pcc.addItem("Present")
        self.value_pcc.addItem("Notpresent")
        self.value_pcc.move(1100, 640)
        self.value_pcc.resize(150, 30)

        self.label_ba = QLabel("Bacteria:", self)
        self.label_ba.move(900, 700)
        self.label_ba.setFont(self.font)
        self.value_ba = QComboBox(self)
        self.value_ba.addItem("Present")
        self.value_ba.addItem("Notpresent")
        self.value_ba.move(1100, 700)
        self.value_ba.resize(150, 30)

        # Initialize the plot window as an attribute of MainWindow
        self.plot_window = Window(self, data)
        # self.addTab(self.plot_window, "Prediction Model")    #Don

        # Connect input change events for QLineEdit to update function
        self.value_age.textChanged.connect(self.update_plots_and_prediction)
        self.value_bp.textChanged.connect(self.update_plots_and_prediction)
        self.value_sg.textChanged.connect(self.update_plots_and_prediction)
        self.value_al.textChanged.connect(self.update_plots_and_prediction)
        self.value_su.textChanged.connect(self.update_plots_and_prediction)

        # Connect input change events for QComboBox to update function
        self.value_rbc.currentTextChanged.connect(self.update_plots_and_prediction)
        self.value_pc.currentTextChanged.connect(self.update_plots_and_prediction)
        self.value_pcc.currentTextChanged.connect(self.update_plots_and_prediction)
        self.value_ba.currentTextChanged.connect(self.update_plots_and_prediction)

        # submit button
        self.button = QPushButton("Submit", self)
        self.button.setCheckable(True)
        self.button.move(900, 770)
        self.button.setCheckable(True)
        self.button.clicked.connect(self.predict_disease)  # Signal-slot connection remains the same

        # output label
        self.prediction = QLabel("", self)
        self.prediction.setGeometry(900, 720, 300, 50)
        self.head.setFont(QFont("Arial", 12, QFont.Weight.Bold))

        # Save image button
        self.save_image_button = QPushButton("Save Plot", self)
        self.save_image_button.setCheckable(True)
        self.save_image_button.move(980, 770)
        self.save_image_button.setCheckable(True)
        self.save_image_button.clicked.connect(self.save_image)

        main = AppMain()
        self.tab = QWidget(self)
        self.addTab(main, "Prediction Model")

    def predict_disease(self):
        age = int(self.value_age.text())
        bp = int(self.value_bp.text())
        sg = float(self.value_sg.text())
        al = int(self.value_al.text())
        su = int(self.value_su.text())

        # Convert categorical variables to the format expected by the model
        rbc = 'normal' if self.value_rbc.currentText() == "Normal" else 'abnormal'
        pc = 'normal' if self.value_pc.currentText() == "Normal" else 'abnormal'
        pcc = 'present' if self.value_pcc.currentText() == "Present" else 'notpresent'
        ba = 'present' if self.value_ba.currentText() == "Present" else 'notpresent'

        # Create a temporary DataFrame for user input
        user_input_temp = pd.DataFrame([[age, bp, sg, al, su, rbc, pc, pcc, ba]],
                                       columns=['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba'])

        # Print the processed user input
        print(f"Processed user input: {user_input_temp}")

        # Apply one-hot encoding to the user input as was done for the training data
        user_input_encoded = pd.get_dummies(user_input_temp, drop_first=True)

        # Ensure the user input has the same features as the training data
        # Add missing columns with 0 value (for categories not present in the user input)
        for col in X.columns:
            if col not in user_input_encoded.columns:
                user_input_encoded[col] = 0

        # Reorder columns to match the training data
        user_input_encoded = user_input_encoded[X.columns]

        # Scale the input features
        user_input_scaled = scaler.transform(user_input_encoded)

        # Predict the disease
        prediction = model.predict(user_input_scaled)

        # Print the prediction
        print(f"Prediction: {'Chronic Kidney Disease' if prediction[0] == 1 else 'No Chronic Kidney Disease'}")

        # Display the prediction result
        if prediction[0] == 1:
            self.prediction.setText("Patient has Chronic Kidney Disease.")
        else:
            self.prediction.setText("Patient doesn't have Chronic Kidney Disease.")

    def update_plots_and_prediction(self):
        # Extract current values from input fields
        age = get_input_value(self.value_age)
        bp = get_input_value(self.value_bp)

        # Update plots
        self.plot_window.update_plots(age, bp)

    def save_image(self):
        self.plot_window.save_current_plot()

# Main application execution
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Apply dark theme to the application
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    app.setPalette(palette)

    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec())
