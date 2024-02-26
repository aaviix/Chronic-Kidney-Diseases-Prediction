import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning
import warnings

# Load your dataset
data = pd.read_csv('Chronic-kidney-disease-final.csv')

# Separate features (X) and target variable (y)
X = data.drop(columns='classification', axis=1)
y = data['classification']

# Identify numeric and categorical columns
numeric_columns = X.select_dtypes(include=np.number).columns
categorical_columns = X.select_dtypes(exclude=np.number).columns

# Create transformers for numeric and categorical columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
])

# Create a column transformer to apply transformers to the correct columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_columns),
        ('cat', categorical_transformer, categorical_columns),
    ])

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2, stratify=y)

# Create a logistic regression model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000)),
])

# Ignore ConvergenceWarning for the purpose of this example
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Model Training
model.fit(x_train, y_train)

# Reset warnings to default
warnings.resetwarnings()

# Evaluate the model on training data
y_train_pred = model.predict(x_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Accuracy on training data: ", train_accuracy)

# Evaluate the model on test data
y_test_pred = model.predict(x_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Accuracy on test data: ", test_accuracy)

# Assuming these are the correct columns based on your training data
all_columns = ['age', 'bp', 'sg', 'al', 'su', 'rbc_normal', 'pc_normal', 'pcc_present', 'ba_present',
               'rbc', 'pc', 'pcc', 'ba']  # Add all other necessary columns here

# Initialize all columns to some default value, for example, 0 or 'missing'
default_values = {col: 0 for col in all_columns}
default_values.update({'rbc_normal': 1, 'pc_normal': 1, 'pcc_present': 0, 'ba_present': 0})

# Update the test inputs with specific values
test_input_values1 = {'age': 62, 'bp': 80, 'sg': 1.01, 'al': 2, 'su': 3}
test_input_values2 = {'age': 48, 'bp': 80, 'sg': 1.025, 'al': 0, 'su': 0}

# Merge default values with specific test input values
test_input1 = {**default_values, **test_input_values1}
test_input2 = {**default_values, **test_input_values2}

# Convert to DataFrame
test_input1 = pd.DataFrame([test_input1])
test_input2 = pd.DataFrame([test_input2])

# Now, make predictions
predict1 = model.predict(test_input1)
predict2 = model.predict(test_input2)

print(f'predict1: {predict1}')
print(f'predict2: {predict2}')
