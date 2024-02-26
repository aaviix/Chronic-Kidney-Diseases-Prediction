import pandas as pd

# Specify the file name of your current CSV
input_file_name = 'Chronic-kidney-disease-final.csv'  # Replace with your actual file name

# Specify the file name for the new, modified CSV
output_file_name = 'Chronic-kidney-disease-final-edited.csv'

# Load the CSV file
df = pd.read_csv(input_file_name)

# List of columns to convert to integers
columns_to_convert = ['age', 'bp', 'al', 'su']

# Convert columns to integers
for col in columns_to_convert:
    # Check if the column exists in the dataframe
    if col in df.columns:
        # Fill NaN values to avoid errors during conversion
        df[col] = df[col].fillna(0)  # You might want to handle NaNs differently
        # Convert to integers (rounding down to the nearest integer)
        df[col] = df[col].apply(lambda x: int(x))

# Save the modified dataframe back to a new CSV file
df.to_csv(output_file_name, index=False)

print(f"File '{output_file_name}' has been created with the modifications.")
