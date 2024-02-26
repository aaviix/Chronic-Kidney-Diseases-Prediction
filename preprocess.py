import pandas as pd

# Load the dataset
df = pd.read_csv('kidney_disease.csv')

# Open the description file in write mode
with open('description.txt', 'w') as f:
    # Write the description of the dataset
    f.write('Description of Dataset:\n')
    for column in df.columns:
        f.write(f'{column}: Description of {column}\n')

    # Write the variables ignored
    f.write('\nvariables ignored:\n')
    ignored_variables = ['id','bgr','bu','sc','pcv','hemo','htn','dm','cad','appet','pe','ane','sod','pot','wc','rc'] # replace with your actual ignored variables
    for variable in ignored_variables:
        f.write(f'{variable}\n')


# Remove the specified columns
df = df.drop(columns=ignored_variables)

# Save the modified data to a new file
df.to_csv('Chronic-kidney-disease-final.csv', index=False)
