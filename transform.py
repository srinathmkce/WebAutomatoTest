import pandas as pd

# Read TableA, TableB, and Template tables
tableA = pd.read_csv('table_A.csv')
tableB = pd.read_csv('table_B.csv')
template = pd.read_csv('template.csv')

# Apply transformations for each column in the Template table
result = pd.DataFrame()

# Date column transformation
result['Date'] = pd.to_datetime(tableA['Date_of_Policy']).dt.strftime('%m-%d-%Y')

# EmployeeName column transformation
result['EmployeeName'] = tableA['FullName']

# Plan column transformation
result['Plan'] = tableA['Insurance_Plan'].str.replace(' Plan', '')

# PolicyNumber column transformation
result['PolicyNumber'] = tableA['Policy_No']

# Premium column transformation
result['Premium'] = tableA['Monthly_Premium']

# Display the resulting table
print(result)