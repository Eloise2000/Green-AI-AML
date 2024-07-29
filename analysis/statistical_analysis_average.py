import pandas as pd
from tabulate import tabulate

filefolder = "output_extracted/"
# List of treatments and their corresponding CSV files
techniques = {
    'Baseline': filefolder + 'perf_output_0_normal_xgboost.csv',
    'RUS': filefolder + 'perf_output_1_rus_xgboost.csv',
    'SUS': filefolder + 'perf_output_2_stratified_xgboost.csv',
    'KNN': filefolder + 'perf_output_3_knn_xgboost.csv',
    'RFE-11': filefolder + 'perf_output_4_rfe_xgboost_features_11.csv',
    'RFE-10': filefolder + 'perf_output_4_rfe_xgboost_features_10.csv',
    'RFE-9': filefolder + 'perf_output_4_rfe_xgboost_features_9.csv',
    'RFE-8': filefolder + 'perf_output_4_rfe_xgboost_features_8.csv',
    'RFE-7': filefolder + 'perf_output_4_rfe_xgboost_features_7.csv'
}

# Mapping of original column names to final column names
column_mapping = {
    'Test_AUC': 'AUC',
    'Accuracy': 'Accuracy',
    'Precision': 'Precision',
    'Recall': 'Recall (TPR)',
    'F1_Score': 'F1',
    'FPR': 'FPR',
    'TPR': 'TPR',
    'Energy_pkg': 'Energy_pkg',
    'Energy_ram': 'Energy_ram',
    'Time_elapsed': 'Runtime'
}

# Columns of interest (final column names)
columns_of_interest = [
    'AUC', 'Energy', 'Runtime', 'Recall (TPR)', 'Accuracy', 'Precision', 'F1', 'FPR'
]

# Dictionary to store the results
results = {}

# Process each treatment
for technique, file in techniques.items():
    # Read the CSV file
    df = pd.read_csv(file)
    
    # Rename the columns according to the mapping
    df = df.rename(columns=column_mapping)
    
    # Compute the Energy feature
    df['Energy'] = df['Energy_pkg'] + df['Energy_ram']
    
    # Calculate the average of the columns of interest
    averages = df[columns_of_interest].mean()
    
    # Store the results in the dictionary
    results[technique] = averages

# Create a DataFrame from the results dictionary
results_df = pd.DataFrame(results).T

# Round the values to 3 decimal places
results_df = results_df.round(3)

# Print the results in a table format
print(tabulate(results_df, headers='keys', tablefmt='pretty'))
