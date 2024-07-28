import re
import pandas as pd

# Function to process the text file and extract metrics
def process_text_file(input_file, output_file):
    # Define regex patterns to match each metric
    patterns = {
        'Run': r'\*\*\*Run #(\d+)',
        'Random_seed': r'Random seed used: (\d+)',
        'Training_time': r'Time spent on training: ([\d.]+) ms',
        'Validation_AUC': r'Validation AUC: ([\d.]+)',
        'Test_AUC': r'Test AUC: ([\d.]+)',
        'Accuracy': r'Accuracy: ([\d.]+)',
        'Precision': r'Precision: ([\d.]+)',
        'Recall': r'Recall: ([\d.]+)',
        'F1_Score': r'F1 Score: ([\d.]+)',
        'FPR': r'False Positive Rate \(FPR\): ([\d.]+)',
        'TPR': r'True Positive Rate \(TPR\): ([\d.]+)',
        'Energy_pkg': r'([\d.]+) Joules power/energy-pkg/',
        'Energy_ram': r'([\d.]+) Joules power/energy-ram/',
        'Time_elapsed': r'([\d.]+) seconds time elapsed'
    }

    # Initialize a list to store extracted metrics
    metrics_list = []
    metrics = {}
    
    # Open and read the input file line by line
    with open(input_file, 'r') as file:
        for line in file:
            line = line.strip()
            
            if line.startswith('***'):
                if metrics:  # Save the previously collected metrics
                    metrics_list.append(metrics)
                metrics = {}  # Reset metrics for the new run

            for key, pattern in patterns.items():
                match = re.match(pattern, line)
                if match:
                    metrics[key] = match.group(1)
                    
        if metrics:  # Don't forget to save the last collected metrics
            metrics_list.append(metrics)

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(metrics_list)
    
    # Convert numeric columns to appropriate types
    df = df.apply(pd.to_numeric, errors='ignore')
    
    # Save the DataFrame to a CSV file
    df.to_csv(output_file, index=False)

# Define input and output file paths
folderpath = '/home/eloise/eloise/'
input_file = folderpath + 'output/' + 'perf_output_4_rfe_xgboost_features_7.txt'  # Replace with your input text file path
output_file = folderpath + 'analysis/output_extracted/' + 'perf_output_4_rfe_xgboost_features_7.csv'  # Replace with your desired output CSV file path

# Process the text file and save to CSV
process_text_file(input_file, output_file)

print(f"Metrics have been extracted and saved to {output_file}.")
