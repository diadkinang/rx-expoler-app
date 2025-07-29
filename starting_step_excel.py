import argparse
import pandas as pd
import numpy as np
import itertools
import random

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--start_cond', type=str, action='store', default=r'./start_cond',
                    help='cond to save files to.')
parser.add_argument('-b', '--boundaries', type=str, action='store', default='Boundaries.xlsx',
                    help='Excel file containing boundaries for the reactions.')
parser.add_argument('-n', '--n_samples', type=int, action='store', default=10,
                    help='Number of conditions for the following reactions.')
args = parser.parse_args()

np.random.seed(6)

# Define headers
headers = ["Bb", "CP", "Solvent", "Electrode", "Additives", "Prior_score"]

# Read the Excel file
try:
    df = pd.read_excel(args.boundaries)
except Exception as e:
    print(f"Error reading Excel file: {e}")
    exit(1)

# Check that all required headers exist
for col in headers:
    if col not in df.columns:
        print(f"Missing expected column: {col}")
        exit(1)

# Convert each column to a list of options
total_list = [df[col].dropna().tolist() for col in headers]

# Create all possible combinations
combinations = list(itertools.product(*total_list))

# Sample random conditions
random_state = random.sample(combinations, k=args.n_samples)

# Write all combinations
with open("all_possible_combinations.txt", "w") as output:
    header_row = "\t".join(headers)
    output.write(header_row + '\n')
    for row in combinations:
        s = '\t'.join(map(str, row))
        output.write(s + '\n')


# Write training data sample
with open("training_data.txt", "w") as output:
    header_row = "\t".join(headers)
    output.write(header_row + '\n')
    for row in random_state:
        s = '\t'.join(map(str, row))
        output.write(s + '\n')

