import numpy as np
import pandas as pd
import yaml
import itertools
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--start_cond', type=str, action='store', default=r'./start_cond',
                    help='cond to save files to.')
parser.add_argument('-b', '--boundaries', type=str, action='store', default='Boundaries.yaml',
                    help='File containing boundaries for the reactions.')
parser.add_argument('-n', '--n_samples', type=int, action='store', default=10,
                    help='Number conditions for the following reactions.')
args = parser.parse_args()

np.random.seed(6)
random_state = np.random.seed(6)

# Import search space file from yaml file
headers = ["Bb", "CP", "Solvent", "Electrode", "Additives", "Prior_score"]
with open('Boundaries.yaml', 'r') as pd.DataFrame:
    try:
        input_all_reactions = yaml.safe_load(pd.DataFrame)
    except yaml.YAMLError as exc:
        print(exc)
    total_list = [input_all_reactions[key] for key in input_all_reactions]
    combinations = list((itertools.product(*total_list)))
    random_state = random.sample(combinations, k=20)
    print(random.sample(combinations, k=20))
    with open("all_possible_combinations1.txt", "w") as output:
        s = "\t".join(map(str, headers))
        output.write(s + '\n')
        for row in combinations:
            s = "\t".join(map(str, row))
            output.write(s + '\n')
