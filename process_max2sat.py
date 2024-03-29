"""
File to process MAX2SAT instances from https://www-or.amp.i.kyoto-u.ac.jp/~yagiura/sat/max2sat/

"""

import os
import sys  
import numpy as np
import pandas as pd
from maxsat import Formula
import pickle

def process_max2sat(file_name, separator='_'):
    """
    Function to process MAX2SAT instances from https://www-or.amp.i.kyoto-u.ac.jp/~yagiura/sat/max2sat/
    
    Parameters:
    - file_path: str, path to the file to process
    
    Returns:
    - formula: Formula, the processed formula

    """
    
    with open("datasets/MAX2SAT/" + file_name, 'r') as f:
        lines = f.readlines()
        
    n_vars, n_clauses = file_name.split('.')[0].split(separator)[1], file_name.split('.')[0].split(separator)[2]
    n_vars, n_clauses = int(n_vars), int(n_clauses)

    clauses = []
    for line in [line for line in lines if line[0] != 'c' and line[0] != 'p']:
        try:
            clause = line.split()[1:3]
            clause = tuple([int(x) for x in clause])
            if clause != ():
                clauses.append(clause)
        except:
            break

    formula = Formula(n_vars, n_clauses, clauses)

    # Store the formula
    if not os.path.exists("datasets/pmax2sat/processed"):
        os.makedirs("datasets/pmax2sat/processed")

    with open("datasets/pmax2sat/processed/" + file_name.split('.')[0] + ".pkl", 'wb') as f:
        pickle.dump(formula, f)

if __name__ == '__main__':
    for file_name in [file for file in os.listdir("datasets/MAX2SAT/")if "-" not in file and ".wcnf" in file]:
        if  500 <= int(file_name.split("_")[1]) <= 5000:
            process_max2sat(file_name)
    for file_name in [file for file in os.listdir("datasets/MAX2SAT/")if "-" in file and ".wcnf" in file]:
        if  100 <= int(file_name.split("-")[1]) <= 1000:
            process_max2sat(file_name, separator='-')
            