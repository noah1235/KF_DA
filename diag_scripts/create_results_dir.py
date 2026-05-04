import os

def create_results_dir():

    with open("../kf_da_configs/data_dir.txt", "r") as f:
        root = f.read().rstrip("\n")
    
    return root
