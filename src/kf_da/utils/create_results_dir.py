import os

def create_results_dir():
    with open("../kf-da-configs/data_dir.txt", "r") as f:
        root = f.read().rstrip("\n")
    
    return root
