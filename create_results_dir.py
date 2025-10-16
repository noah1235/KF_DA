import os

def create_results_dir():
    root = "E:/NF/KF_Results"
    os.makedirs(root, exist_ok=True)
    return root