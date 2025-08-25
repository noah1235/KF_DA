import os

def create_results_dir():
    root = "/Users/noahfrank/Desktop/Kolmogorov_flow_LPT/Results"
    os.makedirs(root, exist_ok=True)
    return root