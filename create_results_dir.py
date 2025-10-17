import os

def create_results_dir():
    root = "/Users/noahfrank/Desktop/Kolmogorov_flow_LPT/Results"
    os.makedirs(root, exist_ok=True)

    #parent_dir = os.path.dirname(os.getcwd())          # directory above current
    #root = os.path.join(parent_dir, "Results")         # add 'Results' subdir
    #os.makedirs(root, exist_ok=True)
    return root
