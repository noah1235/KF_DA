import matplotlib.pyplot as plt

def save_svg(mpl, fig, path):
    mpl.rcParams['svg.fonttype'] = 'none'
    mpl.rcParams['svg.hashsalt'] = ''
    plt.rcParams['font.family'] = 'Arial'


    # optional: tighter bounding box & transparent background
    fig.savefig(
    path,
    bbox_inches="tight",
    pad_inches=0.02,
    transparent=True,
    metadata={"Title": "Nice vector plot", "Creator": "matplotlib"}
    )