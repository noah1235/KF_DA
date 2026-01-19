import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from SRC.utils import Specteral_Upsampling
import matplotlib.colors as colors
from SRC.plotting_utils import balanced_cmap

def plot_particles(xs, ys, L, ax=None, s=20):
    """
    Plot particle positions (black dots) from a packed state vector:
    z = [x1, y1, u1, v1, x2, y2, u2, v2, ..., xN, yN, uN, vN].

    Parameters
    ----------
    z : array_like, shape (4*N,)
        Packed particle state vector.
    L : float or tuple(float, float)
        Domain size. If float, uses (0, L) for both x and y.
        If tuple, interpreted as (Lx, Ly).
    ax : matplotlib.axes.Axes or None
        Axes to plot on. If None, creates a new figure.
    s : float
        Marker size for scatter points.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """

    # Domain extents
    if isinstance(L, (tuple, list, np.ndarray)):
        Lx, Ly = float(L[0]), float(L[1])
    else:
        Lx = Ly = float(L)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.scatter(xs, ys, c="black", s=s, marker="o")
    ax.set_xlim(0.0, Lx)
    ax.set_ylim(0.0, Ly)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Particle Positions")
    ax.set_aspect("equal", adjustable="box")

    return fig, ax

def plot_vorticity(omega, L=2*np.pi, cmap="icefire", ax=None, target_res=512):
    """
    Plot ω_z with imshow and a colorbar. Returns (fig, ax, im, omega).

    Parameters
    ----------
    omega : 2D array
        Vorticity field to plot.
    Lx, Ly : float
        Domain lengths used for the axes extents.
    cmap : str
        Seaborn perceptual colormap name (e.g., "icefire", "flare").
    ax : matplotlib.axes.Axes or None
        If provided, draw on this axes; otherwise create a new figure/axes.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    r = int(round(target_res/omega.shape[0]))
    if r > 1:
        omega_hat = np.fft.rfft2(omega)
        omega = Specteral_Upsampling.spectral_upsample_from_hat2d_rfft(omega_hat, r)
    # pick seaborn colormap
    #cmap_obj = sns.color_palette(cmap, as_cmap=True)

    norm = colors.TwoSlopeNorm(vmin=-10, vcenter=0.0, vmax=10)

    im = ax.imshow(
        omega,
        origin="lower",
        extent=[0, L, 0, L],
        cmap=balanced_cmap,
        norm=norm,
        aspect="equal",
    )

    fig.colorbar(im, ax=ax, shrink=0.8)

    #cbar = fig.colorbar(im, ax=ax, label=r"$\omega_z$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    return fig, ax, im, omega


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_div(div, Lx=None, Ly=None, ax=None,
             title=r"Divergence $\nabla\!\cdot\!\mathbf{u}$",
             cmap="coolwarm", symmetric=True, vlim=None,
             cbar=True, cbar_size="5%", cbar_pad=0.1,
             add_zero_contour=True):
    """
    Plot a divergence field with zero-centered colormap and a slim colorbar.

    Parameters
    ----------
    div : (Ny, Nx) array-like
        Divergence field in physical space.
    Lx, Ly : float or None
        Domain lengths for labeled axes. Defaults: Lx=Nx, Ly=Ny.
    ax : matplotlib Axes or None
        Axes to draw on; creates a new one if None.
    title : str
        Plot title.
    cmap : str
        Matplotlib colormap.
    symmetric : bool
        If True, use a zero-centered normalization (TwoSlopeNorm).
    vlim : None | float | (float, float)
        If symmetric=True and vlim=float, uses [-vlim, +vlim].
        If symmetric=False and vlim=(vmin, vmax), uses those.
        If None, computed from data (max abs for symmetric).
    cbar : bool
        Whether to draw a colorbar.
    cbar_size : str
        Width of the colorbar (e.g., "5%").
    cbar_pad : float
        Padding between axes and colorbar (in inches).
    add_zero_contour : bool
        Overlay a thin 0-contour for reference.

    Returns
    -------
    fig, ax, im, cb : (Figure, Axes, AxesImage, Colorbar or None)
    """
    div = np.asarray(div)
    Ny, Nx = div.shape
    if Lx is None: Lx = Nx
    if Ly is None: Ly = Ny

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Determine scaling
    if symmetric:
        if vlim is None:
            vmax = float(np.nanmax(np.abs(div))) or 1.0
        elif np.isscalar(vlim):
            vmax = float(vlim)
        else:
            raise ValueError("For symmetric=True, vlim should be None or a single float.")
        vmin = -vmax
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        im = ax.imshow(div, origin="lower", extent=[0, Lx, 0, Ly],
                       aspect="equal", cmap=cmap, norm=norm)
    else:
        if vlim is None:
            vmin = float(np.nanmin(div))
            vmax = float(np.nanmax(div))
        else:
            vmin, vmax = map(float, vlim)
        im = ax.imshow(div, origin="lower", extent=[0, Lx, 0, Ly],
                       aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)

    if add_zero_contour:
        x = np.linspace(0, Lx, Nx)
        y = np.linspace(0, Ly, Ny)
        ax.contour(x, y, div, levels=[0.0], colors="k", linewidths=0.6, alpha=0.6)

    cb = None
    if cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=cbar_size, pad=cbar_pad)
        cb = fig.colorbar(im, cax=cax, label=r"$\nabla\!\cdot\!\mathbf{u}$")

    return fig, ax, im, cb

def plot_D_vs_time(D_list, time,
                      figsize=(6,6)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(time, D_list, label="Normalized Dissipation")

    fig.tight_layout()
    return fig