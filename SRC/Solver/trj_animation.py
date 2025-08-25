import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from SRC.utils import Specteral_Upsampling


def extract_state(snapshot, n_particles, NDOF):
    snapshot = np.asarray(snapshot)
    particles = snapshot[:n_particles * 4].real
    U = snapshot[n_particles * 4:].reshape((2, NDOF, NDOF))

    xp = particles[0::4]
    yp = particles[1::4]

    return xp, yp, U[0], U[1]

def animate_particles_and_flow(
    traj, L, n_particles, NDOF,
    interval=40, s=20, qskip=2,
    repeat=True, blit=True, dpi=120, ax=None,
    title="Particles + Velocity Field", skip=1
):
    traj = np.asarray(traj)
    T = traj.shape[0]

    # Domain extents
    if isinstance(L, (tuple, list, np.ndarray)):
        Lx, Ly = float(L[0]), float(L[1])
    else:
        Lx = Ly = float(L)

    # Grid for quiver (downsampled)
    x = np.linspace(0.0, Lx, NDOF, endpoint=False)
    y = np.linspace(0.0, Ly, NDOF, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="xy")
    Xq = X[::qskip, ::qskip]
    Yq = Y[::qskip, ::qskip]

    # Figure/Axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=dpi)
    else:
        fig = ax.figure

    # Initial data
    xp0, yp0, u0, v0 = extract_state(traj[0], n_particles, NDOF)

    # Artists
    scat = ax.scatter(xp0, yp0, c="red", s=s, marker="o")
    U0 = u0[::qskip, ::qskip]
    V0 = v0[::qskip, ::qskip]
    qv = ax.quiver(
        Xq, Yq, U0, V0,
        angles="xy", scale_units="xy", scale=5.0,  # fixed scale for visibility
        width=0.002
    )

    # Mark as animated for blitting backends
    scat.set_animated(True)
    qv.set_animated(True)

    ax.set_xlim(0.0, Lx)
    ax.set_ylim(0.0, Ly)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)

    # init_func ensures first frame is drawn correctly with blit=True
    def init():
        scat.set_offsets(np.column_stack([xp0, yp0]))
        qv.set_UVC(U0, V0)
        return (scat, qv)

    def update(frame_idx):
        snapshot = traj[frame_idx]
        xp, yp, u, v = extract_state(snapshot, n_particles, NDOF)
        scat.set_offsets(np.column_stack([xp, yp]))

        U = u[::qskip, ::qskip]
        V = v[::qskip, ::qskip]
        qv.set_UVC(U, V)

        return (scat, qv)

    frames = range(0, T, skip)
    anim = FuncAnimation(
        fig, update, frames=frames, init_func=init,
        interval=interval, blit=blit, repeat=repeat,
        cache_frame_data=False
    )

    # Keep a reference to avoid garbage collection if you show later
    fig._anim_ref = anim
    return fig, anim

# optional seaborn colormap helper (kept from your pattern)
def _get_cmap(name):
    try:
        import seaborn as sns
        return sns.color_palette(name, as_cmap=True)
    except Exception:
        return plt.get_cmap(name)

def animate_vorticity(vel_hat_trj, NDOF, Lx, Ly, omega_fn,
                      cmap="icefire", interval=40, repeat=True, blit=False,
                      dpi=120, skip=1, cbar=True, sym=True, clim=None, ax=None,
                      title=r"Vorticity $\omega_z$", target_res=512):
    """
    Animate vorticity over time from velocity Fourier fields.

    Parameters
    ----------
    vel_hat_trj : array_like, shape (T, 2*NDOF*NDOF)
        Time series of packed velocity-hat fields: [..., U_hat_x(:), U_hat_y(:)] per frame.
        (Or shape (T, 2, NDOF, NDOF) is also accepted.)
    NDOF : int
        Grid size in each direction (Ny = Nx = NDOF).
    Lx, Ly : float
        Domain lengths.
    skip : int
        Use every `skip`-th frame.

    Returns
    -------
    fig, anim : matplotlib Figure and FuncAnimation
    """
    r = int(round(target_res/NDOF))
    VHT = np.asarray(vel_hat_trj)
    T = VHT.shape[0]

    # reshape to (T, 2, Ny, Nx)
    if VHT.ndim == 2:
        VHT = VHT.reshape(T, 2, NDOF, NDOF//2+1)
    elif VHT.ndim == 4:
        assert VHT.shape[1:] == (2, NDOF, NDOF//2+1), "vel_hat_trj must be (T, 2, NDOF, NDOF)"
    else:
        raise ValueError("vel_hat_trj must be 2D (T, 2*NDOF*NDOF) or 4D (T, 2, NDOF, NDOF).")

    frames = list(range(0, T, skip))
    cmap_obj = _get_cmap(cmap)
    extent = [0.0, float(Lx), 0.0, float(Ly)]

    # Precompute vorticity limits for a fixed color scale (avoids flicker)
    if clim is None:
        # Sample all selected frames to get global min/max
        vmins = []
        vmaxs = []
        for ti in frames:
            U_hat_x = VHT[ti, 0]
            U_hat_y = VHT[ti, 1]
            #U_hat_x = np.fft.rfft2(spectral_upsample_from_hat2d_rfft(U_hat_x, r))
            #U_hat_y = np.fft.rfft2(spectral_upsample_from_hat2d_rfft(U_hat_y, r))
            omg = omega_fn(U_hat_x, U_hat_y)
            omg = Specteral_Upsampling.spectral_upsample_from_hat2d_rfft(np.fft.rfft2(omg), r)

            vmins.append(np.nanmin(omg))
            vmaxs.append(np.nanmax(omg))
        if sym:
            vabs = max(abs(min(vmins)), abs(max(vmaxs)))
            vmin, vmax = -vabs, vabs
        else:
            vmin, vmax = min(vmins), max(vmaxs)
    else:
        vmin, vmax = clim

    # Figure/Axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=dpi)
    else:
        fig = ax.figure

    # Initial image
    U_hat_x0 = VHT[frames[0], 0]
    U_hat_y0 = VHT[frames[0], 1]
    #U_hat_x0 = np.fft.rfft2(spectral_upsample_from_hat2d_rfft(U_hat_x0, r))
    #U_hat_y0 = np.fft.rfft2(spectral_upsample_from_hat2d_rfft(U_hat_y0, r))
    omega0 = omega_fn(U_hat_x0, U_hat_y0)
    omg = Specteral_Upsampling.spectral_upsample_from_hat2d_rfft(np.fft.rfft2(omg), r)

    im = ax.imshow(omega0, origin="lower", extent=extent, cmap=cmap_obj,
                   vmin=vmin, vmax=vmax, aspect="equal", interpolation="nearest")

    if cbar:
        fig.colorbar(im, ax=ax, label=r"$\omega_z$", fraction=0.035)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    if blit:
        im.set_animated(True)

    def init():
        im.set_data(omega0)
        return (im,)

    def update(ti):
        U_hat_x = VHT[ti, 0]
        U_hat_y = VHT[ti, 1]
       #U_hat_x = np.fft.rfft2(spectral_upsample_from_hat2d_rfft(U_hat_x, r))
       # U_hat_y = np.fft.rfft2(spectral_upsample_from_hat2d_rfft(U_hat_y, r))
        omg = omega_fn(U_hat_x, U_hat_y)
        omg = Specteral_Upsampling.spectral_upsample_from_hat2d_rfft(np.fft.rfft2(omg), r)
        im.set_data(omg)
        return (im,)

    anim = FuncAnimation(
        fig, update, frames=frames, init_func=init,
        interval=interval, blit=blit, repeat=repeat, cache_frame_data=False
    )

    fig._anim_ref = anim  # avoid GC
    return fig, anim