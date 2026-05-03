import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_particles_and_flow(
    u, v, xp, yp,
    L, NDOF,
    interval=40, s=20, qskip=2,
    repeat=True, blit=True, dpi=120, ax=None,
    title="Particles + Velocity Field", skip=1
):
    u_np  = np.asarray(u)
    v_np  = np.asarray(v)
    xp_np = np.asarray(xp)
    yp_np = np.asarray(yp)

    T = u_np.shape[0]

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

    # Initial frame data
    xp0 = xp_np[0]
    yp0 = yp_np[0]
    u0  = u_np[0]
    v0  = v_np[0]

    # Artists
    scat = ax.scatter(xp0, yp0, c="red", s=s, marker="o")

    U0 = u0[::qskip, ::qskip]
    V0 = v0[::qskip, ::qskip]

    # Let matplotlib autoscale unless you have a reason to fix it
    qv = ax.quiver(
        Xq, Yq, U0, V0,
        angles="xy", scale_units="xy", scale=10,
        width=0.002
    )

    ax.set_xlim(0.0, Lx)
    ax.set_ylim(0.0, Ly)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)

    # init_func (for blit)
    def init():
        scat.set_offsets(np.column_stack([xp0, yp0]))
        qv.set_UVC(U0, V0)
        return (scat, qv)

    def update(frame_idx):
        # Pull frame data without shadowing outer variables
        xpf = xp_np[frame_idx]
        ypf = yp_np[frame_idx]
        uf  = u_np[frame_idx]
        vf  = v_np[frame_idx]

        scat.set_offsets(np.column_stack([xpf, ypf]))

        U = uf[::qskip, ::qskip]
        V = vf[::qskip, ::qskip]
        qv.set_UVC(U, V)

        return (scat, qv)

    frames = range(0, T, skip)

    # If quiver + blit gives trouble on your backend, set blit=False.
    anim = FuncAnimation(
        fig, update, frames=frames, init_func=init,
        interval=interval, blit=blit, repeat=repeat,
        cache_frame_data=False
    )

    fig._anim_ref = anim
    return fig, anim
