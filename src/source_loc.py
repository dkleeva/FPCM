"""
source_loc.py
--------------
Dipole localization tools for spike-related source modeling.

This module provides an interface for performing source localization of detected
epileptic spikes using RAP-MUSIC algorithm, visualizing modeled topographies,
and plotting dipole positions on anatomical MRI templates.

Authors : Daria Kleeva
Email   : dkleeva@gmail.com
"""

import numpy as np
import mne
from typing import Tuple, Callable, List, Dict, Optional
import math
import matplotlib.pyplot as plt
import nibabel as nib
from nibabel.affines import apply_affine

def fit_spike_dipoles(
    epochs: mne.Epochs,
    fwd:    mne.Forward,
    *,
    t_window: Tuple[float, float] = (-0.1, 0.1),   # seconds (rel. to 0 ms)
    thr_music: float = 0.8,
    thr_svd:   float = 0.95,
    verbose: bool = True,
) -> Dict[str, List]:
    """
    Apply RAP-MUSIC to localize dipoles for each spike epoch.

    Parameters
    ----------
    epochs : mne.Epochs
        Spike-centered epochs (0 ms = apex).
    fwd : mne.Forward
        Forward model (must match epochs in space and channels).
    t_window : tuple of float
        Time window (in seconds) relative to spike apex to crop before fitting.
    thr_music : float
        Minimal subspace correlation value to accept a dipole.
    thr_svd : float
        Variance threshold for selecting the number of SVD components.
    verbose : bool
        Print dipole count and subcorr values for each spike.

    Returns
    -------
    fit_res : dict
        Keys:
            - 'coords': list of (N_dipoles × 3) arrays with dipole coordinates.
            - 'vals'  : list of subcorr values for each dipole.
            - 'index' : list of source space indices for each spike.
    """
    
    gain = fwd["sol"]["data"]   
    src_rr = fwd["source_rr"]         
    coords_all = []          
    val_all    = []          
    idx_all=[]
    for ep_i in range(len(epochs)):
        ep = epochs[ep_i]
        epo = ep.copy().crop(*t_window)      
        data = epo.get_data()[0]            
        v, idx = rap_music_scan(data, gain, thr_music, thr_svd)
        xyz = fwd["source_rr"][idx]         
        coords_all.append(xyz)
        val_all.append(v)
        idx_all.append(idx)
        print(f"Spike {ep_i:3d}: found {len(idx)} dipole(s)  corr={np.array(v)}")

    coords_all = [np.atleast_2d(c) for c in coords_all]
    return dict(coords=coords_all, vals=val_all, index=idx_all)

def plot_modeled_topos(
    epochs:   mne.Epochs,
    fit_res:  Dict[str, List],
    fwd:      mne.Forward,
    *,
    n_cols: int = 4,
    cmap: str = "RdBu_r",
    vmax: float | None = None,
):
    """
    Show topography of each spike and its RAP-MUSIC-based model on a grid.

    Each spike is shown as a pair:
        - true topography at apex (left),
        - modeled topography from RAP-MUSIC dipoles (right).

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs of detected spikes.
    fit_res : dict
        Output of `fit_spike_dipoles` with dipole indices.
    fwd : mne.Forward
        Forward model matching the epochs.
    n_cols : int
        Number of spike-pairs per row (each spike = 2 columns).
    cmap : str
        Colormap for topomaps.
    vmax : float | None
        Color range for plots (symmetric ±vmax). If None, computed per spike.

    Returns
    -------
    fig : matplotlib.Figure
    """
    gain = fwd["sol"]["data"]               
    idx_all = fit_res["index"]               


    valid = [i for i, arr in enumerate(idx_all) if len(arr) > 0]
    n_spk  = len(valid)
    if n_spk == 0:
        raise RuntimeError("No spikes with located dipoles to plot.")

    n_rows = math.ceil(n_spk / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols * 2,
        figsize=(3.4 * n_cols * 2, 3 * n_rows),
        squeeze=False,
    )

    t0 = epochs.time_as_index(0.)[0]

    for k, spk_i in enumerate(valid):
        r, c_pair = divmod(k, n_cols)
        ax_true   = axes[r][2 * c_pair]
        ax_fit    = axes[r][2 * c_pair + 1]

        ep     = epochs[spk_i]
        topo_t = ep.get_data()[0, :, t0]   
        idx    = np.array(idx_all[spk_i]).astype(int)
        G_sel  = np.hstack([gain[:, 3*i : 3*i + 3] for i in idx])
        amps, *_ = np.linalg.lstsq(G_sel, topo_t, rcond=None)
        topo_f = G_sel @ amps


        resid = topo_t - topo_f
        gof = 1.0 - np.sum(resid**2) / np.sum((topo_t - topo_t.mean())**2)

        vlim = dict(vmax=vmax) if vmax is not None else dict()

        mne.viz.plot_topomap(
            topo_t, epochs.info, axes=ax_true, cmap=cmap, show=False,
            names=None, **vlim)
        ax_true.set_title(f"S{spk_i} true", fontsize=9)

        mne.viz.plot_topomap(
            topo_f, epochs.info, axes=ax_fit, cmap=cmap, show=False,
            names=None, **vlim)
        ax_fit.set_title(f"fit  GOF={gof:.2f}", fontsize=9)


    for ax in axes.flat[2*n_spk:]:
        ax.axis("off")

    plt.tight_layout()
    return fig

def plot_dipoles_2d(
    fit_res: Dict[str, List],
    *,
    trans: str | mne.transforms.Transform,
    subject: str,
    subjects_dir: str,
    color: str = "crimson",
    title: Optional[str] = "All RAP‑MUSIC dipoles (2‑D outlines)",
    fig_kwargs: Optional[dict] = None,
) -> plt.Figure:
    """
    Plot all RAP-MUSIC dipoles in 2D brain projections using MNE's dipole viewer.

    Parameters
    ----------
    fit_res : dict
        Output of `fit_spike_dipoles`.
    trans : str | mne.Transform
        MRI→head transform used in forward model.
    subject : str
        Freesurfer subject name (e.g. 'fsaverage').
    subjects_dir : str
        Path to SUBJECTS_DIR.
    color : str
        Dipole marker color (e.g. 'crimson').
    title : str | None
        Title for the figure. Use None to suppress.
    fig_kwargs : dict | None
        Extra kwargs passed to MNE’s `dipole.plot_locations()`.

    Returns
    -------
        fig : matplotlib.Figure
"""

    xyz_all = np.vstack(fit_res["coords"]) if fit_res["coords"] else np.empty((0, 3))
    if xyz_all.size == 0:
        raise RuntimeError("No dipoles to plot.")

    amp_all = (np.hstack(fit_res["vals"])
               if fit_res["vals"] and len(fit_res["vals"][0]) > 0
               else np.ones(xyz_all.shape[0]))

    dip = mne.Dipole(
        times      = np.zeros(len(xyz_all)),
        pos        = xyz_all,  
        amplitude  = amp_all,
        ori        = np.zeros((len(xyz_all), 3)),
        gof        = np.zeros(len(xyz_all))
    )


    kw = dict(mode="outlines", color=color, show_all=False)
    if fig_kwargs:
        kw.update(fig_kwargs)

    fig = dip.plot_locations(
        trans=trans, subject=subject, subjects_dir=subjects_dir, **kw
    )
    if title is not None:
        fig.suptitle(title, y=0.95)
    return fig

def rap_music_scan(spike, Gain, thresh, thr_svd):
    """
    Perform RAP-MUSIC iterative dipole scanning for a single spike.

    Parameters
    ----------
    spike : np.ndarray
        MEG data matrix (n_channels × timepoints) of a single spike.
    Gain : np.ndarray
        Forward model (n_channels × 3·N_sources).
    thresh : float
        Minimum correlation to accept a dipole.
    thr_svd : float
        Variance threshold to select signal subspace.

    Returns
    -------
    Valmax : list of float
        Subspace correlation values of accepted dipoles.
    Indmax : list of int
        Indices of located sources (0-based).
    """
    G2, G2d0, Nsites = g3_to_g2(Gain)
    
    Ns, Nsrc2 = G2.shape
    Nsrc = Nsrc2 // 2

    Valmax = []
    Indmax = []


    U, S, Vh = np.linalg.svd(spike, full_matrices=False)
    h = np.cumsum(S) / np.sum(S)
    n = np.where(h >= thr_svd)[0][0] + 1

    corr = music_scan(G2, U[:, :n])
    valmax, indmax = np.max(corr), np.argmax(corr)

    while valmax > thresh:
        Valmax.append(valmax)
        Indmax.append(indmax)

        A = Gain[:, 3*indmax : 3*(indmax + 1)] 
        P = np.eye(Ns) - A @ np.linalg.inv(A.T @ A) @ A.T
        spike_proj = P @ spike
        G_proj = P @ Gain
        Gain = G_proj

        G2 = np.zeros((Ns, 2 * Nsrc))
        for i in range(Nsrc):
            g = G_proj[:, 3 * i:3 * (i + 1)]
            u, sv, vh = np.linalg.svd(g, full_matrices=False)
            gt = g @ vh.T[:, :2]
            G2[:, 2 * i:2 * (i + 1)] = gt / np.sqrt(np.sum(gt**2, axis=0, keepdims=True))

        U, S, Vh = np.linalg.svd(spike_proj, full_matrices=False)
        h = np.cumsum(S) / np.sum(S)
        n = np.where(h >= thr_svd)[0][0] + 1

        corr = music_scan(G2, U[:, :n])
        valmax, indmax = np.max(corr), np.argmax(corr)

    return Valmax, Indmax

def music_scan(G2, U):
    """
    One step of RAP-MUSIC scan

    Parameters
    ----------
    G2 : np.ndarray
        Forward model in tangential components (n_channels × 2·N_sources).
    U : np.ndarray
        Orthonormal signal subspace (n_channels × n_components).

    Returns
    -------
    corr : np.ndarray
        Subspace correlation values (length N_sources).
    """
    _, Nsrc2 = G2.shape
    Nsrc = Nsrc2 // 2


    tmp = U.T @ G2


    c11c22 = np.sum(tmp**2, axis=0)
    tmp1 = tmp[:, ::2]  
    tmp2 = tmp[:, 1::2] 
    c12 = np.sum(tmp1 * tmp2, axis=0)

    tr = c11c22[::2] + c11c22[1::2]
    d = c11c22[::2] * c11c22[1::2] - c12**2


    l1 = np.sqrt(0.5 * (tr + np.sqrt(tr**2 - 4 * d)))
    l2 = np.sqrt(tr - l1**2)


    corr = np.maximum(l1, l2)

    return corr

def g3_to_g2(G3):
    """
    Compute the forward operator for MEG without the radial component

    Parameters
    ----------
    G3 : np.ndarray
        3D forward model (n_channels × 3·N_sources).

    Returns
    -------
    G2d : np.ndarray
        Normalized tangential forward model (n_channels × 2·N_sources).
    G2d0 : np.ndarray
        Unnormalized tangential forward model.
    Nsites : int
        Number of source locations in the cortical model.
    """

    G_pure = G3.copy()  
    Nch, _ = G_pure.shape
    Nsites = G3[:, 0::3].shape[1]
    G2d = np.zeros((Nch, Nsites * 2))
    G2d0 = np.zeros((Nch, Nsites * 2))

    for i in range(Nsites):
        g = np.column_stack((
            G_pure[:, 3 * i],
            G_pure[:, 3 * i + 1],
            G_pure[:, 3 * i + 2]
        ))
        u, sv, vh = np.linalg.svd(g, full_matrices=False)
        gt = g @ vh.T[:, :2]
        norm_factors = np.sqrt(np.sum(gt**2, axis=0, keepdims=True))
        G2d[:, 2 * i:2 * i + 2] = gt / norm_factors
        G2d0[:, 2 * i:2 * i + 2] = gt

    return G2d, G2d0, Nsites


def _mm_from_m(coords_m):
    return np.asarray(coords_m, dtype=float) * 1000.0

def _coords_surfaceRASmm_to_vox(coords_mm, vox2ras_tkr):
    ras2vox_tkr = np.linalg.inv(vox2ras_tkr)
    homo = np.c_[coords_mm, np.ones((coords_mm.shape[0], 1))]
    vox = (ras2vox_tkr @ homo.T).T[:, :3]
    return np.round(vox).astype(int)

def _paint_sphere(volume, center_xyz, radius_vox):
    x0, y0, z0 = center_xyz
    nx, ny, nz = volume.shape
    x_min, x_max = max(0, x0 - radius_vox), min(nx - 1, x0 + radius_vox)
    y_min, y_max = max(0, y0 - radius_vox), min(ny - 1, y0 + radius_vox)
    z_min, z_max = max(0, z0 - radius_vox), min(nz - 1, z0 + radius_vox)

    xs = np.arange(x_min, x_max + 1)
    ys = np.arange(y_min, y_max + 1)
    zs = np.arange(z_min, z_max + 1)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    mask = (X - x0)**2 + (Y - y0)**2 + (Z - z0)**2 <= (radius_vox ** 2)
    volume[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1][mask] = volume.max()

def save_dipoles_as_nifti_with_white_dots(
    fit_res,
    t1_mgz_path,
    out_path,
    fwd=None,    
    trans=None,
    radius_vox=3
):
    """
    Put dipole markers into a copy of the T1 volume and save to NIfTI/MGZ.
    """
    if not os.path.exists(t1_mgz_path):
        raise FileNotFoundError(f"Cannot find MRI file: {t1_mgz_path}")

    # ---- Load MRI ----
    img = nib.load(t1_mgz_path)
    data = img.get_fdata()
    hdr = img.header
    vox2ras_tkr = hdr.get_vox2ras_tkr()
    ras2vox_tkr = np.linalg.inv(vox2ras_tkr)

    # ---- Collect dipole coordinates ----
    rr_list = [np.atleast_2d(c) for c in fit_res.get("coords", [])]
    if not rr_list:
        raise ValueError("fit_res['coords'] is empty — nothing to plot.")
    rr = np.vstack(rr_list)  # in meters

    # ---- Detect coordinate frame ----
    frame = None
    if fwd is not None:
        frame = fwd.get('coord_frame', None)

    # ---- Convert to MRI ----
    if frame == mne.io.constants.FIFF.FIFFV_COORD_HEAD and trans is not None:
        t = mne.read_trans(trans) if isinstance(trans, str) else trans
        rr = mne.transforms.apply_trans(t['trans'], rr)
        rr_mm = _mm_from_m(rr)
    else:
        rr_mm = _mm_from_m(rr)

    # ---- Convert to voxel ----
    homo = np.c_[rr_mm, np.ones((rr_mm.shape[0], 1))]
    vox = (ras2vox_tkr @ homo.T).T[:, :3]
    vox = np.round(vox).astype(int)

    # ---- Paint spheres ----
    out_vol = data.copy()
    vox_unique = np.unique(vox, axis=0)
    for (x, y, z) in vox_unique:
        if (0 <= x < out_vol.shape[0] and
            0 <= y < out_vol.shape[1] and
            0 <= z < out_vol.shape[2]):
            _paint_sphere(out_vol, (x, y, z), radius_vox=radius_vox)

    # ---- Save the file ----
    out_img = img.__class__(out_vol, img.affine, header=hdr)
    nib.save(out_img, out_path)
    print(f"Saved dipoles to: {out_path}")
    return out_path
