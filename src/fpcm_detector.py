from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from typing import Dict, List, Tuple
import mne
from tqdm import tqdm 

# =============================================================================
# CONSTANTS
# =============================================================================
PEAK_HW_MS = 35.0
WAVE_HW_MS = 60.0
WAVE_POWER = 3.0
BCK_COEFF = 3.0
ERR_PEAK_TH = 0.3
ERR_WAVE_TH = 0.7
HIT_THRESHOLD = 4

SLOPES_HEIGHTS_RATIO = 0.3
SLOPE_TOLERANCE = 2
APEX_WAVE_RATIO1 = 2
APEX_WAVE_RATIO2 = 5

# --------------------------------------------------------------------
# 1.  Spline model 
# --------------------------------------------------------------------
def _build_model(
    peak_hw_s: int,
    wave_hw_s: int,
    wave_power: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Construct the mixed‑spline model.

    Parameters
    ----------
    peak_hw_s : int
        Half‑width of the sharp peak in samples.
    wave_hw_s : int
        Half‑width of the slow wave in samples.
    wave_power : int
        Exponent of the polynomial that models the slow wave.

    Returns
    -------
    piBp : ndarray, shape (6, T)
        Pseudo‑inverse of the projected spline model.
    Bp : ndarray, shape (T, 6)
        Projected spline model.
    H : ndarray, shape (6,)
        Row vector used to read out the spike apex (peak intercept).
    T : int
        Total template window size.
    """
    # Time axes    
    t1_l = np.arange(-peak_hw_s, 1, 1,  dtype=float)
    t1_r = np.arange(0, peak_hw_s + 1, 1, dtype=float)
    t2   = np.arange(-wave_hw_s, wave_hw_s + 1, 1, dtype=float)

    # Segment design matrices
    A1 = np.c_[t1_l, np.ones_like(t1_l)]                    # left slope
    A2 = np.c_[t1_r, np.ones_like(t1_r)]                    # right slope
    A3 = np.c_[np.abs(t2) ** wave_power, np.ones_like(t2)]  # slow wave
    
    # dA3= np.c_[wave_power*abs(t2)**(wave_power-1), np.zeros((len(t2), 1))]


    # --- Continuity  constraint ----------
    v12 = np.hstack([
        A1[-1,:],             
        -A2[0,:],        
        0., 0.              
    ]).astype(float)        
    v12 /= np.linalg.norm(v12)

    v23 = np.hstack([
        0., 0.,             
        A2[-1,:],              
        -A3[0,:],             
    ]).astype(float)     
    v23 /= np.linalg.norm(v23)

    # d23 = np.array([0., 0., 1., 0.,
    #                 wave_power * np.abs(t2[0]) ** (wave_power - 1),
    #                 0.], dtype=float)
    
    # d23 /= np.linalg.norm(d23)

    V   = np.c_[v12, v23]         
    

    A1 = np.delete(A1, A1.shape[0]-1, 0)
    A2 = np.delete(A2, A2.shape[0]-1, 0)

    # --- Full block‑diag basis B (T × 6) -----------------------------
    B1 = np.vstack([A1, np.zeros((A2.shape[0], 2)),
                        np.zeros((A3.shape[0], 2))])
    B2 = np.vstack([np.zeros((A1.shape[0], 2)),
                        A2, np.zeros((A3.shape[0], 2))])
    B3 = np.vstack([np.zeros((A1.shape[0], 2)),
                        np.zeros((A2.shape[0], 2)), A3])

    B = np.hstack([B1, B2, B3])
    T = B.shape[0]
    P  = np.eye(6) - V @ np.linalg.pinv(V)
    Bp = np.matmul(P, B.T).T
            
    piBp = np.matmul(np.linalg.pinv(np.matmul(Bp.T, Bp)), Bp.T)

    H = np.hstack([A1[-1, :], 0, 0, 0, 0]) 
    return piBp, Bp, H, T

def _convolve_filters(X: np.ndarray, piBp: np.ndarray) -> np.ndarray:
    """
    Apply the spline model to the data.

    Parameters
    ----------
    X : ndarray, shape (n_times,)
        1D input signal (single channel).
    piBp : ndarray, shape (6, T)
        Pseudo‑inverse of the spline model.

    Returns
    -------
    C : ndarray, shape (6, n_times)
        Spline coefficients at each time point.
    - C[0]: slope of the left (pre-peak) ramp
    - C[1]: intercept of the left (pre-peak) ramp
    - C[2]: slope of the right (post-peak) ramp
    - C[3]: intercept of the right (post-peak) ramp
    - C[4]: quadratic coefficient (or |t|^p term) of the slow wave
    - C[5]: intercept of the slow wave segment
    """
    flipped_model = np.fliplr(piBp)        
    C = np.vstack([np.convolve(f, X, mode='same') for f in flipped_model])
    return C

# --------------------------------------------------------------------
# 2.  Logical predicates
# --------------------------------------------------------------------
def _apply_predicates(
    C: np.ndarray,
    HC: np.ndarray,
    bkg_left: np.ndarray,
    bkg_right: np.ndarray,
    flat_peak: np.ndarray,
    bkg_coeff: float = BCK_COEFF,
    slope_tol: float = SLOPE_TOLERANCE,
    apex_wave_ratio1: float = APEX_WAVE_RATIO1,
    apex_wave_ratio2: float = APEX_WAVE_RATIO2,    
) -> np.ndarray:
    """
    Apply a set of logical rules to identify spike-like waveforms.

    Parameters
    ----------
    C : ndarray, shape (6, n_times)
        Spline coefficients.
    HC : ndarray, shape (n_times,)
        Estimated spike apex from the spline fit.
    bkg_left, bkg_right : ndarray
        Background estimates on each side of candidate.
    flat_peak : ndarray, bool
        Flatness criterion (computed from spline fit).
    bkg_coeff : float
        Amplitude ratio threshold relative to background.
    slope_tol : float
        Allowed asymmetry between left and right slopes.

    Returns
    -------
    mask : ndarray, shape (n_times,)
        Boolean mask of valid spike candidates.
    """
    neg_peak   = (C[0,:] < 0) & (C[2,:] > 0)
    pos_peak   = (C[0,:] > 0) & (C[2,:] < 0)

    spike_below = (HC < 0) & (C[4,:] < 0)  
    wave_above  = (C[5,:]>0)

    spike_above = (HC > 0) & (C[4,:] > 0)
    wave_below  = (C[5,:]<0) 

    ratio_ok_neg = (np.abs(HC) > apex_wave_ratio1*np.abs(C[5,:])) & (np.abs(HC) < apex_wave_ratio2*np.abs(C[5,:]))
    ratio_ok_pos = ratio_ok_neg.copy()

    slopes_equal = np.abs(np.abs(C[0,:]/C[2,:]) - 1) < slope_tol
    above_bkg    = (np.abs(HC) > bkg_coeff*bkg_left) & (np.abs(HC) > bkg_coeff*bkg_right)

    ind_neg = neg_peak & spike_below & wave_above & ratio_ok_neg & slopes_equal & above_bkg & flat_peak
    ind_pos = pos_peak & spike_above & wave_below & ratio_ok_pos & slopes_equal & above_bkg & flat_peak
    return ind_neg | ind_pos

# --------------------------------------------------------------------
# 3.  Detection
# --------------------------------------------------------------------
def detect_spikes_fpcm(
    raw: mne.io.BaseRaw,
    *,
    peak_hw_ms: float = PEAK_HW_MS,
    wave_hw_ms: float = WAVE_HW_MS,
    wave_power: int   = WAVE_POWER,
    bkg_coeff: float  = BCK_COEFF,
    err_peak_th: float = ERR_PEAK_TH,
    err_wave_th: float = ERR_WAVE_TH,
    hit_threshold: int = HIT_THRESHOLD,
    slopes_heights_ratio: float = SLOPES_HEIGHTS_RATIO,
) -> Dict[str, object]:
    """
    Run FPCM spike detection on an MNE‑Raw object.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Input signal. Must be filtered and restricted to the desired channel type.
    peak_hw_ms, wave_hw_ms : float
        Half-widths of the sharp peak and slow wave in milliseconds.
    wave_power : int
        Exponent of the polynomial wave segment (default: 3).
    bkg_coeff : float
        Background multiplier threshold for amplitude comparison.
    err_peak_th, err_wave_th : float
        Maximum relative fitting error allowed for peak and wave segments.
    hit_threshold : int
        Minimum number of channels that must agree at a given time point.

    Returns
    -------
    results : dict
        {
            'peaks_samples'  : ndarray of spike latencies (samples),
            'events'         : array ready for mne.Epochs construction,
            'hits'           : bool array of shape (n_channels, n_peaks),
            'coeffs'         : list of spline coefficient arrays (per channel),
            'synth_splines'  : dicts of time-domain reconstructions,
            'bp_matrix'      : spline basis matrix,
            'window_len'     : length of template window in samples,
            'peak_hw_s'      : half-width of the peak in samples,
            'wave_hw_s'      : half-width of the wave in samples,
        }
    """
    sfreq     = raw.info['sfreq']
    peak_hw_s = int(round(peak_hw_ms / 1000 * sfreq))
    wave_hw_s = int(round(wave_hw_ms / 1000 * sfreq))

    piBp, Bp, H, T = _build_model(peak_hw_s, wave_hw_s, wave_power)

    data = raw.get_data()           
    n_ch, n_times = data.shape

    err_peak  = np.ones((n_ch, n_times))
    err_wave  = np.ones((n_ch, n_times))
    hit_mask  = np.zeros((n_ch, n_times), bool)
    coeffs    = []
    synth_all = []

    box = np.ones(T) / T

    for ch in tqdm(range(n_ch)):
        x = data[ch,:]
        C = _convolve_filters(x, piBp)     # (6, n_times)
        coeffs.append(C)
        hc       = H @ C
        synthdic = {}

        a0 = Bp[0,:]            @ C
        b0 = Bp[peak_hw_s,:]    @ C
        c0 = Bp[2*peak_hw_s,:]  @ C
        flat = np.abs((a0 - b0) / (c0 - b0)) - 1 < slopes_heights_ratio

        inst_power = np.convolve(box, x**2, mode='same')
        bkg_left  = np.sqrt(np.roll(inst_power, shift=T))
        bkg_left[:T//2] = 0
        bkg_right = np.sqrt(np.roll(inst_power, shift=-T))
        bkg_right[-T//2:] = 0
        
        morph_ok = _apply_predicates(C, hc, bkg_left, bkg_right, flat, bkg_coeff)
        
        cand_idx = np.where(morph_ok)[0]
        for t in cand_idx:
            if T % 2 == 0:
                st, en = t - T // 2, t + T // 2
            else:
                st, en = t - T // 2, t + T // 2 + 1

            synth = Bp @ C[:, t]            
            synthdic[t] = synth

            if st >= 0 and en <= len(x):
                rng = slice(st, en)
                err_peak[ch, t] = np.linalg.norm(x[rng][:2*peak_hw_s] - synth[:2*peak_hw_s]) \
                                  / np.linalg.norm(x[rng][:2*peak_hw_s])
                err_wave[ch, t] = np.linalg.norm(x[rng][2*peak_hw_s:] - synth[2*peak_hw_s:]) \
                                  / np.linalg.norm(x[rng][2*peak_hw_s:])
            else:
                err_peak[ch, t] = 1
                err_wave[ch, t] = 1

        synth_all.append(synthdic)
        good = (err_peak[ch] < err_peak_th) & (err_wave[ch] < err_wave_th)
        hit_mask[ch] = morph_ok & good

    acc_hits = hit_mask.sum(0)
    spike_centres = np.where(acc_hits >= hit_threshold)[0]

    min_separation_samp = int(round(0.040 * sfreq))
    suppressed = []
    last_t = -np.inf
    for t in sorted(spike_centres):
        if t - last_t >= min_separation_samp:
            suppressed.append(t)
            last_t = t
    
    unique_peaks = np.array(suppressed)
    hits_selection = hit_mask[:,unique_peaks]
    if T % 2 == 0:
        unique_peaks = unique_peaks-T//2+peak_hw_s
    else:
        unique_peaks = unique_peaks-T//2+peak_hw_s

    events = np.column_stack([
        unique_peaks + raw.first_samp,
        np.zeros(unique_peaks.size, int),      
        np.ones (unique_peaks.size,  int)      
    ])

    return dict(
        peaks_samples = unique_peaks,
        events        = events,
        hits          = hits_selection,
        coeffs        = coeffs,
        synth_splines = synth_all,
        bp_matrix     = Bp,
        window_len    = T,
        peak_hw_s     = peak_hw_s,
        wave_hw_s     = wave_hw_s
    )
