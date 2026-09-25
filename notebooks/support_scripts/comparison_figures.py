"""Three-way component and spectrum pages for the comparison notebook."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

if __package__:
    from .comparison_lib import FS, BAND, lock_frequencies
else:
    from comparison_lib import FS, BAND, lock_frequencies

COLORS = {"real": "#1f4e79", "retrained": "#c81e3c", "official": "#7851a9"}


def fit_components(values, frequencies, fit_window=None, joint=True):
    y = np.asarray(values, float)
    if fit_window is not None:
        y = y[:min(len(y), int(fit_window))]
    if not frequencies:
        return []
    t = np.arange(len(y))/FS
    def fit(fs):
        columns = [v for f in fs for v in (np.cos(2*np.pi*f*t), np.sin(2*np.pi*f*t))]
        matrix = np.stack(columns+[np.ones_like(t)], axis=1)
        beta = np.linalg.lstsq(matrix, y, rcond=None)[0]
        condition = float(np.linalg.cond(matrix))
        return [(float(np.hypot(beta[2*i], beta[2*i+1])),
                 float(np.arctan2(beta[2*i+1], beta[2*i])), condition) for i in range(len(fs))]
    return fit(frequencies) if joint else [fit([f])[0] for f in frequencies]


def _save(fig, path, pdf=False, stub=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if stub:
        fig.text(.5, .5, "STUB: NOT A MEASUREMENT", fontsize=34, rotation=25,
                 color="red", alpha=.2, ha="center")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    if pdf:
        fig.savefig(path.with_suffix('.pdf'), bbox_inches="tight")
    plt.close(fig)
    return path


def components_page(groups, path, title, fit_real=None, fit_pred=64, joint=True,
                    pdf=False, stub=False, planted=None, component_window="signal",
                    component_cycles=4, amp_scale="real", match_time_scale=True):
    """Groups explicitly carry their own geometry, selection and absolute time origin."""
    import pandas as pd
    n = max(1, max(len(g["frequencies"]) for g in groups))
    fig, axes = plt.subplots(n+1, len(groups), figsize=(5.3*len(groups), 2.1*(n+1)+1.1),
                             squeeze=False, layout="constrained")
    fits = [fit_components(g['values'], g['frequencies'], fit_real if g['side']=='real' else fit_pred,
                           joint=joint) for g in groups]
    real_max = max([a for a, _, _ in fits[0]]+[1e-9])
    all_max = max([a for fs in fits for a, _, _ in fs]+[real_max])
    rows = []
    start = min(g['t0'] for g in groups)/FS*1000
    end = max(g['t0']+len(g['values']) for g in groups)/FS*1000
    for col, (g, params) in enumerate(zip(groups, fits)):
        p, s = g['P'], g['S']
        color = COLORS[g['side']]
        values = np.asarray(g['values'])
        t = (np.arange(len(values))+g['t0'])/FS*1000
        ax = axes[0, col]
        ax.plot(t, values, color=color, lw=1)
        ax.set_title(f"{g['label']}\nP={p}, S={s}; {len(values)} samples", fontsize=10)
        ax.set_xlabel("time [ms]")
        ax.set_ylabel("signal amplitude")
        if match_time_scale:
            ax.set_xlim(start, end)
        for i in range(n):
            ax = axes[i+1, col]
            if i >= len(params):
                ax.set_axis_off()
                if i == 0:
                    ax.text(.5, .5, "No selected components", ha='center', transform=ax.transAxes)
                continue
            f = g['frequencies'][i]
            amplitude, phase, cond = params[i]
            if component_window == "cycles":
                fine = np.linspace(0, component_cycles/f, 600)
            else:
                fine = np.linspace(0, len(values)/FS, 600)
            xx = (fine+g['t0']/FS)*1000
            ax.plot(xx, amplitude*np.cos(2*np.pi*f*fine-phase), color=color, lw=1.1)
            ax.axhline(0, color='.8', lw=.5)
            lim = 1.15*(all_max if amp_scale == "real" else max([a for a, _, _ in params]+[1e-9]))
            ax.set_ylim(-lim, lim)
            if match_time_scale and component_window == "signal":
                ax.set_xlim(start, end)
            locked = min(abs(lock_frequencies(p, s)-f)) <= 1.
            note = " [lock +/-1 Hz]" if locked else ""
            if planted is not None and abs(f-planted) <= 1:
                note += " [injected +/-1 Hz]"
            if cond > 1e3:
                note += " [ill-conditioned fit]"
            window = fit_real if g['side']=='real' else fit_pred
            cycles = f*min(len(values), window or len(values))/FS
            if cycles < 1.5:
                note += f" [{cycles:.1f} cycles fitted]"
            ax.set_title(f"{f:.2f} Hz | {f*p/FS:.3g} cpp / {f*s/FS:.3g} cps\n"
                         f"A={amplitude:.3g}{note}", fontsize=8)
            ax.set_xlabel("time [ms]", fontsize=8)
            ax.set_ylabel("amplitude", fontsize=8)
            ax.tick_params(labelsize=8)
            rows.append(dict(side=g['side'], model=g['label'], P=p, S=s, frequency_hz=f,
                             amplitude=amplitude, phase=phase, fit_condition=cond,
                             spectrum_mag=g.get('selection', {}).get('magnitude_by_frequency', {}).get(f),
                             spectral_floor=g.get('selection', {}).get('floor')))
    fig.suptitle(title+"\nEach prediction uses its own selected spectral peaks", fontsize=12)
    return _save(fig, path, pdf, stub), pd.DataFrame(rows)


def spectra_page(groups, path, title, drawn=None, planted=None, pdf=False, stub=False, bin_hz=1.):
    fig, axes = plt.subplots(len(groups), 1, figsize=(12, 3.9*len(groups)), layout="constrained")
    for ax, g in zip(np.atleast_1d(axes), groups):
        p, s = g['P'], g['S']
        x = np.asarray(g['values'], float)
        w = np.hanning(len(x))
        nfft = max(int(round(FS/bin_hz)), len(x))
        fr = np.fft.rfftfreq(nfft, 1/FS)
        mag = abs(np.fft.rfft((x-x.mean())*w, nfft))*2/w.sum()
        cpp = fr*p/FS
        color = COLORS[g['side']]
        ax.plot(cpp, mag, color=color, lw=1)
        for f in lock_frequencies(p, s):
            ax.axvline(f*p/FS, color='.4', lw=.6, ls='--', alpha=.6)
        if drawn is not None and g['side']=='real':
            for f in drawn:
                ax.axvline(f*p/FS, color='#26883d', lw=.8)
        if planted is not None:
            ax.axvline(planted*p/FS, color='#bc7c00', lw=1.4, label='injected tone')
        for f in g.get('spectrum_peaks', g['frequencies']):
            j = np.argmin(abs(fr-f))
            ax.plot(fr[j]*p/FS, mag[j], 'v', color='black', ms=4)
            ax.annotate(f"{f*p/FS:.3g}", (f*p/FS, mag[j]), xytext=(0, 6),
                        textcoords='offset points', ha='center', fontsize=7)
        if 'selection' in g:
            ax.axhline(g['selection']['floor'], color=color, ls=':', lw=1,
                       label=f"spectral cut: M >= {g['selection']['floor']:.3g}")
        ax.set_xlim(0, p/2)
        ax.set_ylim(0, max(float(mag.max())*1.22, 1e-12))
        ax.set_xlabel('frequency [cpp]')
        ax.set_ylabel('Hann magnitude')
        sec = ax.secondary_xaxis('top', functions=(lambda v, p=p, s=s: v*s/p,
                                                    lambda v, p=p, s=s: v*p/s))
        sec.set_xlabel('frequency [cps]')
        ax.set_title(f"{g['label']} | P={p}, S={s} | {len(x)} samples | native spacing {FS/len(x):g} Hz",
                     fontsize=10, pad=42)
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Line2D([0], [0], color='.4', ls='--', label='geometry locks'))
        ax.legend(handles=handles, fontsize=8, loc='best')
        ax.axvspan(0, BAND[0]*p/FS, color='.94', zorder=-1)
        ax.axvspan(BAND[1]*p/FS, p/2, color='.94', zorder=-1)
    fig.suptitle(title+f"\nRequested grid {bin_hz:g} Hz; zero padding does not increase spectral resolution", fontsize=12)
    return _save(fig, path, pdf, stub)
