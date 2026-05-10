"""
Diagnostic plots for the ST-GNN multi-period AC-OPF forecaster.

Generates combined time-series + parity figures for automatically
selected buses:
    - PQ bus with highest total Pd sum
    - PQ bus with highest Pd variance
    - PV bus with highest total Pg sum (aggregated from generators)
    - PV bus with highest Pg variance (aggregated from generators)

Each figure contains 4 rows × 3 columns:
    col 0: time-series over the full test period
    col 1: time-series zoomed to a 2-week window around peak demand
    col 2: parity plot (y_true vs y_pred) over the full test period
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _aggregate_pg_to_bus(gen_tensor, gen_to_bus_index, N_bus):
    """
    Scatter-add generator Pg to bus level.

    Parameters
    ----------
    gen_tensor : Tensor [S, N_gen, n]
        Per-generator Pg values (predictions or targets).
    gen_to_bus_index : Tensor [N_gen]
        Maps each generator to its bus index (within a single graph).
    N_bus : int
        Number of buses per graph.

    Returns
    -------
    Tensor [S, N_bus, n]
        Bus-level aggregated Pg.
    """
    S, N_gen, n = gen_tensor.shape
    # Flatten to [S*n, N_gen] then scatter to [S*n, N_bus]
    flat = gen_tensor.permute(0, 2, 1).reshape(S * n, N_gen)  # [S*n, N_gen]

    result = torch.zeros(S * n, N_bus, dtype=flat.dtype, device=flat.device)
    idx = gen_to_bus_index.unsqueeze(0).expand(S * n, -1)  # [S*n, N_gen]
    result.scatter_add_(1, idx, flat)

    return result.view(S, n, N_bus).permute(0, 2, 1)  # [S, N_bus, n]


def _select_buses(all_tgt_bus_pd, all_tgt_pg_bus, mask_pq, mask_pv):
    """
    Select 4 buses based on sum and variance criteria.

    Parameters
    ----------
    all_tgt_bus_pd : ndarray [S, N_bus]
        Pd target at the chosen horizon step for all test samples.
    all_tgt_pg_bus : ndarray [S, N_bus]
        Pg target (bus-aggregated) at the chosen horizon step.
    mask_pq : ndarray [N_bus] bool
        PQ bus mask.
    mask_pv : ndarray [N_bus] bool
        PV bus mask.

    Returns
    -------
    list of (bus_idx, label, variable_name)
        4 entries describing the selected buses.
    """
    pq_indices = np.where(mask_pq)[0]
    pv_indices = np.where(mask_pv)[0]

    selections = []

    # --- PQ buses by Pd ---
    if len(pq_indices) > 0:
        pd_pq = all_tgt_bus_pd[:, pq_indices]  # [S, #PQ]
        sum_pd = pd_pq.sum(axis=0)
        var_pd = pd_pq.var(axis=0)

        best_sum_pq = pq_indices[np.argmax(sum_pd)]
        best_var_pq = pq_indices[np.argmax(var_pd)]

        selections.append((best_sum_pq, f"PQ Bus {best_sum_pq} (highest Pd sum)", "Pd (MW)"))
        selections.append((best_var_pq, f"PQ Bus {best_var_pq} (highest Pd var)", "Pd (MW)"))

    # --- PV buses by Pg ---
    if len(pv_indices) > 0:
        pg_pv = all_tgt_pg_bus[:, pv_indices]  # [S, #PV]
        sum_pg = pg_pv.sum(axis=0)
        var_pg = pg_pv.var(axis=0)

        best_sum_pv = pv_indices[np.argmax(sum_pg)]
        best_var_pv = pv_indices[np.argmax(var_pg)]

        selections.append((best_sum_pv, f"PV Bus {best_sum_pv} (highest Pg sum)", "Pg (MW)"))
        selections.append((best_var_pv, f"PV Bus {best_var_pv} (highest Pg var)", "Pg (MW)"))

    return selections


def _find_peak_2week_window(all_tgt_bus_pd):
    """
    Find a 2-week (336-hour) window centered on the hour with the
    highest total Pd across all buses.

    Parameters
    ----------
    all_tgt_bus_pd : ndarray [S, N_bus]
        Pd target at the chosen horizon for all test samples.

    Returns
    -------
    (start, end) : tuple of int
        Slice indices into the sample dimension.
    """
    total_pd_per_step = all_tgt_bus_pd.sum(axis=1)  # [S]
    peak_idx = int(np.argmax(total_pd_per_step))

    half_window = 336 // 2  # 168 hours each side
    S = len(total_pd_per_step)

    start = max(0, peak_idx - half_window)
    end = min(S, peak_idx + half_window)

    # If window clips at the boundary, shift
    if end - start < 336:
        if start == 0:
            end = min(S, 336)
        else:
            start = max(0, end - 336)

    return start, end


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_diagnostic_plots(
    all_pred_bus,       # [S, N_bus, n, 5]
    all_tgt_bus,        # [S, N_bus, n, 5]
    all_pred_gen,       # [S, N_gen, n, 1]
    all_tgt_gen,        # [S, N_gen, n, 1]
    bus_type_mask_pq,   # [N_bus] bool tensor
    bus_type_mask_pv,   # [N_bus] bool tensor
    gen_to_bus_index,   # [N_gen] int tensor
    N_bus,              # int
    forecast_horizon,   # int (n)
    dataset_name,       # str
    save_dir,           # str
):
    """
    Generate combined time-series + parity diagnostic figures.

    Creates one figure for t+1, and one for t+n (if n > 1).
    Each figure is a 4×3 grid (4 selected buses × {full, 2-week, parity}).
    """
    mask_pq = bus_type_mask_pq.numpy()
    mask_pv = bus_type_mask_pv.numpy()
    g2b = gen_to_bus_index.long()

    # Aggregate Pg to bus level: [S, N_bus, n]
    pred_pg_bus = _aggregate_pg_to_bus(all_pred_gen[..., 0], g2b, N_bus)
    tgt_pg_bus = _aggregate_pg_to_bus(all_tgt_gen[..., 0], g2b, N_bus)

    # Horizons to plot
    horizons = [0]  # t+1 (index 0)
    if forecast_horizon > 1:
        horizons.append(forecast_horizon - 1)  # t+n (last index)

    for h_idx in horizons:
        h_label = h_idx + 1  # human-readable (1-indexed)

        # Extract data at this horizon step
        pred_pd = all_pred_bus[:, :, h_idx, 0].numpy()   # [S, N_bus]  Pd
        tgt_pd = all_tgt_bus[:, :, h_idx, 0].numpy()
        pred_pg = pred_pg_bus[:, :, h_idx].numpy()         # [S, N_bus]  Pg (bus-agg)
        tgt_pg = tgt_pg_bus[:, :, h_idx].numpy()

        # Select buses (using targets at this horizon)
        selections = _select_buses(tgt_pd, tgt_pg, mask_pq, mask_pv)

        if not selections:
            print(f"[st_forecast_plots] No buses selected for {dataset_name} t+{h_label}, skipping.")
            continue

        n_rows = len(selections)
        # Find 2-week window (based on Pd at this horizon)
        win_start, win_end = _find_peak_2week_window(tgt_pd)

        # --- Create figure ---
        fig, axes = plt.subplots(n_rows, 3, figsize=(22, 5 * n_rows))
        if n_rows == 1:
            axes = axes[np.newaxis, :]  # ensure 2D

        fig.suptitle(
            f"{dataset_name} — Forecast Diagnostics (t+{h_label})",
            fontsize=16, fontweight="bold", y=1.01,
        )

        for row, (bus_idx, label, var_name) in enumerate(selections):
            # Choose Pd or Pg data based on variable
            if "Pd" in var_name:
                pred_full = pred_pd[:, bus_idx]
                tgt_full = tgt_pd[:, bus_idx]
            else:  # Pg
                pred_full = pred_pg[:, bus_idx]
                tgt_full = tgt_pg[:, bus_idx]

            # ── Col 0: Full test period time-series ──
            ax0 = axes[row, 0]
            ax0.plot(tgt_full, label="True", alpha=0.8, linewidth=0.8)
            ax0.plot(pred_full, label="Forecast", alpha=0.6, linewidth=0.8)
            ax0.set_xlabel("Test Timestep")
            ax0.set_ylabel(var_name)
            ax0.set_title(f"{label} — Full Period")
            ax0.legend(fontsize=8)
            ax0.grid(True, alpha=0.3)

            # ── Col 1: 2-week zoom ──
            ax1 = axes[row, 1]
            ax1.plot(
                range(win_start, win_end),
                tgt_full[win_start:win_end],
                label="True", alpha=0.8, linewidth=1.0,
            )
            ax1.plot(
                range(win_start, win_end),
                pred_full[win_start:win_end],
                label="Forecast", alpha=0.6, linewidth=1.0,
            )
            ax1.set_xlabel("Test Timestep")
            ax1.set_ylabel(var_name)
            ax1.set_title(f"{label} — 2-Week Zoom (peak demand)")
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3)

            # ── Col 2: Parity plot ──
            ax2 = axes[row, 2]
            ax2.scatter(tgt_full, pred_full, s=4, alpha=0.3, edgecolors="none")

            # Identity line y = x
            vmin = min(tgt_full.min(), pred_full.min())
            vmax = max(tgt_full.max(), pred_full.max())
            margin = 0.05 * (vmax - vmin) if vmax > vmin else 1.0
            ax2.plot(
                [vmin - margin, vmax + margin],
                [vmin - margin, vmax + margin],
                "k--", linewidth=1.0, alpha=0.7, label="y = x",
            )

            # Correlation coefficient
            if len(tgt_full) > 1:
                corr = np.corrcoef(tgt_full, pred_full)[0, 1]
                ax2.text(
                    0.05, 0.95, f"R = {corr:.4f}",
                    transform=ax2.transAxes, fontsize=10,
                    verticalalignment="top",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"),
                )

            ax2.set_xlabel(f"True {var_name}")
            ax2.set_ylabel(f"Predicted {var_name}")
            ax2.set_title(f"{label} — Parity")
            ax2.set_aspect("equal", adjustable="datalim")
            ax2.legend(fontsize=8, loc="lower right")
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(save_dir, f"{dataset_name}_forecast_diagnostics_t{h_label}.png")
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[st_forecast_plots] Saved: {plot_path}")
