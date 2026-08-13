"""
Canonical AC constraint-violation metrics for the ML-surrogate and E2E arms.

This module is the graphkit-side twin of Thesis_Repo/exp1/generate_metrics
``compute_feasibility_metrics``. It produces the *identical* canonical quintet
per constraint type so the three thesis arms (exact / surrogate / E2E) report
comparable feasibility numbers under the same metric names.

Per constraint type ``ct`` in {vm, pg, qg, thermal, angle}:
    - ``{ct} viol. count``          number of violating elements
    - ``{ct} viol. mean``          mean magnitude over ALL elements (one-sided)
    - ``{ct} viol. max``           max magnitude
    - ``{ct} viol. rate``          violating elements / total
    - ``{ct} scenario viol. rate`` scenarios with >=1 violation / all scenarios
plus ``fully feasible scenario viol. rate`` (any-type violation per scenario).

Violation magnitude is one-sided and two-sided per limit:
    two-sided (vm, pg, qg, angle):  max(0, x - upper) + max(0, lower - x)
    upper-only (thermal):           max(0, |S| - rate_a)   with |S| = max(S_from, S_to)

Units (physical, matching compare.py): vm p.u., pg MW, qg MVAr, thermal MVA,
angle deg. Values are accumulated as Python scalars so the accumulator is
device-agnostic, memory-flat, and batch-size invariant (exact global stats,
not a cross-batch mean).
"""

import torch
import torch.nn.functional as F
from torch_scatter import scatter_add

from gridfm_graphkit.datasets.globals import (
    VM_OUT, VA_OUT, PG_OUT, QG_OUT,
    MIN_VM_H, MAX_VM_H, MIN_QG_H, MAX_QG_H,
    MIN_PG, MAX_PG, ANG_MIN, ANG_MAX, RATE_A,
)


CT_UNITS = {"vm": "p.u.", "pg": "MW", "qg": "MVAr", "thermal": "MVA", "angle": "deg"}


def _two_sided(x, lower, upper):
    """Element-wise one-sided, two-sided violation magnitude: max(0,x-u)+max(0,l-x)."""
    return torch.clamp(x - upper, min=0.0) + torch.clamp(lower - x, min=0.0)


class ConstraintViolationAccumulator:
    """Accumulates sufficient statistics across test batches, then finalizes the
    canonical quintet. Exact global stats (not a Lightning cross-batch mean).

    Usage:
        acc = ConstraintViolationAccumulator()
        for batch: acc.update(per_type, num_scen)   # per_type from per_type_from_opf_batch
        rows = acc.finalize_rows()                   # list of {Metric, Value, Unit}
    """

    def __init__(self, tol: float = 1e-5):
        self.tol = tol
        self._ct = {
            ct: {"sum": 0.0, "n": 0, "n_viol": 0, "max": 0.0, "n_scen_viol": 0}
            for ct in CT_UNITS
        }
        self.n_scen = 0
        self.n_fully_infeasible = 0

    def update(self, per_type: dict, num_scen: int):
        """Ingest one batch.

        Args:
            per_type: {ct: (mag, scen)} where ``mag`` is the per-element violation
                magnitude (already restricted to valid elements, incl. zeros for
                non-violating ones) and ``scen`` is the per-element scenario id in
                [0, num_scen). Missing/empty cts are skipped.
            num_scen: number of scenarios (graphs) in this batch.
        """
        self.n_scen += int(num_scen)
        fully_any = torch.zeros(int(num_scen), dtype=torch.bool)
        for ct, (mag, scen) in per_type.items():
            if mag is None or mag.numel() == 0:
                continue
            mag = mag.detach()
            scen = scen.detach().to(torch.long)
            viol = mag > self.tol
            s = self._ct[ct]
            s["sum"] += float(mag.sum().item())
            s["n"] += int(mag.numel())
            s["n_viol"] += int(viol.sum().item())
            s["max"] = max(s["max"], float(mag.max().item()))
            # per-scenario "any violation of this ct" -> count violating scenarios
            per_scen = scatter_add(viol.to(torch.long), scen, dim=0, dim_size=int(num_scen))
            any_scen = per_scen > 0
            s["n_scen_viol"] += int(any_scen.sum().item())
            fully_any |= any_scen.cpu()
        self.n_fully_infeasible += int(fully_any.sum().item())

    def reduce_across_ranks(self, device=None):
        """Sum/max the accumulated statistics across DDP ranks (no-op if not
        distributed). Call once before ``finalize_rows`` under multi-GPU test."""
        import torch.distributed as dist
        if not (dist.is_available() and dist.is_initialized()):
            return
        for ct in CT_UNITS:
            s = self._ct[ct]
            t = torch.tensor([s["sum"], s["n"], s["n_viol"], s["n_scen_viol"]],
                             dtype=torch.float64, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            s["sum"], s["n"], s["n_viol"], s["n_scen_viol"] = (
                float(t[0].item()), int(t[1].item()), int(t[2].item()), int(t[3].item()))
            m = torch.tensor([s["max"]], dtype=torch.float64, device=device)
            dist.all_reduce(m, op=dist.ReduceOp.MAX)
            s["max"] = float(m[0].item())
        g = torch.tensor([self.n_scen, self.n_fully_infeasible],
                         dtype=torch.float64, device=device)
        dist.all_reduce(g, op=dist.ReduceOp.SUM)
        self.n_scen, self.n_fully_infeasible = int(g[0].item()), int(g[1].item())

    def finalize_rows(self, prefix: str = "") -> list:
        """Return canonical rows as [{"Metric","Value","Unit"}, ...]."""
        n_scen = self.n_scen or 1
        rows = []
        for ct, unit in CT_UNITS.items():
            s = self._ct[ct]
            if s["n"] == 0:
                continue
            rows.extend([
                {"Metric": f"{prefix}{ct} viol. count", "Value": float(s["n_viol"]), "Unit": "count"},
                {"Metric": f"{prefix}{ct} viol. mean", "Value": s["sum"] / s["n"], "Unit": unit},
                {"Metric": f"{prefix}{ct} viol. max", "Value": s["max"], "Unit": unit},
                {"Metric": f"{prefix}{ct} viol. rate", "Value": s["n_viol"] / s["n"], "Unit": "frac"},
                {"Metric": f"{prefix}{ct} scenario viol. rate", "Value": s["n_scen_viol"] / n_scen, "Unit": "frac"},
            ])
        rows.append({
            "Metric": f"{prefix}fully feasible scenario viol. rate",
            "Value": self.n_fully_infeasible / n_scen,
            "Unit": "frac",
        })
        return rows


def per_bus_type_mae(output: dict, target: dict, batch) -> dict:
    """MAE (L1) mirroring opf_task.py per bus type MSE 

    Called from each arm's own hook / call site (surrogate: _after_opf_metrics;
    E2E: inline after _compute_opf_metrics) 
    """
    mask_PQ = batch.mask_dict["PQ"]
    mask_PV = batch.mask_dict["PV"]
    mask_REF = batch.mask_dict["REF"]
    out = {}
    for t_name, mask in (("PQ", mask_PQ), ("PV", mask_PV), ("REF", mask_REF)):
        mae = F.l1_loss(output["bus"][mask], target["bus"][mask], reduction="none").mean(dim=0)
        out[f"MAE {t_name} nodes - VM"] = mae[VM_OUT]
        out[f"MAE {t_name} nodes - VA"] = mae[VA_OUT]
        out[f"MAE {t_name} nodes - PG"] = mae[PG_OUT]
        out[f"MAE {t_name} nodes - QG"] = mae[QG_OUT]
    out["MAE PG"] = F.l1_loss(output["gen"], target["gen"], reduction="none").mean()
    return out


def bus_type_counts(batch) -> dict:
    return {t: int(batch.mask_dict[t].sum()) for t in ("PQ", "PV", "REF")}


def canonical_scalar_rows(
    m: dict,
    pd_mae: float,
    pd_rmse: float,
    qd_mae: float,
    qd_rmse: float,
    counts: dict,
) -> list:
    """shared by the surrogate and E2E arms
    
     Metric names matching 2-step arms "compare.py"
    
    ``m`` is one dataset's grouped callback_metrics: base "MSE {type} nodes - X"
    keys (co-owned opf_task.py) + "MAE {type} nodes - X" keys (per_bus_type_mae,
    self-logged by the caller). 
    - Va is converted rad->deg to match compare.py

    ``counts`` is per-bus-type node counts ({PQ,PV,REF} -> int, from
    bus_type_counts): the per-type MSE/MAE means are collapsed weighted by these
    so every BUS counts equally (per-bus-equal), 

    Pd/Qd MAE/RMSE are computed differently per arm (surrogate: forecast vs true
    realized load; E2E: its own forecast vs true), so they're passed in rather
    than read from ``m``.
    """
    import math

    def g(k):
        v = m.get(k)
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    def agg(prefix, feat, scale=1.0, sqrt=False):
        # Count-weighted collapse of per-type means. For RMSE (sqrt=True) the
        # per-type value is MSE, so sum(n*MSE)/sum(n) is the exact pooled MSE 
        num = den = 0.0
        for t in ("PQ", "PV", "REF"):
            v, w = g(f"{prefix} {t} nodes - {feat}"), counts.get(t, 0)
            if v == v and v >= 0.0 and w > 0:  # skip NaN / empty type
                num += w * v
                den += w
        if den == 0:
            return float("nan")
        pooled = num / den
        return (pooled ** 0.5 if sqrt else pooled) * scale

    mse_pg, mae_pg = g("MSE PG"), g("MAE PG")
    va_scale = 180.0 / math.pi

    return [
        {"Metric": "Mean Optimality Gap", "Value": g("Opt gap"), "Unit": "%"},
        {"Metric": "Residual P (MAE)", "Value": g("Active Power Loss"), "Unit": "MW"},
        {"Metric": "Residual Q (MAE)", "Value": g("Reactive Power Loss"), "Unit": "MVAr"},
        {"Metric": "MAE Pd", "Value": pd_mae, "Unit": "MW"},
        {"Metric": "RMSE Pd", "Value": pd_rmse, "Unit": "MW"},
        {"Metric": "MAE Qd", "Value": qd_mae, "Unit": "MVAr"},
        {"Metric": "RMSE Qd", "Value": qd_rmse, "Unit": "MVAr"},
        {"Metric": "Vm MAE", "Value": agg("MAE", "VM"), "Unit": "p.u."},
        {"Metric": "Vm RMSE", "Value": agg("MSE", "VM", sqrt=True), "Unit": "p.u."},
        {"Metric": "Va MAE", "Value": agg("MAE", "VA", scale=va_scale), "Unit": "deg"},
        {"Metric": "Va RMSE", "Value": agg("MSE", "VA", scale=va_scale, sqrt=True), "Unit": "deg"},
        {"Metric": "Pg bus MAE", "Value": agg("MAE", "PG"), "Unit": "MW"},
        {"Metric": "Pg bus RMSE", "Value": agg("MSE", "PG", sqrt=True), "Unit": "MW"},
        {"Metric": "Qg MAE", "Value": agg("MAE", "QG"), "Unit": "MVAr"},
        {"Metric": "Qg RMSE", "Value": agg("MSE", "QG", sqrt=True), "Unit": "MVAr"},
        {"Metric": "Generator Pg MAE", "Value": mae_pg, "Unit": "MW"},
        {"Metric": "Generator Pg RMSE", "Value": mse_pg ** 0.5 if mse_pg == mse_pg else float("nan"), "Unit": "MW"},
    ]


def per_type_from_opf_batch(output, batch, branch_flow_layer, tol: float = 1e-5) -> tuple:
    """Extract per-element violation magnitudes + scenario ids from one OPF batch.

    Mirrors compare.py's ``compute_feasibility_metrics`` element construction, in
    physical units. ``output`` is OPF-format: output["bus"]=[Vm,Va,Pg_agg,Qg]
    (denormalized), output["gen"]=[Pg]. ``batch`` provides limits + topology.

    Returns:
        (per_type, num_scen) where per_type = {ct: (mag, scen)}.
    """
    bus_x = batch.x_dict["bus"]
    gen_x = batch.x_dict["gen"]
    bus_scen = batch.batch_dict["bus"].to(torch.long)
    gen_scen = batch.batch_dict["gen"].to(torch.long)
    num_scen = int(bus_scen.max().item()) + 1 if bus_scen.numel() else 0

    edge_index = batch.edge_index_dict[("bus", "connects", "bus")]
    edge_attr = batch.edge_attr_dict[("bus", "connects", "bus")]

    per_type = {}

    # --- Vm (all buses, two-sided, p.u.) ---
    vm = output["bus"][:, VM_OUT]
    vm_mag = _two_sided(vm, bus_x[:, MIN_VM_H], bus_x[:, MAX_VM_H])
    per_type["vm"] = (vm_mag, bus_scen)

    # --- Pg (all generators, two-sided, MW) ---
    pg = output["gen"][:, 0]
    pg_mag = _two_sided(pg, gen_x[:, MIN_PG], gen_x[:, MAX_PG])
    per_type["pg"] = (pg_mag, gen_scen)

    # --- Qg (generator buses only = PV|REF, two-sided, MVAr) ---
    gen_bus_mask = batch.mask_dict["PV"] | batch.mask_dict["REF"]
    qg = output["bus"][:, QG_OUT]
    qg_mag_all = _two_sided(qg, bus_x[:, MIN_QG_H], bus_x[:, MAX_QG_H])
    per_type["qg"] = (qg_mag_all[gen_bus_mask], bus_scen[gen_bus_mask])

    # --- Branch flows (physical MVA) via shared physics layer ---
    Pft, Qft = branch_flow_layer(output["bus"], edge_index, edge_attr)
    S = torch.sqrt(Pft ** 2 + Qft ** 2)
    rate_a = edge_attr[:, RATE_A]
    num_edges = edge_index.size(1)
    half = num_edges // 2  # edges are bidirectional: [0:half] forward, [half:] reverse

    # --- Thermal: max(S_from, S_to) per line vs rate_a (upper-only, MVA) ---
    s_from = S[:half]
    s_to = S[half:]
    s_line = torch.maximum(s_from, s_to)
    rate_line = rate_a[:half]
    valid_rate = (rate_line > 0.0) & torch.isfinite(rate_line)
    from_fwd = edge_index[0, :half]
    line_scen = bus_scen[from_fwd]
    thermal_mag = torch.clamp(s_line - rate_line, min=0.0)
    per_type["thermal"] = (thermal_mag[valid_rate], line_scen[valid_rate])

    # --- Angle diff (signed va_from - va_to, degrees, two-sided, checked against the pglib limits (usually 30 degrees)) ---
    va_deg = output["bus"][:, VA_OUT] * (180.0 / torch.pi)
    ang_diff = va_deg[edge_index[0, :half]] - va_deg[edge_index[1, :half]]
    ang_min = edge_attr[:half, ANG_MIN]
    ang_max = edge_attr[:half, ANG_MAX]
    valid_ang = (
        torch.isfinite(ang_min) & torch.isfinite(ang_max)
        & (ang_min > -359.0) & (ang_max < 359.0)
    )
    ang_mag = _two_sided(ang_diff, ang_min, ang_max)
    per_type["angle"] = (ang_mag[valid_ang], line_scen[valid_ang])

    return per_type, num_scen
