from gridfm_graphkit.io.registries import NORMALIZERS_REGISTRY
import os
import torch
from abc import ABC, abstractmethod
from typing import Any, Optional, List
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
from gridfm_graphkit.datasets.globals import (
    # Bus feature indices
    PD_H,
    QD_H,
    QG_H,
    VM_H,
    VA_H,
    MIN_QG_H,
    MAX_QG_H,
    GS,
    BS,
    VN_KV,
    # Output feature indices
    PG_OUT,
    QG_OUT,
    PG_OUT_GEN,
    # Generator feature indices
    PG_H,
    MIN_PG,
    MAX_PG,
    C0_H,
    C1_H,
    C2_H,
    # Edge feature indices
    P_E,
    Q_E,
    YFF_TT_R,
    YFT_TF_I,
    ANG_MIN,
    ANG_MAX,
    RATE_A,
)


class Normalizer(ABC):
    """
    Abstract base class for all normalization strategies.
    """

    # Subclasses should set this to "fit_on_train" or "fit_on_dataset"
    fit_strategy: str = "fit_on_train"

    @abstractmethod
    def fit(self, data_path: str, scenario_ids: List[int]) -> dict:
        """
        Fit normalization parameters from raw data on disk.

        Args:
            data_path: Path to the raw data directory containing parquet files.
            scenario_ids: List of scenario IDs to use for fitting.

        Returns:
            Dictionary of computed parameters.
        """

    @abstractmethod
    def fit_from_dict(self, params: dict):
        """
        Set parameters from a precomputed dictionary.

        Args:
            params: Dictionary of parameters.
        """

    @abstractmethod
    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Normalize the input data.

        Args:
            data: Input tensor.

        Returns:
            Normalized tensor.
        """

    @abstractmethod
    def inverse_transform(self, normalized_data: torch.Tensor) -> torch.Tensor:
        """
        Undo normalization.

        Args:
            normalized_data: Normalized tensor.

        Returns:
            Original tensor.
        """

    @abstractmethod
    def get_stats(self) -> dict:
        """
        Return the stored normalization statistics for logging/inspection.
        """


@NORMALIZERS_REGISTRY.register("HeteroDataMVANormalizer")
class HeteroDataMVANormalizer(Normalizer):
    """
    In power systems, a suitable normalization strategy must preserve the physical properties of
    the system. A known method is the conversion to the per-unit (p.u.) system, which expresses
    electrical quantities such as voltage, current, power, and impedance as fractions of predefined
    base values. These base values are usually chosen based on system parameters, such as rated
    voltage. The per-unit conversion ensures that power system equations remain scale-invariant,
    preserving fundamental physical relationships.
    """

    fit_strategy = "fit_on_train"

    def __init__(self, args):
        """
        Args:
            args (NestedNamespace): Parameters

        Attributes:
            baseMVA (float): baseMVA found in casefile. From ``args.data.baseMVA``.
            task_name (str): Task name for format-aware denormalization.
        """
        self.baseMVA_orig = getattr(args.data, "baseMVA", 100)
        self.baseMVA = None
        self.task_name = getattr(args.task, "task_name", "OPF")

    def to(self, device):
        pass

    def fit(self, data_path: str, scenario_ids: List[int]) -> dict:
        """
        Fit normalization parameters by loading raw parquet data from disk.

        Args:
            data_path: Path to the raw data directory containing bus_data.parquet and gen_data.parquet.
            scenario_ids: List of scenario IDs to use for fitting.

        Returns:
            Dictionary of computed parameters.
        """
        bus_data = pd.read_parquet(os.path.join(data_path, "bus_data.parquet"))
        gen_data = pd.read_parquet(os.path.join(data_path, "gen_data.parquet"))
        
        assert bus_data.scenario.min() == 0 and bus_data.scenario.max() == len(bus_data.scenario.unique()) - 1

        bus_data = bus_data[bus_data["scenario"].isin(scenario_ids)]
        gen_data = gen_data[gen_data["scenario"].isin(scenario_ids)]

        if self.baseMVA is None:
            pd_values = bus_data["Pd"]
            qd_values = bus_data["Qd"]
            pg_values = gen_data["p_mw"]
            qg_values = bus_data["Qg"]

            non_zero_values = pd.concat(
                [
                    pd_values[pd_values != 0],
                    qd_values[qd_values != 0],
                    pg_values[pg_values != 0],
                    qg_values[qg_values != 0],
                ],
            )

            self.baseMVA = np.percentile(non_zero_values, 95)
        self.vn_kv_max = float(bus_data["vn_kv"].max())

        return {
            "baseMVA_orig": torch.tensor(self.baseMVA_orig, dtype=torch.float),
            "baseMVA": torch.tensor(self.baseMVA, dtype=torch.float),
            "vn_kv_max": torch.tensor(self.vn_kv_max, dtype=torch.float),
        }

    def fit_from_dict(self, params: dict):
        # Base MVA
        self.baseMVA = params.get("baseMVA").item()
        self.baseMVA_orig = params.get("baseMVA_orig").item()

        # vn_kv
        self.vn_kv_max = params.get("vn_kv_max").item()

    def transform(self, data: HeteroData):
        if self.baseMVA is None or self.baseMVA == 0:
            raise ValueError("BaseMVA not properly set")

        # --- Bus input normalization --- PD, QD, QG, MIN_QG, MAX_QG, VA, GS, BS, VN_KV (9)
        data.x_dict["bus"][:, PD_H] /= self.baseMVA
        data.x_dict["bus"][:, QD_H] /= self.baseMVA
        data.x_dict["bus"][:, QG_H] /= self.baseMVA
        data.x_dict["bus"][:, MIN_QG_H] /= self.baseMVA
        data.x_dict["bus"][:, MAX_QG_H] /= self.baseMVA
        data.x_dict["bus"][:, VA_H] *= torch.pi / 180.0
        data.x_dict["bus"][:, GS] *= self.baseMVA_orig / self.baseMVA
        data.x_dict["bus"][:, BS] *= self.baseMVA_orig / self.baseMVA
        data.x_dict["bus"][:, VN_KV] /= self.vn_kv_max

        # --- Bus label normalization --- PD, QD, QG, VA (4)
        data.y_dict["bus"][:, PD_H] /= self.baseMVA
        data.y_dict["bus"][:, QD_H] /= self.baseMVA
        data.y_dict["bus"][:, QG_H] /= self.baseMVA
        data.y_dict["bus"][:, VA_H] *= torch.pi / 180.0
        
        # --- Generator input normalization --- PG, MIN_PG, MAX_PG, C0, C1, C2 (6)
        data.x_dict["gen"][:, PG_H] /= self.baseMVA
        data.x_dict["gen"][:, MIN_PG] /= self.baseMVA
        data.x_dict["gen"][:, MAX_PG] /= self.baseMVA
        data.x_dict["gen"][:, C0_H] = torch.sign(
            data.x_dict["gen"][:, C0_H],
        ) * torch.log1p(torch.abs(data.x_dict["gen"][:, C0_H]))
        data.x_dict["gen"][:, C1_H] = torch.sign(
            data.x_dict["gen"][:, C1_H],
        ) * torch.log1p(torch.abs(data.x_dict["gen"][:, C1_H]))
        data.x_dict["gen"][:, C2_H] = torch.sign(
            data.x_dict["gen"][:, C2_H],
        ) * torch.log1p(torch.abs(data.x_dict["gen"][:, C2_H]))

        # --- Generator label normalization --- PG (1)
        data.y_dict["gen"][:, PG_H] /= self.baseMVA

        # --- Edge input normalization --- P_E, Q_E , Ys, ANG_MIN, ANG_MAX, RATE_A
        data.edge_attr_dict[("bus", "connects", "bus")][:, P_E] /= self.baseMVA
        data.edge_attr_dict[("bus", "connects", "bus")][:, Q_E] /= self.baseMVA
        data.edge_attr_dict[("bus", "connects", "bus")][:, YFF_TT_R : YFT_TF_I + 1] *= (
            self.baseMVA_orig / self.baseMVA
        )
        data.edge_attr_dict[("bus", "connects", "bus")][:, ANG_MIN] *= torch.pi / 180.0
        data.edge_attr_dict[("bus", "connects", "bus")][:, ANG_MAX] *= torch.pi / 180.0
        data.edge_attr_dict[("bus", "connects", "bus")][:, RATE_A] /= self.baseMVA
        data.baseMVA = self.baseMVA
        data.is_normalized = True

    def inverse_transform(self, data: HeteroData):
        if self.baseMVA is None or self.baseMVA == 0:
            raise ValueError("BaseMVA not properly set")

        if not data.is_normalized.all():
            raise ValueError("Attempting to denormalize data which is not normalized")

        if (data.baseMVA != self.baseMVA).any():
            raise ValueError(
                f"Normalizer baseMVA was {self.baseMVA} but Data object baseMVA is {data.baseMVA}",
            )

        # -------- BUS INPUT INVERSE NORMALIZATION --------
        # NOTE: VA (bus input & label) are intentionally kept in
        # radians after inverse_transform -- the physics layers (ComputeBranchFlow,
        # ComputeNodeResiduals, etc.) expect radians.
        #
        # WARNING: GS, BS, and edge admittances (Y) are NOT restored to their
        # original casefile per-unit values. The transform scales them by
        # (baseMVA_orig / baseMVA), but the inverse multiplies by baseMVA
        # (not baseMVA / baseMVA_orig), yielding physical SI units
        # (original * baseMVA_orig). This is intentional for the physics layers.
        data.x_dict["bus"][:, PD_H] *= self.baseMVA
        data.x_dict["bus"][:, QD_H] *= self.baseMVA
        data.x_dict["bus"][:, QG_H] *= self.baseMVA
        data.x_dict["bus"][:, MIN_QG_H] *= self.baseMVA
        data.x_dict["bus"][:, MAX_QG_H] *= self.baseMVA
        data.x_dict["bus"][:, GS] *= self.baseMVA  # -> physical units (not original p.u.)
        data.x_dict["bus"][:, BS] *= self.baseMVA  # -> physical units (not original p.u.)
        data.x_dict["bus"][:, VN_KV] *= self.vn_kv_max

        # -------- BUS LABEL INVERSE NORMALIZATION --------
        data.y_dict["bus"][:, PD_H] *= self.baseMVA
        data.y_dict["bus"][:, QD_H] *= self.baseMVA
        data.y_dict["bus"][:, QG_H] *= self.baseMVA

        # -------- GENERATOR INPUT INVERSE NORMALIZATION --------
        data.x_dict["gen"][:, PG_H] *= self.baseMVA
        data.x_dict["gen"][:, MIN_PG] *= self.baseMVA
        data.x_dict["gen"][:, MAX_PG] *= self.baseMVA
        data.x_dict["gen"][:, C0_H] = torch.sign(data.x_dict["gen"][:, C0_H]) * (
            torch.exp(torch.abs(data.x_dict["gen"][:, C0_H])) - 1
        )
        data.x_dict["gen"][:, C1_H] = torch.sign(data.x_dict["gen"][:, C1_H]) * (
            torch.exp(torch.abs(data.x_dict["gen"][:, C1_H])) - 1
        )
        data.x_dict["gen"][:, C2_H] = torch.sign(data.x_dict["gen"][:, C2_H]) * (
            torch.exp(torch.abs(data.x_dict["gen"][:, C2_H])) - 1
        )

        # -------- GENERATOR LABEL INVERSE NORMALIZATION --------
        data.y_dict["gen"][:, PG_H] *= self.baseMVA 

        # -------- EDGE INPUT INVERSE NORMALIZATION --------
        data.edge_attr_dict[("bus", "connects", "bus")][:, P_E] *= self.baseMVA
        data.edge_attr_dict[("bus", "connects", "bus")][:, Q_E] *= self.baseMVA
        data.edge_attr_dict[("bus", "connects", "bus")][:, YFF_TT_R : YFT_TF_I + 1] *= (
            self.baseMVA  # -> physical units (not original p.u.), see WARNING above
        )

        data.edge_attr_dict[("bus", "connects", "bus")][:, ANG_MIN] *= 180.0 / torch.pi
        data.edge_attr_dict[("bus", "connects", "bus")][:, ANG_MAX] *= 180.0 / torch.pi

        data.edge_attr_dict[("bus", "connects", "bus")][:, RATE_A] *= self.baseMVA
        data.is_normalized = False

    def inverse_output(self, output, batch):
        bus_output = output["bus"]
        gen_output = output["gen"]
        
        if self.task_name in ("ForecastOPF", "ST_ForecastOPF"):
            # ForecastOPF format: [Pd, Qd, Qg, Vm, Va] (5 features)
            bus_output[:, PD_H] *= self.baseMVA  # Pd at index 0
            bus_output[:, QD_H] *= self.baseMVA  # Qd at index 1
            bus_output[:, QG_H] *= self.baseMVA  # Qg at index 2
            # Vm (index 3) and Va (index 4) remain as-is (p.u. and radians)
            gen_output[:, PG_OUT_GEN] *= self.baseMVA  # Pg
        else:
            # OPF format: [Vm, Va, Pg, Qg] (4 features)
            bus_output[:, PG_OUT] *= self.baseMVA  # Pg at index 2
            bus_output[:, QG_OUT] *= self.baseMVA  # Qg at index 3
            # Vm (index 0) and Va (index 1) remain as-is (p.u. and radians)
            gen_output[:, PG_OUT_GEN] *= self.baseMVA  # Pg

    def get_stats(self) -> dict:
        return {
            "baseMVA_orig": torch.tensor(self.baseMVA_orig, dtype=torch.float),
            "baseMVA": torch.tensor(self.baseMVA, dtype=torch.float),
            "vn_kv_max": torch.tensor(self.vn_kv_max, dtype=torch.float),
        }


@NORMALIZERS_REGISTRY.register("HeteroDataPerSampleMVANormalizer")
class HeteroDataPerSampleMVANormalizer(Normalizer):
    """
    Per-sample MVA normalizer: each scenario (sample) gets its own baseMVA and vn_kv_max,
    computed as the 95th percentile of Pd, Qd, Pg, Qg for that scenario. Same per-unit
    formulas as HeteroDataMVANormalizer, but applied with per-scenario scales so that
    batched data with different scenarios is normalized correctly.
    """

    fit_strategy = "fit_on_dataset"

    def __init__(self, args):
        self.baseMVA_orig = getattr(args.data, "baseMVA", 100)  # casefile base MVA (for GS/BS scaling)
        self._baseMVA_lookup = None  # tensor indexed by scenario_id
        self._vn_kv_max_lookup = None
        self._scenario_ids = None  # scenario ids that were fitted (for save/load)
        self.task_name = getattr(args.task, "task_name", "OPF")

    def to(self, device):
        pass

    def fit(self, data_path: str, scenario_ids: List[int]) -> dict:
        """
        Compute per-scenario baseMVA and vn_kv_max by loading raw parquet data from disk.
        For each scenario: concat Pd, Qd, Pg, Qg; take 95th percentile of non-zero as baseMVA;
        max vn_kv as vn_kv_max. Build lookup tensors indexed by scenario_id (no dicts).

        Args:
            data_path: Path to the raw data directory containing bus_data.parquet and gen_data.parquet.
            scenario_ids: List of scenario IDs to use for fitting.

        Returns:
            Dictionary of computed parameters.
        """
        bus_data = pd.read_parquet(os.path.join(data_path, "bus_data.parquet"))
        gen_data = pd.read_parquet(os.path.join(data_path, "gen_data.parquet"))

        bus_data = bus_data[bus_data["scenario"].isin(scenario_ids)]
        gen_data = gen_data[gen_data["scenario"].isin(scenario_ids)]

        baseMVA = []
        vn_kv_max = []
        scenarios = []

        bus_groups = bus_data.groupby("scenario")
        gen_groups = gen_data.groupby("scenario")

        for scenario in sorted(bus_groups.groups.keys()):
            bus_group = bus_groups.get_group(scenario)
            gen_group = gen_groups.get_group(scenario)
            pd_values = bus_group["Pd"]
            qd_values = bus_group["Qd"]
            qg_values = bus_group["Qg"]
            pg_values = gen_group["p_mw"]

            all_values = pd.concat([pd_values, qd_values, pg_values, qg_values])
            non_zero_values = all_values[all_values != 0]
            baseMVA.append(np.percentile(non_zero_values, 95))
            vn_kv_max.append(float(bus_group["vn_kv"].max()))
            scenarios.append(scenario)

        scenarios_t = torch.tensor(scenarios, dtype=torch.long)
        baseMVA_t = torch.tensor(baseMVA, dtype=torch.float)
        vn_kv_max_t = torch.tensor(vn_kv_max, dtype=torch.float)
        max_sid = int(scenarios_t.max().item())
        self._baseMVA_lookup = torch.zeros(max_sid + 1, dtype=torch.float)
        self._vn_kv_max_lookup = torch.zeros(max_sid + 1, dtype=torch.float)
        self._baseMVA_lookup[scenarios_t] = baseMVA_t
        self._vn_kv_max_lookup[scenarios_t] = vn_kv_max_t
        self._scenario_ids = scenarios_t

        return {
            "baseMVA_orig": torch.tensor(self.baseMVA_orig, dtype=torch.float),
            "scenarios": scenarios_t,
            "baseMVA": baseMVA_t,
            "vn_kv_max": vn_kv_max_t,
        }

    def fit_from_dict(self, params: dict):
        """Restore lookups and baseMVA_orig from saved params (scenarios, baseMVA, vn_kv_max tensors)."""
        scenarios = params.get("scenarios")
        baseMVA = params.get("baseMVA")
        vn_kv_max = params.get("vn_kv_max")
        max_sid = int(scenarios.max().item())
        self._baseMVA_lookup = torch.zeros(max_sid + 1, dtype=torch.float)
        self._vn_kv_max_lookup = torch.zeros(max_sid + 1, dtype=torch.float)
        self._baseMVA_lookup[scenarios] = baseMVA
        self._vn_kv_max_lookup[scenarios] = vn_kv_max
        self._scenario_ids = scenarios
        bmo = params.get("baseMVA_orig")
        self.baseMVA_orig = bmo.item() if hasattr(bmo, "item") else bmo

    def _per_node_mva(self, data: HeteroData):
        """
        Get per-node and per-edge baseMVA/vn_kv_max from data.scenario_id (single sample or batch).
        Returns (b, b_orig, vn, g, e_b, e_b_orig) with shapes (n, 1) for bus, gen, edge so they broadcast.
        Fully GPU/CPU safe.
        """
        if self._baseMVA_lookup is None:
            raise ValueError("Normalizer not fitted or lookups not built")
        
        device = data.x_dict["bus"].device
        dtype = data.x_dict["bus"].dtype

        bus_batch = getattr(data["bus"], "batch", None)
        gen_batch = getattr(data["gen"], "batch", None)
        n_bus = data.x_dict["bus"].size(0)
        n_gen = data.x_dict["gen"].size(0)
        edge_index = data["bus", "connects", "bus"].edge_index
        n_edge = edge_index.size(1)

        scenario_id = data["scenario_id"]

        # Scenario id per node/edge
        if bus_batch is not None:
            sid_bus = scenario_id[bus_batch]
            sid_gen = scenario_id[gen_batch]
            sid_edge = scenario_id[bus_batch[edge_index[0]]]
        else:
            sid = scenario_id.item()
            sid_bus = torch.full((n_bus,), sid, device=device, dtype=torch.long)
            sid_gen = torch.full((n_gen,), sid, device=device, dtype=torch.long)
            sid_edge = torch.full((n_edge,), sid, device=device, dtype=torch.long)

        # Move lookups to correct device/dtype before indexing
        baseMVA_lookup = self._baseMVA_lookup.to(device=device, dtype=dtype)
        vn_kv_max_lookup = self._vn_kv_max_lookup.to(device=device, dtype=dtype)

        b = baseMVA_lookup[sid_bus]
        vn = vn_kv_max_lookup[sid_bus]
        g = baseMVA_lookup[sid_gen]
        e_b = baseMVA_lookup[sid_edge]

        b_orig_val = self.baseMVA_orig if isinstance(self.baseMVA_orig, (int, float)) else self.baseMVA_orig.item()
        b_orig = torch.full_like(b, b_orig_val)
        e_b_orig = torch.full_like(e_b, b_orig_val)

        return b, b_orig, vn, g, e_b, e_b_orig


    def transform(self, data: HeteroData):
        """Apply per-unit normalization using per-scenario baseMVA/vn_kv_max (same formulas as base MVA normalizer)."""
        if self._baseMVA_lookup is None:
            raise ValueError("Normalizer not fitted or lookups not loaded")
        b, b_orig, vn, g, e_b, e_b_orig = self._per_node_mva(data)
        # --- Bus input normalization --- 
        data.x_dict["bus"][:, PD_H] /= b
        data.x_dict["bus"][:, QD_H] /= b
        data.x_dict["bus"][:, QG_H] /= b
        data.x_dict["bus"][:, MIN_QG_H] /= b
        data.x_dict["bus"][:, MAX_QG_H] /= b
        data.x_dict["bus"][:, VA_H] *= torch.pi / 180.0
        data.x_dict["bus"][:, GS] *= b_orig / b
        data.x_dict["bus"][:, BS] *= b_orig / b
        data.x_dict["bus"][:, VN_KV] /= vn

        # --- Bus label normalization ---
        data.y_dict["bus"][:, PD_H] /= b
        data.y_dict["bus"][:, QD_H] /= b
        data.y_dict["bus"][:, QG_H] /= b
        data.y_dict["bus"][:, VA_H] *= torch.pi / 180.0
        
        # --- Generator input normalization ---
        data.x_dict["gen"][:, PG_H] /= g
        data.x_dict["gen"][:, MIN_PG] /= g
        data.x_dict["gen"][:, MAX_PG] /= g
        data.x_dict["gen"][:, C0_H] = torch.sign(
            data.x_dict["gen"][:, C0_H],
        ) * torch.log1p(torch.abs(data.x_dict["gen"][:, C0_H]))
        data.x_dict["gen"][:, C1_H] = torch.sign(
            data.x_dict["gen"][:, C1_H],
        ) * torch.log1p(torch.abs(data.x_dict["gen"][:, C1_H]))
        data.x_dict["gen"][:, C2_H] = torch.sign(
            data.x_dict["gen"][:, C2_H],
        ) * torch.log1p(torch.abs(data.x_dict["gen"][:, C2_H]))

        # --- Generator label normalization ---
        data.y_dict["gen"][:, PG_H] /= g

        # --- Edge input normalization ---
        data.edge_attr_dict[("bus", "connects", "bus")][:, P_E] /= e_b
        data.edge_attr_dict[("bus", "connects", "bus")][:, Q_E] /= e_b
        data.edge_attr_dict[("bus", "connects", "bus")][:, YFF_TT_R : YFT_TF_I + 1] *= (
            e_b_orig.unsqueeze(1) / e_b.unsqueeze(1)
        )
        data.edge_attr_dict[("bus", "connects", "bus")][:, ANG_MIN] *= torch.pi / 180.0
        data.edge_attr_dict[("bus", "connects", "bus")][:, ANG_MAX] *= torch.pi / 180.0
        data.edge_attr_dict[("bus", "connects", "bus")][:, RATE_A] /= e_b
        data.is_normalized = True

    def inverse_transform(self, data: HeteroData):
        """Undo per-unit normalization (multiply by baseMVA, rad->deg, inverse log1p for cost coeffs)."""
        if self._baseMVA_lookup is None:
            raise ValueError("Normalizer not fitted or lookups not loaded")
        if not data.is_normalized.all():
            raise ValueError("Attempting to denormalize data which is not normalized")
        b, _, vn, g, e_b, _ = self._per_node_mva(data) # b_orig and e_b_orig are not used

        # -------- BUS INPUT INVERSE NORMALIZATION --------
        # NOTE: VA (bus input & label) are intentionally kept in
        # radians after inverse_transform -- the physics layers (ComputeBranchFlow,
        # ComputeNodeResiduals, etc.) expect radians.
        #
        # WARNING: GS, BS, and edge admittances (Y) are NOT restored to their
        # original casefile per-unit values. The transform scales them by
        # (b_orig / b), but the inverse multiplies by b (not b / b_orig),
        # yielding physical SI units (original * b_orig). This is intentional
        # for the physics layers.
        data.x_dict["bus"][:, PD_H] *= b
        data.x_dict["bus"][:, QD_H] *= b
        data.x_dict["bus"][:, QG_H] *= b
        data.x_dict["bus"][:, MIN_QG_H] *= b
        data.x_dict["bus"][:, MAX_QG_H] *= b
        data.x_dict["bus"][:, GS] *= b  # -> physical units (not original p.u.)
        data.x_dict["bus"][:, BS] *= b  # -> physical units (not original p.u.)
        data.x_dict["bus"][:, VN_KV] *= vn

        # -------- BUS LABEL INVERSE NORMALIZATION --------
        data.y_dict["bus"][:, PD_H] *= b
        data.y_dict["bus"][:, QD_H] *= b
        data.y_dict["bus"][:, QG_H] *= b

        # -------- GENERATOR INPUT INVERSE NORMALIZATION --------
        data.x_dict["gen"][:, PG_H] *= g
        data.x_dict["gen"][:, MIN_PG] *= g
        data.x_dict["gen"][:, MAX_PG] *= g
        data.x_dict["gen"][:, C0_H] = torch.sign(data.x_dict["gen"][:, C0_H]) * (
            torch.exp(torch.abs(data.x_dict["gen"][:, C0_H])) - 1
        )
        data.x_dict["gen"][:, C1_H] = torch.sign(data.x_dict["gen"][:, C1_H]) * (
            torch.exp(torch.abs(data.x_dict["gen"][:, C1_H])) - 1
        )
        data.x_dict["gen"][:, C2_H] = torch.sign(data.x_dict["gen"][:, C2_H]) * (
            torch.exp(torch.abs(data.x_dict["gen"][:, C2_H])) - 1
        )

        # -------- GENERATOR LABEL INVERSE NORMALIZATION --------
        data.y_dict["gen"][:, PG_H] *= g

        # -------- EDGE INPUT INVERSE NORMALIZATION --------
        data.edge_attr_dict[("bus", "connects", "bus")][:, P_E] *= e_b
        data.edge_attr_dict[("bus", "connects", "bus")][:, Q_E] *= e_b
        data.edge_attr_dict[("bus", "connects", "bus")][:, YFF_TT_R : YFT_TF_I + 1] *= (
            e_b.unsqueeze(1)
        )  # -> physical units (not original p.u.), see WARNING above
        data.edge_attr_dict[("bus", "connects", "bus")][:, ANG_MIN] *= 180.0 / torch.pi
        data.edge_attr_dict[("bus", "connects", "bus")][:, ANG_MAX] *= 180.0 / torch.pi

        data.edge_attr_dict[("bus", "connects", "bus")][:, RATE_A] *= e_b
        data.is_normalized = False




    def inverse_output(self, output, batch):
        """
        Denormalize model output (bus PG/QG, gen PG) using per-sample baseMVA from lookups.
        Fully GPU/CPU safe.
        """
        bus_output = output["bus"]
        gen_output = output["gen"]

        bus_batch = getattr(batch["bus"], "batch", None)

        # Move lookup tensor to correct device
        baseMVA_lookup = self._baseMVA_lookup.to(device=bus_output.device, dtype=bus_output.dtype)

        if bus_batch is not None:
            # Batched: scenario_id per node via batch index; lookup base MVA per node
            sid_bus = batch["scenario_id"][bus_batch]
            sid_gen = batch["scenario_id"][batch["gen"].batch]
            b_bus = baseMVA_lookup[sid_bus]
            b_gen = baseMVA_lookup[sid_gen]
        else:
            # Single graph: one scenario_id; use its base MVA
            sid = batch["scenario_id"].item()
            b_bus = baseMVA_lookup[sid]
            b_gen = baseMVA_lookup[sid]

        # Scale per-unit power back to MW/Mvar
        if self.task_name in ("ForecastOPF", "ST_ForecastOPF"):
            # ForecastOPF format: [Pd, Qd, Qg, Vm, Va] (5 features)
            bus_output[:, PD_H] *= b_bus  # Pd at index 0
            bus_output[:, QD_H] *= b_bus  # Qd at index 1
            bus_output[:, QG_H] *= b_bus  # Qg at index 2
            # Vm (index 3) and Va (index 4) remain as-is (p.u. and radians)
            gen_output[:, PG_OUT_GEN] *= b_gen  # Pg
        else:
            # OPF format: [Vm, Va, Pg, Qg] (4 features)
            bus_output[:, PG_OUT] *= b_bus  # Pg at index 2
            bus_output[:, QG_OUT] *= b_bus  # Qg at index 3
            # Vm (index 0) and Va (index 1) remain as-is (p.u. and radians)
            gen_output[:, PG_OUT_GEN] *= b_gen  # Pg

    def get_stats(self) -> dict:
        """Return dict of stats for saving (baseMVA_orig, scenarios, baseMVA, vn_kv_max tensors)."""
        return {
            "baseMVA_orig": torch.tensor(self.baseMVA_orig, dtype=torch.float),
            "scenarios": self._scenario_ids,
            "baseMVA": self._baseMVA_lookup[self._scenario_ids],
            "vn_kv_max": self._vn_kv_max_lookup[self._scenario_ids],
        }
   


@NORMALIZERS_REGISTRY.register("HeteroDataWindowMVANormalizer")
class HeteroDataWindowMVANormalizer(Normalizer):
    """
    Per-window MVA normalizer for temporal (sliding-window) ST-GNN datasets.

    Motivation
    ----------
    With 20+ years of data and a temporal train/test split, test windows have
    higher absolute load (secular growth). A global baseMVA fitted on training
    data causes distribution shift at test time.

    Per-window normalisation removes this: every lookback window is normalised
    by its own 95th-percentile Pd/Qd/Pg baseMVA. The model learns load
    *patterns*; the per-window baseMVA reconstructs absolute MW values at
    denormalisation time without any leakage (baseMVA is computed purely from
    the observable lookback window).

    Architecture
    ------------
    * ``transform()`` (preload time, per individual scenario):
        Static/time-invariant transforms only: VA deg->rad, VN_KV/vn_kv_max,
        log1p for generator costs, ANG_MIN/MAX deg->rad, GS/BS/Y scaled by
        baseMVA_static (fitted on training data).
        Power features (Pd, Qd, Qg, Pg, P_E, Q_E, RATE_A, limits) left in MW.

    * ``compute_window_baseMVA(batch_list)`` + ``apply_window_power_norm(...)``
        Called from collate_temporal_window_norm() after batching.
        Computes per-sample baseMVA from the lookback Pd/Qd/Pg values, then
        normalises power features in-place on the collated batch tensors.

    * ``inverse_transform(data, window_baseMVA)`` /
      ``inverse_output(output, batch, window_baseMVA)``:
        Require the [B] window_baseMVA tensor from the batch dict.
    """

    fit_strategy = "fit_on_train"

    def __init__(self, args):
        self.baseMVA_orig   = getattr(args.data, "baseMVA", 100)
        self.baseMVA_static = None   # 95th-pct from training Pd/Qd/Pg (for GS/BS/Y)
        self.vn_kv_max      = None
        self.task_name      = getattr(args.task, "task_name", "ST_ForecastOPF")

    def to(self, device):
        pass

    def fit(self, data_path: str, scenario_ids: List[int]) -> dict:
        bus_data = pd.read_parquet(os.path.join(data_path, "bus_data.parquet"))
        gen_data = pd.read_parquet(os.path.join(data_path, "gen_data.parquet"))
        bus_data = bus_data[bus_data["scenario"].isin(scenario_ids)]
        gen_data = gen_data[gen_data["scenario"].isin(scenario_ids)]

        non_zero = pd.concat([
            bus_data["Pd"][bus_data["Pd"] != 0],
            bus_data["Qd"][bus_data["Qd"] != 0],
            gen_data["p_mw"][gen_data["p_mw"] != 0],
            bus_data["Qg"][bus_data["Qg"] != 0],
        ])
        self.baseMVA_static = float(np.percentile(non_zero, 95))
        self.vn_kv_max      = float(bus_data["vn_kv"].max())

        return {
            "baseMVA_orig":   torch.tensor(self.baseMVA_orig,   dtype=torch.float),
            "baseMVA_static": torch.tensor(self.baseMVA_static, dtype=torch.float),
            "vn_kv_max":      torch.tensor(self.vn_kv_max,      dtype=torch.float),
        }

    def fit_from_dict(self, params: dict):
        self.baseMVA_static = params["baseMVA_static"].item()
        self.vn_kv_max      = params["vn_kv_max"].item()
        bmo = params.get("baseMVA_orig")
        self.baseMVA_orig   = bmo.item() if hasattr(bmo, "item") else float(bmo)

    def transform(self, data: HeteroData):
        """Apply only time-invariant transforms; power features stay in MW."""
        if self.vn_kv_max is None or self.baseMVA_static is None:
            raise ValueError("HeteroDataWindowMVANormalizer not fitted.")

        ratio = self.baseMVA_orig / self.baseMVA_static

        # Bus input
        data.x_dict["bus"][:, VA_H]   *= torch.pi / 180.0
        data.x_dict["bus"][:, GS]     *= ratio
        data.x_dict["bus"][:, BS]     *= ratio
        data.x_dict["bus"][:, VN_KV] /= self.vn_kv_max

        # Bus label — angle only
        data.y_dict["bus"][:, VA_H]   *= torch.pi / 180.0

        # Generator costs — log-compression
        for feat in [C0_H, C1_H, C2_H]:
            v = data.x_dict["gen"][:, feat]
            data.x_dict["gen"][:, feat] = torch.sign(v) * torch.log1p(v.abs())

        # Edge — admittances & angle limits
        ea = data.edge_attr_dict[("bus", "connects", "bus")]
        ea[:, YFF_TT_R: YFT_TF_I + 1] *= ratio
        ea[:, ANG_MIN]                  *= torch.pi / 180.0
        ea[:, ANG_MAX]                  *= torch.pi / 180.0

        # Power features left in raw MW; window normalisation applied in collate.
        data.is_normalized = False

    @staticmethod
    def compute_window_baseMVA(batch_list) -> torch.Tensor:
        """
        Compute per-sample baseMVA from the lookback window Pd/Qd/Pg values.
        All power values are still in MW at this point (before power normalisation).

        Returns:
            window_baseMVA: [B] float tensor.
        """
        baseMVAs = []
        for window_graphs, _ in batch_list:
            vals = []
            for g in window_graphs:
                vals.append(g["bus"].x[:, PD_H].abs())
                vals.append(g["bus"].x[:, QD_H].abs())
                vals.append(g["bus"].x[:, QG_H].abs()) 
                vals.append(g["gen"].x[:, PG_H].abs())
            all_vals = torch.cat(vals)
            non_zero = all_vals[all_vals > 1e-6]
            baseMVA  = float(torch.quantile(non_zero, 0.95).item()) if non_zero.numel() > 0 else 1.0
            baseMVAs.append(max(baseMVA, 1.0))
        return torch.tensor(baseMVAs, dtype=torch.float)

    @staticmethod
    def apply_window_power_norm(folded_batch, target_batch,
                                window_baseMVA: torch.Tensor,
                                W: int, n: int, N_bus: int, N_gen: int):
        """
        Normalise power features in-place on already-collated batch tensors.

        folded_batch / target_batch : PyG Batch (B*W and B*n graphs)
        window_baseMVA              : [B] per-sample baseMVA (MW)

        Note:
            folded_batch and target_batch use the same per-sample `window_baseMVA`.
            `b_bus_f`/`b_gen_f` and `b_bus_t`/`b_gen_t` differ only in expanded length
            because folded has `W` timesteps and target has `n` timesteps.
        """
        device = folded_batch["bus"].x.device
        wbmva  = window_baseMVA.to(device)

        # per-node scaling vectors 
        b_bus_f = wbmva.repeat_interleave(W * N_bus)   # [B*W*N_bus]
        b_gen_f = wbmva.repeat_interleave(W * N_gen)
        b_bus_t = wbmva.repeat_interleave(n * N_bus)   # [B*n*N_bus]
        b_gen_t = wbmva.repeat_interleave(n * N_gen)

        # per-edge scaling - memory layout is sample major, time minor
        ei_f  = folded_batch[("bus","connects","bus")].edge_index
        b_e_f = wbmva[folded_batch["bus"].batch[ei_f[0]] // W]
        ei_t  = target_batch[("bus","connects","bus")].edge_index
        b_e_t = wbmva[target_batch["bus"].batch[ei_t[0]] // n]

        # apply scaling
        folded_batch["bus"].x[:, PD_H]     /= b_bus_f
        folded_batch["bus"].x[:, QD_H]     /= b_bus_f
        folded_batch["bus"].x[:, QG_H]     /= b_bus_f
        folded_batch["bus"].x[:, MIN_QG_H] /= b_bus_f
        folded_batch["bus"].x[:, MAX_QG_H] /= b_bus_f

        folded_batch["gen"].x[:, PG_H]   /= b_gen_f
        folded_batch["gen"].x[:, MIN_PG] /= b_gen_f
        folded_batch["gen"].x[:, MAX_PG] /= b_gen_f

        folded_batch[("bus", "connects", "bus")].edge_attr[:, P_E]    /= b_e_f
        folded_batch[("bus", "connects", "bus")].edge_attr[:, Q_E]    /= b_e_f
        folded_batch[("bus", "connects", "bus")].edge_attr[:, RATE_A] /= b_e_f

        target_batch["bus"].x[:, PD_H]     /= b_bus_t
        target_batch["bus"].x[:, QD_H]     /= b_bus_t
        target_batch["bus"].x[:, QG_H]     /= b_bus_t
        target_batch["bus"].x[:, MIN_QG_H] /= b_bus_t
        target_batch["bus"].x[:, MAX_QG_H] /= b_bus_t
        target_batch["gen"].x[:, PG_H] /= b_gen_t
        target_batch["gen"].x[:, MIN_PG] /= b_gen_t
        target_batch["gen"].x[:, MAX_PG] /= b_gen_t

        target_batch[("bus", "connects", "bus")].edge_attr[:, P_E]    /= b_e_t
        target_batch[("bus", "connects", "bus")].edge_attr[:, Q_E]    /= b_e_t
        target_batch[("bus", "connects", "bus")].edge_attr[:, RATE_A] /= b_e_t

        target_batch["bus"].y[:, PD_H]     /= b_bus_t
        target_batch["bus"].y[:, QD_H]     /= b_bus_t
        target_batch["bus"].y[:, QG_H]     /= b_bus_t
        target_batch["gen"].y[:, PG_H] /= b_gen_t
            
        folded_batch.is_normalized = True
        target_batch.is_normalized = True

    def inverse_transform(self, data: HeteroData, window_baseMVA: torch.Tensor):
        """
        Denormalise power features. VA/VN_KV/costs stay in transformed form
        (physics layers expect radians, log-space, etc.).

        Args:
            data           : collated PyG Batch (target_batch or folded_batch)
            window_baseMVA : [B] tensor from batch dict
        """
        device   = data["bus"].x.device
        wbmva    = window_baseMVA.to(device)
        bus_b    = data["bus"].batch
        gen_b    = data["gen"].batch
        n_graphs = int(bus_b.max().item()) + 1
        gpb      = n_graphs // wbmva.size(0)          # graphs-per-sample (W or n)

        b_bus = wbmva[bus_b // gpb]
        b_gen = wbmva[gen_b // gpb]

        xb = data["bus"].x
        xb[:, PD_H] *= b_bus;  xb[:, QD_H] *= b_bus
        xb[:, QG_H] *= b_bus;  xb[:, MIN_QG_H] *= b_bus;  xb[:, MAX_QG_H] *= b_bus

        # Restore GS and BS to physical Siemens units using baseMVA_static
        xb[:, GS] *= self.baseMVA_static
        xb[:, BS] *= self.baseMVA_static

        xg = data["gen"].x
        xg[:, PG_H] *= b_gen;  xg[:, MIN_PG] *= b_gen;  xg[:, MAX_PG] *= b_gen
        
        # Restore edge admittances
        if ("bus", "connects", "bus") in data.edge_attr_dict:
            ea = data.edge_attr_dict[("bus", "connects", "bus")]
            ea[:, YFF_TT_R: YFT_TF_I + 1] *= self.baseMVA_static
        if getattr(data["bus"], "y", None) is not None:
            data["bus"].y[:, PD_H] *= b_bus
            data["bus"].y[:, QD_H] *= b_bus
            data["bus"].y[:, QG_H] *= b_bus
        if getattr(data["gen"], "y", None) is not None:
            data["gen"].y[:, PG_H] *= b_gen

        data.is_normalized = False

    def inverse_output(self, output: dict, batch, window_baseMVA: torch.Tensor):
        """
        Denormalise model predictions.

        Args:
            output         : {"bus": [B*n*N_bus, 5], "gen": [B*n*N_gen, 1]}
            batch          : target_batch (for .batch attribute)
            window_baseMVA : [B] tensor from batch dict
        """
        device   = output["bus"].device
        wbmva    = window_baseMVA.to(device)
        bus_b    = batch["bus"].batch
        gen_b    = batch["gen"].batch
        n_graphs = int(bus_b.max().item()) + 1
        n        = n_graphs // wbmva.size(0)

        b_bus = wbmva[bus_b // n]
        b_gen = wbmva[gen_b // n]

        # ForecastOPF output format: [Pd, Qd, Qg, Vm, Va]
        output["bus"][:, PD_H] *= b_bus   # Pd
        output["bus"][:, QD_H] *= b_bus   # Qd
        output["bus"][:, QG_H] *= b_bus   # Qg
        # Vm (3) and Va (4) — stay as p.u. and radians
        output["gen"][:, PG_OUT_GEN] *= b_gen

    def get_stats(self) -> dict:
        return {
            "baseMVA_orig":   torch.tensor(self.baseMVA_orig,   dtype=torch.float),
            "baseMVA_static": torch.tensor(self.baseMVA_static, dtype=torch.float),
            "vn_kv_max":      torch.tensor(self.vn_kv_max,      dtype=torch.float),
        }
