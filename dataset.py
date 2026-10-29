"""Read the small, portable CSV archive and construct the four-node-type V3 graph."""
from pathlib import Path
import json
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

ROOT = Path(__file__).resolve().parent
DOFS = ("U1", "U2", "U3", "UR1", "UR2", "UR3")
TARGETS = ("stress", "U1", "U2", "U3")


def relation_files():
    """Preserve the original V3 relation names, directions, and insertion order."""
    relations = {}
    for suffix, offset in (("", 0), ("v", 4)):
        for src, dst, prefix in (("edge", "plate", "Ae2p"), ("plate", "edge", "Ap2e")):
            for i in range(1, 5):
                relations[(src, f"e{i + offset}", dst)] = f"{prefix}_{suffix}{i}.csv"
    relations.update({
        ("plate", "e9", "force"): "Ap2f.csv",
        ("force", "e10", "plate"): "Af2p.csv",
        ("Disp", "e11", "edge"): "Ad2e.csv",
        ("edge", "e12", "Disp"): "Ae2d.csv",
    })
    return relations


def load_dataset(archive=None):
    """Load CSVs directly from ZIP; no download, extraction, or pickle required.

    One CSV row is one graph. Padding NaNs are removed per row, following the
    research loader. Adjacency is stored as all source indices, then destinations.
    """
    archive = Path(archive) if archive else ROOT / "data" / "case1_demo.zip"
    with ZipFile(archive) as z:
        manifest = json.loads(z.read("manifest.json"))
        tables = {name: pd.read_csv(z.open(name)) for name in manifest["csv_files"]}
    n = len(manifest["samples"])
    if any(len(table) != n for table in tables.values()):
        raise ValueError("CSV sample counts disagree with manifest")

    def tensor(name, row, width, integer=False):
        values = tables[name].iloc[row].dropna().to_numpy(dtype=np.float64)
        if values.size % width or not np.isfinite(values).all():
            raise ValueError(f"Invalid values/shape in {name}, row {row}")
        if integer and not np.equal(values, np.floor(values)).all():
            raise ValueError(f"Non-integer adjacency in {name}, row {row}")
        dtype = torch.long if integer else torch.float32
        return torch.tensor(values, dtype=dtype).reshape(-1, width)

    graphs = []
    dims = manifest["feature_dims"]
    for row, sample in enumerate(manifest["samples"]):
        g = HeteroData()
        # Node insertion order matches FHGT_YC's shared BatchNorm training order.
        g["plate"].x = tensor("Node_plate.csv", row, dims["plate"])
        g["force"].x = tensor("Node_force.csv", row, dims["force"])
        g["Disp"].x = torch.cat([tensor(f"{d}.csv", row, dims["Disp"] // 6) for d in DOFS], dim=1)
        g["edge"].x = tensor("Node_edge.csv", row, dims["edge"])
        for target in TARGETS:
            name = "Plate_stress_top.csv" if target == "stress" else f"Plate_{target}.csv"
            g["plate"][target] = tensor(name, row, manifest["output_dim"])
            if g["plate"][target].size(0) != g["plate"].num_nodes:
                raise ValueError(f"Plate target count mismatch in row {row}")
        g["plate"].Location = tensor("Panel_location.csv", row, 4)
        for relation, name in relation_files().items():
            # tensor() flattened row, then reshape to [2, E], NOT [E, 2].
            g[relation].edge_index = tensor(name, row, 2, integer=True).reshape(2, -1)
        g.sample_id = sample["source_index"]
        g.validate(raise_on_error=True)
        graphs.append(g)
    splits = {name: [graphs[i] for i, s in enumerate(manifest["samples"]) if s["split"] == name]
              for name in ("train", "val", "test")}
    return graphs, splits, manifest
