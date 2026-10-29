"""Train, evaluate, and visualize the V3 stiffened-panel demonstration."""
import argparse
import json
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from dataset import ROOT, TARGETS, load_dataset
from model import FHGT_YC_V2


def make_model(graph, config=None):
    return FHGT_YC_V2(in_channels=dict(graph.num_node_features), metadata=graph.metadata(), **(config or {}))


def evaluate(model, graphs, device="cpu", target="stress", batch_size=4):
    """Report pointwise metrics weighted by the number of plate field values."""
    if not graphs:
        raise ValueError("Evaluation split is empty")
    model.eval()
    truths, predictions = [], []
    with torch.no_grad():
        for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
            batch = batch.to(device)
            predictions.append(model(batch.x_dict, batch.edge_index_dict)["plate"].cpu())
            truths.append(batch["plate"][target].cpu())
    true, pred = torch.cat(truths).numpy(), torch.cat(predictions).numpy()
    error = pred - true
    metrics = {"rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.mean(np.abs(error))),
               "n_graphs": len(graphs), "n_plate_nodes": len(true), "n_field_values": int(true.size)}
    if not np.isfinite(pred).all():
        raise RuntimeError("Model produced non-finite predictions")
    return metrics, true, pred


def load_checkpoint(path, graph, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["feature_dims"] != dict(graph.num_node_features):
        raise ValueError("Checkpoint and graph feature dimensions differ")
    if checkpoint["metadata"] != graph.metadata():
        raise ValueError("Checkpoint and graph relations/order differ")
    model = make_model(graph, checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device), checkpoint


def train(splits, output, device="cpu", target="stress", epochs=3, batch_size=8,
          hidden_channels=64, num_layers=4, lr=1e-3, seed=42):
    if epochs < 1 or batch_size < 1 or not splits["train"] or not splits["val"]:
        raise ValueError("Training requires positive epochs/batch size and nonempty train/val splits")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    config = dict(hidden_channels=hidden_channels, num_layers=num_layers, heads=4,
                  out_channels=500, seed=seed)
    graph = splits["train"][0]
    model = make_model(graph, config)
    model.fit_normalization(splits["train"], target)
    model = model.to(device)
    loader = DataLoader(splits["train"], batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # A simple constant learning rate keeps this short demonstration transparent.
    best_loss, best_state, best_epoch = float("inf"), None, None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        squared_error, n_values = 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            truth = batch["plate"][target]
            pred = model(batch.x_dict, batch.edge_index_dict)["plate"]
            loss = torch.nn.functional.mse_loss(pred, truth) / model.target_scale.square()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            squared_error += loss.item() * model.target_scale.item() ** 2 * truth.numel()
            n_values += truth.numel()
        metrics, _, _ = evaluate(model, splits["val"], device, target)
        val_mse = metrics["rmse"] ** 2
        history.append({"epoch": epoch, "train_mse": squared_error / n_values, "val_mse": val_mse})
        print(f"Epoch {epoch}/{epochs}: train MSE={squared_error/n_values:.4f}, val MSE={val_mse:.4f}", flush=True)
        if val_mse < best_loss:
            best_loss, best_epoch = val_mse, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": best_state, "model_config": config, "target": target,
                "feature_dims": dict(graph.num_node_features), "metadata": graph.metadata(),
                "best_epoch": best_epoch, "training_source_indices": [g.sample_id for g in splits["train"]],
                "validation_source_indices": [g.sample_id for g in splits["val"]],
                "provenance": "Case 1 training; exact graph IDs recorded in this checkpoint"},
               output / "best_model.pt")
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    model.load_state_dict(best_state)
    return model, history


def plot_prediction(true, pred, target="stress", plate=0):
    """Original MATLAB reshape [10,50] uses column-major ordering."""
    if not 0 <= plate < len(true):
        raise ValueError("Plate index out of range")
    truth = true[plate].reshape(10, 50, order="F")
    prediction = pred[plate].reshape(10, 50, order="F")
    error = np.abs(prediction - truth)
    lo, hi = min(truth.min(), prediction.min()), max(truth.max(), prediction.max())
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), constrained_layout=True)
    for ax, values, title in zip(axes, (truth, prediction, error), ("Reference", "HGT prediction", "Absolute error")):
        kwargs = {"vmin": lo, "vmax": hi} if title != "Absolute error" else {"vmin": 0}
        im = ax.imshow(values, origin="lower", aspect="auto", extent=(0, 1, 0, 1), **kwargs)
        ax.set(title=title, xlabel="Normalized plate length", ylabel="Normalized plate width")
        fig.colorbar(im, ax=ax, label="MPa" if target == "stress" else "Stored displacement units")
    fig.suptitle(f"{target}: plate {plate} (zero-based)")
    return fig


def run_inference(checkpoint_path, splits, output, device="cpu", split="test", plate=0):
    model, checkpoint = load_checkpoint(checkpoint_path, splits[split][0], device)
    target = checkpoint["target"]
    metrics, true, pred = evaluate(model, splits[split], device, target)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Contour from the FIRST graph only; metrics cover the entire chosen split.
    first_metrics, first_true, first_pred = evaluate(model, splits[split][:1], device, target)
    fig = plot_prediction(first_true, first_pred, target, plate)
    fig.savefig(output / "prediction.png", dpi=160)
    plt.close(fig)
    np.savez_compressed(output / "predictions.npz", true=true, predicted=pred,
                        plate_counts=[g["plate"].num_nodes for g in splits[split]],
                        source_indices=[g.sample_id for g in splits[split]])
    report = {"target": target, "split": split, **metrics,
              "first_graph": {"source_index": splits[split][0].sample_id, **first_metrics}}
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect", "train", "predict"))
    parser.add_argument("--data", type=Path, default=ROOT / "data/case1_demo.zip")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "pretrained/case1_stress.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/demo")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--target", choices=TARGETS, default="stress")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--plate", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    graphs, splits, manifest = load_dataset(args.data)
    if args.command == "inspect":
        print(json.dumps({"samples": len(graphs), "splits": {k: len(v) for k, v in splits.items()},
                          "features": manifest["feature_dims"], "relations": len(graphs[0].edge_types)}, indent=2))
        print(graphs[0])
    elif args.command == "train":
        train(splits, args.output, args.device, args.target, args.epochs, args.batch_size, args.hidden, args.layers, args.lr)
        run_inference(args.output / "best_model.pt", splits, args.output, args.device)
    else:
        run_inference(args.checkpoint, splits, args.output, args.device, args.split, args.plate)


if __name__ == "__main__":
    main()
