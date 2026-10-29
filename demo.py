"""Train, evaluate, and visualize the improved heterogeneous graph representation."""
import argparse
import json
from pathlib import Path
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from dataset import ROOT, DEFAULT_ARCHIVE, TARGETS, load_dataset
from model import FHGT_YC, FHGT_YC_V2


def make_model(graph, config=None, architecture="research_hgt"):
    classes = {"research_hgt": FHGT_YC, "residual_hgt": FHGT_YC_V2}
    if architecture not in classes:
        raise ValueError(f"Unknown architecture: {architecture}")
    return classes[architecture](in_channels=dict(graph.num_node_features), metadata=graph.metadata(), **(config or {}))


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
    model = make_model(graph, checkpoint["model_config"], checkpoint.get("architecture", "residual_hgt"))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device), checkpoint


def train(splits, output, device="cpu", target="stress", epochs=3, batch_size=8,
          hidden_channels=64, num_layers=10, lr=1e-4, seed=42,
          patience=0, min_epochs=0, scheduler_patience=15, case="demo",
          architecture="research_hgt"):
    if epochs < 1 or batch_size < 1 or not splits["train"] or not splits["val"]:
        raise ValueError("Training requires positive epochs/batch size and nonempty train/val splits")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    config = dict(hidden_channels=hidden_channels, num_layers=num_layers, heads=4,
                  out_channels=500, seed=seed)
    graph = splits["train"][0]
    model = make_model(graph, config, architecture)
    if architecture == "residual_hgt":
        model.fit_normalization(splits["train"], target)
    model = model.to(device)
    loader = DataLoader(splits["train"], batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=scheduler_patience, min_lr=1e-5)
    best_loss, best_state, best_epoch = float("inf"), None, None
    history = []
    started = time.monotonic()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        squared_error, n_values = 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            truth = batch["plate"][target]
            pred = model(batch.x_dict, batch.edge_index_dict)["plate"]
            scale = float(model.target_scale.item()) if architecture == "residual_hgt" else 1.0
            loss = torch.nn.functional.mse_loss(pred, truth) / scale**2
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            squared_error += loss.item() * scale**2 * truth.numel()
            n_values += truth.numel()
        metrics, _, _ = evaluate(model, splits["val"], device, target, batch_size=max(4, batch_size))
        val_mse = metrics["rmse"] ** 2
        history.append({"epoch": epoch, "train_mse": squared_error / n_values, "val_mse": val_mse,
                        "lr": optimizer.param_groups[0]["lr"], "elapsed_seconds": time.monotonic() - started})
        scheduler.step(val_mse)
        print(f"Epoch {epoch}/{epochs}: train MSE={squared_error/n_values:.4f}, val MSE={val_mse:.4f}, elapsed={time.monotonic()-started:.1f}s", flush=True)
        if val_mse < best_loss:
            best_loss, best_epoch = val_mse, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if patience and epoch >= min_epochs and epoch - best_epoch >= patience:
            print(f"Early stopping after {epoch} epochs; best validation epoch {best_epoch}.", flush=True)
            break
    torch.save({"model_state_dict": best_state, "model_config": config, "architecture": architecture, "target": target,
                "feature_dims": dict(graph.num_node_features), "metadata": graph.metadata(),
                "best_epoch": best_epoch, "training_source_indices": [g.sample_id for g in splits["train"]],
                "validation_source_indices": [g.sample_id for g in splits["val"]],
                "case": case, "provenance": "Exact training and validation graph IDs recorded in this checkpoint",
                "training_config": {"epochs_requested": epochs, "epochs_run": len(history),
                                    "batch_size": batch_size, "learning_rate": lr, "seed": seed,
                                    "scheduler": "ReduceLROnPlateau", "scheduler_patience": scheduler_patience,
                                    "early_stopping_patience": patience, "min_epochs": min_epochs}},
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
    parser.add_argument("--data", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "pretrained/case3_stress.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/demo")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--target", choices=TARGETS, default="stress")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=0, help="Validation early stopping; 0 disables it")
    parser.add_argument("--min-epochs", type=int, default=0)
    parser.add_argument("--scheduler-patience", type=int, default=15)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--plate", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    graphs, splits, manifest = load_dataset(args.data)
    if args.command == "inspect":
        print(json.dumps({"case": manifest["case"], "representation": manifest["representation"],
                          "samples": len(graphs), "splits": {k: len(v) for k, v in splits.items()},
                          "features": manifest["feature_dims"], "relations": len(graphs[0].edge_types)}, indent=2))
        print(graphs[0])
    elif args.command == "train":
        train(splits, args.output, args.device, args.target, args.epochs, args.batch_size, args.hidden,
              args.layers, args.lr, patience=args.patience, min_epochs=args.min_epochs,
              scheduler_patience=args.scheduler_patience, case=manifest["case"])
        run_inference(args.output / "best_model.pt", splits, args.output, args.device)
    else:
        run_inference(args.checkpoint, splits, args.output, args.device, args.split, args.plate)


if __name__ == "__main__":
    main()
