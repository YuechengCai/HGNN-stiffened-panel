"""Research residual HGT architecture with portable normalization statistics."""
import torch
from torch_geometric.nn import HGTConv


class FHGT_YC_V2(torch.nn.Module):
    """The research V2 residual HGT, with training-only normalization buffers.

    The improved graph representation follows the subsequent global–local paper.
    Only plate nodes have an output head. Predictions are returned in stored
    target units. All normalization statistics travel with the state dict.
    """
    def __init__(self, in_channels, metadata, hidden_channels=64, out_channels=500,
                 num_layers=4, heads=4, seed=42, dropout=0.1):
        super().__init__()
        if num_layers < 1 or hidden_channels % heads:
            raise ValueError("Use >=1 layers and hidden channels divisible by heads")
        torch.manual_seed(seed)
        self.node_types = list(metadata[0])
        self.in_proj = torch.nn.ModuleDict({nt: torch.nn.Linear(in_channels[nt], hidden_channels) for nt in self.node_types})
        self.convs = torch.nn.ModuleList([
            HGTConv(hidden_channels, hidden_channels, metadata, heads=heads) for _ in range(num_layers)])
        self.norms = torch.nn.ModuleList([
            torch.nn.ModuleDict({nt: torch.nn.LayerNorm(hidden_channels) for nt in self.node_types})
            for _ in range(num_layers)])
        self.act = torch.nn.LeakyReLU(0.1)
        self.drop = torch.nn.Dropout(dropout)
        self.head = torch.nn.ModuleDict({"plate": torch.nn.Linear(hidden_channels, out_channels)})
        for nt in self.node_types:
            self.register_buffer(f"input_mean_{nt}", torch.zeros(in_channels[nt]))
            self.register_buffer(f"input_scale_{nt}", torch.ones(in_channels[nt]))
        self.register_buffer("target_mean", torch.tensor(0.0))
        self.register_buffer("target_scale", torch.tensor(1.0))

    @torch.no_grad()
    def fit_normalization(self, training_graphs, target):
        for nt in self.node_types:
            x = torch.cat([g[nt].x for g in training_graphs])
            scale = x.std(0, unbiased=False)
            getattr(self, f"input_mean_{nt}").copy_(x.mean(0))
            getattr(self, f"input_scale_{nt}").copy_(torch.where(scale > 1e-6, scale, torch.ones_like(scale)))
        y = torch.cat([g["plate"][target].flatten() for g in training_graphs])
        self.target_mean.copy_(y.mean())
        self.target_scale.copy_(y.std(unbiased=False).clamp_min(1e-6))

    def forward(self, x_dict, edge_index_dict):
        h = {nt: self.in_proj[nt]((x - getattr(self, f"input_mean_{nt}")) /
                                 getattr(self, f"input_scale_{nt}")) for nt, x in x_dict.items()}
        for norms, conv in zip(self.norms, self.convs):
            messages = conv({nt: norms[nt](x) for nt, x in h.items()}, edge_index_dict)
            h = {nt: x + self.drop(self.act(messages[nt])) for nt, x in h.items()}
        return {"plate": self.head["plate"](h["plate"]) * self.target_scale + self.target_mean}
