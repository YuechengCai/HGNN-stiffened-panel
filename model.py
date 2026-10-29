"""Full research HGT with node-type-specific BatchNorm inference moments."""
import torch
from torch_geometric.nn import HGTConv


class FHGT_YC(torch.nn.Module):
    """Original full-size research HGT, preserving its checkpoint layout.

    The final field prediction uses LeakyReLU directly on the final HGT output,
    as in the research source. The final BatchNorm module is retained for exact
    compatibility with the archived research checkpoints; it does not transform
    the prediction returned by that source implementation.
    """
    def __init__(self, in_channels, metadata, hidden_channels=64, out_channels=500,
                 num_layers=10, heads=4, seed=42, node_type_batch_stats=True):
        super().__init__()
        if num_layers < 2 or hidden_channels % heads or out_channels % heads:
            raise ValueError("Use >=2 layers and channel dimensions divisible by heads")
        torch.manual_seed(seed)
        self.num_layers = num_layers
        self.node_types = list(metadata[0])
        self.node_type_batch_stats = node_type_batch_stats
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.convs.append(HGTConv(in_channels, hidden_channels, metadata, heads=heads))
        self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(HGTConv(hidden_channels, hidden_channels, metadata, heads=heads))
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(HGTConv(hidden_channels, out_channels, metadata, heads=heads))
        self.batch_norms.append(torch.nn.BatchNorm1d(out_channels))
        self.activation = torch.nn.LeakyReLU()
        # Training calls BatchNorm separately for each node type. Running moments
        # must also remain type-specific for consistent inference. Affine weights
        # remain shared, preserving the original trainable parameter count.
        for i in range(num_layers - 1):
            for nt in self.node_types:
                self.register_buffer(f"type_mean_{i}_{nt}", torch.zeros(hidden_channels))
                self.register_buffer(f"type_var_{i}_{nt}", torch.ones(hidden_channels))

    def load_research_state_dict(self, state):
        """Import archived weights and initialize per-type moments for calibration."""
        expanded = dict(state)
        for i in range(self.num_layers - 1):
            for nt in self.node_types:
                expanded[f"type_mean_{i}_{nt}"] = state[f"batch_norms.{i}.running_mean"].clone()
                expanded[f"type_var_{i}_{nt}"] = state[f"batch_norms.{i}.running_var"].clone()
        self.load_state_dict(expanded, strict=True)

    def normalize(self, i, nt, x):
        norm = self.batch_norms[i]
        if not self.node_type_batch_stats:
            return norm(x)
        return torch.nn.functional.batch_norm(
            x, getattr(self, f"type_mean_{i}_{nt}"), getattr(self, f"type_var_{i}_{nt}"),
            norm.weight, norm.bias, self.training, norm.momentum, norm.eps)

    def forward(self, x_dict, edge_index_dict):
        for i in range(self.num_layers - 1):
            x_dict = self.convs[i](x_dict, edge_index_dict)
            x_dict = {nt: self.normalize(i, nt, x) for nt, x in x_dict.items()}
        x_dict = self.convs[-1](x_dict, edge_index_dict)
        for nt, x in x_dict.items():
            self.batch_norms[-1](x)
            x_dict[nt] = self.activation(x)
        return x_dict


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
