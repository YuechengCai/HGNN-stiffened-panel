"""Checks for the released graphs, normalization, and pretrained inference."""
from pathlib import Path
import sys
import unittest

import json
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import ROOT, load_dataset
from demo import evaluate, load_checkpoint, make_model
from model import FHGT_YC


class DemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.graphs, cls.splits, cls.manifest = load_dataset()

    def test_graph_endpoints_and_reverse_relations(self):
        for graph in self.graphs:
            self.assertTrue(graph.validate(raise_on_error=True))
            self.assertEqual(set(graph.node_types), {"plate", "force", "Disp", "edge"})
            self.assertEqual(len(graph.edge_types), 20)
            for i in range(1, 9):
                self.assertTrue(torch.equal(graph["edge", f"e{i}", "plate"].edge_index.flip(0),
                                            graph["plate", f"e{i}", "edge"].edge_index))
            self.assertTrue(torch.equal(graph["Disp", "e11", "edge"].edge_index.flip(0),
                                        graph["edge", "e12", "Disp"].edge_index))
            self.assertTrue(torch.equal(graph["plate", "e9", "force"].edge_index.flip(0),
                                        graph["force", "e10", "plate"].edge_index))

    def test_splits_and_shapes(self):
        self.assertEqual([len(self.splits[k]) for k in ("train", "val", "test")], [24, 4, 4])
        ids = [g.sample_id for g in self.graphs]
        self.assertEqual(len(ids), len(set(ids)))
        for graph in self.graphs:
            self.assertEqual(dict(graph.num_node_features), self.manifest["feature_dims"])
            self.assertEqual(graph["plate"].stress.shape, (graph["plate"].num_nodes, 500))
            self.assertEqual(graph["Disp"].x.size(1), 360)

    def test_node_type_running_statistics_remain_separate(self):
        graph = self.graphs[0]
        model = FHGT_YC(dict(graph.num_node_features), graph.metadata(),
                        hidden_channels=8, out_channels=8, num_layers=2, heads=2)
        noise = torch.linspace(-1, 1, 256).reshape(32, 8)
        plate, edge = noise + 10, noise - 10
        model.train()
        with torch.no_grad():
            for _ in range(100):
                model.normalize(0, "plate", plate)
                model.normalize(0, "edge", edge)
        model.eval()
        # Reusing a single running mean would bias both node types at inference.
        with torch.no_grad():
            self.assertLess(float(model.normalize(0, "plate", plate).mean(0).abs().max()), 0.01)
            self.assertLess(float(model.normalize(0, "edge", edge).mean(0).abs().max()), 0.01)

    def test_targets_are_not_inputs(self):
        model, _ = load_checkpoint(ROOT / "pretrained/case3_stress.pt", self.graphs[0])
        model.eval()
        original = self.graphs[0]
        changed = original.clone()
        changed["plate"].stress += 10000
        changed["plate"].Location += 10000
        with torch.no_grad():
            a = model(original.x_dict, original.edge_index_dict)["plate"]
            b = model(changed.x_dict, changed.edge_index_dict)["plate"]
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_pretrained_batching_and_reference_metrics(self):
        model, checkpoint = load_checkpoint(ROOT / "pretrained/case3_stress.pt", self.graphs[0])
        self.assertEqual(checkpoint["architecture"], "research_hgt")
        self.assertEqual(sum(p.numel() for p in model.parameters()), 4887056)
        self.assertEqual(model.num_layers, 10)
        self.assertEqual(checkpoint["training_source_indices"], list(range(4800)))
        self.assertEqual(checkpoint["validation_source_indices"], list(range(5400, 5600)))
        self.assertTrue(set(checkpoint["training_source_indices"]).isdisjoint(g.sample_id for g in self.splits["test"]))
        full_test = set(range(4800, 5400)) | set(range(5600, 5999))
        self.assertTrue(set(checkpoint["training_source_indices"]).isdisjoint(full_test))
        self.assertTrue(set(checkpoint["validation_source_indices"]).isdisjoint(full_test))
        m1, y1, p1 = evaluate(model, self.splits["test"], batch_size=1)
        m4, y4, p4 = evaluate(model, self.splits["test"], batch_size=4)
        np.testing.assert_array_equal(y1, y4)
        np.testing.assert_allclose(p1, p4, atol=0.002, rtol=0.0002)
        self.assertAlmostEqual(m1["rmse"], m4["rmse"], places=3)
        reference = json.loads((ROOT / "pretrained/metrics.json").read_text())
        self.assertAlmostEqual(m4["rmse"], reference["rmse"], places=3)
        self.assertGreater(float(np.std(p4)), 1.0)


if __name__ == "__main__":
    unittest.main()
