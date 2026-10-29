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

    def test_normalization_uses_training_only(self):
        model = make_model(self.graphs[0])
        model.fit_normalization(self.splits["train"], "stress")
        for nt in self.graphs[0].node_types:
            training_mean = torch.cat([g[nt].x for g in self.splits["train"]]).mean(0)
            torch.testing.assert_close(getattr(model, f"input_mean_{nt}"), training_mean)
        y = torch.cat([g["plate"].stress.flatten() for g in self.splits["train"]])
        torch.testing.assert_close(model.target_mean, y.mean())
        torch.testing.assert_close(model.target_scale, y.std(unbiased=False))

    def test_pretrained_batching_and_reference_metrics(self):
        model, checkpoint = load_checkpoint(ROOT / "pretrained/case3_stress.pt", self.graphs[0])
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
