# Example data

`case1_demo.zip` contains 34 CSV files and `manifest.json`. It is a lossless float32 export of 32 existing Case 1 processed graphs. The entire research database is not included.

## Samples and splits

The manifest records every original zero-based sample index and its split. The 24 training indices are evenly spaced over the original training range 0–399; four validation indices are spaced over 5100–5299; four held-out demonstration indices are spaced over 5300–5378. Splits never overlap. These are fixed graph-level subsets of the available Case 1 cache; they are not a new reproduction of the paper's database split or an assertion of independence between all related physical configurations.

One CSV row corresponds to one graph. Columns are `c0`, `c1`, ... and rows are padded with empty cells because node counts differ. The loader removes padding, reshapes values, and validates graph endpoints. CSVs use an explicit header, unlike some original MATLAB exports; `dataset.py` handles that header.

## Node features and outputs

| CSV | Tensor |
|---|---|
| `Node_plate.csv` | `plate.x`: `[number_of_plates, 3]` |
| `Node_force.csv` | `force.x`: `[number_of_loading_nodes, 9]` |
| `Node_edge.csv` | `edge.x`: `[number_of_structural_edges, 10]` |
| `U1.csv`, `U2.csv`, `U3.csv`, `UR1.csv`, `UR2.csv`, `UR3.csv` | Each `[number_of_boundary_nodes, 60]`; concatenated in that order into `Disp.x` with 360 columns |
| `Plate_stress_top.csv` | `plate.stress`: `[number_of_plates, 500]`, MPa |
| `Plate_U1.csv`, `Plate_U2.csv`, `Plate_U3.csv` | The corresponding displacement target, each `[number_of_plates, 500]`, in original stored/scaled units |
| `Panel_location.csv` | `plate.Location`: `[number_of_plates, 4]`; metadata, not an input to the model |

Feature values preserve the research preprocessing and scaling. `force` is nine-dimensional for this Case 1 dataset; the legacy ablation loader's hard-coded value of ten does not match these data. The demo preserves the actual nine-dimensional inputs. No output targets or location metadata are passed into the HGT's `x_dict`.

The six `U*.csv`/`UR*.csv` inputs are prescribed boundary information. The `Plate_U*.csv` files are supervised plate field targets and are distinct from those inputs.

## Directed relations

Each adjacency row flattens a `[2, number_of_connections]` integer array: **all source indices first, then all destination indices**. Indices are zero-based.

| CSVs | Relation triplets |
|---|---|
| `Ae2p_1.csv` ... `Ae2p_4.csv` | `('edge', 'e1' ... 'e4', 'plate')` |
| `Ap2e_1.csv` ... `Ap2e_4.csv` | `('plate', 'e1' ... 'e4', 'edge')` |
| `Ae2p_v1.csv` ... `Ae2p_v4.csv` | `('edge', 'e5' ... 'e8', 'plate')` |
| `Ap2e_v1.csv` ... `Ap2e_v4.csv` | `('plate', 'e5' ... 'e8', 'edge')` |
| `Ap2f.csv` | `('plate', 'e9', 'force')` |
| `Af2p.csv` | `('force', 'e10', 'plate')` |
| `Ad2e.csv` | `('Disp', 'e11', 'edge')` |
| `Ae2d.csv` | `('edge', 'e12', 'Disp')` |

The exporter reverses the endpoint rows of the two legacy cached `Disp ↔ edge` arrays. Their original orientation assigned structural-edge IDs to the smaller `Disp` node set and produced invalid source/destination indices. The released arrays follow the relation triplets above. Features, plate targets, and all other relations are unchanged; every exported tensor was checked against the corrected source graph.

## Field grid

The 500 field entries form a 10 × 50 grid. Use `field.reshape(10, 50, order='F')` to match MATLAB's `reshape(field, [10, 50])`. Plots use normalized local plate coordinates. They do not infer global panel coordinates from the location metadata.
