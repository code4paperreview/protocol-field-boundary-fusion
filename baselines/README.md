# Baseline implementations

Paper-based implementations and adapters for the byte-boundary comparison:

- ProsegDL (Zhao et al., ICNP 2022), including NetPlier grouping
- SLMSP (Li et al., IEEE TIFS 2026)
- TransRE (Zhao et al., 2026), including retrospective source selection
- Packet Field Tree (Rohl et al., TMA 2023)

The implementations follow the published workflows and the evaluation
protocol described in the manuscript.

## Requirements

From the package root:

```bash
pip install -r requirements.txt
```

The runners use PyTorch. A CUDA device is recommended; `--device cpu` is
supported but slow for the UNet branch of ProsegDL.

## Data layout

The runners expect one subdirectory per protocol:

```text
<NHF_DATA_ROOT>/
├── dnp3/dnp3.csv
├── modbus/modbus.csv
└── ...
```

Each CSV needs a `Hex` column; the target `Segment` column is read only after
predictions are written. Set the data directory with:

```bash
export NHF_DATA_ROOT=/path/to/format_data
```

If unset, the package looks for `data/` next to the package root.

## Run

```bash
python -m baselines.external_dl.run \
  --method prosegdl --protocol mqtt --seed 20261021 \
  --output runs/prosegdl_mqtt

python -m baselines.external_dl.run \
  --method slmsp --protocol mqtt --seed 20261021 \
  --output runs/slmsp_mqtt

python -m baselines.external_dl.run \
  --method transre --protocol mqtt --seed 20261021 \
  --output runs/transre_mqtt
```

`--smoke` runs a short integration check and marks the output as `smoke` in
`manifest.json`.

Each run writes `manifest.json`, `predictions.jsonl`, and `metrics.json`.
Evaluation uses `nhf.evaluation.metrics`.

### ProsegDL with NetPlier grouping

`prepare_netplier.py`, `cluster_netplier.py`, and
`infer_prosegdl_netplier.py` reproduce the NetPlier-grouped ProsegDL setting.
Set `NHF_PROJECT_ROOT` to the directory containing `format_label/`, and
`NETPLIER_ROOT` to the upstream NetPlier tree. The cluster and inference
commands are:

```bash
python -m baselines.external_dl.prepare_netplier --protocol mqtt --output runs/netplier_mqtt
python -m baselines.external_dl.cluster_netplier --metadata runs/netplier_mqtt/metadata.json --output runs/netplier_mqtt/clusters.json
python -m baselines.external_dl.infer_prosegdl_netplier \
  --protocol mqtt --training-run runs/prosegdl_mqtt \
  --clusters runs/netplier_mqtt/clusters.json --output runs/prosegdl_mqtt_netplier
```

### TransRE retrospective source selection

`run_transre_pairwise.py` runs TransRE for every source-target pair and
records the best source per target:

```bash
python -m baselines.external_dl.run_transre_pairwise --output runs/transre_pairwise
```

### Packet Field Tree

`pft.py`, `pft_numerics.py`, and `run_pft.py` adapt the public Packet Field
Tree implementation. Set `PFT_ROOT` to the PacketFieldTree tree and
`PFT_DEPENDENCIES` to its Python dependencies, then run:

```bash
python -m baselines.external_dl.run_pft --output runs/pft
```

## Scope

This package covers byte-boundary recovery for the four baseline
implementations.
