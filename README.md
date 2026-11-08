# Code for structure-constrained field boundary inference

Reference implementation of the heuristic evidence extraction and
structure-constrained fusion decoding used in the manuscript.

Repository: <https://github.com/code4paperreview/protocol-field-boundary-fusion>

## Contents

```text
nhf/
├── common/       message, constraint, and boundary-prior data structures
├── heuristic/    structural and semantic evidence detectors
├── evaluation/   boundary F1 and perfection metrics
├── fusion/       ConstrainedBoundaryDecoder
└── neural/       dual-stream predictor architecture
baselines/
└── external_dl/  ProsegDL, SLMSP, TransRE, NetPlier, and PFT scripts
examples/
├── sample_messages.csv
└── sample_boundary_scores.json
scripts/
└── run_demo.py
```

`nhf.neural` contains the predictor architecture. `baselines/external_dl`
contains the paper-based baseline implementations; see `baselines/README.md`.

## Requirements

Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

The core heuristic and decoding demo needs `numpy`. The neural module also
needs `torch` and `torchcrf`.

## Demo

```bash
python scripts/run_demo.py
```

The script reads grouped messages and per-position boundary probabilities,
extracts heuristic constraints, and decodes a field partition. The scores in
`examples/sample_boundary_scores.json` are illustrative; replace them with the
dual-stream predictor output for new traffic.

Input format:

- messages CSV: columns `group` and `hex`
- scores JSON: group name to per-byte boundary probabilities

The output contains the heuristic boundaries, decoded boundaries, and the
number of extracted constraints for each group.

## Data

The benchmark messages and annotations come from the public sources below.
The manuscript cites:

- <https://github.com/hi-zer/Industrial-protocols-dataset>
- <https://github.com/ITI/ICS-Security-Tools>
- <https://github.com/Bright000026/Dual_stream_format_inference-PRE>

## Upstream baseline repositories

The classical and hybrid baselines use the following public implementations:

- Netzob: <https://github.com/netzob/netzob>
- BinaryInferno: <https://github.com/binaryinferno/binaryinferno>
- NEMESYS: <https://github.com/vs-uulm/nemesys>
- NetPlier: <https://github.com/yapengye/NetPlier>
- Packet Field Tree: <https://github.com/rohlex18/apre-benchmark-database>

## License

MIT License. See [LICENSE](LICENSE).

`nhf/heuristic/N_entropy_result.csv` is an entropy reference table from the
BinaryInferno project (Chandler et al., NDSS 2023). The MIT license applies to
the code in this repository; the table remains subject to the upstream
BinaryInferno license.
