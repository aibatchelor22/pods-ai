# Copilot Instructions for PODS-AI

## Project Overview

PODS-AI (Programmatic Orca Detection System using Artificial Intelligence) is a Python project for detecting and classifying orca and other whale vocalizations from audio recordings provided by the [Orcasound](https://www.orcasound.net/) hydrophone network.

## Repository Layout

```
pods-ai/
├── .github/
│   ├── copilot-instructions.md   # This file
│   ├── dependabot.yml
│   └── workflows/
│       ├── check_csv.yml         # CI: regenerates and validates detections.csv
│       └── validate-yaml.yml     # CI: yamllint on all YAML files
├── requirements.txt
├── output/                          # Generated output (detections.csv, wav files, spectrograms)
└── src/
    ├── make_csv.py                  # Step 1: query APIs → detections.csv
    ├── extract_training_samples.py  # Step 2: detections.csv → training_samples.csv
    ├── download_wavs.py             # Step 3: download wav files
    ├── make_spectrograms.py         # Step 4: wav → PNG spectrograms
    └── spectrogram_visualizer.py    # Helper: visualize spectrograms
```

## Data Sources

- **Orcasite API** (`https://live.orcasound.net/api/json/`) – Human and machine detections, feed metadata.
- **OrcaHello** – Azure Cosmos DB (`aifororcasmetadatastore`) storing Southern Resident Killer Whale (SRKW) review results; accessed via the `COSMOS_KEY` secret.
- **Orcasound S3** (`audio-orcasound-net`, region `us-west-2`) – HLS audio streams and wav segments.

## Key Environment Variables / Secrets

| Variable | Default | Purpose |
|---|---|---|
| `COSMOS_URL` | `https://aifororcasmetadatastore.documents.azure.com:443/` | Cosmos DB endpoint |
| `COSMOS_KEY` | *(required secret)* | Cosmos DB primary key |
| `COSMOS_DB` | `predictions` | Cosmos DB database name |
| `COSMOS_CONTAINER` | `metadata` | Cosmos DB container name |

## Coding Conventions

- **Language**: Python 3.11+
- **License header** (required at the top of every new source file):
  ```python
  # Copyright (c) PODS-AI contributors
  # SPDX-License-Identifier: MIT
  ```
- **Typing**: Use `dataclasses`, built-in generic types (`list`, `tuple`), and `typing` utilities (`Optional`) with type annotations throughout.
- **Docstrings**: All public functions and classes must have Google-style or plain docstrings describing parameters and return values.
- **Error handling**: Catch exceptions at I/O boundaries (network, file), print a descriptive error message, and return an empty list or `None` as appropriate—do not let exceptions propagate silently.
- **Constants**: Define module-level constants for magic values (e.g., `NEAR_MIN`, `MAX_DETECTION_PAGES`).
- **Comments**: Comments should end in punctuation (typically a period).

## pods-ai Pipeline

The scripts in `src/` are meant to be run in order:

1. `make_csv.py` – Queries Orcasite and OrcaHello APIs; writes `output/csv/detections.csv`.
2. `process_humpback_wavs.py` – Processes humpback signal files from `signals-humpback_*.wav`; extracts 2-second segments.
3. `extract_training_samples.py` – Reads `detections.csv`; writes `output/csv/training_samples.csv` (adjusts humpback count based on existing signal files).
4. `download_wavs.py` – Reads `training_samples.csv`; downloads wav files into subdirectories under `output/wav/`.
5. `make_spectrograms.py` – Generates a PNG spectrogram alongside each wav file.

Detection labels: `resident`, `transient`, `humpback`, `other`.  
Classification kinds: `tp_human_only`, `tp_machine_only`, `fp_machine_only`, `tp_both`, `skip`.

## CI / CD

- **check_csv.yml** – Runs on every PR; re-executes `make_csv.py` and `extract_training_samples.py` and asserts no diff in the committed CSV files (requires `COSMOS_KEY` secret).
- **validate-yaml.yml** – Runs `yamllint` against all YAML files using the rules in `.yamllint.yml` (line-length disabled, truthy check-keys disabled).
- **dependabot.yml** – Weekly updates for GitHub Actions dependencies.

## Dependencies

Install project requirements before running scripts:

```bash
pip install -r requirements.txt
```

Key packages: `azure-cosmos`, `boto3`, `librosa`, `ffmpeg-python`, `matplotlib`, `numpy`, `requests`, `pytz`.  
