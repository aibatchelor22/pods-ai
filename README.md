# Programmatic Orca Detection System using Artificial Intelligence (PODS-AI)

This repository contains scripts for preparing training data for orca detection models.

Bootstrap-only generation scripts and archived CSV inputs now live under [`bootstrap/`](bootstrap/README.md).

## Overview

The ongoing sample CSVs are:

- `output/csv/training_3s_samples.csv`
- `output/csv/testing_60s_samples.csv`

Both files can be updated manually by editing rows directly, or via scripts (for example
`add_samples.py`, `process_false_positives.py`, and `process_false_negatives.py`).

The active scripts in `src` include:

1. **download_wavs.py**: Uses `output/csv/training_3s_samples.csv` and `output/csv/testing_60s_samples.csv` to download wav files.
2. **make_spectrograms.py**: Creates a png file for each wav file in a subdirectory of `output/png`.
3. **train_podsai_model.py**: Trains a PODS-AI model on the generated training samples.
4. **compare_models.py**: Evaluates models on `output/csv/testing_60s_samples.csv`.

```mermaid
flowchart TD;
    podsaiModel[(HuggingFace davethaler/whale-call-detector)];
    orcaHelloModel[(HuggingFace orcasound/orcahello-srkw-detector-v1)];
    trainingSamples@{ shape: doc, label: "training_3s_samples.csv" };
    testingSamples@{ shape: doc, label: "testing_60s_samples.csv" };
    wav@{ shape: docs, label: "wav/*" };
    testingWav@{ shape: docs, label: "testing-wav/*" };
    concatenated@{ shape: docs, label: "concatenated.wav" };

    downloadWavs@{ shape: rect, label: "download_wavs.py" };
    trainPodsaiModel@{ shape: rect, label: "train_podsai_model.py" };
    compareModels@{ shape: rect, label: "compare_models.py" };
    concatenateWavs@{ shape: rect, label: "concatenate_wavs.py" };

    trainingSamples-->downloadWavs-->wav;
    testingSamples-->downloadWavs-->testingWav;

    wav-->trainPodsaiModel-->podsaiModel;

    podsaiModel-->compareModels;
    testingSamples-->compareModels;
    testingWav-->compareModels;
    orcaHelloModel-->compareModels;

    wav-->concatenateWavs-->concatenated;
```

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `boto3`: For accessing S3 audio files
- `ffmpeg-python`: For audio processing
- `librosa>=0.10.0`: For audio analysis
- `m3u8`: For HLS stream parsing
- `pytz`: For timezone handling
- `fastai==1.0.61`: For FastAI model support
- `torch>=2.1.0`: PyTorch deep learning framework
- `torchvision>=0.16.0`: Computer vision models and utilities
- `torchaudio>=2.1.0`: Audio processing for PyTorch
- `soundfile`: Audio file I/O
- `fastai_audio`: FastAI audio extensions (from GitHub)
- `pandas`, `pydub`: Data processing and audio manipulation

## Helper Scripts

- **spectrogram_visualizer.py**: Adapted from [aifororcas-livesystem](https://github.com/orcasound/aifororcas-livesystem/blob/main/InferenceSystem/src/spectrogram_visualizer.py)
- **model_inference.py**: Provides model inference interface for scoring audio samples
- **orcasite_feeds.py**: Lightweight module providing the `OrcasiteFeed` dataclass and
  `get_orcasite_feeds()` helper. Depends only on `requests` — no `azure-cosmos` — so
  scripts that only need the feeds REST API (e.g. `add_samples.py`) can import it
  without pulling in the full `make_csv` dependency tree.
- **add_samples.py**: Splits a WAV file into 3-second segments (2-second hop), saves each
  segment to a `new/` directory using the standard filename convention, and prints the
  predicted class for each segment. Useful for labelling new recordings and adding them
  to the training set. See [add_samples.py](#add_samplespy) below.
- **concatenate_wavs.py**: Concatenates WAV files in a directory into a single output file,
  adding a short beep between clips to make quick listen-through review easier. See
  [concatenate_wavs.py](#concatenate_wavspy) below.
- **process_false_positives.py**: Re-checks rejected OrcaHello detections by
  downloading the 60-second WAV, re-running PODS-AI, and appending whale-class
  sub-segments with corrected classes to `output/csv/training_3s_samples.csv`.
  The corrected class is inferred from the human-authored portion of the moderation
  comments (auto-generated "AI: …" lines are ignored).  Explicit negations in the
  comments are understood: "No humpback" suppresses the humpback match, and
  "No humpback nor vessel" resolves the corrected class to `water`.
  Supports `--category CATEGORY` to process only detections whose inferred
  actual category matches the provided value.
- **process_false_negatives.py**: Re-checks confirmed OrcaHello detections by
  downloading the 60-second WAV, re-running PODS-AI and OrcaHello segment inference,
  and appending segments where OrcaHello predicts resident but PODS-AI does not to
  `output/csv/training_3s_samples.csv` with corrected class `resident`. Supports
  `--category CATEGORY` to process only detections whose PODS-AI predicted category
  matches the provided value.
- **run_inference.py**: Runs a model on a wav file and prints the global prediction,
  confidence, and per-class probabilities.
- **LiveInferenceOrchestrator.py**: Runs live/date-range HLS inference with the multiclass
  PODS-AI model and can upload positive detections (resident/transient/humpback)
  to Azure Blob Storage and Cosmos DB.
- **compare_models.py**: Evaluates and compares fastai, orcahello, podsai (AST), and oldpodsai (Wav2Vec2) models
  on the test set loaded from `output/csv/testing_60s_samples.csv` and downloaded by `download_wavs.py`).
  Reports correct identifications, false positives, false negatives, and average prediction time for each model.

### add_samples.py

Split a WAV recording into 3-second segments (with a 2-second hop — the same settings
used by `run_inference.py`), save each segment to a `new/` directory using the standard
filename convention, and print the predicted class for each segment.  The timestamp
encoded in each filename reflects the **actual start time** of that sample inside the
original recording.

Output files follow the same naming convention as `output/wav/humpback/` etc.:

```
{node_name_with_hyphens}_{YYYY_MM_DD_HH_MM_SS_PST}.wav
```

Inference always uses the **PODS-AI (podsai)** model type.  The default model is
`davethaler/whale-call-detector` on HuggingFace Hub; override with `--model-path`.

If `--node-name` and `--timestamp` are omitted, the script infers them from the input
filename.  The filename must follow the same convention:
`{node_name_with_hyphens}_{YYYY_MM_DD_HH_MM_SS_PST}.wav`
(e.g. `rpi-orcasound-lab_2025_12_17_22_34_03_PST.wav` → node `rpi_orcasound_lab`,
timestamp `2025_12_17_22_34_03_PST`).

After reviewing the predictions you can move the segments into the appropriate
`output/wav/<category>/` directory to add them to the training set.

```
usage: python add_samples.py <wav_file> [--node-name NAME] [--timestamp TIMESTAMP]
                             [--output-dir DIR] [--model-path PATH] [--uri URI]
```

| Argument | Description |
|---|---|
| `wav_file` | Path to the input WAV file to segment |
| `--node-name` | Hydrophone node name (e.g. `rpi_orcasound_lab`). Underscores are replaced with hyphens in output filenames. **Inferred from the input filename if omitted.** |
| `--timestamp` | PST timestamp of the **start** of the recording (e.g. `2025_01_15_12_30_00_PST`). **Inferred from the input filename if omitted.** |
| `--output-dir` | Directory to save segments (default: `new`) |
| `--model-path` | HuggingFace Hub model ID or path to a local podsai model directory (default: `davethaler/whale-call-detector`) |
| `--uri` | Optional custom URI to use for all segments. If provided, all output rows will use this URI instead of generating one per segment. Useful when all segments come from the same detection. |

**Example — node name and timestamp inferred from filename**

```bash
cd src
python add_samples.py rpi-orcasound-lab_2025_01_15_12_30_00_PST.wav
```

**Example — explicit node name and timestamp with custom model**

```bash
cd src
python add_samples.py /path/to/recording.wav \
    --node-name rpi_orcasound_lab \
    --timestamp 2025_01_15_12_30_00_PST \
    --model-path /path/to/local-model
```


**Example — use custom URI for all segments**

When all segments come from the same detection event, you can specify a single URI
to use for all output rows:

```bash
cd src
python add_samples.py /path/to/recording.wav \
    --node-name rpi_orcasound_lab \
    --timestamp 2025_01_15_12_30_00_PST \
    --uri "https://live.orcasound.net/bouts/new/rpi_orcasound_lab?time=2025-01-15T20%3A30%3A00.000Z"
```

Output:
```
Saved: new/rpi-orcasound-lab_2025_01_15_12_30_00_PST.wav
Saved: new/rpi-orcasound-lab_2025_01_15_12_30_02_PST.wav
Saved: new/rpi-orcasound-lab_2025_01_15_12_30_04_PST.wav
...

Loading podsai model from /path/to/local-model...

Segment predictions:
  rpi-orcasound-lab_2025_01_15_12_30_00_PST.wav: water
  rpi-orcasound-lab_2025_01_15_12_30_02_PST.wav: resident
  rpi-orcasound-lab_2025_01_15_12_30_04_PST.wav: resident
  ...
```

### concatenate_wavs.py

Concatenate all WAV files in a directory into a single WAV file with a short beep
between clips.

```bash
cd src
python concatenate_wavs.py <directory> [--output OUTPUT_FILENAME]
```

Example:

```bash
cd src
python concatenate_wavs.py ../output/wav/resident --output concatenated.wav
```

### run_inference.py

Run model inference on a wav file and display the global prediction, confidence score,
and per-class probabilities.  For PODS-AI models the per-class probability is the
mean of all `local_confidence` values (from windows predicting that class) that exceed
the model's threshold — the same statistic used for `global_confidence`.  For the FastAI
binary model, `resident = global_confidence` and `other = 1 - global_confidence`.

```
usage: python run_inference.py [wav_file]
       [--node-name NODE_NAME]
       [--end-timestamp-str YYYY_MM_DD_HH_MM_SS_PST | --start-timestamp-utc YYYY-MM-DDTHH:MM:SSZ]
       [--model {podsai,fastai,orcahello}] [--type {ast,wav2vec2}] [--model-path PATH]
```

| Argument | Description |
|---|---|
| `wav_file` | Path to the wav file to score |
| `--node-name` | Hydrophone feed node name (for download mode) |
| `--end-timestamp-str` | PST **end** timestamp used with `--node-name` (format: `YYYY_MM_DD_HH_MM_SS_PST`) |
| `--start-timestamp-utc` | UTC **start** timestamp used with `--node-name` (format: `YYYY-MM-DDTHH:MM:SSZ`) |
| `--model` | Model type: `podsai` (default), `fastai`, or `orcahello` |
| `--type` | PODS-AI model variant used with `--model podsai`: `ast` (default) or `wav2vec2` (older model variant). These map to the currently pinned revisions in `src/run_inference.py` |
| `--model-path` | Path to model directory or HuggingFace Hub model ID. Required for `podsai`; defaults to `./model` for `fastai`; defaults to `orcasound/orcahello-srkw-detector-v1` for `orcahello`; defaults to `davethaler/whale-call-detector` for `podsai` |

When using `--node-name`, provide exactly one timestamp argument:
`--end-timestamp-str` or `--start-timestamp-utc`.

**Example — PODS-AI model**

```bash
cd src
python run_inference.py sample.wav --model podsai
```

Output:
```
Model type: podsai
Global prediction: resident (confidence: 0.7000)
Prediction time: 1.23s

Per-class probabilities:
  humpback: 0.0000
  human: 0.0000
  jingle: 0.0000
  resident: 0.7000
  transient: 0.0000
  vessel: 0.0000
  water: 0.0000
```

**Example — FastAI model**

```bash
cd src
python run_inference.py sample.wav --model fastai --model-path ../model
```

Output:
```
Model type: fastai
Global prediction: resident (confidence: 0.7500)
Prediction time: 0.85s

Per-class probabilities:
  other: 0.2500
  resident: 0.7500
```

**Example — OrcaHello SRKW Detector**

Uses the [`orcasound/orcahello-srkw-detector-v1`](https://huggingface.co/orcasound/orcahello-srkw-detector-v1)
model from HuggingFace Hub. This is a binary SRKW (Southern Resident Killer Whale) detector
based on the new OrcaHello inference pipeline (ResNet50 + mel spectrograms, no fastai_audio dependency).

The model implementation is loaded from the `orcasound/orcahello` submodule. Initialize it first:

```bash
git submodule update --init external/orcahello
```

Then run inference:

```bash
cd src
python run_inference.py sample.wav --model orcahello
```

Output:
```
Model type: orcahello
Global prediction: resident (confidence: 0.8000)
Prediction time: 0.92s

Per-class probabilities:
  other: 0.2000
  resident: 0.8000
```

You can compare results between models by running each on the
same file and comparing the output.

### compare_models.py

Evaluate and compare fastai, orcahello, podsai (AST), and oldpodsai (Wav2Vec2) models on the same test set of
60-second audio samples.  Loads the test set directly from `output/csv/testing_60s_samples.csv`, then runs each enabled model on the
corresponding WAV files under `output/testing-wav/`
(downloaded by `download_wavs.py`), and reports a summary table
with correct identifications, whale-class F1, per-whale-class false positive/false negative rates,
and average prediction time.

Evaluation uses model-specific correctness plus per-whale-class error counts:
- **Correct** – for `fastai` and `orcahello`, model predicted "resident" (SRKW) when the label is
  `resident`, or anything other than `resident` when the label is not `resident`; for
  `oldpodsai` and `podsai`, the predicted category exactly matches the label.
- **F1** – macro F1 over the whale classes `humpback`, `resident`, and `transient` that are
  present in the evaluated samples.
- **R/T/H false positive** – model predicted `resident`, `transient`, or `humpback`
  when the correct label was a different class.
- **R/T/H false negative** – the correct label was `resident`, `transient`, or `humpback`,
  but the model predicted a different class. Because `fastai` and `orcahello` are binary
  resident-vs-other models, their transient/humpback FP% values stay at `0.0%` and their
  transient/humpback FN% values are `100.0%` whenever those classes are present.
- `compare_models.py` evaluates end-to-end 60-second WAV inference from `output/testing-wav`, so
  its results will differ from the training workflow's held-out evaluation metrics, which score the
  model directly on the trainer's test split.

```
usage: python compare_models.py [--testing-csv PATH] [--max-samples N]
                                [--wav-dir PATH] [--models MODEL_LIST]
                                [--fastai-model-path PATH]
                                [--orcahello-model-path PATH]
                                [--podsai-model-path PATH]
                                [--category CATEGORY]
```

| Argument | Description |
|---|---|
| `--testing-csv` | Path to `testing_60s_samples.csv` (default: `output/csv/testing_60s_samples.csv`) |
| `--max-samples` | Maximum number of test samples to process. If not specified, all samples are processed |
| `--wav-dir` | Root directory of testing WAV files (default: `output/testing-wav`) |
| `--models` | Comma-separated list of models to evaluate (default: `fastai,orcahello,podsai,oldpodsai`) |
| `--fastai-model-path` | Path to FastAI model directory. Defaults to `model` when not specified |
| `--orcahello-model-path` | HuggingFace Hub ID or path for OrcaHello model. Defaults to `orcasound/orcahello-srkw-detector-v1` when not specified |
| `--podsai-model-path` | Path or Hub ID for PODS-AI model. Used by both `podsai` (AST) and `oldpodsai` (Wav2Vec2). Defaults to `davethaler/whale-call-detector` when not specified |
| `--category` | Only evaluate samples from this category (e.g. `resident`, `humpback`, `water`). If not specified, all categories are evaluated |

**Example — compare all four models**

```bash
python src/compare_models.py \
    --models fastai,orcahello,podsai,oldpodsai \
    --fastai-model-path model \
    --podsai-model-path /path/to/podsai-model
```

Example output layout (actual metric values vary with the evaluated dataset):
```
Loaded 160 test samples from output\csv\testing_60s_samples.csv
WAV directory: output/testing-wav
Models to evaluate: fastai, orcahello, podsai, oldpodsai

  ...

==============================================================================================================================================
Model Comparison Summary
==============================================================================================================================================
Model           Evaluated   Correct  Accuracy      F1    RFP%    RFN%    TFP%    TFN%    HFP%    HFN%   Avg Time
----------------------------------------------------------------------------------------------------------------------------------------------
fastai                160        68     42.5%   0.129   61.0%   51.7%    0.0%  100.0%    0.0%  100.0%     11.40s
orcahello             160        41     25.6%   0.128   96.0%   38.3%    0.0%  100.0%    0.0%  100.0%      4.97s
oldpodsai             160        88     55.0%   0.492   21.0%   45.0%   14.6%   63.3%   15.4%   40.0%      4.49s
podsai                160       104     65.0%   0.581   20.0%   35.0%    0.0%   73.3%    6.2%   36.7%      6.77s
==============================================================================================================================================

Definitions:
  Accuracy     = Correct / Evaluated
  Correct      = fastai/orcahello: resident vs other; oldpodsai/podsai: exact category match
  F1           = macro F1 over humpback, resident, and transient classes that are present
  [R|T|H]FP%   = among non-[R|T|H] samples, fraction predicted as that class
  [R|T|H]FN%   = among actual samples of that class, fraction predicted as another class
  Avg Time     = average time spent in model predict() per 60-second WAV file

Confusion Matrix for fastai (rows=actual, cols=predicted):
                 other   resident      total
      human          6          4         10
   humpback         17         13         30
     jingle          8          2         10
   resident         31         29         60
  transient          4         26         30
     vessel          4          6         10
      water          0         10         10

Confusion Matrix for orcahello (rows=actual, cols=predicted):
                 other   resident      total
      human          0         10         10
   humpback          5         25         30
     jingle          0         10         10
   resident         23         37         60
  transient          0         30         30
     vessel          0         10         10
      water          0         10         10

Confusion Matrix for oldpodsai (rows=actual, cols=predicted):
                 human   humpback     jingle   resident  transient     vessel      water      total
      human          7          1          0          1          1          0          0         10
   humpback          1         18          0          8          3          0          0         30
     jingle          0          8          2          0          0          0          0         10
   resident          6          2          0         33         15          1          3         60
  transient          1          9          0          9         11          0          0         30
     vessel          0          0          0          3          0          7          0         10
      water          0          0          0          0          0          0         10         10

Confusion Matrix for podsai (rows=actual, cols=predicted):
                 human   humpback     jingle   resident  transient     vessel      water      total
      human          9          1          0          0          0          0          0         10
   humpback          0         19          0          3          0          7          1         30
     jingle          0          0          9          0          0          1          0         10
   resident          1          3          0         39          0         12          5         60
  transient          0          4          0         17          8          1          0         30
     vessel          0          0          0          0          0         10          0         10
      water          0          0          0          0          0          0         10         10
```

**Example - compare only fastai and orcahello**

```bash
python src/compare_models.py --models fastai,orcahello --fastai-model-path model
```

**Example - limit to 10 test samples**

```bash
python src/compare_models.py --max-samples 10 --fastai-model-path model
```

**Example - evaluate only resident samples**

```bash
python src/compare_models.py --category resident --fastai-model-path model
```

## LiveInferenceSystem Container

`LiveInferenceSystem/` packages `src/LiveInferenceOrchestrator.py` as a Docker container for
production deployment to Azure Kubernetes Service (AKS), following the same pattern used by
[OrcaHello's InferenceSystem](https://github.com/orcasound/orcahello/tree/main/InferenceSystem).
The two containers can run side-by-side in the same Kubernetes cluster without conflicts.

### Quick Start

Build the image from the repo root (requires the `external/orcahello` submodule):

```bash
git submodule update --init external/orcahello
docker build -f LiveInferenceSystem/Dockerfile -t pods-ai-live-inference-system .
```

> **macOS M-series:** prefix with `docker buildx build --platform linux/amd64`

Run locally by mounting an orchestrator config at `/config/config.yml`:

```bash
# Linux/Mac
docker run --rm -it --env-file .env \
  -v $PWD/LiveInferenceSystem/tests/orch_configs/LiveHLS/LiveHLS_OrcasoundLab.yml:/config/config.yml \
  pods-ai-live-inference-system \
  --max_live_iterations 2

# Windows
docker run --rm -it --env-file .env ^
  -v %cd%/LiveInferenceSystem/tests/orch_configs/LiveHLS/LiveHLS_OrcasoundLab.yml:/config/config.yml ^
  pods-ai-live-inference-system ^
  --max_live_iterations 2
```

The `.env` file should contain Azure credentials (see `LiveInferenceOrchestrator.py` for required
environment variables).

### Deployment

In production each hydrophone location runs as a separate deployment in its own Kubernetes
namespace.  The `LiveInferenceSystem/deploy/` directory contains the Kubernetes manifests:

- `<location>.yaml` — deployment spec
- `<location>-configmap.yaml` — hydrophone-specific orchestrator configuration

To release a new container image, push a tag of the form `LiveInferenceSystem.v#.#.#`.
This triggers the `LiveInferenceSystem-deploy` workflow, which builds the image and pushes it to
`orcaconservancycr.azurecr.io/pods-ai-live-inference-system`.

To deploy to a hydrophone location:

```bash
NAMESPACE=orcasound-lab  # or andrews-bay, bush-point, etc.
kubectl apply -f LiveInferenceSystem/deploy/$NAMESPACE-configmap.yaml
# Scale to 0 first — required by the Recreate strategy on memory-constrained nodes
# so that the old pod is fully terminated before the new pod starts.
kubectl scale deployment pods-ai-inference-system -n $NAMESPACE --replicas=0
kubectl apply -f LiveInferenceSystem/deploy/$NAMESPACE.yaml
```

To add a new hydrophone location, create `deploy/<namespace>-configmap.yaml` and
`deploy/<namespace>.yaml` using an existing pair as a template, then create the namespace and
secret:

```bash
kubectl create namespace <namespace>
kubectl create secret generic pods-ai-inference-system -n <namespace> \
    --from-literal=AZURE_COSMOSDB_PRIMARY_KEY='<key>' \
    --from-literal=AZURE_STORAGE_CONNECTION_STRING='<string>' \
    --from-literal=INFERENCESYSTEM_APPINSIGHTS_CONNECTION_STRING='<string>'
```

## Architecture

The timestamp correction implementation follows the architecture described in the [aifororcas-livesystem](https://github.com/orcasound/aifororcas-livesystem):

- Uses `DateRangeHLSStream` approach to download audio from specific time ranges
- Downloads from Orcasound S3 buckets: `s3-us-west-2.amazonaws.com/audio-orcasound-net/`
- Processes HLS streams with m3u8 playlists
- Uses FFmpeg for audio format conversion
- Returns `local_confidences` array with scores for each segment

## Example Configuration

Similar to [aifororcas-livesystem config files](https://github.com/orcasound/aifororcas-livesystem/blob/main/InferenceSystem/config/Test/Positive/FastAI_DateRangeHLS_AndrewsBay.yml):

```yaml
model_type: "FastAI"
model_local_threshold: 0.5
model_global_threshold: 3
model_path: "./model"
model_name: "model.pkl"
```

## GitHub CI configuration

The following repository secrets must be configured using information obtained
from HuggingFace:

* HF_TOKEN — Get this from https://huggingface.co/settings/tokens after logging in as the account used to publish the model (e.g., "davethaler").  This is used by train_model.yml.

or from portal.azure.com:

* COSMOS_KEY — "aifororcasmetadatastore" CosmosDB account → "Keys" → "Read-only Keys" → primary key.  This is used by bootstrap make_csv.py and train_model.yml.
* AZURE_COSMOSDB_PRIMARY_KEY — "aifororcasmetadatastore" CosmosDB account → "Keys" → "Read-write Keys" → primary key.  This is used by LiveInferenceOrchestrator.py.
* AZURE_STORAGE_CONNECTION_STRING — "livemlaudiospecstorage" storage account. See the "Connection String" section in [these instructions](https://learn.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-python?tabs=connection-string%2Croles-azure-portal%2Csign-in-azure-cli&pivots=blob-storage-quickstart-scratch#authenticate-to-azure-and-authorize-access-to-blob-data).  This is used by LiveInferenceOrchestrator.py.
* INFERENCESYSTEM_APPINSIGHTS_CONNECTION_STRING — "InferenceSystemInsights" Application Insights → "Overview" → connection string.  This is used by LiveInferenceOrchestrator.py.
* ACR_USERNAME — "orcaconservancycr" Container registry → "Access keys" → "Username".  This is used by LiveInferenceSystem-deploy.yaml.
* ACR_PASSWORD — "orcaconservancycr" Container registry → "Access keys" → "password".  This is used by LiveInferenceSystem-deploy.yaml.
* ACR_REGISTRY — "orcaconservancycr" Container registry → "Access keys" → "Registry name".  This is used by LiveInferenceSystem-deploy.yaml.
* KUBE_CONFIG — This is used by LiveInferenceSystem-deploy-configmaps.yaml.  To obtain the KUBE_CONFIG value, run the following:

```
az aks get-credentials --resource-group LiveSRKWNotificationSystem --name inference-system-AKS --admin --file kubeconfig
```

This produces a file named `kubeconfig`, the contents of which can be used as the KUBE_CONFIG value.
