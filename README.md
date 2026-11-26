# C147/247 Final Project
### Fall 2026 

This course project is built upon the emg2qwerty work from Meta. The first section of this README provides some guidance for working with the repo and contains a running list of FAQs. **Note that the rest of the README is from the original repo and we encourage you to take a look at their work.**

## Course project experiments

### What to run first

| If you want to… | Start with | Recorded result |
|---|---|---|
| Verify the setup or reproduce the reference | [TDS baseline](benchmarks/tds_baseline/) | **25.07% test CER** in the sweep rerun. This is the Colab default. |
| Try our best measured offline model | [TDS + BiGRU](benchmarks/tds_bigru_offline/) | **22.50% test CER** on the same held-out session. The backward GRU uses future frames. |
| Test whether more context helps | [Width-48 TDS](experiments/tds_longer_context/) | Tests **1.536 s** of context. A result has yet to be recorded in this repository. |
| Reproduce the completed comparison | [Context and depth sweep](experiments/tds_context_depth/) | Four cases. The width-32 baseline won on validation. |

These CERs are from one subject. Start with the baseline in smoke mode to
check the setup. Choose BiGRU when offline accuracy is the goal.

`benchmarks/` contains runnable reference configurations with recorded results.
`experiments/` contains research comparisons and follow-ups. Each folder has
an `experiment.yaml` that uses the same launcher and an explanation of what the
run is for. See the [experiment record](docs/tds-context-sweep.md) for training
commits, measured results, and artifact paths.

### Data split and default protocol

The course experiments use **subject 89335547** from the released emg2qwerty
dataset. There are **16 training sessions, one validation session, and one
held-out test session**. [`config/user/single_user.yaml`](config/user/single_user.yaml)
contains the exact IDs. Use the extracted directory containing the 18
`.hdf5` files as `--data`. On Colab the default input is
`My Drive/ECE247A_final_project/89335547.zip`. The session list and file
checksums are saved with each run. The data source is linked below under
[Setup](#setup).

The manifests use seed **1501**, **40 epochs**, batch size **32**, one GPU,
Adam with a 10-epoch warm-up followed by cosine decay, log spectrograms, and
greedy CTC decoding. Train and validation targets come from 4-second windows.
The width-48 case explicitly extends the input on the past side while
preserving the target interval. Select checkpoints and compare variants using
validation character error rate (CER). After all declared cases finish,
the runner evaluates the validation winner on the held-out full session.

### Run an experiment

Use the environment described under [Setup](#setup) or the pinned packages
in `requirements.txt` with Python 3.10 for the Colab workflow. From the repo
root on a clean checkout, run:

```sh
python scripts/run_experiment.py \
  --experiment benchmarks/tds_baseline/experiment.yaml \
  --data /path/to/hdf5-sessions \
  --output-root /path/to/experiment-results
```

Add `--plan` to print the exact commands without using a GPU or reading
recordings. Add `--check-data` in a separate invocation to verify the split
files. Use `--smoke` for two training batches and one validation batch per
case. It skips full-session testing and writes to a `-smoke` directory when
using `--output-root`. The regular run stores a `summary.md`, checkpoints,
resolved Hydra configurations, per-case logs, the source commit, file
checksums and package versions. Resume using the same commit, manifest,
environment, and output directory. For a new run with changed settings,
choose a new output root or exact `--output` path.

**Browser Colab.** Import [`Colab_setup.ipynb`](Colab_setup.ipynb), select a
hosted GPU, and set `EXPERIMENT` to the desired YAML path. It defaults to
`benchmarks/tds_baseline/experiment.yaml`. For published code, set `PROJECT_BUNDLE = None` and set
`PROJECT_REF` to the full Git commit SHA. To use a local commit, create
a Git bundle with `git bundle create emg2qwerty-experiments.bundle HEAD`
and put it next to `89335547.zip` in Drive. Run the settings and launch
cells. Reports are saved under `ECE247A_final_project/experiments/manifest-runs`
and a successful full run releases the GPU. Release any idle runtime after
a stopped or failed job.

### Add and merge experiments

1. Branch from the agreed integration commit. Add a unique folder under
   `experiments/` and copy a nearby manifest. Give it a unique `id`. Put
   common Hydra overrides in `overrides`, case-specific overrides in
   `cases`, and optional validation comparisons in `comparisons`.
2. Put experiment-specific model code in that folder if you need it. Use a
   Python-friendly folder name, for example `my_encoder`. A case can select
   its class through a Hydra override such as
   `module._target_=experiments.my_encoder.model.MyEncoder`. Use a unique
   local Hydra config group entry if extra configuration is needed. A new
   manifest needs no launcher registry change.
3. Keep model-specific options in the manifest or its directory. For
   shared changes, preserve the existing defaults and run the neighboring
   baseline and BiGRU checks. The harness expects a Lightning model with
   `forward(inputs, input_lengths=...)`, validation metric `val/CER`,
   test metric `test/CER`, and an output checkpoint selected by validation.
4. Commit the manifest, model code and results README. Store recordings, logs,
   and checkpoints outside Git. A results README should say which commit,
   split, seed, and hardware produced it and link to the artifacts.
5. When integrating, preserve separate experiment folders and their provenance.
   Resolve overlap in shared code explicitly and verify both manifests before
   merging. See [AGENTS.md](AGENTS.md) for agent-specific guidance.

## Guiding Tips + FAQs
_Last updated 2/13/2025_
- Read through the Project Guidelines to ensure that you have a clear understanding of what we expect
- Familiarize yourself with the prediction task and get a high-level understanding of their base architecture (it would be beneficial to read about CTC loss)
- Get comfortable with the codebase
  - ```lightning.py``` + ```modules.py``` - where most of your model architecture development will take place
  - ```data.py``` - defines PyTorch dataset (likely will not need to touch this much)
  - ```transforms.py``` - implement more data transforms and other preprocessing techniques
  - ```config/*.yaml``` - modify model hyperparameters and PyTorch Lightning training configuration
    - **Q: How do we update these configuration files?** A: Note the structure of YAML files include basic key-value pairs (i.e. ```<key>: <value>```) and hierarchical structure. So, for instance, if we wanted to update the ```mlp_features``` hyperparameter of the ```TDSConvCTCModule```, we would change the value at line 5 of ```config/model/tds_conv_ctc.yaml``` (under ```module```). _Read more details [here](https://pytorch-lightning.readthedocs.io/en/1.3.8/common/lightning_cli.html)._
    - **Q: Where do we configure data splitting?** A: Refer to ```config/user/single_user.yaml```. Be careful with your edits, so that you don't accidentally move the test data into your training set.

# emg2qwerty
[ [`Paper`](https://arxiv.org/abs/2410.20081) ] [ [`Dataset`](https://fb-ctrl-oss.s3.amazonaws.com/emg2qwerty/emg2qwerty-data-2021-08.tar.gz) ] [ [`Blog`](https://ai.meta.com/blog/open-sourcing-surface-electromyography-datasets-neurips-2024/) ] [ [`BibTeX`](#citing-emg2qwerty) ]

A dataset of surface electromyography (sEMG) recordings while touch typing on a QWERTY keyboard with ground-truth, benchmarks and baselines.

<p align="center">
  <img src="https://github.com/user-attachments/assets/71a9f361-7685-4188-83c3-099a009b6b81" height="80%" width="80%" alt="alt="sEMG recording" >
</p>

## Setup

```shell
# Install [git-lfs](https://git-lfs.github.com/) (for pretrained checkpoints)
git lfs install

# Clone the repo, setup environment, and install local package
git clone git@github.com:joe-lin-tech/emg2qwerty.git ~/emg2qwerty 
cd ~/emg2qwerty
conda env create -f environment.yml
conda activate emg2qwerty
pip install -e .

# Download the dataset, extract, and symlink to ~/emg2qwerty/data
cd ~ && wget https://fb-ctrl-oss.s3.amazonaws.com/emg2qwerty/emg2qwerty-data-2021-08.tar.gz
tar -xvzf emg2qwerty-data-2021-08.tar.gz
ln -s ~/emg2qwerty-data-2021-08 ~/emg2qwerty/data
```

## Data

The dataset consists of 1,136 files in total - 1,135 session files spanning 108 users and 346 hours of recording, and one `metadata.csv` file. Each session file is in a simple HDF5 format and includes the left and right sEMG signal data, prompted text, keylogger ground-truth, and their corresponding timestamps. `emg2qwerty.data.EMGSessionData` offers a programmatic read-only interface into the HDF5 session files.

To load the `metadata.csv` file and print dataset statistics,

```shell
python scripts/print_dataset_stats.py
```

<p align="center">
  <img src="https://user-images.githubusercontent.com/172884/131012947-66cab4c4-963c-4f1a-af12-47fea1681f09.png" alt="Dataset statistics" height="50%" width="50%">
</p>

To re-generate data splits,

```shell
python scripts/generate_splits.py
```

The following figure visualizes the dataset splits for training, validation and testing of generic and personalized user models. Refer to the paper for details of the benchmark setup and data splits.

<p align="center">
  <img src="https://user-images.githubusercontent.com/172884/131012465-504eccbf-8eac-4432-b8aa-0e453ad85b49.png" alt="Data splits">
</p>

To re-format data in [EEG BIDS format](https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/03-electroencephalography.html),

```shell
python scripts/convert_to_bids.py
```

## Training

Generic user model:

```shell
python -m emg2qwerty.train \
  user=generic \
  trainer.accelerator=gpu trainer.devices=8 \
  --multirun
```

Personalized user models:

```shell
python -m emg2qwerty.train \
  user="single_user" \
  trainer.accelerator=gpu trainer.devices=1
```

If you are using a Slurm cluster, include "cluster=slurm" override in the argument list of above commands to pick up `config/cluster/slurm.yaml`. This overrides the Hydra Launcher to use [Submitit plugin](https://hydra.cc/docs/plugins/submitit_launcher). Refer to Hydra documentation for the list of available launcher plugins if you are not using a Slurm cluster.

## Testing

Greedy decoding:

```shell
python -m emg2qwerty.train \
  user="glob(user*)" \
  checkpoint="${HOME}/emg2qwerty/models/personalized-finetuned/\${user}.ckpt" \
  train=False trainer.accelerator=cpu \
  decoder=ctc_greedy \
  hydra.launcher.mem_gb=64 \
  --multirun
```

Beam-search decoding with 6-gram character-level language model:

```shell
python -m emg2qwerty.train \
  user="glob(user*)" \
  checkpoint="${HOME}/emg2qwerty/models/personalized-finetuned/\${user}.ckpt" \
  train=False trainer.accelerator=cpu \
  decoder=ctc_beam \
  hydra.launcher.mem_gb=64 \
  --multirun
```

The 6-gram character-level language model, used by the first-pass beam-search decoder above, is generated from [WikiText-103 raw dataset](https://huggingface.co/datasets/wikitext), and built using [KenLM](https://github.com/kpu/kenlm). The LM is available under `models/lm/`, both in the binary format, and the human-readable [ARPA format](https://cmusphinx.github.io/wiki/arpaformat/). These can be regenerated as follows:

1. Build kenlm from source: <https://github.com/kpu/kenlm#compiling>
2. Run `./scripts/lm/build_char_lm.sh <ngram_order>`

## License

emg2qwerty is CC-BY-NC-4.0 licensed, as found in the LICENSE file.

## Citing emg2qwerty

```
@misc{sivakumar2024emg2qwertylargedatasetbaselines,
      title={emg2qwerty: A Large Dataset with Baselines for Touch Typing using Surface Electromyography},
      author={Viswanath Sivakumar and Jeffrey Seely and Alan Du and Sean R Bittner and Adam Berenzweig and Anuoluwapo Bolarinwa and Alexandre Gramfort and Michael I Mandel},
      year={2024},
      eprint={2410.20081},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.20081},
}
```

## Single-user TDS context and depth experiment

The [experiment write-up](docs/tds-context-sweep.md) records the four-way
context and depth comparison, earlier CNN and BiGRU results, and instructions
to reproduce the sweep and recover its artifacts.

## CNN-BiGRU experiment

The optional `cnn_bigru_ctc` model keeps the single-user splits, spectrogram
preprocessing, CTC loss, greedy decoding, and metrics used by the TDS baseline.
It applies a small temporal convolution and a two-layer bidirectional GRU. The
baseline remains the default. Select the new architecture with the Hydra
argument `model=cnn_bigru_ctc`; see `config/model/cnn_bigru_ctc.yaml` for its
size settings. Use a published branch or commit containing this file when
running on Colab; local source files are not automatically synchronized.

Start with a short run before committing GPU credits, then compare validation
CER and runtime under the same data splits and decoding settings. The model
uses future context and is therefore intended for offline evaluation. The
existing test loader evaluates an entire session at once: memory and runtime
on a T4 have not yet been measured, so test that path before a long run. Do not
compare truncated test sessions as if they were the full baseline test.

## TDS-BiGRU experiment

Select `model=tds_bigru_ctc` to compare the baseline TDS front end (including
four TDS blocks) with an added two-layer bidirectional GRU. The defaults retain
the baseline's wrist feature size, preprocessing, data splits, CTC loss, and
greedy decoder. The GRU has 128 hidden units per direction, matching the
smaller CNN-BiGRU model. The TDS layers shorten each spectrogram by 124 frames;
sequence lengths are adjusted before packing and computing CTC loss.

This is an offline model because the backward GRU uses future context. The
[benchmark folder](benchmarks/tds_bigru_offline/) records the completed run and a
manifest for repeating it.
