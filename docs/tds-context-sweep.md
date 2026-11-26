# TDS context and depth experiment

This experiment tests how much signal context helps recognize keystrokes, and whether four TDS blocks improve CER over two at the same context. It uses the convolution-only encoder and greedy CTC decoding. The BiGRU runs are separate references.

| Blocks | Kernel width | Spectrogram frames | Raw signal span |
|---:|---:|---:|---:|
| 4 | 32 | 125 | 1,024 ms |
| 4 | 16 | 61 | 512 ms |
| 2 | 63 | 125 | 1,024 ms |
| 2 | 31 | 61 | 512 ms |

The raw span is the 32 ms FFT window plus 8 ms for each additional frame. The count measures the model's receptive field. All cases keep 4 s target windows with 900 ms past and 100 ms future padding, the original training augmentations, the same single-user session split, Adam and the 10-epoch warmup followed by cosine decay. Defaults are seed **1501**, **40 epochs**, batch size **32**, and one GPU. The two depths also differ in parameter count and intermediate sequence lengths. Treat this as a comparison of architectural choices under the same training protocol.

The runner selects each case's best checkpoint by validation CER and runs the held-out test only on the global validation winner. The test session was used in earlier project runs. Report it as a held-out session result and use a fresh session for an independent final estimate if available. Saving the seed and package versions helps repeat the run, but the same GPU setup is needed for a close timing comparison.

## Results from November 25–26, 2026

All four cases completed on a single Tesla T4 using seed 1501 and training commit
`d4e5e0ce922476d248580c33954cb8db02e18a84`. CER is character error rate, with
lower values indicating fewer edits per reference character.

| Configuration | Raw signal span | Validation CER | Test CER | Trainable parameters | Median forward time |
|---|---:|---:|---:|---:|---:|
| Four blocks, width 32 | 1,024 ms | **23.75%** | **25.07%** | 5,293,315 | 3.61 ms |
| Four blocks, width 16 | 512 ms | 43.27% | — | 5,256,451 | 3.34 ms |
| Two blocks, width 63 | 1,024 ms | 32.61% | — | 2,923,603 | 2.38 ms |
| Two blocks, width 31 | 512 ms | 51.13% | — | 2,886,739 | 2.15 ms |

Only the validation winner was evaluated on the held-out test session. The
forward-pass timings use batch size one and one 497-frame spectrogram per pass.

- **Context had the larger effect.** Increasing the raw span from 512 to
  1,024 ms reduced validation CER by **19.52 percentage points** at four
  blocks and **18.52 points** at two blocks. Median forward time increased
  by 0.27 ms and 0.23 ms, respectively.
- **Depth also mattered.** Four blocks reduced validation CER by **7.86 points**
  at 512 ms and **8.86 points** at 1,024 ms. The deeper model also has more
  parameters and does more work per frame. At 1,024 ms it took an additional
  1.22 ms of median forward time.
- **Prioritize keeping about one second of context if reducing compute.** The
  long-context two-block model had 32.61% validation CER and a 2.38 ms
  forward pass. The short-context four-block model had 43.27% validation
  CER and a 3.34 ms forward pass.

The comparison used one seed and one subject, with one validation and one test
session. The measured gains establish what happened on this split. Their
magnitude may change on other sessions or users. There was no recurrent layer in
the sweep. Neither GRU hidden-state width nor the available context for a GRU
was varied. The benchmark measures computation over prepared features. Measure
end-to-end key emission delay separately for a streaming design.

The run manifest, logs, checkpoints, and machine-generated summary are in the
[Drive sweep folder](https://drive.google.com/drive/u/0/folders/18kfsMIYYFjkAOsMmGLukP8v64Mp3m4yH)
for collaborators who already have access.

## Run on Colab

1. Use the included `Colab_setup.ipynb` and a dedicated hosted GPU runtime. Put `tds-context-sweep.bundle` in your Drive project directory alongside the existing `89335547.zip`. The bundle contains Git history and source. The subject ZIP supplies the recordings.
2. Run the settings and training cells. Change `DRIVE_PROJECT` if your files are elsewhere. Authorize the Drive mount when Colab asks. Check that the first configuration starts before stepping away.
3. Open `experiments/tds-context-depth-seed1501/summary.md` in that Drive directory for partial or final results. After a successful sweep the notebook flushes Drive and releases the GPU. If a case failed, check the summary and the affected console log, then release the GPU once it is idle.
4. To resume, use the same bundle, extracted data and Drive output path. Completed cases are reused. A case interrupted during training can resume from its latest saved Lightning checkpoint. Use a separate directory for a changed commit, dataset, seed or package environment. Start only one sweep process per output directory.

To generate a transport bundle from this branch after committing changes:

```sh
git bundle create /path/to/tds-context-sweep.bundle HEAD refs/heads/codex/tds-context-sweep
git bundle verify /path/to/tds-context-sweep.bundle
```

## Run on another GPU machine

Set up the project as described in the repository README and use the same HDF5 recordings. The exact required session names are in `config/user/single_user.yaml`. You can use the released emg2qwerty dataset to obtain them. The bundle by itself is sufficient for the code used by this sweep. The sweep trains from scratch, so pretrained Git LFS weights are unnecessary.

```sh
GIT_LFS_SKIP_SMUDGE=1 git clone --branch codex/tds-context-sweep \
    /path/to/tds-context-sweep.bundle emg2qwerty-sweep
cd emg2qwerty-sweep
python scripts/tds_context_sweep.py \
    --data /path/to/extracted-sessions \
    --output /path/to/persistent-storage/tds-context-depth-seed1501
```

Use the same command in a persistent batch job or terminal session if you want it to survive your local terminal disconnecting. The runner logs each case to disk. It tries the following cases if one fails and skips winner testing until all four have validation results.

## Files to keep

| File or folder | What it captures |
|---|---|
| `summary.md` | Per-case validation CER, parameter count and model forward time, plus the context and depth comparisons and the selected test CER. |
| `manifest.json`, `packages.txt` | Git commit, run settings, exact split and SHA-256 of each recording, and installed package versions. |
| `tds-*/results.json`, `selected-test/results.json` | Structured CER and component rates, checkpoint paths and per-case timing samples. The existing implementation labels insertions and deletions in the prediction-to-reference direction. |
| `tds-*/checkpoints/` | The best and latest Lightning checkpoints for each model. |
| `tds-*/hydra_configs/`, `command.json`, `console.log`, `lightning_logs/` | Resolved run setup and training evidence for reproduction or diagnosis. Commands and logs are also retained for the selected test. |

Timing uses a fixed 497-frame spectrogram, batch size one, 10 warm-up passes, and the median of 30 synchronized forward passes. It excludes spectrogram preparation, the decoder, and the wait for incoming signal. Report wall-clock training duration separately from compute time and live typing delay.

## Earlier project runs

The project workspace already contains the following historical archives next to the repository. They include the Hydra configuration, logs, and trained checkpoints. The commit column points to the corresponding model implementation. The saved Colab jobs did not all run from those commits verbatim.

| Model | Implementation commit | Validation CER | Test CER | Archive relative to repository root |
|---|---|---:|---:|---|
| TDS baseline | `78837e4` | 23.90% | 25.18% | `../baslineResults_20261121-165141.zip` |
| CNN1 + BiGRU | `03f6bc7` | 71.29% | 76.72% | `../old/CNN1-biGRU_20261124-184038.zip` |
| TDS + BiGRU | `9687b95` | 21.93% | 22.50% | `../old/TDS-biGRU_20261124-214853.zip` |

See `historical-run-archives.sha256` for archive checksums. Keep these archives with the project artifacts outside Git. Commit `98c3a35` keeps the original notebook changes and its saved cell outputs.

## Follow-up experiments

1. **Confirm the context effect.** For the next sweep, test an intermediate span around 0.75 s at four blocks and repeat the 1.0 s baseline with another seed. To confirm the original comparison across seeds, repeat both endpoint configurations at the new seed. Reuse the matching case commands with only the intended settings and output paths changed. Repeat on another session or user to test whether the result generalizes.
2. **Decoder headroom.** Evaluate greedy versus CTC beam decoding on a fixed saved checkpoint. Choose beam settings using validation and report the added inference cost along with any change in CER. This needs evaluation rather than model retraining.
3. **Better temporal modeling.** With a frontend selected, compare it against the same frontend followed by a recurrent or attention encoder. For live typing, compare a forward GRU with a causal local-attention Conformer at a similar parameter budget and the same allowed lookahead. Measure time from key-down to emission in addition to CER.
4. **Use key timestamps to test the curriculum idea.** Compare conventional CTC training with the same model and training budget starting on well-spaced key windows before switching to the usual mix. Evaluate both on natural held-out typing. Group errors by inter-key gap to see whether short context mainly hurts fast or overlapping presses.
