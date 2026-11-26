# Four-block TDS baseline

The original encoder uses four width-32 time-depth separable convolution
blocks. Each output sees 125 spectrogram frames covering 1.024 seconds of raw
signal. The prediction head scores keys and the CTC blank. The greedy
decoder turns the emissions into a key sequence.

| Manifest | Cases | Validation CER | Held-out test CER |
|---|---|---:|---:|
| [experiment.yaml](experiment.yaml) | The original four-block, width-32 baseline | 23.90% | 25.18% |
| [context-depth.yaml](context-depth.yaml) | Four-case TDS depth and context sweep. Results shown for the width-32 winner. | 23.75% | 25.07% |

The sweep completed at training commit `d4e5e0c`. The earlier baseline
settings were recorded in the November 21 archive. See the
[experiment record](../../docs/tds-context-sweep.md) for the four variants,
archive names, timings, and conclusions. Select either manifest in the
[shared launcher](../../README.md#run-an-experiment).
