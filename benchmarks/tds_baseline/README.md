# Four-block TDS baseline

**Start here to check the setup or reproduce the reference.** This is the default
in the shared Colab notebook. Run [experiment.yaml](experiment.yaml) with the
[shared launcher](../../README.md#run-an-experiment). Start with smoke mode if
you just want to verify the setup and artifact writes.

The original encoder uses four width-32 time-depth separable convolution
blocks. Each output sees 125 spectrogram frames covering 1.024 seconds of raw
signal. The prediction head scores keys and the CTC blank. The greedy
decoder turns the emissions into a key sequence.

| Recorded run | Validation CER | Held-out test CER |
|---|---:|---:|
| Original baseline | 23.90% | 25.18% |
| Baseline retrained in the four-case sweep | 23.75% | 25.07% |

The sweep completed at training commit `d4e5e0c`. See the
[experiment record](../../docs/tds-context-sweep.md) for the settings and
archives, or the [context and depth experiment](../../experiments/tds_context_depth/)
to repeat all four cases.
