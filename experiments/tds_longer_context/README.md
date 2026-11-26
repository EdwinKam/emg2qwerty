# TDS with 1.536 seconds of context

Tests whether extending the four-block TDS encoder beyond the original
1.024-second context improves character error rate (CER). The only case uses
width 48 in every block, for 189 spectrogram frames per output. The loader
adds 1,024 past raw samples so the encoder's output stays aligned with the
original target interval, including at the start of a session.

Run [experiment.yaml](experiment.yaml) with the [shared launcher](../../README.md#run-an-experiment).
Compare with the baseline's 23.75% validation CER and 25.07% held-out test
CER. Run the baseline on the same GPU when comparing timing. The result for
this configuration has not been recorded in this repository.
