# TDS with a bidirectional GRU

This model keeps the four-block, width-32 TDS frontend and adds two stacked
bidirectional GRU layers with 128 hidden units per direction and 0.1 dropout
between layers. The 1.024 seconds in the manifest describes the convolutional
frontend. The GRU can use the supplied sequence in both directions.

Run [experiment.yaml](experiment.yaml) with the [shared launcher](../../README.md#run-an-experiment).
The earlier run reached **21.93% validation CER** and **22.50% test CER**.
Its implementation was committed as `9687b95`. See the
[experiment record](../../docs/tds-context-sweep.md#earlier-project-runs)
for the run archive. Validation uses windows and test uses a full session,
so this setup is for offline evaluation.
