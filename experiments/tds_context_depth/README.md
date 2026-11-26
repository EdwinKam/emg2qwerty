# TDS context and depth sweep

**Completed.** This experiment crossed two and four TDS blocks with
512 ms and 1,024 ms of raw signal context. It compared accuracy and forward
time on the shared single-user split.

Run [experiment.yaml](experiment.yaml) with the
[shared launcher](../../README.md#run-an-experiment) to repeat all four cases.
The width-32, four-block [baseline](../../benchmarks/tds_baseline/) won on
validation at **23.75% CER** and scored **25.07% CER** on the held-out session.
See the [experiment record](../../docs/tds-context-sweep.md#results-from-november-2526-2026)
for every result, timing, and training commit.
