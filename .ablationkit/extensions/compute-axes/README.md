# compute-axes

The pilot report's §5.4 ("Compute accounting: three axes, three answers") as a comparison view
(Runs → Compare, then **Compute-axis comparison**). Select a reference run (by default the base 8k
transformer) and one or more candidates, for example the three runs of study `poc-pilot-100m`.

For each candidate it takes the final validation point and reads the reference's validation curve
at the same x, on one of four axes:

| Axis | x at a validation step | From |
| --- | --- | --- |
| Equal tokens of text | `tokens_seen` logged at that step (base tokens) | metric history |
| Equal training FLOPs | tokens × FLOPs per base token | **your estimate**, typed into the view (no run records FLOPs) |
| Equal wall-clock | time of the training point at that step − time of the first logged training point | metric timestamps |
| Equal training global positions | step × `batch_size` × `sequence_length` | params |

Rules:

- Reference values between validation points are linearly interpolated, never extrapolated. Out of
  range is shown as such.
- They come from one run's curve before its learning rate annealed, so they are biased in the
  candidates' favour by an unknown amount (as the report notes). There is one seed per configuration.
- The wall-clock axis excludes data preparation, compilation and the first log interval, and is
  measured on whatever hardware each run used.
- Runs without `batch_size`/`sequence_length` params (jobs tracked through the metrics file only)
  have no positions axis.

Capability: `project.runs.read` only (`runs.get` and `runs.metric_history`). The view writes
nothing.
