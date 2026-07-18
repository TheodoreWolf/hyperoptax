# Dashboard design survey: W&B, TensorBoard, Optuna, and Constellation

Status: research summary

Researched: 2026-07-17

## Executive summary

The four dashboards solve different problems:

- **Weights & Biases (W&B)** is a centralized, service-backed collaboration
  product that also supports dedicated and self-managed deployments. Its
  primary abstraction is a project containing runs, a globally filtering runs
  table, configurable workspace panels, and presentation-oriented reports.
- **TensorBoard** is a local, offline-capable viewer over append-only event
  files. Its primary abstraction is a log directory containing runs and tagged
  time series, rendered through data-type-specific plugins.
- **Optuna Dashboard** is a self-hosted, domain-specific viewer over an Optuna
  study database. Its primary abstraction is a study containing trials, with a
  curated set of hyperparameter-optimization views.
- **PufferLib Constellation** is a compact native/WebAssembly explorer over a
  preprocessed experiment corpus. Its primary abstraction is a point whose
  numeric fields can be assigned to axes, colour, and range filters.

The best starting point for Hyperoptax is not to clone any one of them. It is to
combine:

1. TensorBoard's local-first, offline-capable server and loopback-only default.
2. Optuna's curated HPO views, linked trial selection, and low-complexity polling.
3. W&B's globally filtering table, saved views, explicit downsampling, and
   schema-aware agent interface.
4. Constellation's field-composable scatter explorer, direct trial inspection,
   and first-class score-versus-cost Pareto view.

The proposed visualization engine is **Plotly**. It already covers the HPO plots
we need, supports browser-side interaction and incremental redraws, and uses a
serializable JSON figure representation. Vega-Lite is the strongest alternative
for agent-authored charts, but introducing two renderers in the first version
would add more surface area than value.

The dashboard should be a browser application, but the browser must not be the
only access path. Remote users should also be able to use an SSH tunnel, an
existing JupyterHub or editor port proxy, a local snapshot, a static HTML export,
or eventually a small terminal view.

For running Hyperoptax studies, live progress should be a first-release feature:
an optional host-side callback on the ordinary Python optimization loop writes
each completed batch to the local study database, and the browser polls the
resulting revisions. Compiled `optimize_scan` runs remain bulk-import only.

## Comparison at a glance

| Dimension | W&B | TensorBoard | Optuna Dashboard | Constellation |
|---|---|---|---|---|
| Default hosting | W&B-managed web service | Local HTTP server | Local HTTP server | Native executable or static WebAssembly page |
| Source of truth | Central experiment service | User-owned `tfevents` files | Optuna storage backend | Preprocessed JSON experiment corpus |
| Offline path | Local run files, later `wandb sync`; terminal viewer | Native design: read local files with no internet | Local database or journal file | Native build with bundled data |
| Typical cluster access | Job pushes outbound; user opens hosted UI | Local server beside the data; SSH tunnel is the safe derived pattern | Local server beside storage; SSH tunnel is the safe derived pattern | Native graphics forwarding or publish/copy the Web build |
| Main interaction | Projects, runs, workspaces, reports | Plugin tabs, runs, tags/cards | Studies, analytics, trials, selection | Assign fields to X/Y/Z/colour and two range filters |
| Custom plots | GraphQL query plus Vega/Vega-Lite spec | Plugin API or pre-rendered media | Plotly figures and LLM-generated Plotly code | Fixed explorer over arbitrary numeric fields |
| Live update | Streaming to hosted service | Backend log polling | Polling, normally every 10 seconds | Static preprocessed corpus |
| Agent story | Official MCP server and integrated research agent | No polished agent surface | Natural-language filtering and chart generation | No agent API; compact field/view state is transferable |
| Main tradeoff | Rich collaboration requires service infrastructure | Simple files limit arbitrary analytics | Strong HPO fit but weak default access control | Extremely lean, but preprocessing is not a live study service |

## W&B

### Product and hosting model

W&B's common path is a central service. A training process starts a background
client that sends run data to W&B, so a job on a remote cluster needs outbound
HTTPS but does not need to expose an inbound port. The user opens the dashboard
from their laptop. W&B offers multi-tenant cloud, dedicated cloud, and
self-managed deployments; the production self-managed stack requires
Kubernetes, MySQL, object storage, and Redis. A local Docker server exists for
testing, but it is not the recommended production design. See W&B's
[deployment options](https://docs.wandb.ai/platform/hosting/hosting-options) and
[self-managed overview](https://docs.wandb.ai/platform/hosting/hosting-options/self-managed).

If the training host is offline, the SDK writes locally and `wandb sync` uploads
the run later. W&B also has a terminal viewer intended for SSH, `tmux`, HPC, and
pre-sync inspection. These are important counterexamples to the idea that every
remote workflow must use a browser tunnel. See the
[offline workflow](https://docs.wandb.ai/support/models/articles/what-happens-if-internet-connection-is-l)
and [LEET terminal viewer](https://docs.wandb.ai/models/ref/cli/wandb-beta/wandb-beta-leet).

The consequence of W&B's architecture is that cluster visualization is easy
when outbound data transfer is acceptable, but data governance, credentials,
service availability, and deployment cost become product concerns.

### Information architecture

W&B separates live exploration from presentation:

- The project-level runs table contains configuration, summaries, state, notes,
  colors, and visibility.
- Filtering, grouping, and visibility change the data shown by workspace panels.
  Selection and pinning affect compatible comparison panels rather than every
  panel type. The table is therefore a main analytical control, not a secondary
  listing. See W&B's documented
  [comparison limitations](https://docs.wandb.ai/models/runs/compare-runs#limitations).
- Automatic workspaces create panels from logged keys. Manual workspaces start
  empty and favor deliberate curation.
- Personal workspaces act as scratchpads, saved views preserve useful layouts,
  and reports combine prose with panels for communication.

The progression is:

```text
all runs -> filter/select -> exploratory panels -> saved view/report
```

W&B documents this model in its
[workspace guide](https://docs.wandb.ai/models/track/workspaces),
[panel guide](https://docs.wandb.ai/models/app/features/panels), and
[reports guide](https://docs.wandb.ai/models/reports).

Sweep pages start with domain-specific defaults rather than an empty canvas:
parallel coordinates, parameter importance, and scatter plots. This is a better
first-use experience than asking every user to design a dashboard before they
can inspect a sweep. See
[visualizing sweep results](https://docs.wandb.ai/models/sweeps/visualize-sweep-results).

### Plotting and scale

W&B separates a chart's data query from its rendering specification. Custom
charts query visible run data and render it with Vega; field placeholders bind
query columns into reusable chart presets. The spec can be edited with a live
preview and saved for reuse. See the
[custom charts guide](https://docs.wandb.ai/models/app/features/custom-charts).

This separation is more important than the choice of Vega itself:

```text
query/filter/aggregate -> typed table -> declarative chart spec -> renderer
```

W&B also treats plot fidelity as a first-class scale decision. It can aggregate
large curves into buckets, preserve extrema, redraw at a different resolution
after zooming, or sample a bounded number of points. Its documentation states
the limits rather than pretending the browser always receives the complete
dataset. See
[line-plot sampling](https://docs.wandb.ai/models/app/features/panels/line-plot/sampling).

### Programmatic and agent interfaces

W&B exposes APIs for querying runs and for creating workspaces and reports. It
also exposes an official MCP server in hosted and local forms, letting external
coding/chat agents discover projects and schemas, query and compare runs,
inspect artifacts, and create reports. ARIA is a separate in-product research
agent that can create persistent workspaces, panels, custom charts, and reports.
The workspace API is Public Preview and ARIA is a Preview offering, so their
contracts may still change. See the
[Public API guide](https://docs.wandb.ai/models/track/public-api-guide),
[workspace API](https://docs.wandb.ai/models/ref/wandb_workspaces), and
[MCP server](https://docs.wandb.ai/platform/mcp-server), plus the
[ARIA overview](https://wandb.ai/site/agent/).

The transferable agent workflow is:

1. Discover studies, fields, and available operations.
2. Query bounded raw data or server-side aggregates.
3. Generate and validate a declarative plot request.
4. Preview the result.
5. Persist it only through an explicit write operation.

### What to adopt and avoid

Adopt:

- A table that globally filters every visualization.
- Useful automatic panels plus manual saved views.
- Separation between exploratory views and durable reports/exports.
- A query/spec split for custom charts.
- Explicit point limits and fidelity indicators.
- A real agent API rather than UI automation.
- An SSH-friendly fallback for restricted systems.

Avoid initially:

- Building a central hosted service before the local data contract is stable.
- Requiring several production infrastructure services for a small dashboard.
- Supporting multiple query languages for different panel types.
- Generating a panel for every key without a way to control panel explosion.

## TensorBoard

### Product and hosting model

TensorBoard is a self-hosted browser viewer over user-owned append-only event
files. `tensorboard --logdir PATH` starts a local server, normally on port 6006,
and the project is explicitly designed to work without internet access. The
former hosted sharing service, TensorBoard.dev, shut down on 1 January 2024.
The core project therefore remains local-first rather than a freemium hosted
service. See the [TensorBoard repository](https://github.com/tensorflow/tensorboard),
[TensorBoard.dev notice](https://tensorboard.dev/), and
[notebook integration](https://www.tensorflow.org/tensorboard/tensorboard_in_notebooks).

Training code writes tagged, step-indexed summaries to `tfevents` files.
TensorBoard recursively scans the log directory, treats event-containing
subdirectories as runs, and stitches restarted event files together. A backend
data layer reads and downsamples events; a WSGI application and plugin routes
serve the browser UI. The default `--load_fast=auto` selection policy attempts
to use a separate Rust data server where supported, but that fast path remains
experimental. The architecture is visible in TensorBoard's
[program orchestration](https://github.com/tensorflow/tensorboard/blob/master/tensorboard/program.py),
[data server ingester](https://github.com/tensorflow/tensorboard/blob/master/tensorboard/data/server_ingester.py),
and [backend application](https://github.com/tensorflow/tensorboard/blob/master/tensorboard/backend/application.py).

The durable design lesson is to keep four layers distinct:

```text
experiment records -> storage-independent query layer -> HTTP service -> UI
```

### Remote-cluster consequences

TensorBoard listens on localhost by default. Its CLI documents explicit network
exposure through `--bind_all`; running it beside the data and forwarding its
port is our derived safe cluster recommendation:

```sh
# Remote login or visualization host
tensorboard --logdir /remote/results --host localhost --port 6006

# Laptop
ssh -N -L 6006:localhost:6006 user@cluster
```

The service can instead be mounted into Jupyter, placed behind an authenticated
reverse proxy, pointed at supported object storage, or run locally against a
copied snapshot. Binding to all interfaces is possible, but the
[core CLI flags](https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/core/core_plugin.py)
make that an explicit choice. Hyperoptax should preserve the same secure default.

TensorBoard's file-tail behavior also exposes a remote-sync pitfall: replacing
active event files with `rsync` is different from appending to them and can
conflict with fast loading. Copying a completed snapshot is reliable; live file
mirroring needs an explicit snapshot protocol.

### Information architecture and HParams

TensorBoard organizes dashboards by data-type plugin. A run selector, tag
search, card grid, pinning area, and settings panel provide shared controls.
Only plugins with applicable data become prominent. This progressive disclosure
keeps a broad tool from showing irrelevant views.

The HParams plugin is the closest reference for Hyperoptax. It combines:

- A canonical table with one row per run.
- Global parameter, metric, state, sort, and count filters.
- Parallel coordinates with reorderable, brushable axes.
- A scatter-plot matrix.
- Linked selection: a row, line, or point refers to the same run everywhere.
- Drill-down from a final outcome into that run's metric history.

See the official
[HParams dashboard tutorial](https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams).
The interaction model is valuable even though the tutorial still describes the
plugin as preview-stage.

### Plotting and extensions

TensorBoard's mature frontend uses Angular, NgRx/RxJS, D3, Plottable, and some
legacy Polymer. It is a custom product stack, not a sensible dependency choice
for a small Python library. The current dependencies are listed in its
[package manifest](https://github.com/tensorflow/tensorboard/blob/master/package.json).

TensorBoard plugins can own a summary type, add Python backend routes, and load
an isolated browser frontend. Plugins read through shared event-multiplexer or
data-provider abstractions rather than reaching directly into log files. See the
[plugin guide](https://github.com/tensorflow/tensorboard/blob/master/ADDING_A_PLUGIN.md).

This is good modularity, but Hyperoptax does not need third-party UI plugins in
its first version. A stable query API and chart-specification contract give
agents and Python callers most of the useful extensibility at much lower cost.

### Programmatic and agent interfaces

TensorBoard has programmatic building blocks but not a polished end-user agent
surface. `DataProvider` abstracts run and series queries, its gRPC implementation
separates those query semantics from transport, `EventAccumulator` reads local
event data, and `summary_iterator` exposes raw records. These APIs show how an
agent-facing query layer could work, but they expose TensorBoard's own storage
concepts and some remain experimental. The documented high-level DataFrame API
only supported the now-closed TensorBoard.dev service, so it is not a current
local analysis path. See
[DataProvider](https://github.com/tensorflow/tensorboard/blob/master/tensorboard/data/provider.py)
and its [gRPC transport](https://github.com/tensorflow/tensorboard/blob/master/tensorboard/data/grpc_provider.py),
[EventAccumulator](https://github.com/tensorflow/tensorboard/blob/master/tensorboard/backend/event_processing/event_accumulator.py),
[`summary_iterator`](https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/summary_iterator),
and the obsolete [DataFrame API](https://www.tensorflow.org/tensorboard/dataframe_api).

### What to adopt and avoid

Adopt:

- One-command, local, offline serving.
- Loopback binding with tunnelling as the cluster path Hyperoptax documents.
- A storage-independent query layer.
- Linked filters and selections across tables and plots.
- Progressive disclosure based on available data.
- Server-side downsampling, augmented with visible fidelity metadata that
  TensorBoard itself does not expose.
- Notebook or cluster-gateway embedding as an alternative entry point.

Avoid:

- A bespoke low-level plotting frontend.
- Making agents parse event files or drive the UI.
- Coupling programmatic data access to a hosted service.
- Treating live `rsync` of open files as the primary remote workflow.

## Optuna Dashboard

### Product and hosting model

Optuna Dashboard is an open-source, self-hosted browser application. Its Python
package bundles a React/TypeScript single-page application and serves it with a
Bottle/WSGI JSON backend. The backend reads through Optuna's storage abstraction,
so the dashboard does not need a second experiment database. The current source
uses React, Material UI, TanStack Query/Table, Jotai, and Plotly.js. See the
[server application](https://github.com/optuna/optuna-dashboard/blob/af7fd1aa89ec8a5384ee3bf4e0571e7dc9fe0aba/optuna_dashboard/_app.py),
[Python dependencies](https://github.com/optuna/optuna-dashboard/blob/af7fd1aa89ec8a5384ee3bf4e0571e7dc9fe0aba/pyproject.toml),
and [frontend dependencies](https://github.com/optuna/optuna-dashboard/blob/af7fd1aa89ec8a5384ee3bf4e0571e7dc9fe0aba/optuna_dashboard/package.json).

The simplest command points the dashboard at SQLite, a relational database URL,
a journal file, or Redis. Python and production WSGI interfaces are available,
as is an official Docker image. See
[Getting Started](https://optuna-dashboard.readthedocs.io/en/latest/getting-started.html).

Optuna also demonstrates several alternatives to a raw SSH tunnel:

- A JupyterLab extension mounts the app into the existing Jupyter server and
  supports JupyterHub base paths.
- VS Code/code-server can open a database through an editor webview.
- A [static browser-only WASM app](https://optuna.github.io/optuna-dashboard/)
  can open a copied SQLite or journal file without a Python server, although it
  is not live and supports fewer features.
- A local dashboard can connect to a secured, network-reachable PostgreSQL or
  MySQL database.

For a live database that only exists on a remote filesystem, the safe default is
still to run the dashboard beside the storage and tunnel port 8080.

### Storage and security consequences

Optuna's storage abstraction supports single-machine and distributed workflows,
but not every backend is safe for every topology. The official distributed
guidance discourages SQLite for parallel optimization and recommends PostgreSQL
or MySQL for multi-node work. File-journal locking on shared filesystems is
filesystem-dependent. See
[distributed optimization](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html)
and the [SQLite FAQ](https://optuna.readthedocs.io/en/stable/faq.html#how-can-i-solve-the-error-that-occurs-when-performing-parallel-optimization-with-sqlite3).

The dashboard is not merely a viewer: its API can mutate studies, trials, notes,
and artifacts. It has no documented built-in authentication or TLS. Loopback is
therefore an important safety boundary, and a team deployment needs an
authenticated reverse proxy plus an explicit permission model. Hyperoptax
should offer a genuinely read-only mode from the beginning.

### Information architecture and live updates

Optuna begins with a searchable study catalogue, then gives each study a
task-oriented set of views:

- History: objective history, best trials, importance, timeline, intermediate
  values, attributes, notes, and artifacts.
- Analytics: slice, contour, rank, and empirical-distribution plots.
- Trials: a virtualized, filterable table with details and artifacts.
- Smart Selection: linked parallel coordinates, history or Pareto plot, and
  trial table.
- Compare Studies: compatible cross-study history and distribution views.

This is a stronger MVP model for Hyperoptax than W&B's fully configurable canvas.
Users receive useful HPO analysis immediately, while custom views can be added
later.

"Real-time" updates are deliberately simple: the frontend polls, normally every
10 seconds, retains the immutable trial prefix, and requests the mutable tail.
Plots update with `Plotly.react`; long trial lists are virtualized. This is likely
enough for sweep monitoring and avoids a WebSocket or event-streaming subsystem.

### Plotting and agent interface

Optuna uses Plotly.js, with optional Plotly.py generation on the server. Plotly
provides native HPO-relevant chart types and interactions: parallel coordinates,
contours, heatmaps, Pareto fronts, timelines, brushing, hover, and zoom. Python
figures and browser figures share a JSON representation. Optuna can also persist
custom Plotly figures; see its
[custom plot API](https://optuna-dashboard.readthedocs.io/en/latest/_generated/optuna_dashboard.save_plotly_graph_object.html).

The current dashboard has a directly relevant LLM feature. Natural language can
produce trial filters or custom Plotly charts. The model receives the prompt and
a typed study/trial contract rather than the full study data, then returns a
JavaScript transformation. The browser shows the generated code for approval and
evaluates it in an isolated iframe with network APIs disabled. See the
[LLM tutorial](https://optuna-dashboard.readthedocs.io/en/latest/tutorials/llm-integration.html),
[generation prompt](https://github.com/optuna/optuna-dashboard/blob/af7fd1aa89ec8a5384ee3bf4e0571e7dc9fe0aba/optuna_dashboard/llm/prompt_templates/_generate_plotly_graph.py),
and [sandbox implementation](https://github.com/optuna/optuna-dashboard/blob/af7fd1aa89ec8a5384ee3bf4e0571e7dc9fe0aba/tslib/react/src/hooks/useEvalFunctionInSandbox.tsx).

This is clever and keeps bulk data local, but the Optuna documentation correctly
warns that generated JavaScript cannot be guaranteed safe. Hyperoptax should
borrow the typed transformation boundary and local execution, but accept only
built-in view recipes or validated, bounded Plotly figure JSON in the default
agent path.

### What to adopt and avoid

Adopt:

- A curated study/trial UI with domain-specific plots.
- Linked selection across the table, history, and parallel coordinates.
- Plotly as a shared Python/browser figure engine.
- Polling and incremental-tail queries before server push.
- Storage independence and separate artifact storage.
- JupyterHub/editor integration as a first-class cluster path.
- Typed context for agent-generated plots, executed close to the data.

Avoid:

- Mutating routes in a server advertised as a viewer without a read-only mode.
- Direct public exposure without authentication or TLS.
- Arbitrary generated JavaScript as the only custom-plot interface.
- Persisting large duplicated trace arrays when a compact query and plot recipe
  would reproduce the figure.

## PufferLib Constellation

### Hosting and data flow

Constellation is implemented as a compact C/raylib application and can be built
as either a desktop executable or a WebAssembly page. PufferLib documents
`bash build.sh constellation --[local|fast|web]`; the web build preloads its
experiment resources into the browser bundle. See the
[build script](https://github.com/PufferAI/PufferLib/blob/373723f10be77e175a879223d872bd92a57a8fc4/build.sh#L96-L185)
and [PufferLib docs](https://puffer.ai/docs.html).

Its Python preprocessing step merges experiment JSON logs, flattens nested
fields, normalizes hyperparameters, optionally computes a t-SNE projection, and
by default retains points that are non-dominated across score, steps, and
wall-clock cost. The resulting JSON corpus is read entirely by the renderer.
See [cache_data.py](https://github.com/PufferAI/PufferLib/blob/373723f10be77e175a879223d872bd92a57a8fc4/constellation/cache_data.py).

This makes Constellation excellent for fast, portable aggregate exploration,
but it does not provide the append-only storage/query service needed for a live
Hyperoptax run. Hyperoptax should retain its local FastAPI/SQLite process for
live data and eventually offer a static snapshot export as the analogous
portable mode.

### Interaction model

Constellation's strongest idea is that the data fields, not a catalogue of
hard-coded chart names, drive the main view. A single toolbar assigns fields to
X, Y, Z, and colour, chooses linear/log/logit transforms, and applies two range
filters. The same filtered points appear in 3-D and 2-D scatter views, a t-SNE
projection, and parameter box plots. Right-clicking the nearest point shows
score/cost/steps and copies its hyperparameters. See
[constellation.c](https://github.com/PufferAI/PufferLib/blob/373723f10be77e175a879223d872bd92a57a8fc4/constellation/constellation.c#L840-L1220).

The transferable design is a small declarative view recipe:

```json
{
  "version": 1,
  "chart": "scatter",
  "x": {"field": "duration_seconds", "scale": "linear"},
  "y": {"field": "objective_value", "scale": "linear"},
  "color": {"field": "params.learning_rate"},
  "filters": [],
  "pareto": {
    "cost": {"field": "duration_seconds", "direction": "minimize"},
    "score": {"field": "objective_value", "direction": "maximize"}
  }
}
```

This is now the model used by Hyperoptax's experiment explorer and copied view
JSON. Unlike Constellation, the Hyperoptax recipe is encoded into a shareable
URL and backed by the typed trial-query and Pareto APIs so agents do not need to
drive the browser.

### What to adopt and avoid

Adopt:

- Numeric-field composition for axes, colour, transforms, and range filters.
- Objective-versus-recorded-cost Pareto analysis as a default when timing
  exists.
- One-click trial inspection and copyable configurations.
- A portable snapshot/export mode over the same declarative view contract.

Avoid:

- Replacing the Python/Web stack with a native renderer solely for speed.
- Filtering the durable dataset down to Pareto points; keep every trial and
  render the frontier as an overlay.
- Treating callback batch duration as hardware-independent cost. Compilation,
  vectorization, and contention must remain visible provenance.

## Remote-cluster access patterns

SSH tunnelling is the right default for a local-first Hyperoptax server, but it
is not the only workable design.

| Pattern | When it fits | Benefits | Costs or risks |
|---|---|---|---|
| Push to a hosted service | Cluster has outbound HTTPS and data may leave | No inbound port; live links and collaboration | Service, credentials, privacy, and operations |
| SSH local port forward | One user, private cluster data | Simple, encrypted, preserves loopback binding | Tunnel lifetime and jump-host setup |
| JupyterHub/code-server/editor proxy | Cluster already provides a web gateway | Reuses authentication, TLS, and routing | Integration work and base-path handling |
| Authenticated reverse proxy | Persistent team dashboard | Stable URL and multi-user access | Must add TLS, identity, authorization, and operations |
| Local snapshot | Completed or intermittently copied study | Dashboard runs entirely on laptop | Not live; active databases need a consistent snapshot |
| Static HTML export | Sharing a fixed result | No server or database exposure | No live queries; export can be large |
| Browser-only file viewer | User can download a SQLite/snapshot file | No Python server; data stays in browser | Limited features and no live update |
| Object/shared database | Storage is already network-reachable | Dashboard may run locally or centrally | Database security and backend-specific concurrency |
| Terminal summary/TUI | SSH-only or air-gapped workflow | No tunnel or graphical environment | Less expressive interaction |

The standard tunnel documentation for Hyperoptax should be:

```sh
# Remote host running the recorder/store, or holding a completed snapshot
hyperoptax dashboard /path/to/results.sqlite3 --host 127.0.0.1 --port 8080

# Laptop
ssh -N -L 8080:127.0.0.1:8080 user@login.cluster
```

For a compute node behind a login node, run the viewer on the host that safely
owns the live store and use `ssh -J`. An approved login/visualization node is an
alternative only when the recorder also runs there, the storage backend supports
cross-host access, or the input is a completed snapshot. VS Code's
[Remote SSH port forwarding](https://code.visualstudio.com/docs/remote/ssh#_forwarding-a-port-creating-ssh-tunnel)
provides the same model through its UI.

Snapshot refreshes should publish immutable, revisioned files and reopen the
viewer on the new copy. Replacing a SQLite pathname underneath an existing
connection is not a reliable live-update mechanism.

The server should bind to `127.0.0.1` by default. Binding to `0.0.0.0` must print
a warning unless authentication is configured. A public or team deployment is a
different mode, not a different spelling of the development command.

## Dynamic plotting library decision

### Requirements

The renderer needs to support:

- Interactive hover, zoom, pan, selection, and linked brushing.
- Parallel coordinates, scatter matrices, contour/heatmap plots, distributions,
  timelines, and ordinary line/bar/scatter plots.
- Browser rendering and standalone HTML export.
- A serializable specification that can be validated, saved, diffed, and
  generated by an agent.
- Python-side authoring without requiring users to write JavaScript.
- Efficient redraws and a route to WebGL for larger point sets.
- Local/offline asset serving with no CDN dependency.

### Options

| Library | Strengths | Weaknesses | Verdict |
|---|---|---|---|
| Plotly.js, optionally Plotly.py | Broad scientific/HPO chart set, built-in interactions, JSON figure tree, standalone HTML, WebGL traces, `Plotly.react` | Large browser bundle; raw figure JSON is verbose; large embedded datasets still need limits | **Use Plotly.js first; add Plotly.py only for a measured server need** |
| Vega-Lite/Altair | Concise declarative grammar, strong transforms and selections, JSON schema, excellent agent target | Some HPO views need full Vega or custom transforms; a second renderer complicates consistency | Best alternative; reconsider for custom panels later |
| Bokeh/Panel | Strong Python reactivity, streaming, server callbacks, multiple plotting backends | Stateful browser/server sessions and WebSockets couple UI to Python runtime; saved specs are less portable | Useful for a prototype, not the preferred architecture |
| Matplotlib | Already familiar in the notebooks; excellent static output | No native linked browser interaction or reusable interactive spec | Keep for notebooks/static export only |
| TensorBoard's D3 stack | Maximum control | High frontend maintenance cost | Do not reproduce |

Plotly figures are tree-shaped structures serialized to JSON and rendered by
Plotly.js; they can also be exported to HTML or static images. See Plotly's
[graph-object model](https://plotly.com/python/graph-objects/) and
[rendering options](https://plotly.com/python/renderers/). Browser interactions
render with SVG or WebGL and do not require a round trip for every hover or zoom.

This decision is about the **chart engine**, not the application framework.
Dash is the fastest way to make an all-Python Plotly prototype, but its callback
model should not become the data or agent API. An API-first service with a thin
browser client keeps queries, plot compilation, exports, and agents on the same
semantic boundary.

## Agent-interface design lessons

Agents should not need browser control, direct database access, arbitrary SQL,
or executable JavaScript. The dashboard should expose the same semantic service
used by its own UI.

The minimum capabilities are:

```text
list_studies
describe_study
query_trials(filter, sort, columns, limit)
get_trial
preview_view(query, built_in_view_spec)
preview_figure(plotly_figure_json)
export_data(query, format)
save_view(query, view_or_frozen_figure, title)  # later and opt-in
```

Important boundaries:

- Return schema metadata and typed tabular results, not rendered HTML alone.
- Make filters, aggregation, selection, point limits, and downsampling explicit.
- Separate a read-only preview from a persistent write.
- Store the query, built-in recipe or frozen figure, study revision, renderer
  version, and agent/user provenance together.
- Validate field names and built-in view variants against the study schema;
  validate custom figure JSON against strict type and size limits.
- Keep raw code execution out of the normal server.
- Default agent access to read-only; enabling saved-view writes is a separate
  permission.
- Run queries and transforms near the data and return only the bounded result.

REST/OpenAPI provides a general programmatic surface. A small MCP adapter can
then expose the same operations to coding agents through local STDIO or secured
HTTP. MCP distinguishes contextual resources from model-invocable tools; either
can expose read-only data, while Hyperoptax reserves mutations for explicitly
enabled tools. See its
[architecture overview](https://modelcontextprotocol.io/docs/learn/architecture).

## Decisions carried into the Hyperoptax plan

| Area | Draft decision |
|---|---|
| Product | Local-first, offline-capable live study viewer; no hosted service in the initial scope |
| Default network | Bind to `127.0.0.1`; document SSH forwarding |
| Remote alternatives | Jupyter/editor proxy, consistent snapshot, static HTML, reverse proxy for teams |
| Storage | Host-local SQLite first, with one recorder writing trials; shared-filesystem SQLite is unsupported |
| Recording | Optional synchronous callback persists each completed Python-loop batch; compiled scans bulk-import on return |
| UI | Curated HPO views first; custom saved views second |
| Global state | Trial table/filter is canonical and controls every plot |
| Live update | Adaptive two-second revision/tail polling while a study is running, before WebSockets or SSE |
| Plot engine | Bundled Plotly.js; Plotly.py only if a spike proves a server need |
| Custom plots | Four built-in dynamic `ViewSpec` variants plus bounded, validated Plotly figure snapshots from agents |
| Agent access | Shared query service, REST/OpenAPI, then MCP adapter |
| Agent writes | Read-only by default; preview and save are separate operations |
| Scale | Load and virtualize 10,000 trials first; add pagination/aggregation only if measured, with visible fidelity metadata |
| Packaging | Optional dashboard dependencies; keep the core JAX package lean |

The implementation draft is in
[dashboard-implementation-plan.md](dashboard-implementation-plan.md).
