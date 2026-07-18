const APP_BASE = new URL("../", import.meta.url);
const API_BASE = new URL("api/v1/", APP_BASE);
const LIVE_STATES = new Set(["created", "pending", "running"]);
const LIVE_POLL_INTERVAL_MS = 2_000;
const MAX_IDLE_POLL_INTERVAL_MS = 10_000;
const MAX_ERROR_POLL_INTERVAL_MS = 60_000;
const MAX_TABLE_ROWS = 1_000;
const MAX_PLOT_POINTS = 5_000;
const MAX_PARALLEL_DIMENSIONS = 7;
const IMPORTANCE_SIGNIFICANT_DIGITS = 3;

const elements = Object.fromEntries(
  [
    "loading-state",
    "empty-state",
    "error-state",
    "error-message",
    "error-retry",
    "empty-refresh",
    "dashboard",
    "study-select",
    "theme-toggle",
    "connection-status",
    "study-title",
    "study-description",
    "study-tags",
    "summary-best",
    "summary-best-context",
    "summary-trials",
    "summary-trials-context",
    "summary-status",
    "summary-updated",
    "summary-optimizer",
    "summary-direction",
    "trial-search",
    "state-filter",
    "reset-filters",
    "filter-summary",
    "history-chart",
    "best-chart",
    "parallel-chart",
    "explorer-chart",
    "explorer-note",
    "explorer-x",
    "explorer-x-scale",
    "explorer-y",
    "explorer-y-scale",
    "explorer-color",
    "explorer-pareto",
    "explorer-filter-1-field",
    "explorer-filter-1-min",
    "explorer-filter-1-max",
    "explorer-filter-2-field",
    "explorer-filter-2-min",
    "explorer-filter-2-max",
    "copy-view-link",
    "copy-view-spec",
    "view-feedback",
    "importance-chart",
    "importance-note",
    "trial-head",
    "trial-body",
    "table-summary",
    "table-note",
    "trial-inspector",
    "inspector-title",
    "inspector-summary",
    "inspector-config",
    "inspector-extra",
    "inspector-feedback",
    "close-inspector",
    "copy-trial-config",
  ].map((id) => [id, document.getElementById(id)]),
);

const state = {
  studies: [],
  studyDescription: null,
  importance: null,
  study: null,
  trials: [],
  filteredTrials: [],
  revision: 0,
  queryMetadata: {},
  selectedIds: new Set(),
  search: "",
  trialState: "",
  sort: { field: "evaluation_index", direction: "asc" },
  explorer: {
    x: "",
    y: "objective_value",
    color: "",
    xScale: "linear",
    yScale: "linear",
    pareto: true,
    filters: [
      { field: "", minimum: "", maximum: "" },
      { field: "", minimum: "", maximum: "" },
    ],
  },
  inspectedTrialId: "",
  parallelFields: [],
  parallelConstraints: new Map(),
  pollTimer: null,
  pollFailures: 0,
  unchangedPolls: 0,
  loadToken: 0,
  theme: localStorage.getItem("hyperoptax-theme") || "system",
};

function studyId(study) {
  return String(study?.study_id ?? study?.id ?? "");
}

function trialId(trial) {
  return String(
    trial?.trial_id ??
      trial?.id ??
      `${trial?.batch_index ?? "batch"}:${trial?.batch_slot ?? "slot"}`,
  );
}

function trialName(trial) {
  if (trial?.trial_name) return String(trial.trial_name);
  const index = evaluationIndex(trial, -1);
  return index >= 0 ? `trial-${index + 1}` : trialId(trial);
}

function studyRecord(description) {
  return description?.study ?? description?.metadata ?? description ?? {};
}

function studyStatus() {
  return String(state.study?.status ?? "unknown").toLowerCase();
}

function objectiveValue(trial) {
  const value = trial?.objective_value ?? trial?.objective ?? trial?.result;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function evaluationIndex(trial, fallback = 0) {
  const value =
    trial?.evaluation_index ??
    trial?.trial_index ??
    trial?.batch_index ??
    fallback;
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function asArray(value) {
  if (Array.isArray(value)) return value;
  if (value == null) return [];
  return [...value];
}

function studiesFrom(payload) {
  if (Array.isArray(payload)) return payload;
  return asArray(payload?.studies ?? payload?.rows ?? payload?.items);
}

function trialsFrom(payload) {
  if (Array.isArray(payload)) return payload;
  return asArray(payload?.rows ?? payload?.trials ?? payload?.items);
}

function flattenValues(value, prefix = "params", output = {}) {
  if (value == null || typeof value !== "object") {
    output[prefix] = value;
    return output;
  }
  if (Array.isArray(value)) {
    if (value.length === 0) output[prefix] = [];
    value.forEach((item, index) => flattenValues(item, `${prefix}[${index}]`, output));
    return output;
  }
  const entries = Object.entries(value);
  if (entries.length === 0) output[prefix] = {};
  entries.forEach(([key, item]) => {
    const path = prefix ? `${prefix}.${key}` : key;
    flattenValues(item, path, output);
  });
  return output;
}

function prepareTrial(trial) {
  const flat = {};
  Object.entries(trial || {}).forEach(([key, value]) => {
    if (key.startsWith("params.")) flat[key] = value;
  });
  if (trial?.params && typeof trial.params === "object") {
    flattenValues(trial.params, "params", flat);
  }
  return {
    ...trial,
    _flatParams: flat,
    _searchText: `${trialName(trial)} ${JSON.stringify(trial)}`.toLocaleLowerCase(),
  };
}

function fieldValue(trial, field) {
  if (field in trial) return trial[field];
  if (field in (trial._flatParams || {})) return trial._flatParams[field];
  return field.split(".").reduce((value, key) => value?.[key], trial);
}

function formatNumber(value, maximumSignificantDigits = 6) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  const absolute = Math.abs(number);
  if ((absolute > 0 && absolute < 1e-4) || absolute >= 1e6) {
    return number.toExponential(
      Math.min(3, Math.max(0, maximumSignificantDigits - 1)),
    );
  }
  return new Intl.NumberFormat(undefined, { maximumSignificantDigits }).format(number);
}

function formatValue(value) {
  if (value == null || value === "") return "—";
  if (typeof value === "number") return formatNumber(value);
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function formatStatus(value) {
  const text = String(value || "unknown").replaceAll("_", " ");
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function apiUrl(path) {
  return new URL(path.replace(/^\//, ""), API_BASE);
}

async function request(path, options = {}) {
  const response = await fetch(apiUrl(path), {
    ...options,
    headers: {
      Accept: "application/json",
      ...(options.body ? { "Content-Type": "application/json" } : {}),
      ...(options.headers || {}),
    },
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      detail = payload.detail || detail;
    } catch {
      // Keep the HTTP status when an intermediary returns non-JSON.
    }
    throw new Error(detail);
  }
  return response.json();
}

function setConnection(mode, label) {
  elements["connection-status"].dataset.state = mode;
  elements["connection-status"].lastChild.textContent = label;
}

function showSurface(name, error = null) {
  elements["loading-state"].hidden = name !== "loading";
  elements["empty-state"].hidden = name !== "empty";
  elements["error-state"].hidden = name !== "error";
  elements.dashboard.hidden = name !== "dashboard";
  if (error) elements["error-message"].textContent = error.message || String(error);
}

function populateStudySelect(selectedId = "") {
  const select = elements["study-select"];
  select.replaceChildren();
  for (const study of state.studies) {
    const option = document.createElement("option");
    option.value = studyId(study);
    option.textContent = study.name || option.value || "Unnamed study";
    option.selected = option.value === selectedId;
    select.append(option);
  }
  select.disabled = state.studies.length === 0;
}

async function loadStudies(preferredId = "") {
  clearPolling();
  setConnection("syncing", "Loading");
  showSurface("loading");
  try {
    const payload = await request("studies");
    state.studies = studiesFrom(payload);
    if (state.studies.length === 0) {
      populateStudySelect();
      setConnection("connected", "Local");
      showSurface("empty");
      return;
    }

    const requested = new URLSearchParams(window.location.search).get("study");
    const candidate = preferredId || requested;
    const selected =
      state.studies.find((study) => studyId(study) === candidate) || state.studies[0];
    populateStudySelect(studyId(selected));
    await loadStudy(studyId(selected));
  } catch (error) {
    setConnection("error", "Disconnected");
    showSurface("error", error);
  }
}

async function loadStudy(id) {
  const token = ++state.loadToken;
  clearPolling();
  setConnection("syncing", "Loading study");
  try {
    const [description, queryResult, importance] = await Promise.all([
      request(`studies/${encodeURIComponent(id)}`),
      request(`studies/${encodeURIComponent(id)}/trials/query`, {
        method: "POST",
        body: JSON.stringify({ limit: 10_000 }),
      }),
      request(`studies/${encodeURIComponent(id)}/importance`).catch((error) => ({
        parameters: [],
        reason: error.message,
      })),
    ]);
    if (token !== state.loadToken) return;

    state.studyDescription = description;
    state.study = studyRecord(description);
    state.importance = importance;
    state.trials = trialsFrom(queryResult).map(prepareTrial);
    state.queryMetadata = queryResult || {};
    state.revision = Number(
      queryResult?.study_revision ?? state.study?.revision ?? description?.revision ?? 0,
    );
    state.pollFailures = 0;
    state.unchangedPolls = 0;
    const restoredView = viewStateFromLocation();
    state.selectedIds.clear();
    state.parallelConstraints.clear();
    state.search = restoredView.search;
    state.trialState = restoredView.trialState;
    state.sort = restoredView.sort;
    state.explorer = restoredView.explorer;
    state.inspectedTrialId = restoredView.trial;
    elements["trial-search"].value = state.search;

    configureFilterChoices();
    updateStudyLocation(id);
    updateViewLocation();
    renderDashboard();
    showSurface("dashboard");
    restoreTrialInspector();
    setConnection("connected", LIVE_STATES.has(studyStatus()) ? "Live" : "Local");
    schedulePoll();
  } catch (error) {
    if (token !== state.loadToken) return;
    setConnection("error", "Disconnected");
    showSurface("error", error);
  }
}

function updateStudyLocation(id) {
  const url = new URL(window.location.href);
  url.searchParams.set("study", id);
  window.history.replaceState({}, "", url);
  elements["study-select"].value = id;
}

function viewStateFromLocation() {
  const params = new URLSearchParams(window.location.search);
  const direction = params.get("order") === "desc" ? "desc" : "asc";
  const readFilter = (index) => ({
    field: params.get(`f${index}`) || "",
    minimum: params.get(`f${index}min`) || "",
    maximum: params.get(`f${index}max`) || "",
  });
  return {
    search: params.get("q") || "",
    trialState: params.get("state") || "",
    trial: params.get("trial") || "",
    sort: {
      field: params.get("sort") || "evaluation_index",
      direction,
    },
    explorer: {
      x: params.get("x") || params.get("parameter") || "",
      y: params.get("y") || "objective_value",
      color: params.get("color") || "",
      xScale: params.get("xscale") === "log" ? "log" : "linear",
      yScale: params.get("yscale") === "log" ? "log" : "linear",
      pareto: params.get("pareto") !== "0",
      filters: [readFilter(1), readFilter(2)],
    },
  };
}

function updateViewLocation() {
  const url = new URL(window.location.href);
  const setOrDelete = (key, value, defaultValue = "") => {
    if (value && value !== defaultValue) url.searchParams.set(key, value);
    else url.searchParams.delete(key);
  };
  setOrDelete("q", state.search);
  setOrDelete("state", state.trialState);
  url.searchParams.delete("parameter");
  setOrDelete("sort", state.sort.field, "evaluation_index");
  setOrDelete("order", state.sort.direction, "asc");
  setOrDelete("trial", state.inspectedTrialId);
  setOrDelete("x", state.explorer.x);
  setOrDelete("y", state.explorer.y, "objective_value");
  setOrDelete("color", state.explorer.color);
  setOrDelete("xscale", state.explorer.xScale, "linear");
  setOrDelete("yscale", state.explorer.yScale, "linear");
  setOrDelete("pareto", state.explorer.pareto ? "1" : "0", "1");
  state.explorer.filters.forEach((filter, index) => {
    const number = index + 1;
    setOrDelete(`f${number}`, filter.field);
    setOrDelete(`f${number}min`, filter.minimum);
    setOrDelete(`f${number}max`, filter.maximum);
  });
  window.history.replaceState({}, "", url);
}

function configureFilterChoices() {
  const stateSelect = elements["state-filter"];
  const requestedState = state.trialState || stateSelect.value;
  stateSelect.replaceChildren(new Option("All states", ""));
  const states = [...new Set(state.trials.map((trial) => trial.state).filter(Boolean))].sort();
  states.forEach((trialState) => {
    stateSelect.append(new Option(formatStatus(trialState), String(trialState)));
  });
  state.trialState = states.includes(requestedState) ? requestedState : "";
  stateSelect.value = state.trialState;

  const fields = numericParameterFields(state.trials);
  const sortableFields = new Set([
    "trial_id",
    "evaluation_index",
    "batch_index",
    "batch_slot",
    "state",
    "objective_value",
    "duration_seconds",
    ...fields,
  ]);
  if (!sortableFields.has(state.sort.field)) {
    state.sort = { field: "evaluation_index", direction: "asc" };
  }
  configureExplorerChoices();
}

function numericExplorerFields(trials = state.trials) {
  const base = [
    "objective_value",
    "duration_seconds",
    "evaluation_index",
    "batch_index",
    "batch_slot",
  ].filter((field) =>
    trials.some((trial) => numericFieldValue(trial, field) != null),
  );
  return [...new Set([...base, ...numericParameterFields(trials)])];
}

function configureExplorerChoices() {
  const fields = numericExplorerFields();
  const parameterFallback = fields.find((field) => field.startsWith("params."));
  const defaultX = fields.includes("duration_seconds")
    ? "duration_seconds"
    : parameterFallback || fields.find((field) => field !== "objective_value") || "";
  state.explorer.x = fields.includes(state.explorer.x) ? state.explorer.x : defaultX;
  state.explorer.y = fields.includes(state.explorer.y)
    ? state.explorer.y
    : fields.includes("objective_value")
      ? "objective_value"
      : fields[0] || "";
  state.explorer.color = fields.includes(state.explorer.color)
    ? state.explorer.color
    : "";

  const axisOptions = fields.map((field) => new Option(displayField(field), field));
  elements["explorer-x"].replaceChildren(...axisOptions.map((option) => option.cloneNode(true)));
  elements["explorer-y"].replaceChildren(...axisOptions.map((option) => option.cloneNode(true)));
  elements["explorer-color"].replaceChildren(
    new Option("Single colour", ""),
    ...axisOptions.map((option) => option.cloneNode(true)),
  );
  elements["explorer-x"].value = state.explorer.x;
  elements["explorer-y"].value = state.explorer.y;
  elements["explorer-color"].value = state.explorer.color;
  elements["explorer-x-scale"].value = state.explorer.xScale;
  elements["explorer-y-scale"].value = state.explorer.yScale;

  state.explorer.filters.forEach((filter, index) => {
    const number = index + 1;
    const fieldElement = elements[`explorer-filter-${number}-field`];
    fieldElement.replaceChildren(
      new Option("No range filter", ""),
      ...axisOptions.map((option) => option.cloneNode(true)),
    );
    filter.field = fields.includes(filter.field) ? filter.field : "";
    fieldElement.value = filter.field;
    syncRangeFilter(index, { preserveValues: true });
  });
  syncParetoControl();
}

function numericParameterFields(trials) {
  const candidates = new Map();
  trials.forEach((trial) => {
    Object.entries(trial._flatParams || {}).forEach(([field, value]) => {
      const record = candidates.get(field) || { numeric: 0, present: 0 };
      record.present += 1;
      if (
        value != null &&
        value !== "" &&
        typeof value !== "boolean" &&
        Number.isFinite(Number(value))
      ) {
        record.numeric += 1;
      }
      candidates.set(field, record);
    });
  });
  return [...candidates.entries()]
    .filter(([, counts]) => counts.numeric > 0 && counts.numeric === counts.present)
    .map(([field]) => field)
    .sort();
}

function displayField(field) {
  const labels = {
    trial_id: "Trial",
    evaluation_index: "Evaluation",
    batch_index: "Batch",
    batch_slot: "Batch slot",
    objective_value: "Objective",
    duration_seconds: "Duration (s)",
    state: "State",
  };
  if (labels[field]) return labels[field];
  return field.replace(/^params\./, "").replaceAll(".", " › ");
}

function applyFilters() {
  const search = state.search.trim().toLocaleLowerCase();
  let trials = state.trials.filter((trial) => {
    if (state.trialState && String(trial.state) !== state.trialState) return false;
    if (search && !trial._searchText.includes(search)) return false;
    return true;
  });

  const { field, direction } = state.sort;
  const multiplier = direction === "desc" ? -1 : 1;
  trials = [...trials].sort((left, right) => {
    let leftValue = canonicalFieldValue(left, field);
    let rightValue = canonicalFieldValue(right, field);
    if (leftValue == null && rightValue == null) return 0;
    if (leftValue == null) return 1;
    if (rightValue == null) return -1;
    if (typeof leftValue === "number" && typeof rightValue === "number") {
      return (leftValue - rightValue) * multiplier;
    }
    return String(leftValue).localeCompare(String(rightValue)) * multiplier;
  });
  state.filteredTrials = trials;
}

function canonicalFieldValue(trial, field) {
  if (field === "trial_id") return trialName(trial);
  if (field === "evaluation_index") return evaluationIndex(trial);
  if (field === "objective_value") return objectiveValue(trial);
  return fieldValue(trial, field);
}

function numericFieldValue(trial, field) {
  const value = canonicalFieldValue(trial, field);
  if (value == null || value === "" || typeof value === "boolean") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function renderDashboard({ skipParallel = false } = {}) {
  applyFilters();
  renderHeading();
  renderSummary();
  renderFilterSummary();
  renderTable();
  renderHistory();
  renderBestSoFar();
  renderExplorer();
  if (!skipParallel) renderParallelCoordinates();
  renderImportance();
  renderTrialInspector();
}

function renderHeading() {
  const study = state.study;
  elements["study-title"].textContent = study.name || studyId(study) || "Unnamed study";
  const parts = [
    study.optimizer_name || study.optimizer,
    study.n_parallel ? `${study.n_parallel} parallel` : null,
    study.metadata?.recording_mode === "bulk_history" ? "Imported history" : null,
    study.created_at ? new Date(study.created_at).toLocaleString() : null,
  ].filter(Boolean);
  elements["study-description"].textContent = parts.join(" · ");

  const tags = asArray(study.tags);
  elements["study-tags"].replaceChildren(
    ...tags.map((tagValue) => {
      const tag = document.createElement("span");
      tag.className = "tag";
      tag.textContent = String(tagValue);
      return tag;
    }),
  );
}

function completedTrials(trials = state.filteredTrials) {
  return trials.filter(
    (trial) => String(trial.state).toLowerCase() === "completed" && objectiveValue(trial) != null,
  );
}

function bestTrial(trials = state.filteredTrials) {
  const completed = completedTrials(trials);
  if (completed.length === 0) return null;
  const minimise = String(state.study.direction).toLowerCase() === "minimize";
  return completed.reduce((best, trial) => {
    const candidate = objectiveValue(trial);
    const current = objectiveValue(best);
    return minimise ? (candidate < current ? trial : best) : candidate > current ? trial : best;
  });
}

function renderSummary() {
  const best = bestTrial();
  elements["summary-best"].textContent = best ? formatNumber(objectiveValue(best)) : "—";
  elements["summary-best-context"].textContent = best
    ? `Trial ${trialName(best)}`
    : "No completed trials";

  elements["summary-trials"].textContent = new Intl.NumberFormat().format(
    state.filteredTrials.length,
  );
  const allCount = state.trials.length;
  elements["summary-trials-context"].textContent =
    state.filteredTrials.length === allCount
      ? `${completedTrials(state.trials).length} completed`
      : `${allCount} loaded total`;

  elements["summary-status"].textContent = formatStatus(studyStatus());
  elements["summary-updated"].textContent = `Revision ${state.revision}`;
  elements["summary-optimizer"].textContent =
    state.study.optimizer_name || state.study.optimizer || "Custom";
  elements["summary-direction"].textContent = `${formatStatus(
    state.study.direction || "unknown",
  )} objective`;
}

function renderFilterSummary() {
  const selectedVisible = state.filteredTrials.filter((trial) =>
    state.selectedIds.has(trialId(trial)),
  ).length;
  const completedCount = completedTrials().length;
  const parts = [
    `${state.filteredTrials.length} of ${state.trials.length} trials`,
    selectedVisible ? `${selectedVisible} selected` : null,
    state.parallelConstraints.size ? "parallel brush active" : null,
    completedCount > MAX_PLOT_POINTS
      ? `${MAX_PLOT_POINTS.toLocaleString()} plot points evenly sampled from ${completedCount.toLocaleString()}`
      : null,
  ].filter(Boolean);
  elements["filter-summary"].textContent = parts.join(" · ");
}

function tableFields() {
  const base = [
    "trial_id",
    "evaluation_index",
    "state",
    "objective_value",
    "duration_seconds",
  ];
  return [...base, ...numericParameterFields(state.trials).slice(0, 6)];
}

function renderTable() {
  const fields = tableFields();
  const headerRow = document.createElement("tr");
  const selectHeader = document.createElement("th");
  selectHeader.className = "trial-select";
  selectHeader.scope = "col";
  const selectAll = document.createElement("input");
  selectAll.type = "checkbox";
  selectAll.setAttribute("aria-label", "Select all visible trials");
  const visibleIds = state.filteredTrials.map(trialId);
  const selectedVisible = visibleIds.filter((id) => state.selectedIds.has(id)).length;
  selectAll.checked = visibleIds.length > 0 && selectedVisible === visibleIds.length;
  selectAll.indeterminate = selectedVisible > 0 && selectedVisible < visibleIds.length;
  selectAll.addEventListener("change", () => {
    visibleIds.forEach((id) =>
      selectAll.checked ? state.selectedIds.add(id) : state.selectedIds.delete(id),
    );
    renderDashboard({ skipParallel: true });
  });
  selectHeader.append(selectAll);
  headerRow.append(selectHeader);

  fields.forEach((field) => {
    const header = document.createElement("th");
    header.scope = "col";
    const button = document.createElement("button");
    button.type = "button";
    const active = state.sort.field === field;
    button.textContent = `${displayField(field)} ${active ? (state.sort.direction === "asc" ? "↑" : "↓") : ""}`;
    button.setAttribute(
      "aria-label",
      `Sort by ${displayField(field)}${active ? `, currently ${state.sort.direction}` : ""}`,
    );
    button.addEventListener("click", () => {
      state.sort = {
        field,
        direction: active && state.sort.direction === "asc" ? "desc" : "asc",
      };
      updateViewLocation();
      renderDashboard({ skipParallel: true });
    });
    header.append(button);
    headerRow.append(header);
  });
  elements["trial-head"].replaceChildren(headerRow);

  const rows = state.filteredTrials.slice(0, MAX_TABLE_ROWS).map((trial) => {
    const row = document.createElement("tr");
    const id = trialId(trial);
    if (state.selectedIds.has(id)) row.classList.add("is-selected");

    const selectCell = document.createElement("td");
    selectCell.className = "trial-select";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = state.selectedIds.has(id);
    checkbox.setAttribute("aria-label", `Select trial ${trialName(trial)}`);
    checkbox.addEventListener("change", () => {
      checkbox.checked ? state.selectedIds.add(id) : state.selectedIds.delete(id);
      renderDashboard({ skipParallel: true });
    });
    selectCell.append(checkbox);
    row.append(selectCell);

    fields.forEach((field) => {
      const cell = document.createElement("td");
      const value = canonicalFieldValue(trial, field);
      if (field === "state") {
        const label = document.createElement("span");
        label.className = "state-label";
        label.dataset.state = String(value || "unknown").toLowerCase();
        label.textContent = formatStatus(value);
        cell.append(label);
      } else if (field === "trial_id") {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "trial-link";
        button.textContent = trialName(trial);
        button.setAttribute("aria-label", `Inspect trial ${trialName(trial)}`);
        button.addEventListener("click", () => openTrialInspector(id));
        cell.append(button);
      } else {
        cell.textContent = formatValue(value);
      }
      cell.dataset.label = displayField(field);
      row.append(cell);
    });
    return row;
  });
  elements["trial-body"].replaceChildren(...rows);
  elements["table-summary"].textContent = `${state.filteredTrials.length} matching trials`;

  const notes = [];
  if (state.filteredTrials.length > MAX_TABLE_ROWS) {
    notes.push(`Showing the first ${MAX_TABLE_ROWS.toLocaleString()} table rows.`);
  }
  const apiRowsTotal = Number(state.queryMetadata.rows_total ?? state.trials.length);
  const apiRowsReturned = Number(
    state.queryMetadata.rows_returned ?? state.trials.length,
  );
  if (apiRowsTotal > apiRowsReturned) {
    notes.push(
      `Loaded ${apiRowsReturned.toLocaleString()} of ${apiRowsTotal.toLocaleString()} API rows.`,
    );
  } else if (state.queryMetadata.sampled) {
    notes.push(`The API marked these ${apiRowsReturned.toLocaleString()} rows as sampled.`);
  }
  elements["table-note"].textContent = notes.join(" ");
}

function plotColours() {
  const styles = getComputedStyle(document.documentElement);
  return {
    foreground: styles.getPropertyValue("--foreground").trim(),
    muted: styles.getPropertyValue("--muted").trim(),
    surface: styles.getPropertyValue("--surface-raised").trim(),
    grid: styles.getPropertyValue("--chart-grid").trim(),
    primary: styles.getPropertyValue("--chart-1").trim(),
    secondary: styles.getPropertyValue("--chart-2").trim(),
    tertiary: styles.getPropertyValue("--chart-3").trim(),
  };
}

function plotLayout(overrides = {}) {
  const colours = plotColours();
  return {
    autosize: true,
    paper_bgcolor: colours.surface,
    plot_bgcolor: colours.surface,
    font: { color: colours.foreground, family: "Inter, system-ui, sans-serif", size: 12 },
    margin: { l: 62, r: 20, t: 20, b: 52 },
    hoverlabel: { bgcolor: colours.surface, bordercolor: colours.grid, font: { color: colours.foreground } },
    xaxis: {
      gridcolor: colours.grid,
      linecolor: colours.grid,
      zerolinecolor: colours.grid,
      automargin: true,
    },
    yaxis: {
      gridcolor: colours.grid,
      linecolor: colours.grid,
      zerolinecolor: colours.grid,
      automargin: true,
    },
    showlegend: false,
    ...overrides,
  };
}

const plotConfig = {
  responsive: true,
  displaylogo: false,
  scrollZoom: true,
  modeBarButtonsToRemove: ["toImage", "sendDataToCloud"],
};

function plotAvailable(element) {
  if (window.Plotly) return true;
  showEmptyPlot(element, "Plotly.js is missing from the offline dashboard assets.");
  return false;
}

function showEmptyPlot(element, message) {
  if (window.Plotly && element.data) window.Plotly.purge(element);
  element.classList.add("plot-empty");
  element.textContent = message;
}

function preparePlot(element) {
  element.classList.remove("plot-empty");
  if (!element.data) element.textContent = "";
}

function completedOrdered() {
  return completedTrials().sort(
    (left, right) => evaluationIndex(left) - evaluationIndex(right),
  );
}

function samplePlotPoints(items) {
  if (items.length <= MAX_PLOT_POINTS) return items;
  const lastIndex = items.length - 1;
  return Array.from({ length: MAX_PLOT_POINTS }, (_, index) =>
    items[Math.round((index * lastIndex) / (MAX_PLOT_POINTS - 1))],
  );
}

function renderHistory() {
  const element = elements["history-chart"];
  const trials = samplePlotPoints(completedOrdered());
  if (!plotAvailable(element) || trials.length === 0) {
    showEmptyPlot(element, "No numeric objective history for these filters.");
    return;
  }
  preparePlot(element);
  const colours = plotColours();
  const ids = trials.map(trialId);
  window.Plotly.react(
    element,
    [
      {
        type: "scattergl",
        mode: "lines+markers",
        x: trials.map((trial, index) => evaluationIndex(trial, index)),
        y: trials.map(objectiveValue),
        customdata: ids,
        text: trials.map(trialName),
        line: { color: colours.primary, width: 2 },
        marker: {
          color: ids.map((id) =>
            state.selectedIds.has(id) ? colours.secondary : colours.primary,
          ),
          size: ids.map((id) => (state.selectedIds.has(id) ? 10 : 6)),
          symbol: ids.map((id) => (state.selectedIds.has(id) ? "diamond" : "circle")),
        },
        hovertemplate: "Evaluation %{x}<br>Objective %{y:.6g}<br>Trial %{text}<extra></extra>",
      },
    ],
    plotLayout({
      xaxis: { ...plotLayout().xaxis, title: "Evaluation" },
      yaxis: { ...plotLayout().yaxis, title: "Objective" },
      uirevision: `history-${studyId(state.study)}`,
    }),
    plotConfig,
  );
  bindPointSelection(element);
}

function renderBestSoFar() {
  const element = elements["best-chart"];
  const allTrials = completedOrdered();
  if (!plotAvailable(element) || allTrials.length === 0) {
    showEmptyPlot(element, "No completed objective values for these filters.");
    return;
  }
  preparePlot(element);
  const minimise = String(state.study.direction).toLowerCase() === "minimize";
  let best = objectiveValue(allTrials[0]);
  const points = allTrials.map((trial) => {
    const value = objectiveValue(trial);
    best = minimise ? Math.min(best, value) : Math.max(best, value);
    return { trial, best };
  });
  const sampledPoints = samplePlotPoints(points);
  const colours = plotColours();
  window.Plotly.react(
    element,
    [
      {
        type: "scatter",
        mode: "lines",
        line: { color: colours.tertiary, width: 2.5, shape: "hv" },
        fill: "tozeroy",
        fillcolor: colorWithAlpha(colours.tertiary, 0.1),
        x: sampledPoints.map(({ trial }, index) => evaluationIndex(trial, index)),
        y: sampledPoints.map(({ best: value }) => value),
        customdata: sampledPoints.map(({ trial }) => trialId(trial)),
        hovertemplate: "Evaluation %{x}<br>Best %{y:.6g}<extra></extra>",
      },
    ],
    plotLayout({
      xaxis: { ...plotLayout().xaxis, title: "Evaluation" },
      yaxis: { ...plotLayout().yaxis, title: `Best (${minimise ? "min" : "max"})` },
      uirevision: `best-${studyId(state.study)}`,
    }),
    plotConfig,
  );
}

function colorWithAlpha(colour, alpha) {
  if (!colour.startsWith("#")) return colour;
  const hex = colour.slice(1);
  const expanded = hex.length === 3 ? [...hex].map((part) => part + part).join("") : hex;
  const red = parseInt(expanded.slice(0, 2), 16);
  const green = parseInt(expanded.slice(2, 4), 16);
  const blue = parseInt(expanded.slice(4, 6), 16);
  return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
}

function renderParallelCoordinates() {
  const element = elements["parallel-chart"];
  const trials = samplePlotPoints(completedTrials());
  const fields = numericParameterFields(trials).slice(0, MAX_PARALLEL_DIMENSIONS);
  if (!plotAvailable(element) || trials.length < 2 || fields.length < 2) {
    state.parallelFields = [];
    showEmptyPlot(element, "At least two numeric parameters and two trials are required.");
    return;
  }
  preparePlot(element);
  state.parallelFields = [...fields, "objective_value"];
  const colours = plotColours();
  const objectives = trials.map(objectiveValue);
  const dimensions = fields.map((field) => ({
    label: displayField(field),
    values: trials.map((trial) => Number(fieldValue(trial, field))),
    constraintrange: state.parallelConstraints.get(field),
  }));
  dimensions.push({
    label: "Objective",
    values: objectives,
    constraintrange: state.parallelConstraints.get("objective_value"),
  });
  window.Plotly.react(
    element,
    [
      {
        type: "parcoords",
        dimensions,
        labelangle: -24,
        labelside: "top",
        labelfont: { color: colours.foreground, size: 12 },
        tickfont: { color: colours.muted, size: 10 },
        rangefont: { color: colours.muted, size: 10 },
        line: {
          color: objectives,
          colorscale: [
            [0, colours.primary],
            [1, colours.secondary],
          ],
          showscale: true,
          colorbar: { title: "Objective", thickness: 12 },
        },
      },
    ],
    plotLayout({ margin: { l: 48, r: 64, t: 72, b: 24 } }),
    { ...plotConfig, scrollZoom: false },
  );
  bindParallelSelection(element);
}

function renderImportance() {
  const element = elements["importance-chart"];
  const analysis = state.importance || {};
  const parameters = asArray(analysis.parameters).filter(
    (item) => Number.isFinite(Number(item?.importance)),
  );
  if (!plotAvailable(element) || parameters.length === 0) {
    elements["importance-note"].textContent =
      analysis.reason || "No numeric hyperparameter importance is available.";
    showEmptyPlot(element, analysis.reason || "No importance estimates are available.");
    return;
  }

  preparePlot(element);
  const colours = plotColours();
  const rows = [...parameters].sort(
    (left, right) => Number(left.importance) - Number(right.importance),
  );
  const labels = rows.map((item) => displayField(String(item.field)));
  const importances = rows.map((item) => Number(item.importance));
  const correlations = rows.map((item) =>
    Number.isFinite(Number(item.correlation)) ? Number(item.correlation) : null,
  );
  const maximumImportance = Math.max(...importances, 0.01);
  const trialLabel = `${Number(analysis.n_trials).toLocaleString()} completed trials`;
  elements["importance-note"].textContent =
    `${trialLabel} · RandomForestRegressor (${analysis.n_estimators} trees) · Pearson r`;

  window.Plotly.react(
    element,
    [
      {
        type: "bar",
        orientation: "h",
        x: importances,
        y: labels,
        xaxis: "x",
        marker: { color: colours.primary },
        text: importances.map((value) =>
          formatNumber(value, IMPORTANCE_SIGNIFICANT_DIGITS),
        ),
        textposition: "outside",
        cliponaxis: false,
        hovertemplate: "%{y}<br>Random forest importance %{x:.3g}<extra></extra>",
      },
      {
        type: "bar",
        orientation: "h",
        x: correlations,
        y: labels,
        xaxis: "x2",
        marker: { color: colours.tertiary },
        text: correlations.map((value) =>
          value == null
            ? "—"
            : formatNumber(value, IMPORTANCE_SIGNIFICANT_DIGITS),
        ),
        textposition: "outside",
        cliponaxis: false,
        hovertemplate: "%{y}<br>Pearson r %{x:.3g}<extra></extra>",
      },
    ],
    plotLayout({
      margin: { l: 138, r: 30, t: 58, b: 56 },
      bargap: 0.32,
      annotations: [
        {
          xref: "paper",
          yref: "paper",
          x: 0.2,
          y: 1.16,
          text: "Random forest importance",
          showarrow: false,
          font: { color: colours.foreground, size: 12 },
        },
        {
          xref: "paper",
          yref: "paper",
          x: 0.79,
          y: 1.16,
          text: "Pearson r",
          showarrow: false,
          font: { color: colours.foreground, size: 12 },
        },
      ],
      xaxis: {
        ...plotLayout().xaxis,
        domain: [0, 0.4],
        range: [0, maximumImportance * 1.28],
        title: "Random forest importance",
        tickformat: ".3g",
      },
      xaxis2: {
        ...plotLayout().xaxis,
        domain: [0.58, 1],
        range: [-1.18, 1.18],
        title: "Pearson r",
        tickformat: ".3g",
        zeroline: true,
        zerolinecolor: colours.foreground,
      },
      yaxis: {
        ...plotLayout().yaxis,
        categoryorder: "array",
        categoryarray: labels,
        automargin: true,
      },
      uirevision: `importance-${studyId(state.study)}-${analysis.study_revision}`,
    }),
    { ...plotConfig, scrollZoom: false },
  );
}

function normaliseConstraint(value) {
  if (value == null) return null;
  if (Array.isArray(value) && value.length === 2 && !Array.isArray(value[0])) {
    return value.map(Number);
  }
  if (Array.isArray(value)) return value.map((range) => range.map(Number));
  return null;
}

function matchesConstraint(value, constraint) {
  const number = Number(value);
  if (!Number.isFinite(number) || !constraint) return false;
  const ranges = Array.isArray(constraint[0]) ? constraint : [constraint];
  return ranges.some(([minimum, maximum]) => number >= minimum && number <= maximum);
}

function bindParallelSelection(element) {
  if (element.dataset.selectionBound === "true") return;
  element.dataset.selectionBound = "true";
  element.on("plotly_restyle", (update) => {
    let changed = false;
    Object.entries(update || {}).forEach(([key, rawValue]) => {
      const match = key.match(/^dimensions\[(\d+)]\.constraintrange$/);
      if (!match) return;
      const field = state.parallelFields[Number(match[1])];
      if (!field) return;
      const value = Array.isArray(rawValue) && rawValue.length === 1 ? rawValue[0] : rawValue;
      const constraint = normaliseConstraint(value);
      constraint
        ? state.parallelConstraints.set(field, constraint)
        : state.parallelConstraints.delete(field);
      changed = true;
    });
    if (!changed) return;
    state.selectedIds.clear();
    if (state.parallelConstraints.size > 0) {
      completedTrials()
        .filter((trial) =>
          [...state.parallelConstraints.entries()].every(([field, constraint]) =>
            matchesConstraint(canonicalFieldValue(trial, field), constraint),
          ),
        )
        .forEach((trial) => state.selectedIds.add(trialId(trial)));
    }
    renderDashboard({ skipParallel: true });
  });
}

function fieldScale(field) {
  const path = field.replace(/^params\./, "").split(".");
  const schema = state.study.search_space;
  const definition = path.reduce((value, key) => value?.[key], schema);
  return String(definition?.scale || definition?.type || "").toLowerCase().includes("log")
    ? "log"
    : "linear";
}

function fieldExtent(field) {
  const values = state.trials
    .map((trial) => numericFieldValue(trial, field))
    .filter((value) => value != null);
  return values.length === 0 ? null : [Math.min(...values), Math.max(...values)];
}

function syncRangeFilter(index, { preserveValues = false } = {}) {
  const filter = state.explorer.filters[index];
  const number = index + 1;
  const minimum = elements[`explorer-filter-${number}-min`];
  const maximum = elements[`explorer-filter-${number}-max`];
  const extent = filter.field ? fieldExtent(filter.field) : null;
  if (!preserveValues) {
    filter.minimum = "";
    filter.maximum = "";
  }
  minimum.disabled = !extent;
  maximum.disabled = !extent;
  minimum.value = filter.minimum;
  maximum.value = filter.maximum;
  minimum.placeholder = extent ? formatNumber(extent[0]) : "—";
  maximum.placeholder = extent ? formatNumber(extent[1]) : "—";
}

function paretoCompatible() {
  return new Set([state.explorer.x, state.explorer.y]).size === 2 &&
    [state.explorer.x, state.explorer.y].includes("duration_seconds") &&
    [state.explorer.x, state.explorer.y].includes("objective_value");
}

function syncParetoControl() {
  const compatible = paretoCompatible();
  if (!compatible) state.explorer.pareto = false;
  elements["explorer-pareto"].disabled = !compatible;
  elements["explorer-pareto"].checked = compatible && state.explorer.pareto;
}

function explorerViewSpec() {
  return {
    version: 1,
    study_id: studyId(state.study),
    chart: "scatter",
    x: { field: state.explorer.x, scale: state.explorer.xScale },
    y: { field: state.explorer.y, scale: state.explorer.yScale },
    color: state.explorer.color ? { field: state.explorer.color } : null,
    filters: state.explorer.filters
      .filter((filter) => filter.field)
      .map((filter) => ({
        field: filter.field,
        minimum: filter.minimum === "" ? null : Number(filter.minimum),
        maximum: filter.maximum === "" ? null : Number(filter.maximum),
      })),
    pareto: state.explorer.pareto && paretoCompatible()
      ? {
          cost: { field: "duration_seconds", direction: "minimize" },
          score: {
            field: "objective_value",
            direction: String(state.study.direction || "maximize").toLowerCase(),
          },
        }
      : null,
  };
}

function explorerTrials() {
  const { x, y, color, xScale, yScale } = state.explorer;
  return completedTrials().filter((trial) => {
    const xValue = numericFieldValue(trial, x);
    const yValue = numericFieldValue(trial, y);
    if (xValue == null || yValue == null) return false;
    if ((xScale === "log" && xValue <= 0) || (yScale === "log" && yValue <= 0)) {
      return false;
    }
    if (color && numericFieldValue(trial, color) == null) {
      return false;
    }
    return state.explorer.filters.every((filter) => {
      if (!filter.field) return true;
      const value = numericFieldValue(trial, filter.field);
      if (value == null) return false;
      const minimum = filter.minimum === "" ? null : Number(filter.minimum);
      const maximum = filter.maximum === "" ? null : Number(filter.maximum);
      if (Number.isFinite(minimum) && value < minimum) return false;
      if (Number.isFinite(maximum) && value > maximum) return false;
      return true;
    });
  });
}

function objectiveDurationPareto(trials) {
  const minimiseObjective = String(state.study.direction).toLowerCase() === "minimize";
  const sorted = [...trials].sort((left, right) => {
    const costDifference = Number(left.duration_seconds) - Number(right.duration_seconds);
    if (costDifference !== 0) return costDifference;
    const objectiveDifference = objectiveValue(left) - objectiveValue(right);
    return minimiseObjective ? objectiveDifference : -objectiveDifference;
  });
  const frontier = [];
  let bestObjective = minimiseObjective ? Number.POSITIVE_INFINITY : Number.NEGATIVE_INFINITY;
  for (let index = 0; index < sorted.length; ) {
    const cost = Number(sorted[index].duration_seconds);
    const group = [];
    while (index < sorted.length && Number(sorted[index].duration_seconds) === cost) {
      group.push(sorted[index]);
      index += 1;
    }
    const groupBest = group.reduce((best, trial) => {
      const value = objectiveValue(trial);
      return minimiseObjective ? Math.min(best, value) : Math.max(best, value);
    }, minimiseObjective ? Number.POSITIVE_INFINITY : Number.NEGATIVE_INFINITY);
    const improves = minimiseObjective ? groupBest < bestObjective : groupBest > bestObjective;
    if (improves) {
      frontier.push(...group.filter((trial) => objectiveValue(trial) === groupBest));
      bestObjective = groupBest;
    }
  }
  return frontier;
}

function renderExplorer() {
  const element = elements["explorer-chart"];
  const { x, y, color, xScale, yScale } = state.explorer;
  const allTrials = explorerTrials();
  const trials = samplePlotPoints(allTrials);
  if (!plotAvailable(element) || !x || !y || trials.length === 0) {
    elements["explorer-note"].textContent =
      "Choose two numeric fields with completed trial values.";
    showEmptyPlot(element, "No completed trials match this view.");
    return;
  }

  preparePlot(element);
  const colours = plotColours();
  const ids = trials.map(trialId);
  const marker = {
    color: color
      ? trials.map((trial) => numericFieldValue(trial, color))
      : colours.primary,
    colorscale: color ? "Viridis" : undefined,
    colorbar: color
      ? { title: { text: displayField(color), side: "right" }, thickness: 14 }
      : undefined,
    showscale: Boolean(color),
    size: ids.map((id) => (state.selectedIds.has(id) ? 12 : 8)),
    symbol: ids.map((id) => (state.selectedIds.has(id) ? "diamond" : "circle")),
    line: {
      color: ids.map((id) =>
        state.selectedIds.has(id) ? colours.secondary : colorWithAlpha(colours.surface, 0.8),
      ),
      width: ids.map((id) => (state.selectedIds.has(id) ? 2 : 0.5)),
    },
    opacity: 0.86,
  };
  const traces = [
    {
      type: "scattergl",
      mode: "markers",
      name: "Trials",
      x: trials.map((trial) => numericFieldValue(trial, x)),
      y: trials.map((trial) => numericFieldValue(trial, y)),
      customdata: ids,
      text: trials.map(trialName),
      marker,
      hovertemplate: `${displayField(x)} %{x:.6g}<br>${displayField(y)} %{y:.6g}<br>Trial %{text}<extra></extra>`,
    },
  ];
  let frontierCount = 0;
  if (state.explorer.pareto && paretoCompatible()) {
    const frontier = objectiveDurationPareto(allTrials);
    frontierCount = frontier.length;
    traces.push({
      type: "scatter",
      mode: "lines+markers",
      name: "Pareto front",
      x: frontier.map((trial) => numericFieldValue(trial, x)),
      y: frontier.map((trial) => numericFieldValue(trial, y)),
      customdata: frontier.map(trialId),
      text: frontier.map(trialName),
      line: { color: colours.tertiary, width: 3 },
      marker: { color: colours.tertiary, size: 9, symbol: "diamond-open", line: { width: 2 } },
      hovertemplate: `Pareto trial %{text}<br>${displayField(x)} %{x:.6g}<br>${displayField(y)} %{y:.6g}<extra></extra>`,
    });
  }
  const filterCount = state.explorer.filters.filter((filter) => filter.field).length;
  const notes = [
    `${allTrials.length.toLocaleString()} matching completed trials`,
    color ? `colour: ${displayField(color)}` : null,
    filterCount ? `${filterCount} range ${filterCount === 1 ? "filter" : "filters"}` : null,
    frontierCount ? `${frontierCount} Pareto-optimal` : null,
    frontierCount ? "cost: recorded duration" : null,
    allTrials.length > trials.length ? `${trials.length.toLocaleString()} displayed` : null,
  ].filter(Boolean);
  elements["explorer-note"].textContent = notes.join(" · ");

  window.Plotly.react(
    element,
    traces,
    plotLayout({
      showlegend: frontierCount > 0,
      legend: { orientation: "h", x: 0, y: 1.08 },
      xaxis: {
        ...plotLayout().xaxis,
        title: {
          text: displayField(x),
          standoff: 14,
          font: { color: colours.foreground, size: 13 },
        },
        type: xScale,
      },
      yaxis: {
        ...plotLayout().yaxis,
        title: {
          text: displayField(y),
          standoff: 14,
          font: { color: colours.foreground, size: 13 },
        },
        type: yScale,
      },
      margin: { l: 84, r: color ? 96 : 30, t: frontierCount ? 54 : 24, b: 74 },
      uirevision: `explorer-${studyId(state.study)}-${x}-${y}-${xScale}-${yScale}`,
    }),
    plotConfig,
  );
  bindPointSelection(element);
}

function bindPointSelection(element) {
  if (element.dataset.pointSelectionBound === "true") return;
  element.dataset.pointSelectionBound = "true";
  element.on("plotly_click", (event) => {
    const id = event?.points?.[0]?.customdata;
    if (id == null) return;
    openTrialInspector(String(id));
  });
}

function trialById(id) {
  return state.trials.find((trial) => trialId(trial) === String(id));
}

function summaryItem(label, value) {
  const wrapper = document.createElement("div");
  const term = document.createElement("dt");
  const description = document.createElement("dd");
  term.textContent = label;
  description.textContent = value;
  wrapper.append(term, description);
  return wrapper;
}

function renderTrialInspector() {
  if (!state.inspectedTrialId) return;
  const trial = trialById(state.inspectedTrialId);
  if (!trial) {
    state.inspectedTrialId = "";
    updateViewLocation();
    if (elements["trial-inspector"].open) elements["trial-inspector"].close();
    return;
  }
  elements["inspector-title"].textContent = `Trial ${trialName(trial)}`;
  elements["inspector-summary"].replaceChildren(
    summaryItem("Objective", formatNumber(objectiveValue(trial))),
    summaryItem("State", formatStatus(trial.state)),
    summaryItem("Duration", trial.duration_seconds == null ? "—" : `${formatNumber(trial.duration_seconds)} s`),
    summaryItem("Evaluation", formatValue(evaluationIndex(trial))),
    summaryItem("Batch", `${formatValue(trial.batch_index)} · slot ${formatValue(trial.batch_slot)}`),
    summaryItem("Worker", formatValue(trial.worker)),
  );
  elements["inspector-config"].textContent = JSON.stringify(trial.params ?? {}, null, 2);

  const sections = [];
  const addSection = (title, value, className = "") => {
    if (value == null || value === "" || (typeof value === "object" && Object.keys(value).length === 0)) {
      return;
    }
    const section = document.createElement("section");
    const heading = document.createElement("h3");
    const content = document.createElement("p");
    heading.textContent = title;
    content.textContent = typeof value === "object" ? JSON.stringify(value, null, 2) : String(value);
    if (className) content.className = className;
    section.append(heading, content);
    sections.push(section);
  };
  addSection("Timing", [trial.started_at, trial.finished_at].filter(Boolean).join(" → "));
  addSection("Seed", trial.seed);
  addSection("Metadata", trial.metadata);
  addSection("Error", trial.error, "text-danger");
  elements["inspector-extra"].replaceChildren(...sections);
}

function restoreTrialInspector() {
  if (!state.inspectedTrialId || !trialById(state.inspectedTrialId)) {
    state.inspectedTrialId = "";
    updateViewLocation();
    return;
  }
  state.selectedIds.add(state.inspectedTrialId);
  renderTrialInspector();
  if (!elements["trial-inspector"].open) elements["trial-inspector"].showModal();
}

function openTrialInspector(id) {
  const trial = trialById(id);
  if (!trial) return;
  state.inspectedTrialId = trialId(trial);
  state.selectedIds.add(state.inspectedTrialId);
  updateViewLocation();
  renderDashboard({ skipParallel: true });
  if (!elements["trial-inspector"].open) elements["trial-inspector"].showModal();
}

function closeTrialInspector() {
  state.inspectedTrialId = "";
  updateViewLocation();
  if (elements["trial-inspector"].open) elements["trial-inspector"].close();
}

async function copyText(text) {
  if (!navigator.clipboard?.writeText) {
    throw new Error("Clipboard access is unavailable in this browser.");
  }
  await navigator.clipboard.writeText(text);
}

async function copyViewLink() {
  updateViewLocation();
  try {
    await copyText(window.location.href);
    elements["view-feedback"].textContent = "Shareable view link copied.";
  } catch (error) {
    elements["view-feedback"].textContent = error.message;
  }
}

async function copyViewSpec() {
  try {
    await copyText(JSON.stringify(explorerViewSpec(), null, 2));
    elements["view-feedback"].textContent = "Declarative view JSON copied.";
  } catch (error) {
    elements["view-feedback"].textContent = error.message;
  }
}

async function copyTrialConfig() {
  const trial = trialById(state.inspectedTrialId);
  if (!trial) return;
  try {
    await copyText(JSON.stringify(trial.params ?? {}, null, 2));
    elements["inspector-feedback"].textContent = "Trial configuration copied.";
  } catch (error) {
    elements["inspector-feedback"].textContent = error.message;
  }
}

function mergeChanges(payload) {
  const previousRevision = state.revision;
  const previousStatus = studyStatus();
  const updates = trialsFrom(payload).map(prepareTrial);
  if (updates.length > 0) {
    const byId = new Map(state.trials.map((trial) => [trialId(trial), trial]));
    updates.forEach((trial) => byId.set(trialId(trial), trial));
    state.trials = [...byId.values()];
  }
  const nextStatus = payload?.study_status ?? payload?.status;
  if (nextStatus) state.study = { ...state.study, status: nextStatus };
  const nextRevision = Number(payload?.study_revision ?? payload?.revision ?? state.revision);
  const statusChanged = nextStatus && String(nextStatus).toLowerCase() !== previousStatus;
  const changed = updates.length > 0 || nextRevision !== previousRevision || statusChanged;
  state.revision = nextRevision;
  return changed;
}

function clearPolling() {
  if (state.pollTimer) window.clearTimeout(state.pollTimer);
  state.pollTimer = null;
}

function pollingDelay() {
  if (!LIVE_STATES.has(studyStatus())) return null;
  const idleDelay = Math.min(
    LIVE_POLL_INTERVAL_MS * 2 ** state.unchangedPolls,
    MAX_IDLE_POLL_INTERVAL_MS,
  );
  return Math.min(idleDelay * 2 ** state.pollFailures, MAX_ERROR_POLL_INTERVAL_MS);
}

function schedulePoll(delay = null) {
  clearPolling();
  if (document.hidden || !state.study || !LIVE_STATES.has(studyStatus())) return;
  const nextDelay = delay ?? pollingDelay();
  if (nextDelay == null) return;
  state.pollTimer = window.setTimeout(pollChanges, nextDelay);
}

async function pollChanges() {
  if (!state.study || document.hidden) return;
  const id = studyId(state.study);
  try {
    setConnection("syncing", "Checking updates");
    const payload = await request(
      `studies/${encodeURIComponent(id)}/changes?since_revision=${state.revision}`,
    );
    state.pollFailures = 0;
    const changed = mergeChanges(payload);
    state.unchangedPolls = changed ? 0 : state.unchangedPolls + 1;
    if (changed) {
      state.importance = await request(
        `studies/${encodeURIComponent(id)}/importance`,
      ).catch((error) => ({ parameters: [], reason: error.message }));
      configureFilterChoices();
      renderDashboard();
    }
    setConnection("connected", LIVE_STATES.has(studyStatus()) ? "Live" : "Local");
  } catch (error) {
    state.pollFailures += 1;
    setConnection("error", `Retrying · ${error.message}`);
  }
  schedulePoll();
}

function resetFilters() {
  state.search = "";
  state.trialState = "";
  state.selectedIds.clear();
  state.parallelConstraints.clear();
  state.explorer.filters = [
    { field: "", minimum: "", maximum: "" },
    { field: "", minimum: "", maximum: "" },
  ];
  elements["trial-search"].value = "";
  elements["state-filter"].value = "";
  configureExplorerChoices();
  updateViewLocation();
  renderDashboard();
}

function applyTheme() {
  if (state.theme === "system") {
    delete document.documentElement.dataset.theme;
  } else {
    document.documentElement.dataset.theme = state.theme;
  }
  elements["theme-toggle"].textContent = `Theme: ${state.theme}`;
  elements["theme-toggle"].setAttribute(
    "aria-label",
    `Current theme is ${state.theme}; activate to change theme`,
  );
}

function cycleTheme() {
  const modes = ["system", "light", "dark"];
  state.theme = modes[(modes.indexOf(state.theme) + 1) % modes.length];
  localStorage.setItem("hyperoptax-theme", state.theme);
  applyTheme();
  if (state.study) renderDashboard();
}

elements["study-select"].addEventListener("change", (event) => {
  loadStudy(event.target.value);
});
elements["trial-search"].addEventListener("input", (event) => {
  state.search = event.target.value;
  updateViewLocation();
  renderDashboard({ skipParallel: true });
});
elements["state-filter"].addEventListener("change", (event) => {
  state.trialState = event.target.value;
  updateViewLocation();
  renderDashboard();
});
elements["explorer-x"].addEventListener("change", (event) => {
  state.explorer.x = event.target.value;
  state.explorer.xScale = fieldScale(state.explorer.x);
  elements["explorer-x-scale"].value = state.explorer.xScale;
  syncParetoControl();
  updateViewLocation();
  renderExplorer();
});
elements["explorer-y"].addEventListener("change", (event) => {
  state.explorer.y = event.target.value;
  state.explorer.yScale = fieldScale(state.explorer.y);
  elements["explorer-y-scale"].value = state.explorer.yScale;
  syncParetoControl();
  updateViewLocation();
  renderExplorer();
});
elements["explorer-color"].addEventListener("change", (event) => {
  state.explorer.color = event.target.value;
  updateViewLocation();
  renderExplorer();
});
elements["explorer-x-scale"].addEventListener("change", (event) => {
  state.explorer.xScale = event.target.value;
  updateViewLocation();
  renderExplorer();
});
elements["explorer-y-scale"].addEventListener("change", (event) => {
  state.explorer.yScale = event.target.value;
  updateViewLocation();
  renderExplorer();
});
elements["explorer-pareto"].addEventListener("change", (event) => {
  state.explorer.pareto = event.target.checked;
  updateViewLocation();
  renderExplorer();
});
state.explorer.filters.forEach((_, index) => {
  const number = index + 1;
  elements[`explorer-filter-${number}-field`].addEventListener("change", (event) => {
    state.explorer.filters[index].field = event.target.value;
    syncRangeFilter(index);
    updateViewLocation();
    renderExplorer();
  });
  for (const bound of ["min", "max"]) {
    elements[`explorer-filter-${number}-${bound}`].addEventListener("input", (event) => {
      const key = bound === "min" ? "minimum" : "maximum";
      state.explorer.filters[index][key] = event.target.value;
      updateViewLocation();
      renderExplorer();
    });
  }
});
elements["reset-filters"].addEventListener("click", resetFilters);
elements["theme-toggle"].addEventListener("click", cycleTheme);
elements["copy-view-link"].addEventListener("click", copyViewLink);
elements["copy-view-spec"].addEventListener("click", copyViewSpec);
elements["close-inspector"].addEventListener("click", closeTrialInspector);
elements["copy-trial-config"].addEventListener("click", copyTrialConfig);
elements["trial-inspector"].addEventListener("close", () => {
  if (!state.inspectedTrialId) return;
  state.inspectedTrialId = "";
  updateViewLocation();
});
elements["empty-refresh"].addEventListener("click", () => loadStudies());
elements["error-retry"].addEventListener("click", () => loadStudies(studyId(state.study)));
document.addEventListener("visibilitychange", () => {
  if (document.hidden) clearPolling();
  else if (state.study && LIVE_STATES.has(studyStatus())) pollChanges();
});
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
  if (state.theme === "system" && state.study) renderDashboard();
});

applyTheme();
loadStudies();
