// /static/js/app.js
"use strict";

/* =========================================================
 * Utilidades compartidas
 * =======================================================*/
const $$ = {
  qs: (sel, el = document) => el.querySelector(sel),
  qsa: (sel, el = document) => Array.from(el.querySelectorAll(sel)),
  setText: (el, text) => { if (el) el.textContent = text ?? "—"; },
  show: (el) => el && el.classList.remove("is-hidden"),
  hide: (el) => el && el.classList.add("is-hidden"),
  sleep: (ms) => new Promise((r) => setTimeout(r, ms)),
  safeText: async (resp) => { try { return await resp.text(); } catch { return ""; } },
  safeNumber: (v) => {
    if (v == null || v === "") return "—";
    const n = Number(v);
    if (!Number.isFinite(n)) return String(v);
    if (Math.abs(n) >= 100) return n.toFixed(0);
    if (Math.abs(n) >= 10) return n.toFixed(1);
    return n.toFixed(2);
  },
  setStatus(el, message, type = "info") {
    if (!el) return;
    el.className = `status status--${type}`;
    el.textContent = message;
  },
};

/* =========================================================
 * Página: INDEX (uploader, schema, process)
 * Se activa si existen elementos del index en DOM
 * =======================================================*/
document.addEventListener("DOMContentLoaded", () => {
  const isIndex = !!$$.qs("#uploadForm");
  if (!isIndex) return;

  // ---- Elementos (Uploader) ----
  const uploadModeInputs = $$.qsa('input[name="upload-mode"]');
  const uploadForm = $$.qs("#uploadForm");
  const fileInputRow = $$.qs('.form-row[data-mode="file"]');
  const urlInputRow = $$.qs('.form-row[data-mode="url"]');
  const fileInput = $$.qs("#fileInput");
  const bucketUrl = $$.qs("#bucketUrl");
  const btnResetUpload = $$.qs("#btnResetUpload");
  const uploadStatus = $$.qs("#uploadStatus");
  const uploadMeta = $$.qs("#uploadMeta");

  // ---- Elementos (Schema & Processing) ----
  const fileIdSpan = $$.qs("#fileId");
  const engineNameSpan = $$.qs("#engineName");
  const rowCountSpan = $$.qs("#rowCount");
  const schemaTableBody = $$.qs("#schemaTable tbody");

  const processForm = $$.qs("#processForm");
  const albedoInput = $$.qs("#albedo");
  const autoDeriveInput = $$.qs("#autoDerive");
  const processStatus = $$.qs("#processStatus");
  const processMeta = $$.qs("#processMeta");

  // ---- Elementos (Results quick access) ----
  const linkDownloadCSV = $$.qs("#linkDownloadCSV");
  const linkDownloadJSON = $$.qs("#linkDownloadJSON");

  // ---- Estado ----
  let currentFileId = null;
  let currentEngine = null;
  let currentRowCount = null;

  // ---- Helpers UI ----
  function clearTable(tbody) {
    if (!tbody) return;
    while (tbody.firstChild) tbody.removeChild(tbody.firstChild);
  }
  function addSchemaRow(tbody, detected, mappedTo, status) {
    const tr = document.createElement("tr");
    const td1 = document.createElement("td");
    const td2 = document.createElement("td");
    const td3 = document.createElement("td");
    td1.textContent = detected ?? "—";
    td2.textContent = mappedTo ?? "—";
    td3.textContent = status ?? "ok";
    tr.appendChild(td1);
    tr.appendChild(td2);
    tr.appendChild(td3);
    tbody.appendChild(tr);
  }
  function fillSchemaTable(detectedColumns) {
    clearTable(schemaTableBody);
    if (!detectedColumns || detectedColumns.length === 0) {
      addSchemaRow(schemaTableBody, "—", "—", "no columns detected");
      return;
    }
    detectedColumns.forEach((col) => {
      if (typeof col === "string") {
        addSchemaRow(schemaTableBody, col, col, "ok");
      } else if (col && typeof col === "object") {
        addSchemaRow(
          schemaTableBody,
          col.detected ?? col.source ?? "—",
          col.mapped_to ?? col.mapped ?? col.target ?? "—",
          col.status ?? "ok"
        );
      }
    });
  }
  function resetUploadUI() {
    currentFileId = null;
    currentEngine = null;
    currentRowCount = null;
    $$.setText(fileIdSpan, "—");
    $$.setText(engineNameSpan, "—");
    $$.setText(rowCountSpan, "—");
    clearTable(schemaTableBody);
    uploadForm.reset();
    processForm.reset();
    $$.setStatus(uploadStatus, "Waiting for input…", "info");
    uploadMeta.textContent = "";
    processMeta.textContent = "";
    $$.setStatus(processStatus, "Idle.", "info");
    linkDownloadCSV?.setAttribute("href", "#");
    linkDownloadJSON?.setAttribute("href", "#");
  }

  // ---- Upload mode toggle ----
  function updateUploadMode() {
    const checked = uploadModeInputs.find((x) => x.checked);
    const mode = checked ? checked.value : "file";
    if (mode === "file") {
      $$.show(fileInputRow);
      $$.hide(urlInputRow);
    } else {
      $$.hide(fileInputRow);
      $$.show(urlInputRow);
    }
  }
  uploadModeInputs.forEach((input) => input.addEventListener("change", updateUploadMode));
  updateUploadMode();

  // ---- Upload handler ----
  uploadForm.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    $$.setStatus(uploadStatus, "Uploading…", "info");
    uploadMeta.textContent = "";

    try {
      const checked = uploadModeInputs.find((x) => x.checked);
      const mode = checked ? checked.value : "file";

      let resp;
      if (mode === "file") {
        const file = fileInput.files && fileInput.files[0];
        if (!file) {
          $$.setStatus(uploadStatus, "Please select a .csv or .parquet file.", "warn");
          return;
        }
        const formData = new FormData();
        formData.append("file", file);
        resp = await fetch("/api/upload", { method: "POST", body: formData });
      } else {
        const url = (bucketUrl.value || "").trim();
        if (!url) {
          $$.setStatus(uploadStatus, "Please provide a bucket URL.", "warn");
          return;
        }
        resp = await fetch("/api/upload", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ url }),
        });
      }

      if (!resp.ok) {
        const text = await $$.safeText(resp);
        throw new Error(`Upload failed (${resp.status}): ${text}`);
      }

      const data = await resp.json();
      currentFileId = data.file_id || data.id || null;
      currentEngine = data.engine || "pandas";
      currentRowCount = data.rows ?? data.row_count ?? null;

      $$.setText(fileIdSpan, currentFileId || "—");
      $$.setText(engineNameSpan, currentEngine || "—");
      $$.setText(rowCountSpan, currentRowCount ?? "—");

      fillSchemaTable(data.detected_columns || data.columns || []);

      $$.setStatus(uploadStatus, "Upload successful.", "success");
      uploadMeta.textContent = JSON.stringify(
        { file_id: currentFileId, engine: currentEngine, rows: currentRowCount },
        null,
        2
      );
    } catch (err) {
      console.error(err);
      $$.setStatus(uploadStatus, err.message || "Upload error.", "error");
    }
  });

  btnResetUpload?.addEventListener("click", resetUploadUI);

  // ---- Process handler ----
  processForm.addEventListener("submit", async (ev) => {
    ev.preventDefault();

    if (!currentFileId) {
      $$.setStatus(processStatus, "Please upload a dataset first.", "warn");
      return;
    }

    const albedo = parseFloat(albedoInput.value || "0.3");
    const autoDerive = !!autoDeriveInput.checked;

    $$.setStatus(processStatus, "Processing…", "info");
    processMeta.textContent = "";

    try {
      const resp = await fetch("/api/process", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          file_id: currentFileId,
          assumptions: { albedo },
          options: { auto_derive: autoDerive },
        }),
      });

      if (!resp.ok) {
        const text = await $$.safeText(resp);
        throw new Error(`Process failed (${resp.status}): ${text}`);
      }

      const data = await resp.json();
      if (data.csv_path || data.json_path) {
        publishResultLinks(data.csv_path, data.json_path);
        $$.setStatus(processStatus, "Processing complete.", "success");
        processMeta.textContent = JSON.stringify(data, null, 2);
        return;
      }

      const jobId = data.job_id || data.id;
      if (!jobId) {
        publishResultLinks(data.csv_path, data.json_path);
        $$.setStatus(processStatus, "Processed (no job id returned).", "success");
        processMeta.textContent = JSON.stringify(data, null, 2);
        return;
      }

      $$.setStatus(processStatus, `Queued (job: ${jobId}). Polling results…`, "info");
      const result = await pollResults(jobId, 10, 1500);
      publishResultLinks(result.csv_path, result.json_path);
      $$.setStatus(processStatus, "Processing complete.", "success");
      processMeta.textContent = JSON.stringify(result, null, 2);
    } catch (err) {
      console.error(err);
      $$.setStatus(processStatus, err.message || "Processing error.", "error");
    }
  });

  $$.qs("#btnCancelProcess")?.addEventListener("click", () => {
    $$.setStatus(processStatus, "Cancelled by user.", "warn");
  });

  function publishResultLinks(csvPath, jsonPath) {
    if (csvPath) linkDownloadCSV?.setAttribute("href", csvPath);
    else linkDownloadCSV?.setAttribute("href", "#");
    if (jsonPath) linkDownloadJSON?.setAttribute("href", jsonPath);
    else linkDownloadJSON?.setAttribute("href", "#");
  }

  async function pollResults(jobId, maxAttempts = 10, delayMs = 1500) {
    let attempt = 0;
    while (attempt < maxAttempts) {
      attempt += 1;
      const url = `/api/result/${encodeURIComponent(jobId)}`;
      const resp = await fetch(url, { method: "GET" });
      if (resp.ok) {
        const data = await resp.json();
        if (data.csv_path || data.json_path || (data.status && data.status === "done")) {
          return data;
        }
      }
      await $$.sleep(delayMs);
    }
    throw new Error("Result polling exceeded attempts.");
  }

  // Optional: ping backend
  (async () => {
    try {
      const resp = await fetch("/healthz");
      if (resp.ok) {
        const info = await resp.json();
        console.log("Backend health:", info);
      }
    } catch { /* ignore */ }
  })();
});


/* =========================================================
 * Página: RESULTS (DataTables + coloreo por disposición)
 * Se activa si existe la tabla #resultsTable
 * =======================================================*/
document.addEventListener("DOMContentLoaded", () => {
  const tableEl = $$.qs("#resultsTable");
  if (!tableEl) return; // no es la página de resultados

  const resultsStatus = $$.qs("#resultsStatus");
  const resultsSummary = $$.qs("#resultsSummary");
  const linkDownloadCSV = $$.qs("#linkDownloadCSV");
  const linkDownloadJSON = $$.qs("#linkDownloadJSON");

  const filterForm = $$.qs("#filterForm");
  const searchId = $$.qs("#searchId");
  const categorySelect = $$.qs("#categorySelect");
  const btnClearFilters = $$.qs("#btnClearFilters");
  const btnRefresh = $$.qs("#btnRefreshResults");

  let dataset = [];
  let dt = null;         // instancia DataTable
  let loading = false;

  function setDownloadsEnabled(enabled) {
    if (enabled) {
      linkDownloadCSV?.setAttribute("href", "/api/result/download/csv");
      linkDownloadJSON?.setAttribute("href", "/api/result/download/json");
      linkDownloadCSV?.removeAttribute("aria-disabled");
      linkDownloadJSON?.removeAttribute("aria-disabled");
    } else {
      linkDownloadCSV?.setAttribute("href", "#");
      linkDownloadJSON?.setAttribute("href", "#");
      linkDownloadCSV?.setAttribute("aria-disabled", "true");
      linkDownloadJSON?.setAttribute("aria-disabled", "true");
    }
  }

  function normStr(s) {
    if (!s) return "";
    return String(s).trim().toLowerCase().replace(/\s+/g, "_");
  }

  function computeColorHint(row) {
    if (row && row.color_hint) {
      const c = String(row.color_hint).toLowerCase();
      if (c === "green" || c === "black" || c === "red") return c;
    }
    const disp = normStr(row?.koi_disposition);
    const cat = normStr(row?.category);
    if (!disp) return "green";
    if (disp === cat) return "black";
    return "red";
  }

  function colorClassFromHint(hint) {
    if (hint === "green") return "disp-green";
    if (hint === "red") return "disp-red";
    return "disp-black";
  }

  async function waitForDataTables(maxMs = 5000, interval = 100) {
    let waited = 0;
    while (!window.DataTable && waited < maxMs) {
      await $$.sleep(interval);
      waited += interval;
    }
    return !!window.DataTable;
  }

  // Inicializa o re-renderiza DataTable con dataset actual
  async function renderDataTable(timestamp) {
    const ready = await waitForDataTables(6000, 100);
    if (!ready) {
      $$.setStatus(resultsStatus, "Error loading results: DataTables library not loaded.", "error");
      return;
    }

    const dataRows = dataset.map((r) => {
      const flagsStr = Array.isArray(r.flags) ? r.flags.join("; ") : (r.flags ?? "");
      return {
        _raw: r,
        object_id: r.object_id ?? "—",
        koi_disposition: r.koi_disposition ?? "",
        category: r.category ?? "",
        score: (r.score ?? "") === "" ? "—" : Number(r.score).toFixed(2),
        flags: flagsStr || "—",
        koi_period: $$.safeNumber(r.koi_period),
        koi_prad: $$.safeNumber(r.koi_prad),
        koi_teq: $$.safeNumber(r.koi_teq),
        koi_insol: $$.safeNumber(r.koi_insol),
      };
    });

    if (dt && typeof dt.destroy === "function") {
      dt.destroy();
      dt = null;
    }
    const tbody = tableEl.querySelector("tbody");
    while (tbody.firstChild) tbody.removeChild(tbody.firstChild);

    // Inicialización usando selector, tal como sugiere la documentación:
    dt = new window.DataTable("#resultsTable", {
      data: dataRows,
      columns: [
        { data: "object_id", title: "Object ID" },
        { data: "koi_disposition", title: "Original Disposition" },
        { data: "category", title: "Category (ExoHunter)" },
        { data: "score", title: "Score" },
        { data: "flags", title: "Flags" },
        { data: "koi_period", title: "Period (days)" },
        { data: "koi_prad", title: "Radius (R⊕)" },
        { data: "koi_teq", title: "Teq (K)" },
        { data: "koi_insol", title: "Insol (F/F⊕)" },
      ],
      pageLength: Number(tableEl.dataset.dtPageLength || 25),
      order: [[0, "asc"]],
      deferRender: true,
      responsive: true,
      rowCallback: (rowEl, rowData) => {
        const raw = rowData?._raw || {};
        const hint = computeColorHint(raw);
        const cls = colorClassFromHint(hint);
        rowEl.classList.remove("disp-green", "disp-black", "disp-red");
        rowEl.classList.add(cls);
      },
    });

    applyQuickFilters();

    const summary = { rows_total: dataset.length, rows_shown: dataset.length };
    if (timestamp) summary.timestamp = timestamp;
    $$.setText(resultsSummary, JSON.stringify(summary, null, 2));
  }

  function applyQuickFilters() {
    if (!dt) return;
    const idQuery = (searchId?.value || "").trim();
    const cat = (categorySelect?.value || "").trim();
    dt.columns(0).search(idQuery, true, false);
    dt.columns(2).search(cat ? `^${cat}$` : "", true, false);
    dt.draw();
  }

  // Carga datos del backend
  async function refresh() {
    if (loading) return;
    loading = true;
    btnRefresh?.setAttribute("disabled", "true");
    btnRefresh?.setAttribute("aria-busy", "true");
    $$.setStatus(resultsStatus, "Fetching latest results…", "info");

    try {
      const resp = await fetch("/api/result/latest");
      if (!resp.ok) {
        let errMsg = `HTTP ${resp.status}`;
        try {
          const errData = await resp.json();
          if (errData && errData.error) errMsg = errData.error;
        } catch (_) {}
        throw new Error(errMsg);
      }
      const data = await resp.json();

      const rows = Array.isArray(data.rows)
        ? data.rows.map((r) => {
            const c = { ...r };
            if (typeof c.flags === "string") c.flags = c.flags.split(";").filter(Boolean);
            return c;
          })
        : [];

      dataset = rows;

      setDownloadsEnabled(rows.length > 0);
      $$.setStatus(resultsStatus, "Results loaded.", "success");
      await renderDataTable(data.timestamp);
    } catch (err) {
      console.error(err);
      setDownloadsEnabled(false);
      $$.setStatus(resultsStatus, `Error loading results: ${err && err.message ? err.message : "unknown error"}`, "error");
      dataset = [];
      await renderDataTable(); // dejar tabla vacía o mostrar error de carga de DataTables
    } finally {
      loading = false;
      btnRefresh?.removeAttribute("disabled");
      btnRefresh?.removeAttribute("aria-busy");
    }
  }

  // Eventos de filtros
  filterForm?.addEventListener("submit", (e) => {
    e.preventDefault();
    applyQuickFilters();
  });
  btnClearFilters?.addEventListener("click", () => {
    filterForm?.reset();
    applyQuickFilters();
  });
  btnRefresh?.addEventListener("click", () => refresh());

  // Primera carga
  refresh();
});
