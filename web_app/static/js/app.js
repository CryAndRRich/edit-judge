/* Edit-Judge Studio front end (vanilla JS, no build step).
 * Flow: pick an image (upload or sample) -> type an instruction (live routing preview) -> POST /api/jobs ->
 * poll GET /api/jobs/{id} (per-model progress) -> render the winner, the candidate grid and the leaderboard.
 * URL parameters: ?sample=<id>&instruction=<text>&autorun=1, or ?run=<run id> to reopen a finished run.
 */
(() => {
  "use strict";

  const MAX_BYTES = 25 * 1024 * 1024;
  const QUICK_TAGS = [
    "Make brighter", "Remove background", "Super resolution", "Sharpen details", "Black and white",
    "Remove the object", "Restore old photo", "Colorize", "Crop to square",
    "Làm nét ảnh", "Xóa phông nền", "Làm sáng ảnh",
  ];
  const CATEGORY_LABELS = {
    shared: "Shared", remove: "Remove", add_replace: "Add / Replace", enhance: "Enhance",
    restore: "Restore", color: "Change Color", formatting: "Formatting",
  };

  const state = {
    image: null,          // {kind: "upload"|"sample", id, url, name, width, height}
    samples: [],
    result: null,
    filter: "all",
    sort: "score",
    running: false,
    routeTimer: null,
    routeSeq: 0,
    modalId: null,
    returnFocus: null,
    modelNames: {},
    visible: [],
  };

  // ---------------------------------------------------------------- helpers
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
  const fmtGain = (g) => (g == null || !Number.isFinite(g)) ? "—" : (g > 0 ? "+" : "") + g.toFixed(3);
  const gainClass = (g) => (g == null || Math.abs(g) < 0.0005) ? "zero" : (g > 0 ? "pos" : "neg");
  const fmtMs = (ms) => ms == null ? "—" : ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(ms < 10000 ? 2 : 1)} s`;
  const fmtGb = (gb) => gb == null ? "—" : gb < 0.1 ? `${Math.round(gb * 1000)} MB` : `${gb.toFixed(1)} GB`;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const ICONS = {
    clock: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>',
    chip: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/></svg>',
    crown: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M3 7l4.5 4L12 4l4.5 7L21 7l-2 12H5L3 7z"/></svg>',
    download: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 4v12m0 0-4-4m4 4 4-4M4 20h16"/></svg>',
    zoom: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><path d="m20 20-4-4M11 8v6M8 11h6"/></svg>',
    link: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M10 14a4 4 0 0 0 5.66 0l3-3a4 4 0 0 0-5.66-5.66l-1 1"/><path d="M14 10a4 4 0 0 0-5.66 0l-3 3a4 4 0 0 0 5.66 5.66l1-1"/></svg>',
    close: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M18 6 6 18M6 6l12 12"/></svg>',
  };

  async function api(path, options = {}) {
    const response = await fetch(path, options);
    let body = null;
    try { body = await response.json(); } catch (_) { /* not JSON */ }
    if (!response.ok) {
      const detail = body && body.detail ? (typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail)) : response.statusText;
      const error = new Error(detail || `HTTP ${response.status}`);
      error.status = response.status;
      throw error;
    }
    return body;
  }

  function toast(title, message, kind = "error") {
    const node = document.createElement("div");
    node.className = `toast ${kind}`;
    node.innerHTML = `<b>${esc(title)}</b><div class="muted" style="margin-top:2px;font-size:12.5px">${esc(message)}</div>`;
    $("#toasts").appendChild(node);
    setTimeout(() => { node.style.opacity = "0"; node.style.transform = "translateY(6px)"; node.style.transition = "opacity .3s, transform .3s"; }, 5200);
    setTimeout(() => node.remove(), 5600);
  }

  // ---------------------------------------------------------------- image selection
  function setImage(image) {
    state.image = image;
    const preview = $("#dz-preview"), empty = $("#dz-empty");
    if (image) {
      $("#dz-img").src = image.url;
      $("#dz-name").textContent = image.name;
      $("#img-dims").textContent = image.width ? `${image.width} × ${image.height}` : "";
      preview.classList.remove("hidden");
      empty.classList.add("hidden");
    } else {
      $("#dz-img").removeAttribute("src");
      $("#img-dims").textContent = "";
      preview.classList.add("hidden");
      empty.classList.remove("hidden");
    }
    $$(".sample").forEach((node) => node.classList.toggle("active", !!image && image.kind === "sample" && node.dataset.id === image.id));
    updateRunButton();
  }

  async function uploadFile(file) {
    if (!file) return;
    if (!file.type.startsWith("image/") && !/\.(jpe?g|png|webp|bmp|tiff?|gif)$/i.test(file.name)) {
      toast("Not an image", `${file.name} is not an image file.`);
      return;
    }
    if (file.size > MAX_BYTES) {
      toast("File too large", `${(file.size / 1048576).toFixed(1)} MB; the limit is 25 MB.`);
      return;
    }
    const localUrl = URL.createObjectURL(file);
    setImage({ kind: "pending", id: null, url: localUrl, name: file.name });
    const zone = $("#dropzone");
    zone.classList.add("uploading");
    $("#dz-name").textContent = `Uploading ${file.name}…`;
    try {
      const form = new FormData();
      form.append("file", file, file.name);
      const meta = await api("/api/upload", { method: "POST", body: form });
      setImage({ kind: "upload", id: meta.image_id, url: meta.url, name: meta.filename, width: meta.width, height: meta.height });
    } catch (error) {
      setImage(null);
      toast("Upload failed", error.message);
    } finally {
      zone.classList.remove("uploading");
      URL.revokeObjectURL(localUrl);
    }
  }

  function bindDropzone() {
    const zone = $("#dropzone"), input = $("#file-input");
    zone.addEventListener("click", (event) => {
      if (event.target.closest("#dz-clear")) return;
      input.click();
    });
    zone.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") { event.preventDefault(); input.click(); }
    });
    input.addEventListener("change", () => { uploadFile(input.files[0]); input.value = ""; });
    ["dragenter", "dragover"].forEach((type) => zone.addEventListener(type, (event) => {
      event.preventDefault(); zone.classList.add("dragover");
    }));
    ["dragleave", "dragend"].forEach((type) => zone.addEventListener(type, () => zone.classList.remove("dragover")));
    zone.addEventListener("drop", (event) => {
      event.preventDefault();
      zone.classList.remove("dragover");
      const file = event.dataTransfer.files && event.dataTransfer.files[0];
      if (file) uploadFile(file);
    });
    // a page-wide drop also works
    window.addEventListener("dragover", (event) => event.preventDefault());
    window.addEventListener("drop", (event) => {
      if (event.target.closest && event.target.closest("#dropzone")) return;
      event.preventDefault();
      const file = event.dataTransfer && event.dataTransfer.files[0];
      if (file) uploadFile(file);
    });
    window.addEventListener("paste", (event) => {
      const item = Array.from(event.clipboardData?.items || []).find((i) => i.type.startsWith("image/"));
      if (item) uploadFile(item.getAsFile());
    });
    $("#dz-clear").addEventListener("click", (event) => { event.stopPropagation(); setImage(null); });
  }

  function renderSamples() {
    const box = $("#samples");
    if (!state.samples.length) {
      box.innerHTML = '<div class="faint" style="grid-column:1/-1;font-size:12px">No samples — run <span class="mono">python scripts/fetch_dev_split.py</span></div>';
      return;
    }
    $("#samples-note").textContent = `${state.samples.length} public photos`;
    box.innerHTML = state.samples.map((s) => `
      <button class="sample" type="button" data-id="${esc(s.id)}" title="${esc(s.instruction)} — ${esc(s.author || s.source)} (${esc(s.source)})">
        <img src="${esc(s.url)}" alt="${esc(s.id)}" loading="lazy">
      </button>`).join("");
    $$(".sample", box).forEach((node) => node.addEventListener("click", () => selectSample(node.dataset.id, true)));
  }

  // decorative background: the sample photos laid out as an overlapping, slightly tilted print collage
  const COLLAGE = [
    [-3, 8, -7, "l"], [7, 30, 5, "p"], [15, 14, -3, "l"], [25, 36, 8, "l"], [33, 4, -6, "p"],
    [42, 28, 3, "l"], [51, 10, -8, "l"], [60, 34, 6, "p"], [68, 6, -4, "l"], [77, 30, 7, "l"],
    [86, 12, -5, "p"], [93, 38, 4, "l"],
  ];
  function renderCollage(samples) {
    const box = $("#collage");
    if (!box || !samples.length) return;
    box.innerHTML = COLLAGE.map(([left, top, rot, shape], i) => {
      const s = samples[i % samples.length];
      return `<figure class="collage-tile ${shape}" style="left:${left}%;top:${top}%;--rot:${rot}deg;animation-delay:${i * 60}ms"><img src="${esc(s.url)}" alt="" decoding="async"></figure>`;
    }).join("");
  }

  function selectSample(id, fillInstruction) {
    const sample = state.samples.find((s) => s.id === id);
    if (!sample) return false;
    setImage({ kind: "sample", id: sample.id, url: sample.url, name: `${sample.id} · ${sample.author || sample.source}`, width: sample.width, height: sample.height });
    if (fillInstruction) setInstruction(sample.instruction);
    return true;
  }

  // ---------------------------------------------------------------- instruction + routing preview
  function setInstruction(text) {
    const box = $("#instruction");
    box.value = text;
    onInstruction();
  }

  function onInstruction() {
    const text = $("#instruction").value;
    $("#char-count").textContent = `${text.length} / 500`;
    $$("#quick-tags .chip").forEach((chip) => chip.classList.toggle("active", chip.dataset.tag === text.trim()));
    updateRunButton();
    clearTimeout(state.routeTimer);
    state.routeTimer = setTimeout(previewRoute, 220);
  }

  async function previewRoute() {
    const text = $("#instruction").value.trim();
    const box = $("#route-box");
    if (!text) {
      box.innerHTML = '<div class="row"><span class="label">Routing</span><span class="faint">Type an instruction to see which models will run.</span></div>';
      return;
    }
    const seq = ++state.routeSeq;
    try {
      const route = await api("/api/route", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ instruction: text }) });
      if (seq !== state.routeSeq) return;
      const cats = route.categories.filter((c) => c.id !== "shared");
      const intents = Object.keys(route.intents);
      box.innerHTML = `
        <div class="row"><span class="label">Tasks</span>
          <span class="badge">Shared</span>
          ${cats.map((c) => `<span class="badge cat">${esc(c.label)}</span>`).join("")}
          ${cats.length ? "" : '<span class="faint">generalists only</span>'}
        </div>
        ${intents.length ? `<div class="row"><span class="label">Intents</span>${intents.map((i) => `<span class="intent">${esc(i)}</span>`).join("")}</div>` : ""}
        <div class="row"><span class="label">Models</span><span><b class="mono">${route.models.length}</b> <span class="muted">candidates will run</span></span></div>`;
    } catch (error) {
      if (seq === state.routeSeq) box.innerHTML = `<div class="row"><span class="faint">Routing preview unavailable: ${esc(error.message)}</span></div>`;
    }
  }

  function updateRunButton() {
    const ready = state.image && state.image.id && $("#instruction").value.trim().length > 0 && !state.running;
    $("#run-btn").disabled = !ready;
  }

  // ---------------------------------------------------------------- running
  async function run() {
    if (state.running || !state.image || !state.image.id) return;
    const instruction = $("#instruction").value.trim();
    if (!instruction) return;
    state.running = true;
    updateRunButton();
    const button = $("#run-btn");
    button.classList.add("loading");
    $("#run-label").textContent = "Running models…";
    button.querySelector("svg").outerHTML = '<span class="spinner"></span>';
    $("#progress").classList.remove("hidden");
    $("#progress-text").textContent = "Starting…";
    $("#progress-bar").style.width = "4%";

    const body = { instruction };
    if (state.image.kind === "sample") body.sample_id = state.image.id; else body.image_id = state.image.id;
    try {
      const job = await api("/api/jobs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      renderProgress({ status: "queued", progress: Object.fromEntries(job.models.map((m) => [m, "queued"])), done: 0, total: job.models.length });
      showLive(true);
      let status;
      for (let attempt = 0; attempt < 2400; attempt++) {       // about 15 minutes at most
        await sleep(attempt < 10 ? 250 : 400);
        status = await api(`/api/jobs/${job.job_id}`);
        renderProgress(status);
        if (status.status === "done" || status.status === "error") break;
      }
      if (!status || status.status !== "done") throw new Error(status && status.error ? status.error : "the job did not finish");
      showResult(status.result, true);
      const url = new URL(window.location.href);
      url.search = `?run=${status.result.run_id}`;
      history.replaceState(null, "", url);
    } catch (error) {
      toast("Run failed", error.message);
      $("#progress-text").textContent = "Failed";
    } finally {
      showLive(false);
      state.running = false;
      button.classList.remove("loading");
      $("#run-label").textContent = "Run Models";
      const spinner = button.querySelector(".spinner");
      if (spinner) spinner.outerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M13 2 3 14h9l-1 8 10-12h-9l1-8z"/></svg>';
      updateRunButton();
    }
  }

  function renderProgress(job) {
    const names = state.modelNames || {};
    const total = job.total || Object.keys(job.progress).length || 1;
    const pct = job.status === "done" ? 100 : Math.max(4, Math.round((job.done / total) * (job.status === "scoring" ? 100 : 92)));
    $("#progress-bar").style.width = `${pct}%`;
    $("#progress-count").textContent = `${job.done}/${total}`;
    $("#progress-text").textContent = {
      queued: "Queued…", running: "Running candidate models…", scoring: "Scoring edits against the original…",
      done: "Done", error: "Failed",
    }[job.status] || job.status;
    $("#model-states").innerHTML = Object.entries(job.progress).map(([id, st]) =>
      `<div class="model-state ${esc(st)}"><span class="st"></span><span>${esc(names[id] || id)}</span></div>`).join("");
    // the live panel in the results column
    $("#live-title").textContent = job.status === "scoring" ? "Scoring every edit against the original…" : "Running candidate models…";
    $("#live-count").textContent = `${job.done}/${total}`;
    const src = state.image && state.image.url ? esc(`--src:url(${JSON.stringify(state.image.url)})`) : "";
    $("#live-grid").innerHTML = Object.entries(job.progress).map(([id, st]) =>
      `<div class="live-card ${esc(st)}"><div class="ph ${st === "ok" || st === "failed" ? "" : "skeleton"}" style="${src}"></div><div class="nm"><span class="st"></span><span>${esc(names[id] || id)}</span></div></div>`).join("");
  }

  function showLive(on) {
    $("#live").classList.toggle("hidden", !on);
    if (on) {
      $("#empty-state").classList.add("hidden");
      $("#results").classList.add("hidden");
    } else if (!state.result) {
      $("#empty-state").classList.remove("hidden");
    } else {
      $("#results").classList.remove("hidden");
    }
  }

  // ---------------------------------------------------------------- results
  function showResult(result, scroll) {
    state.result = result;
    state.filter = "all";
    $("#empty-state").classList.add("hidden");
    $("#results").classList.remove("hidden");
    renderHero();
    renderStats();
    renderFilters();
    renderGrid();
    renderTable();
    if (scroll && window.innerWidth < 1080) $("#results").scrollIntoView({ behavior: "smooth" });
  }

  function sameAspect(a, b) {
    if (!a || !b || !a.width || !b.width) return false;
    return Math.abs(Math.log((a.width / a.height) / (b.width / b.height))) < 0.02;
  }

  function compareHtml(original, candidate, maxHeight) {
    const ratio = `${candidate.width} / ${candidate.height}`;
    if (!sameAspect(original, candidate)) {
      return `<div class="side-by-side">
        <figure><img src="${esc(original.url)}" alt="Original"><figcaption>Original · ${original.width}×${original.height}</figcaption></figure>
        <figure class="${candidate.transparent ? "checker" : ""}"><img src="${esc(candidate.image_url)}" alt="${esc(candidate.name)} output"><figcaption>${esc(candidate.name)} · ${candidate.width}×${candidate.height}</figcaption></figure>
      </div>`;
    }
    return `<div class="compare ${candidate.transparent ? "checker" : ""}" style="aspect-ratio:${ratio};max-height:${maxHeight};margin:0 auto;max-width:calc(${maxHeight} * ${candidate.width / candidate.height})">
      <img class="after" src="${esc(candidate.image_url)}" alt="${esc(candidate.name)} output">
      <img class="before" src="${esc(original.url)}" alt="Original">
      <div class="handle"></div>
      <span class="tag l">Original</span><span class="tag r">${esc(candidate.name)}</span>
      <input type="range" min="0" max="100" value="50" aria-label="Compare original and edited image">
    </div>`;
  }

  function bindCompare(root) {
    $$(".compare", root).forEach((box) => {
      const input = $("input[type=range]", box);
      const set = (v) => box.style.setProperty("--pos", `${v}%`);
      input.addEventListener("input", () => set(input.value));
      set(input.value);
    });
  }

  function breakdownHtml(candidate) {
    const p = candidate.proxy || {};
    const rows = [["Alignment", p.alignment], ["Quality", p.quality], ["Fidelity", p.fidelity]];
    return rows.map(([label, value]) => {
      const v = Number.isFinite(value) ? value : 0;
      const width = Math.min(50, Math.abs(v) * 50);
      return `<div class="brow"><span>${label}</span><div class="dbar"><i class="${v >= 0 ? "pos" : "neg"}" style="width:${width}%"></i></div><span class="val">${fmtGain(value)}</span></div>`;
    }).join("");
  }

  function renderHero() {
    const r = state.result, best = r.best, hero = $("#hero");
    const original = r.original;
    if (!best) {
      hero.innerHTML = `<div class="hero-info"><h2>No candidate produced an image</h2><p class="muted">Every model failed for this request; see the cards below for the errors.</p></div>`;
      return;
    }
    const gain = best.score ? best.score.gain : null;
    const scorerLabel = r.scorer.kind === "reward_model" ? r.scorer.label : "Proxy score";
    const beat = r.counts.beat_original;
    const bench = best.benchmark;
    const notices = [
      r.scorer.kind !== "reward_model" ? esc(r.scorer.fallback_reason || "Scored by the heuristic proxy, not EditReward.") : "",
      beat === 0 ? "No candidate beat the unedited photo: the best one is shown, but keeping the original may be the right call." : "",
    ].filter(Boolean);
    hero.innerHTML = `
      <div class="hero-media">${compareHtml(original, best, "62vh")}</div>
      <div class="hero-info">
        <div class="hero-block hero-head">
          <div class="hero-tags">
            <span class="crown">${ICONS.crown} Best edit</span>
            <span class="badge ${best.backend}">${best.backend.toUpperCase()}</span>
          </div>
          <div>
            <h2>${esc(best.name)}</h2>
            <div class="sub">${esc(best.date)} · ${best.categories.map((c) => CATEGORY_LABELS[c] || c).join(", ")}</div>
          </div>
          ${r.instruction ? `<p class="prompt-echo">${esc(r.instruction)}</p>` : ""}
        </div>
        <div class="hero-block hero-score">
          <div class="score-big">
            <span class="v ${gainClass(gain)}">${fmtGain(gain)}</span>
            <span class="l">${esc(scorerLabel)} gain<br>vs. the unedited photo</span>
          </div>
          <dl class="kv" style="margin-top:16px">
            <div><dt>Latency</dt><dd>${fmtMs(best.elapsed_ms)}</dd></div>
            <div><dt>VRAM</dt><dd>${fmtGb(best.vram_gb_estimate)}</dd></div>
            <div><dt>Output</dt><dd>${best.width}×${best.height}</dd></div>
          </dl>
        </div>
        <div class="hero-block">
          <div class="block-label">Score breakdown</div>
          <div class="breakdown">${breakdownHtml(best)}</div>
          ${bench ? `<div class="bench" style="margin-top:12px">Kaggle T4 benchmark · EditReward gain <b class="${gainClass(bench.editreward_gain)} g">${fmtGain(bench.editreward_gain)}</b>, win rate ${Math.round((bench.editreward_win_rate || 0) * 100)}%${bench.seconds_per_image_t4 ? ", " + bench.seconds_per_image_t4.toFixed(2) + " s/image" : ""} (n=${bench.ok})</div>` : ""}
        </div>
        ${notices.length || best.note ? `<div class="hero-block hero-notices">
          ${notices.map((n) => `<div class="notice">${n}</div>`).join("")}
          ${best.note ? `<div class="note">${esc(best.note)}</div>` : ""}
        </div>` : ""}
        <div class="hero-block hero-actions-block">
          <div class="hero-actions">
            <a class="btn primary" href="${esc(best.image_url)}" download="${esc(best.model_id)}-${esc(r.run_id.slice(0, 8))}">${ICONS.download} Download</a>
            <button class="btn" type="button" id="hero-zoom">${ICONS.zoom} Inspect</button>
            <button class="btn" type="button" id="hero-link">${ICONS.link} Copy link</button>
          </div>
        </div>
      </div>`;
    bindCompare(hero);
    $("#hero-zoom").addEventListener("click", () => openModal(best.model_id));
    $("#hero-link").addEventListener("click", async () => {
      const url = `${location.origin}/?run=${r.run_id}`;
      try { await navigator.clipboard.writeText(url); toast("Link copied", url, "ok"); } catch (_) { toast("Link", url, "ok"); }
    });
  }

  function renderStats() {
    const r = state.result, c = r.counts;
    const items = [
      ["Candidates", `${c.ok}<small> / ${c.models} ok</small>`],
      ["Beat original", `${c.beat_original}<small> / ${c.ok}</small>`],
      ["Real · Mock", `${c.real}<small> · ${c.mock}</small>`],
      ["Total time", fmtMs(r.timings_ms.total)],
      ["Primary task", esc(CATEGORY_LABELS[r.route.primary] || r.route.primary)],
    ];
    $("#stats").innerHTML = items.map(([k, v]) => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");
  }

  function renderFilters() {
    const counts = {};
    state.result.candidates.forEach((c) => c.categories.forEach((cat) => { counts[cat] = (counts[cat] || 0) + 1; }));
    const keys = ["all", ...Object.keys(CATEGORY_LABELS).filter((k) => counts[k])];
    $("#filters").innerHTML = keys.map((k) => `<button class="chip ${state.filter === k ? "active" : ""}" data-filter="${k}">${k === "all" ? "All" : CATEGORY_LABELS[k]}<span class="count">${k === "all" ? state.result.candidates.length : counts[k]}</span></button>`).join("");
    $$("#filters .chip").forEach((chip) => chip.addEventListener("click", () => {
      state.filter = chip.dataset.filter;
      renderFilters();
      renderGrid();
    }));
  }

  function sortedCandidates() {
    let list = state.result.candidates.slice();
    if (state.filter !== "all") list = list.filter((c) => c.categories.includes(state.filter));
    const gain = (c) => (c.score && Number.isFinite(c.score.gain)) ? c.score.gain : -Infinity;
    if (state.sort === "speed") list.sort((a, b) => (a.status === "ok" ? 0 : 1) - (b.status === "ok" ? 0 : 1) || a.elapsed_ms - b.elapsed_ms);
    else if (state.sort === "name") list.sort((a, b) => a.name.localeCompare(b.name));
    else list.sort((a, b) => gain(b) - gain(a));
    return list;
  }

  function renderGrid() {
    const list = sortedCandidates();
    state.visible = list.map((c) => c.model_id);
    const winner = state.result.best ? state.result.best.model_id : null;
    // bento: when ranking by score, the top candidate of the current view gets a 2×2 tile
    const featureId = state.sort === "score" && list.length > 2 && list[0].status === "ok" ? list[0].model_id : null;
    $("#grid").innerHTML = list.map((c, i) => {
      const gain = c.score ? c.score.gain : null;
      const rankClass = c.rank && c.rank <= 3 ? `r${c.rank}` : "";
      const feature = c.model_id === featureId;
      const media = c.status === "ok"
        ? `<img class="loading" src="${esc(feature ? c.image_url : c.thumb_url)}" alt="${esc(c.name)} output" loading="lazy">`
        : `<div class="err">${ICONS.close.replace("<svg ", '<svg width="18" height="18" ')}<span>${esc(c.error || "failed")}</span></div>`;
      return `<article class="card ${feature ? "feature" : ""} ${c.model_id === winner ? "winner" : ""} ${c.status !== "ok" ? "failed" : ""}" tabindex="0" data-id="${esc(c.model_id)}" style="animation-delay:${Math.min(i, 12) * 30}ms">
        <div class="media ${c.transparent ? "checker" : ""} ${c.status === "ok" ? "loading-bg" : ""}"${feature ? ` style="${esc(`--src:url(${JSON.stringify(c.thumb_url)})`)}"` : ""}>${media}
          ${c.rank ? `<span class="rank ${rankClass}">#${c.rank}</span>` : ""}
          <div class="badges">${c.status !== "ok" ? '<span class="badge failed">FAILED</span>' : `<span class="badge ${c.backend}">${c.backend.toUpperCase()}</span>`}</div>
        </div>
        <div class="body">
          <div class="title"><b title="${esc(c.name)}">${esc(c.name)}</b><span class="g ${gainClass(gain)}">${fmtGain(gain)}</span></div>
          <div class="meta">
            <span class="badge cat">${esc(c.categories.map((x) => CATEGORY_LABELS[x] || x).join(" · "))}</span>
            <span title="Processing time">${ICONS.clock}${fmtMs(c.elapsed_ms)}</span>
            <span title="Weights in fp16 (VRAM estimate)">${ICONS.chip}${fmtGb(c.vram_gb_estimate)}</span>
          </div>
          ${c.status === "ok" ? `<div class="bars">${["alignment", "quality", "fidelity"].map((k) => {
            const v = c.proxy ? c.proxy[k] : 0;
            return `<div class="mini"><span>${k}</span><div class="dbar"><i class="${v >= 0 ? "pos" : "neg"}" style="width:${Math.min(50, Math.abs(v) * 50)}%"></i></div><span class="val">${fmtGain(v)}</span></div>`;
          }).join("")}</div>` : ""}
        </div>
      </article>`;
    }).join("") || '<div class="empty-grid">No candidate in this category.</div>';
    $$("#grid .card").forEach((card) => {
      const open = () => { if (!card.classList.contains("failed")) openModal(card.dataset.id); };
      card.addEventListener("click", open);
      card.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); } });
      card.addEventListener("pointermove", (e) => {
        const box = card.getBoundingClientRect();
        card.style.setProperty("--mx", `${e.clientX - box.left}px`);
        card.style.setProperty("--my", `${e.clientY - box.top}px`);
      });
    });
    $$("#grid .media img.loading").forEach(revealWhenLoaded);
  }

  // image skeleton: shimmer until the bitmap arrives, then fade the image in
  function revealWhenLoaded(img) {
    const done = () => { img.classList.remove("loading"); img.parentElement && img.parentElement.classList.remove("loading-bg"); };
    if (img.complete && img.naturalWidth) done();
    else { img.addEventListener("load", done, { once: true }); img.addEventListener("error", done, { once: true }); }
  }

  function renderTable() {
    const r = state.result;
    $("#table-note").textContent = `· ${r.scorer.kind === "reward_model" ? r.scorer.label : "proxy scorer"} · run ${r.run_id.slice(0, 8)}`;
    const rows = r.candidates.map((c) => {
      const p = c.proxy || {}, b = c.benchmark || {};
      return `<tr data-id="${esc(c.model_id)}">
        <td class="num">${c.rank ?? "—"}</td>
        <td><b>${esc(c.name)}</b> <span class="faint">${esc(c.date)}</span></td>
        <td>${esc(c.categories.map((x) => CATEGORY_LABELS[x] || x).join(", "))}</td>
        <td><span class="badge ${c.status === "ok" ? c.backend : "failed"}">${c.status === "ok" ? c.backend.toUpperCase() : "FAILED"}</span></td>
        <td class="num g ${gainClass(c.score && c.score.gain)}">${fmtGain(c.score && c.score.gain)}</td>
        <td class="num">${fmtGain(p.alignment)}</td><td class="num">${fmtGain(p.quality)}</td><td class="num">${fmtGain(p.fidelity)}</td>
        <td class="num">${fmtMs(c.elapsed_ms)}</td><td class="num">${fmtGb(c.vram_gb_estimate)}</td>
        <td class="num">${c.width ? `${c.width}×${c.height}` : "—"}</td>
        <td class="num">${b.editreward_gain != null ? fmtGain(b.editreward_gain) : "—"}</td>
      </tr>`;
    }).join("");
    $("#table").innerHTML = `<thead><tr><th class="num">#</th><th>Model</th><th>Task lists</th><th>Backend</th><th class="num">Gain</th><th class="num">Align</th><th class="num">Quality</th><th class="num">Fidelity</th><th class="num">Time</th><th class="num">VRAM</th><th class="num">Output</th><th class="num" title="EditReward gain on the Kaggle T4 Enhance benchmark">T4 bench</th></tr></thead><tbody>${rows}</tbody>`;
    $$("#table tbody tr").forEach((tr) => tr.addEventListener("click", () => {
      const c = r.candidates.find((x) => x.model_id === tr.dataset.id);
      if (c && c.status === "ok") openModal(c.model_id);
    }));
  }

  // ---------------------------------------------------------------- lightbox
  function openModal(modelId) {
    const r = state.result;
    const candidate = r.candidates.find((c) => c.model_id === modelId);
    if (!candidate || candidate.status !== "ok") return;
    if ($("#modal").classList.contains("hidden")) state.returnFocus = document.activeElement;
    state.modalId = modelId;
    const p = candidate.proxy || {}, m = p.metrics || {};
    const s = candidate.score || {};
    const intents = Object.entries(p.intents || {});
    const pair = (v) => Array.isArray(v) ? `${v[0]} → ${v[1]}` : "—";
    const modal = $("#modal");
    modal.innerHTML = `
      <div class="modal-card">
        <button class="icon-btn modal-close" type="button" aria-label="Close">${ICONS.close}</button>
        <div class="modal-media">${compareHtml(r.original, candidate, "78vh")}</div>
        <div class="modal-side">
          <div>
            ${candidate.rank ? `<span class="rank ${candidate.rank <= 3 ? "r" + candidate.rank : ""}" style="position:static;display:inline-grid;margin-bottom:10px">#${candidate.rank}</span>` : ""}
            <h3 id="modal-title">${esc(candidate.name)}</h3>
            <div class="faint" style="font-size:12px;margin-top:6px;display:flex;gap:8px;align-items:center">${esc(candidate.date)} <span class="badge ${candidate.backend}">${candidate.backend.toUpperCase()}</span></div>
          </div>
          <div class="score-big"><span class="v ${gainClass(s.gain)}">${fmtGain(s.gain)}</span><span class="l">gain (${esc(s.scorer || "")})<br>raw ${s.raw != null ? Number(s.raw).toFixed(3) : "—"} · baseline ${s.baseline != null ? Number(s.baseline).toFixed(3) : "—"}</span></div>
          <div><div class="block-label">Score breakdown</div><div class="breakdown">${breakdownHtml(candidate)}</div></div>
          ${intents.length ? `<div><div class="block-label">Per-intent alignment</div>${intents.map(([k, v]) => `<div class="brow"><span class="intent" style="justify-self:start">${esc(k === "_generic" ? "general" : k)}</span><div class="dbar"><i class="${v >= 0 ? "pos" : "neg"}" style="width:${Math.min(50, Math.abs(v) * 50)}%"></i></div><span class="val">${fmtGain(v)}</span></div>`).join("")}</div>` : ""}
          <div><div class="block-label">Metrics · original → edit</div><table class="metrics">
            <tr><td class="muted">Latency</td><td>${fmtMs(candidate.elapsed_ms)}</td></tr>
            <tr><td class="muted">VRAM estimate (fp16 weights)</td><td>${fmtGb(candidate.vram_gb_estimate)}</td></tr>
            <tr><td class="muted">Size</td><td>${pair((m.size || []).map((x) => x.join("×")))}</td></tr>
            <tr><td class="muted">Luminance</td><td>${pair(m.luminance)}</td></tr>
            <tr><td class="muted">Sharpness</td><td>${pair(m.sharpness)}</td></tr>
            <tr><td class="muted">Colourfulness</td><td>${pair(m.colorfulness)}</td></tr>
            <tr><td class="muted">Noise</td><td>${pair(m.noise)}</td></tr>
            <tr><td class="muted">Structure corr.</td><td>${p.structure_correlation ?? "—"}</td></tr>
            ${candidate.benchmark ? `<tr><td class="muted">T4 bench EditReward gain</td><td>${fmtGain(candidate.benchmark.editreward_gain)}</td></tr><tr><td class="muted">T4 s / image</td><td>${candidate.benchmark.seconds_per_image_t4 ? candidate.benchmark.seconds_per_image_t4.toFixed(2) : "—"}</td></tr>` : ""}
          </table></div>
          ${candidate.note ? `<div class="note" style="margin:0">${esc(candidate.note)}</div>` : ""}
          <div class="hero-actions">
            <a class="btn primary" href="${esc(candidate.image_url)}" download="${esc(candidate.model_id)}-${esc(r.run_id.slice(0, 8))}">${ICONS.download} Download</a>
            <a class="btn" href="${esc(candidate.image_url)}" target="_blank" rel="noopener">Full size ↗</a>
          </div>
          <div class="modal-nav"><button class="btn" type="button" data-step="-1" aria-label="Previous candidate">← Prev</button><button class="btn" type="button" data-step="1" aria-label="Next candidate">Next →</button><span class="esc"><span class="kbd">←</span><span class="kbd">→</span> browse · <span class="kbd">esc</span></span></div>
        </div>
      </div>`;
    modal.classList.remove("hidden");
    document.body.style.overflow = "hidden";
    bindCompare(modal);
    $(".modal-close", modal).addEventListener("click", closeModal);
    $$("[data-step]", modal).forEach((b) => b.addEventListener("click", () => stepModal(Number(b.dataset.step))));
    $(".modal-close", modal).focus();
  }

  function stepModal(delta) {
    const r = state.result;
    const order = (state.visible.length ? state.visible : r.candidates.map((c) => c.model_id))
      .filter((id) => (r.candidates.find((c) => c.model_id === id) || {}).status === "ok");
    if (!order.length) return;
    const current = Math.max(0, order.indexOf(state.modalId));
    const next = order[(current + delta + order.length) % order.length];
    openModal(next);
  }

  function closeModal() {
    const modal = $("#modal");
    modal.classList.add("hidden");
    modal.innerHTML = "";
    document.body.style.overflow = "";
    if (state.returnFocus && document.contains(state.returnFocus)) state.returnFocus.focus({ preventScroll: true });
    state.returnFocus = null;
  }

  // ---------------------------------------------------------------- init
  async function init() {
    bindDropzone();
    $("#quick-tags").innerHTML = QUICK_TAGS.map((t) => `<button class="chip" type="button" data-tag="${esc(t)}">${esc(t)}</button>`).join("");
    $$("#quick-tags .chip").forEach((chip) => chip.addEventListener("click", () => { setInstruction(chip.dataset.tag); $("#instruction").focus(); }));
    $("#instruction").addEventListener("input", onInstruction);
    $("#instruction").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); run(); } });
    $("#run-btn").addEventListener("click", run);
    $$("#sorts button").forEach((b) => b.addEventListener("click", () => {
      state.sort = b.dataset.sort;
      $$("#sorts button").forEach((x) => x.classList.toggle("active", x === b));
      renderGrid();
    }));
    $("#modal").addEventListener("click", (e) => { if (e.target.id === "modal") closeModal(); });
    document.addEventListener("keydown", (e) => {
      if ($("#modal").classList.contains("hidden")) {
        const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName) || e.target.isContentEditable;
        if (e.key === "/" && !typing && !e.metaKey && !e.ctrlKey) { e.preventDefault(); $("#instruction").focus(); }
        return;
      }
      if (e.key === "Tab") {                                   // keep focus inside the dialog
        const nodes = $$('button, a[href], input, [tabindex]:not([tabindex="-1"])', $("#modal")).filter((n) => n.offsetParent !== null);
        if (nodes.length) {
          const first = nodes[0], last = nodes[nodes.length - 1];
          if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
          else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
        }
      }
      if (e.key === "Escape") closeModal();
      if (e.key === "ArrowRight" && e.target.type !== "range") stepModal(1);
      if (e.key === "ArrowLeft" && e.target.type !== "range") stepModal(-1);
    });

    const [health, models, samples] = await Promise.allSettled([api("/api/health"), api("/api/models"), api("/api/samples")]);
    if (health.status === "fulfilled") {
      const h = health.value;
      $("#pill-device").innerHTML = `<span class="dot ok"></span>Device <b>${esc(h.device.toUpperCase())}</b>`;
    } else {
      $("#pill-device").innerHTML = '<span class="dot bad"></span>Server <b>offline</b>';
      toast("Server unreachable", "Start it with: python run_web.py");
    }
    if (models.status === "fulfilled") {
      state.modelNames = Object.fromEntries(models.value.models.map((m) => [m.id, m.name]));
      const real = models.value.models.filter((m) => m.backend === "real").length;
      $("#pill-models").innerHTML = `<span class="dot ok"></span><b>${models.value.models.length}</b> models · ${real} real`;
    }
    state.samples = samples.status === "fulfilled" ? samples.value.samples : [];
    renderSamples();
    renderCollage(state.samples);
    onInstruction();

    const params = new URLSearchParams(location.search);
    if (params.get("run")) {
      try {
        const result = await api(`/api/runs/${encodeURIComponent(params.get("run"))}`);
        showResult(result, false);
        // put the run's photo and instruction back in the controls, so it can be tweaked and run again
        const o = result.original;
        if (o.sample_id) selectSample(o.sample_id, false);
        else if (o.image_id) setImage({ kind: "upload", id: o.image_id, url: o.source_url, name: "Uploaded photo", width: o.width, height: o.height });
        setInstruction(result.instruction);
      } catch (error) { toast("Run not found", error.message); }
    }
    if (params.get("sample")) selectSample(params.get("sample"), !params.get("instruction"));
    if (params.get("instruction")) setInstruction(params.get("instruction"));
    if (params.get("autorun") === "1" && state.image) run();
    window.EJ.ready = true;
  }

  document.addEventListener("DOMContentLoaded", () => {
    init().catch((error) => { console.error(error); toast("Startup error", error.message); });
  });
})();
