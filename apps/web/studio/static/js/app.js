// EditJudge Studio: the live assistant (tabs Assistant, Rankings, Dataset). No build step, no framework.
// Assistant flow: pick or upload a photo, write a request, Process. POST /api/jobs reads the request (task, the
// instruction sent to the models) and picks the models; GET /api/jobs/{id} is polled while they run one after
// another. Each card's bar follows the model's expected time (see progressAt); the result shows with its score.
// When every model has run, the cards move into rank order and the best edit opens in the before / after slider.
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  const fmt = (value) => value.toLocaleString("en-US");
  const compact = (value) => (value >= 1000 ? `${(value / 1000).toFixed(1)}k` : String(value));
  const POLL_MS = 300;
  const FINISH_MS = 250;
  const SWEEP_MS = 900;
  const MAX_REQUEST = 100;
  const ACTIVE = ["waiting", "running"];

  const state = {
    image: null,          // {kind: "upload" | "sample", id, url, width, height, name}
    samples: [],
    job: null,            // the last job from the server
    jobId: null,
    polledAt: 0,          // when `job` arrived (ms, performance clock)
    bars: {},             // model -> {progress, finishedAt, from, shown}
    selected: null,       // model shown in the slider
    heroModel: null,      // model the slider shows now (a new one sweeps in)
    ranked: false,        // cards in rank order (after the run)
    appeared: new Set(),  // cards already shown (they rise only once)
    revealed: new Set(),  // results already shown (they develop only once)
    running: false,
    rankings: null,
    dataset: null,
  };

  // ---------------------------------------------------------------- helpers
  async function api(path, options = {}) {
    const response = await fetch(path, options);
    let body = null;
    try { body = await response.json(); } catch (error) { body = null; }
    if (!response.ok) throw new Error((body && body.detail) || `${response.status} ${response.statusText}`);
    return body;
  }
  function toast(message) {
    const el = document.createElement("div");
    el.className = "toast error";
    el.innerHTML = `<b>Something went wrong</b> <span>${esc(message)}</span>`;
    $("toasts").appendChild(el);
    setTimeout(() => el.remove(), 6000);
  }
  const isDone = (candidate) => candidate.status === "ok" || candidate.status === "failed";

  // ---------------------------------------------------------------- progress of one model
  // 0 -> 85% linearly over 85% of the expected time, then slower and slower toward 99% (never 100% before the
  // result). When the result arrives, a short ease-out to 100%.
  function progressAt(elapsed, estimate) {
    const knee = 0.85 * estimate;
    if (elapsed <= knee) return 0.85 * elapsed / knee;
    return 0.85 + 0.14 * (1 - Math.exp(-(elapsed - knee) / (0.6 * estimate)));
  }
  function elapsedNow(candidate) {
    return (candidate.elapsed || 0) + (performance.now() - state.polledAt) / 1000;
  }

  // ---------------------------------------------------------------- theme
  function setTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    const next = theme === "dark" ? "light" : "dark";
    $("theme-btn").setAttribute("aria-label", `Switch to ${next} mode`);
    $("theme-btn").title = `Switch to ${next} mode`;
    try { localStorage.setItem("ej-theme", theme); } catch (error) { /* storage blocked: the choice lasts this visit */ }
  }
  $("theme-btn").addEventListener("click", () => setTheme(document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark"));

  // ---------------------------------------------------------------- views: tabs with a sliding pill
  const VIEWS = ["assistant", "rankings", "dataset"];
  function movePill() {
    const active = document.querySelector(".view-switch button.active");
    const pill = document.querySelector(".tab-pill");
    if (!active || !pill) return;
    pill.style.width = `${active.offsetWidth}px`;
    pill.style.transform = `translateX(${active.offsetLeft}px)`;
  }
  function showView(name) {
    VIEWS.forEach((view) => {
      $("tab-" + view).classList.toggle("active", view === name);
      $("tab-" + view).setAttribute("aria-selected", String(view === name));
      $("view-" + view).classList.toggle("hidden", view !== name);
    });
    $("layout").classList.toggle("wide", name !== "assistant");
    movePill();
    const layout = $("layout");
    layout.classList.remove("enter");
    void layout.offsetWidth;            // restart the entrance animation
    layout.classList.add("enter");
    if (name === "rankings") showRankings();
    if (name === "dataset") showDataset();
  }
  const viewOfHash = () => (VIEWS.includes(location.hash.slice(1)) ? location.hash.slice(1) : "assistant");
  VIEWS.forEach((view) => $("tab-" + view).addEventListener("click", () => { showView(view); history.replaceState(null, "", "#" + view); }));
  window.addEventListener("hashchange", () => showView(viewOfHash()));
  window.addEventListener("resize", movePill);
  if (document.fonts) document.fonts.ready.then(movePill);

  // ---------------------------------------------------------------- backdrop: columns of photos falling
  // 6 columns of 8 photos, each column twice so the fall loops without a jump, tall enough for the whole window
  function fillCollage(urls) {
    if (!urls.length) return;
    $("collage").innerHTML = [0, 1, 2, 3, 4, 5].map((column) => {
      const tiles = [0, 1, 2, 3, 4, 5, 6, 7].map((row) => {
        const rot = ((column * 7 + row * 5) % 9) - 4;
        return `<figure class="collage-tile" style="--rot:${rot}deg"><img src="${esc(urls[(column + row) % urls.length])}" alt="" loading="lazy"></figure>`;
      }).join("");
      return `<div class="fall" style="--speed:${11 + column * 2}s;--delay:-${column * 3.5}s">${tiles}${tiles}</div>`;
    }).join("");
  }

  // ---------------------------------------------------------------- source image
  function resetRun() {
    state.job = null; state.jobId = null; state.selected = null; state.heroModel = null; state.bars = {};
    state.ranked = false; state.appeared = new Set(); state.revealed = new Set();
  }
  function setImage(image) {
    state.image = image;
    resetRun();
    document.querySelectorAll(".sample").forEach((el) => el.classList.toggle("active", image.kind === "sample" && el.dataset.id === image.id));
    $("route-box").classList.add("hidden");
    $("progress").classList.add("hidden");
    updateRunButton();
    renderHero(); renderGrid();
  }
  async function uploadFile(file) {
    if (!file) return;
    if (state.running) { toast("wait until the current request finishes"); return; }
    const form = new FormData();
    form.append("file", file);
    $("dropzone").classList.add("uploading");
    try {
      const meta = await api("/api/upload", { method: "POST", body: form });
      setImage({ kind: "upload", id: meta.image_id, url: meta.url, width: meta.width, height: meta.height, name: meta.filename });
    } catch (error) {
      toast(error.message);
    } finally {
      $("dropzone").classList.remove("uploading");
    }
  }
  $("dropzone").addEventListener("click", () => $("file-input").click());
  $("dropzone").addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); $("file-input").click(); } });
  $("file-input").addEventListener("change", (event) => { uploadFile(event.target.files[0]); event.target.value = ""; });
  ["dragenter", "dragover"].forEach((name) => $("dropzone").addEventListener(name, (event) => { event.preventDefault(); $("dropzone").classList.add("dragover"); }));
  ["dragleave", "drop"].forEach((name) => $("dropzone").addEventListener(name, () => $("dropzone").classList.remove("dragover")));
  $("dropzone").addEventListener("drop", (event) => { event.preventDefault(); uploadFile(event.dataTransfer.files[0]); });
  document.addEventListener("paste", (event) => {
    const item = [...(event.clipboardData?.items || [])].find((entry) => entry.type.startsWith("image/"));
    if (item) uploadFile(item.getAsFile());
  });

  async function loadSamples() {
    try {
      state.samples = (await api("/api/samples")).samples;
    } catch (error) {
      state.samples = [];
    }
    $("samples").innerHTML = state.samples.map((sample) =>
      `<button class="sample" type="button" data-id="${esc(sample.id)}" title="${esc(sample.instruction)}"><img src="${esc(sample.url)}" alt="${esc(sample.instruction)}" loading="lazy"></button>`).join("");
    fillCollage(state.samples.map((sample) => sample.url));
    if (!state.image && state.samples.length) pickSample(state.samples[0]);     // start with a photo in the frame
  }
  function pickSample(sample) {
    setImage({ kind: "sample", id: sample.id, url: sample.url, width: sample.width, height: sample.height, name: sample.file });
    setInstruction(sample.instruction);
  }
  $("samples").addEventListener("click", (event) => {
    const button = event.target.closest(".sample");
    if (!button || state.running) return;
    const sample = state.samples.find((row) => row.id === button.dataset.id);
    if (sample) pickSample(sample);
  });

  // ---------------------------------------------------------------- request: one line that grows with the text
  function fitRequest() {
    const box = $("instruction");
    box.style.height = "auto";
    box.style.height = `${box.scrollHeight}px`;
    $("char-count").textContent = `${box.value.length} / ${MAX_REQUEST}`;
    updateRunButton();
  }
  function setInstruction(text) {
    $("instruction").value = text.slice(0, MAX_REQUEST);
    fitRequest();
  }
  function updateRunButton() {
    $("run-btn").disabled = state.running || !state.image || !$("instruction").value.trim();
  }
  $("instruction").addEventListener("input", fitRequest);
  $("instruction").addEventListener("keydown", (event) => { if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) { event.preventDefault(); processRequest(); } });
  $("run-btn").addEventListener("click", () => processRequest());

  function setRunning(running) {
    state.running = running;
    $("run-btn").classList.toggle("loading", running);
    $("run-label").textContent = running ? "Processing" : "Process";
    updateRunButton();
  }

  // what the assistant read: rows appear one by one, then the models pop in
  function renderRouteBox(understood) {
    const models = (state.job?.candidates || []).map((c, index) => `<span class="intent" style="--k:${index}">${esc(c.name)}</span>`).join("");
    $("route-box").innerHTML = `<div class="fade">
      <div class="row"><span class="label">Task</span><b>${esc(understood.task_label)}</b></div>
      <div class="row"><span class="label">Sent</span><span class="sent">${esc(understood.english)}</span></div>
      ${understood.target ? `<div class="row"><span class="label">Target</span><b>${esc(understood.target)}</b></div>` : ""}
      <div class="row"><span class="label">Models</span>${models}</div></div>`;
    $("route-box").classList.remove("hidden");
  }

  // ---------------------------------------------------------------- run
  async function processRequest() {
    const instruction = $("instruction").value.trim();
    if (!state.image || !instruction || state.running) return;
    const body = { instruction, ...(state.image.kind === "upload" ? { image_id: state.image.id } : { sample_id: state.image.id }) };
    resetRun();
    setRunning(true);
    renderHero(); renderGrid();
    $("route-box").classList.remove("hidden");
    $("route-box").innerHTML = '<div class="thinking"><span class="spinner"></span>Reading your request</div>';
    $("progress").classList.add("hidden");
    let job;
    try {
      job = await api("/api/jobs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    } catch (error) {
      setRunning(false);
      $("route-box").classList.add("hidden");
      toast(error.message);
      return;
    }
    const jobId = job.job_id;
    state.jobId = jobId;
    acceptJob(job);
    renderRouteBox(job.understood);
    $("progress").classList.remove("hidden");
    requestAnimationFrame(tick);
    while (state.jobId === jobId && ACTIVE.includes(state.job.status)) {
      await sleep(POLL_MS);
      if (state.jobId !== jobId) return;
      try {
        acceptJob(await api(`/api/jobs/${jobId}`));
      } catch (error) {
        toast(error.message);
        break;
      }
    }
    if (state.jobId !== jobId) return;
    setRunning(false);
    if (state.job.status === "error") toast(state.job.error || "the job failed");
    await sleep(FINISH_MS + 60);          // let the last bar reach 100%
    if (state.jobId !== jobId) return;
    finishRun();
  }

  // the run is over: open the best edit in the slider, move the cards into rank order
  function finishRun() {
    const best = state.job.candidates.find((c) => c.best);
    if (best && !state.selected) state.selected = best.model;
    updateOverall();
    renderHero();
    const before = new Map([...document.querySelectorAll("#grid .card")].map((el) => [el.id, el.getBoundingClientRect()]));
    state.ranked = true;
    renderGrid();
    document.querySelectorAll("#grid .card").forEach((el, index) => {
      const old = before.get(el.id);
      if (!old) return;
      const now = el.getBoundingClientRect();
      const dx = old.left - now.left;
      const dy = old.top - now.top;
      if (!dx && !dy) return;
      el.animate([{ transform: `translate(${dx}px, ${dy}px)` }, { transform: "none" }],
        { duration: 700, delay: index * 40, easing: "cubic-bezier(0.2, 0.8, 0.2, 1)", fill: "backwards" });
    });
  }

  function acceptJob(job) {
    const before = state.job;
    state.job = job;
    state.polledAt = performance.now();
    const changed = !before || job.candidates.some((c, index) => !before.candidates[index] || before.candidates[index].status !== c.status || before.candidates[index].rank !== c.rank || before.candidates[index].best !== c.best);
    job.candidates.forEach((candidate) => {
      const bar = state.bars[candidate.model] || (state.bars[candidate.model] = { progress: 0, finishedAt: null });
      if (isDone(candidate) && bar.finishedAt === null) bar.finishedAt = performance.now();
    });
    if (changed) renderGrid();
    updateOverall();
  }

  // one animation loop while a job runs: the bar of the running model and the short finish of the one just done
  function tick() {
    const job = state.job;
    if (!job) return;
    let active = ACTIVE.includes(job.status);
    job.candidates.forEach((candidate) => {
      const bar = state.bars[candidate.model];
      if (candidate.status === "running") {
        bar.progress = progressAt(elapsedNow(candidate), candidate.estimate);
        updateCard(candidate, bar);
      } else if (bar.finishedAt !== null && !bar.shown) {
        const k = Math.min(1, (performance.now() - bar.finishedAt) / FINISH_MS);
        const from = bar.from ?? (bar.from = bar.progress);
        bar.progress = from + (1 - from) * (1 - Math.pow(1 - k, 3));
        updateCard(candidate, bar);
        active = true;
        if (k >= 1) { bar.shown = true; renderGrid(); }
      }
    });
    if (active) requestAnimationFrame(tick);
  }

  // the bar under the Process button: all models together; it goes away when the run is over
  function updateOverall() {
    const job = state.job;
    if (!job) return;
    const all = job.candidates;
    const finished = all.filter(isDone).length;
    const total = all.reduce((sum, c) => sum + c.estimate, 0) || 1;
    const doneTime = all.reduce((sum, c) => sum + (isDone(c) ? c.estimate : c.status === "running" ? c.estimate * (state.bars[c.model]?.progress || 0) : 0), 0);
    $("progress").classList.toggle("hidden", !ACTIVE.includes(job.status));
    $("progress-bar").style.width = `${Math.min(100, 100 * doneTime / total)}%`;
    $("progress-count").textContent = `${finished} / ${all.length}`;
    const current = all.find((c) => c.status === "running");
    $("progress-text").textContent = job.status === "waiting" ? "Waiting for another request" : current ? `Running ${current.name}` : "Starting";
  }

  // ---------------------------------------------------------------- big frame: the photo, or the slider
  function renderHero() {
    const hero = $("hero");
    hero.classList.add("solo");
    if (!state.image) {
      hero.innerHTML = `<div class="hero-media"><div class="hero-empty">Upload a photo or pick a sample on the left.</div></div>`;
      state.heroModel = null;
      return;
    }
    const ratio = state.image.width && state.image.height ? state.image.width / state.image.height : 1.5;
    const sizing = `aspect-ratio:${ratio};max-height:50vh;max-width:calc(50vh * ${ratio})`;    // leaves the cards in view
    const candidate = state.job?.candidates.find((c) => c.model === state.selected && c.status === "ok");
    if (!candidate) {
      hero.innerHTML = `<div class="hero-media"><div class="compare" style="${sizing}"><img class="only" src="${esc(state.image.url)}" alt="Original photo"><span class="tag l">Original</span></div></div>`;
      state.heroModel = null;
      return;
    }
    const original = state.job.original?.url || state.image.url;
    const done = state.job.candidates.filter((c) => c.status === "ok").length;
    hero.innerHTML = `
      <div class="hero-media">
        <div class="compare${candidate.transparent ? " checker" : ""}" style="${sizing}">
          <img src="${esc(candidate.image_url)}" alt="${esc(candidate.name)} edit">
          <img class="before" src="${esc(original)}" alt="Original">
          <div class="handle"></div>
          <span class="tag l">Original</span><span class="tag r">Edited</span>
          <input type="range" min="0" max="100" value="50" aria-label="Before and after">
        </div>
        <div class="hero-line"><span>Model <b>${esc(candidate.name)}</b></span><span>Score <b class="g">${candidate.score.toFixed(2)}</b></span><span>Latency <b>${candidate.seconds.toFixed(1)} s</b></span><span>Rank <b>${candidate.rank ? `${candidate.rank} / ${done}` : "-"}</b></span></div>
      </div>`;
    const box = hero.querySelector(".compare");
    const range = hero.querySelector("input[type=range]");
    range.addEventListener("input", () => box.style.setProperty("--pos", `${range.value}%`));
    if (state.heroModel !== candidate.model) {     // a new edit: the divider sweeps in from the right
      const start = performance.now();
      const sweep = (now) => {
        const k = Math.min(1, (now - start) / SWEEP_MS);
        const position = 100 - 50 * (1 - Math.pow(1 - k, 3));
        box.style.setProperty("--pos", `${position}%`);
        range.value = position;
        if (k < 1 && box.isConnected) requestAnimationFrame(sweep);
      };
      requestAnimationFrame(sweep);
    }
    state.heroModel = candidate.model;
  }

  // ---------------------------------------------------------------- result cards
  // running order (fastest first) while the job runs, rank order after it (failed models last)
  function displayOrder(candidates) {
    if (!state.ranked) return candidates;
    return [...candidates].sort((a, b) => (a.rank || 999) - (b.rank || 999));
  }
  function renderGrid() {
    const grid = $("grid");
    const job = state.job;
    if (!job) { grid.innerHTML = ""; return; }
    grid.innerHTML = "";
    displayOrder(job.candidates).forEach((c) => {
      const bar = state.bars[c.model] || { progress: 0 };
      const settled = isDone(c) && (bar.shown || bar.finishedAt === null);
      const shown = settled ? c.status : c.status === "queued" ? "queued" : "running";
      const el = document.createElement("div");
      el.id = `card-${c.model}`;
      el.className = `card ${shown === "ok" ? "done" : shown}${c.best && state.ranked ? " winner" : ""}${state.selected === c.model ? " selected" : ""}${state.appeared.has(c.model) ? "" : " fresh"}`;
      state.appeared.add(c.model);
      let media;
      let body;
      if (shown === "ok") {
        const reveal = state.revealed.has(c.model) ? "" : "reveal";
        state.revealed.add(c.model);
        media = `<div class="media${c.transparent ? " checker" : ""}"><span class="rank r${c.rank}">#${c.rank}</span><img class="${reveal}" src="${esc(c.thumb_url)}" alt="">
          <a class="dl" href="${esc(c.image_url)}" download title="Download" aria-label="Download the ${esc(c.name)} edit"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 4v11m0 0-4.5-4.5M12 15l4.5-4.5"/><path d="M5 19h14"/></svg></a></div>`;
        body = `<div class="line"><b>${esc(c.name)}</b><span class="t">${c.seconds.toFixed(1)} s</span><span class="g">${c.score.toFixed(2)}</span></div>`;
        el.setAttribute("role", "button");
        el.tabIndex = 0;
        el.setAttribute("aria-label", `Compare ${c.name} with the original`);
        el.addEventListener("click", (event) => {
          if (event.target.closest(".dl")) return;     // the download link works on its own
          state.selected = c.model;
          renderHero(); renderGrid();
          $("hero").scrollIntoView({ behavior: "smooth", block: "nearest" });
        });
        el.addEventListener("keydown", (event) => {
          if (event.target !== el || (event.key !== "Enter" && event.key !== " ")) return;
          event.preventDefault();
          el.click();
        });
      } else if (shown === "failed") {
        media = `<div class="media"><div class="err">${esc(c.error || "no image")}</div></div>`;
        body = `<div class="line"><b>${esc(c.name)}</b><span class="t">failed after ${(c.seconds || 0).toFixed(1)} s</span></div>`;
      } else if (shown === "queued") {
        media = `<div class="media"><span class="wait">waiting</span></div>`;
        body = `<div class="title"><b>${esc(c.name)}</b></div><div class="note-line">Usually about ${c.estimate.toFixed(1)} s.</div>`;
      } else {
        media = `<div class="media loading-bg"></div>`;
        body = `<div class="title"><b>${esc(c.name)}</b></div><div class="cbar"><i></i></div><div class="cstatus"><span class="el"></span><span class="pc"></span></div>`;
      }
      el.innerHTML = media + `<div class="body">${body}</div>`;
      grid.appendChild(el);
      if (shown === "running") updateCard(c, bar);
    });
  }
  function updateCard(candidate, bar) {
    const el = $(`card-${candidate.model}`);
    if (!el) return;
    const fill = el.querySelector(".cbar i");
    if (fill) fill.style.transform = `scaleX(${bar.progress})`;
    const elapsedText = el.querySelector(".el");
    const percent = el.querySelector(".pc");
    if (elapsedText) {
      const elapsed = candidate.status === "running" ? elapsedNow(candidate) : candidate.seconds || 0;
      const slow = elapsed > candidate.estimate * 1.15;
      elapsedText.innerHTML = `${elapsed.toFixed(1)} s of about ${candidate.estimate.toFixed(1)} s${slow ? ' <span class="slow">slower than usual</span>' : ""}`;
      percent.textContent = `${Math.floor(bar.progress * 100)}%`;
    }
  }

  // ---------------------------------------------------------------- rankings: drawn again each time the tab opens
  async function showRankings() {
    if (!state.rankings) {
      try {
        state.rankings = await api("/api/rankings");
      } catch (error) {
        toast(error.message);
        return;
      }
    }
    $("rank-grid").innerHTML = state.rankings.tasks.map((task, card) => {
      const rows = task.rows.map((row, index) => `<tr class="${row.place === 1 ? "p1" : ""}" style="--r:${index}">
        <td><span class="pos">${row.place}</span><b>${esc(row.name)}</b>${row.live ? '<span class="live-dot" title="Runs in the live demo" aria-label="Runs in the live demo"></span>' : ""}</td>
        <td class="num">${row.score.toFixed(2)}</td><td class="num">${row.estimate.toFixed(2)}</td></tr>`).join("");
      return `<section class="surface rank-card" style="--i:${card}"><h3>${esc(task.label)}</h3><div class="table-scroll"><table class="lb">
        <thead><tr><th>Model</th><th class="num">Score</th><th class="num" title="estimated on an A30 GPU">s/image</th></tr></thead><tbody>${rows}</tbody></table></div></section>`;
    }).join("");
  }

  // ---------------------------------------------------------------- dataset: the figures of the reports
  // Each block fades up when it scrolls into view, then its figure draws itself (bars grow, cells and points appear).
  function columns(rows) {
    const top = Math.max(...rows.map((row) => row[1]));
    return `<div class="vbars">${rows.map(([label, count], index) => `<div class="vbar" style="--i:${index}"><span class="v">${compact(count)}</span><span class="col"><i style="--h:${(100 * count / top).toFixed(1)}%"></i></span><span class="k">${esc(label)}</span></div>`).join("")}</div>`;
  }
  function donut(parts, colors, unit) {
    const total = parts.reduce((sum, part) => sum + part[1], 0);
    const circle = 2 * Math.PI * 42;
    let offset = 0;
    const segments = parts.map(([, count], index) => {
      const length = circle * count / total;
      const svg = `<circle class="seg" r="42" cx="60" cy="60" style="--len:${length.toFixed(2)};--off:${(-offset).toFixed(2)};--i:${index};stroke:${colors[index]}"/>`;
      offset += length;
      return svg;
    }).join("");
    return `<div class="donut-row"><div class="donut"><svg viewBox="0 0 120 120"><circle class="ring" r="42" cx="60" cy="60"/>${segments}</svg>
      <div class="donut-c"><b data-to="${total}">0</b><span>${esc(unit)}</span></div></div>
      <div class="legend">${parts.map(([label, count], index) => `<span><i style="background:${colors[index]}"></i>${esc(label)}<b>${fmt(count)}</b></span>`).join("")}</div></div>`;
  }
  function pipeline(groups) {
    let index = 0;
    const group = (block, number) => `<div class="pipe-group" style="--g:${number}"><div class="pipe-title">${esc(block.group)}</div><div class="pipe-row">${block.steps.map(([title, detail]) =>
      `<div class="pipe-step" style="--i:${index++}"><b>${esc(title)}</b><span>${esc(detail)}</span></div>`).join("")}</div></div>`;
    return `<div class="pipe2">${groups.map((block, number) => group(block, number)).join('<div class="pipe-link" aria-hidden="true"></div>')}</div>`;
  }
  function heatmap(categories) {
    const counts = categories.counts;
    const max = Math.max(...Object.values(counts).flat());
    const strength = (count) => (count ? Math.log(count + 1) / Math.log(max + 1) : 0);
    const head = `<span></span>${categories.operations.map(([, short]) => `<span class="hd">${esc(short)}</span>`).join("")}<span class="hd">All</span>`;
    const rows = categories.subjects.map((subject, row) => {
      const cells = categories.operations.map(([operation], column) => {
        const count = counts[operation][row];
        const alpha = strength(count);
        return `<span class="cell${count ? "" : " zero"}${alpha > 0.62 ? " dark" : ""}" style="--a:${(0.08 + 0.92 * alpha).toFixed(2)};--d:${row + column}" title="${esc(operation)}, ${esc(subject)}: ${fmt(count)}">${count || ""}</span>`;
      }).join("");
      const total = categories.operations.reduce((sum, [operation]) => sum + counts[operation][row], 0);
      return `<span class="rk">${esc(subject.replace("/", ", ").replace("-", " "))}</span>${cells}<span class="tot">${fmt(total)}</span>`;
    }).join("");
    const totals = `<span class="rk"><b>All</b></span>${categories.operations.map(([operation]) => `<span class="tot col-total">${compact(counts[operation].reduce((sum, count) => sum + count, 0))}</span>`).join("")}<span class="tot">${compact(categories.labels)}</span>`;
    return `<div class="heat">${head}${rows}${totals}</div>`;
  }
  function share(categories, operation, subject) {
    const counts = categories.counts;
    const total = operation ? counts[operation].reduce((sum, count) => sum + count, 0)
      : Object.values(counts).reduce((sum, row) => sum + row[categories.subjects.indexOf(subject)], 0);
    return Math.round(100 * total / categories.labels);
  }
  function agreement(rows) {
    return rows.map(([label, team, chance], index) => `<div class="agree"><div class="agree-k">${esc(label)}</div>
      <div class="pair"><span class="track"><i class="gx" style="--w:${team}%;--i:${index}"></i><u style="left:${chance}%" title="chance ${chance}%"></u></span><b>${team.toFixed(1)}%</b></div></div>`).join("")
      + `<div class="agree-note"><u></u>agreement if everyone ranked at random</div>`;
  }
  function accuracy(data) {
    const high = 0.7;
    const x = (value) => `${(100 * value / high).toFixed(1)}%`;
    const ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7].map((value) => `<span style="left:${x(value)}">${value.toFixed(1)}</span>`).join("");
    return `<div class="acc"><span></span><span class="ticks">${ticks}</span><span></span>${data.rows.map(([label, value, kind], index) => `<span class="k ${kind}">${esc(label)}</span>
      <span class="plot"><i class="gx ${kind}" style="--w:${x(value)};--i:${index}"></i><u style="left:${x(data.chance)}"></u></span>
      <span class="n ${kind}">${value.toFixed(3)}</span>`).join("")}</div>`;
  }
  // test accuracy against the time to score one pair; the time axis is logarithmic (each line about 3 times the last)
  function scatter(points) {
    const px = (ms) => 46 + (Math.log10(ms) - 1.7) / 2.78 * 334;
    const py = (value) => 200 - (value - 0.475) / 0.11 * 182;
    const ticksX = [[100, "0.1 s"], [300, "0.3 s"], [1000, "1 s"], [3000, "3 s"], [10000, "10 s"], [30000, "30 s"]].map(([ms, label]) =>
      `<line class="grid-line" x1="${px(ms)}" x2="${px(ms)}" y1="12" y2="200"/><text class="axis" x="${px(ms)}" y="213" text-anchor="middle">${label}</text>`).join("");
    const ticksY = [0.50, 0.55].map((value) => `<line class="grid-line" x1="46" x2="380" y1="${py(value)}" y2="${py(value)}"/><text class="axis" x="40" y="${py(value) + 3}" text-anchor="end">${value.toFixed(2)}</text>`).join("");
    const dots = points.map((point, index) => {
      const ours = point.ours ? " ours" : "";
      const dx = point.label_left ? -8 : 8;
      return `<circle class="pt${ours}" cx="${px(point.ms).toFixed(1)}" cy="${py(point.accuracy).toFixed(1)}" r="${point.ours ? 6 : 4.5}" style="--i:${index}"/>
        <text class="lbl${ours}" x="${(px(point.ms) + dx).toFixed(1)}" y="${(py(point.accuracy) + 3).toFixed(1)}" text-anchor="${point.label_left ? "end" : "start"}" style="--i:${index}">${esc(point.name)}</text>`;
    }).join("");
    return `<svg class="scatter" viewBox="0 0 400 232" role="img" aria-label="Test accuracy against seconds to score one pair">${ticksX}${ticksY}${dots}
      <text class="axis-title" x="213" y="229" text-anchor="middle">Seconds to score one pair</text>
      <text class="axis-title" transform="translate(10 106) rotate(-90)" text-anchor="middle">Test accuracy</text></svg>`;
  }
  async function showDataset() {
    if (!state.dataset) {
      try {
        state.dataset = await api("/api/dataset");
      } catch (error) {
        toast(error.message);
        return;
      }
    }
    const data = state.dataset;
    const stat = (label, value, small = "", decimals = 0) => `<div class="stat"><div class="k">${label}</div><div class="v"><span data-to="${value}" data-dec="${decimals}">0</span>${small ? ` <small>${small}</small>` : ""}</div></div>`;
    const categories = data.categories;
    const editsTotal = data.edits_per_sample.reduce((sum, row) => sum + row[1], 0);
    $("view-dataset").innerHTML = `
      <section class="stats stats6 anim">${stat("Websites", data.stats.websites)}${stat("Languages", data.stats.languages)}${stat("Images", data.stats.images)}${stat("Tasks", data.stats.tasks)}${stat("Models", data.stats.models)}${stat("Size", data.stats.size_gb, "GB", 1)}</section>
      <div class="ds-grid">
        <section class="surface ds-card ds-full anim"><h3>From community posts to the edit assistant</h3>${pipeline(data.pipeline)}</section>
        <div class="ds-split">
          <section class="surface ds-card anim"><h3>What people ask for: operation x subject</h3>${heatmap(categories)}
            <p class="caption">${fmt(categories.labels)} labels of the ${fmt(categories.samples)} samples. Removing is the most common request (<b>${share(categories, "Remove")}%</b>), most requests are about people (<b>${share(categories, null, "Person")}%</b>) or the background (<b>${share(categories, null, "Background")}%</b>).</p></section>
          <div class="ds-stack">
            <section class="surface ds-card anim"><h3>Image filters</h3>${donut(data.filters, ["#10b981", "#f59e0b", "#e11d48"], "images")}</section>
            <section class="surface ds-card anim"><h3>Splits (pairs, by sample)</h3>${donut(data.splits_pairs, ["#2563eb", "#f59e0b", "#db2777"], "pairs")}</section>
          </div>
        </div>
        <section class="surface ds-card anim"><h3>Request length</h3>${columns(data.request_words.bins)}
          <p class="caption">Words per request, in steps of 10, median <b>${data.request_words.median}</b>.</p></section>
        <section class="surface ds-card anim"><h3>Edits per sample</h3>${columns(data.edits_per_sample)}
          <p class="caption">Two thirds of the samples (<b>${Math.round(100 * data.edits_per_sample[0][1] / editsTotal)}%</b>) have exactly two edits.</p></section>
        <section class="surface ds-card anim"><h3>Team ranking: how often members agree</h3>${agreement(data.agreement)}</section>
        <section class="surface ds-card ds-wide anim"><h3>The scorer: pairwise accuracy on ${fmt(data.scorer_accuracy.test_pairs)} test pairs</h3>${accuracy(data.scorer_accuracy)}
          <p class="caption">Red dashed line: a random ranking (${data.scorer_accuracy.chance}). A small head on frozen SigLIP features is level with the 7B editing judges.</p></section>
        <section class="surface ds-card anim"><h3>Accuracy against speed</h3>${scatter(data.scorer_speed)}
          <p class="caption">Up is more accurate, left is faster. Ours needs 1.9 GB of GPU memory, the 7B judges 18 to 20 GB.</p></section>
      </div>`;
    $("view-dataset").querySelectorAll(".anim").forEach((el) => blockWatcher.observe(el));
  }
  // numbers count up from 0 when their block comes into view
  function countUp(el) {
    const target = Number(el.dataset.to);
    const decimals = Number(el.dataset.dec || 0);
    const start = performance.now();
    const step = (now) => {
      const k = Math.min(1, (now - start) / 1200);
      el.textContent = (target * (1 - Math.pow(1 - k, 3))).toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
      if (k < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }
  const blockWatcher = new IntersectionObserver((entries) => entries.forEach((entry) => {
    if (!entry.isIntersecting) return;
    entry.target.classList.add("in");
    entry.target.querySelectorAll("[data-to]").forEach(countUp);
    blockWatcher.unobserve(entry.target);
  }), { threshold: 0.12 });

  // ---------------------------------------------------------------- start
  let savedTheme = null;
  try { savedTheme = localStorage.getItem("ej-theme"); } catch (error) { savedTheme = null; }
  setTheme(savedTheme === "dark" ? "dark" : "light");
  fitRequest();
  renderHero();
  loadSamples();
  showView(viewOfHash());
})();
