# FULL-AUTONOMOUS WEB APP DEVELOPMENT PLAN & CLAUDE MODEL ALLOCATION

This document outlines the end-to-end development roadmap for the **AI Image Editing Benchmark & Auto-Selection System**. It integrates **Karpathy's Iterative Refinement Loop**, **Modern UI/UX Design Benchmarks**, and **Zero-Human-Intervention Directives** for Claude Code CLI.

---

## 🔄 CORE EXECUTION PRINCIPLES

### 1. Karpathy Iterative Loop (Build $\rightarrow$ Test $\rightarrow$ Inspect $\rightarrow$ Refine)
Every phase must execute in an autonomous feedback loop:
1. **Draft Implementation:** Generate clean, modular code.
2. **Execute & Test:** Launch local server/endpoint and run synthetic test requests.
3. **Inspect Output & Logs:** Parse console logs, response status codes, latency, and visual layout flaws.
4. **Self-Correct & Refine:** Automatically fix bugs, handle edge cases, and enhance styling without stopping.
5. **Repeat:** Iterate until zero bugs and optimal performance/visual metrics are achieved.

### 2. Design Inspiration & Modern Aesthetics Benchmark
* **Reference UI Standards:** Benchmark visual styling against top-tier AI platforms (*Midjourney web UI, Leonardo.ai, RunwayML, Clipdrop, Tailwind UI Dark components*).
* **Styling Rules:** Dark slate aesthetic (`#0f172a` background, `#1e293b` cards), smooth CSS glassmorphism, accent glow gradients (emerald for high score gains, violet for active models), micro-interactions, responsive grid, real-time progress indicators, and crisp typography.

### 3. Fully Autonomous Protocol (Zero Human Intervention)
* Claude Code runs with `auto mode on`.
* Auto-detect missing dependencies and run `pip install` automatically.
* Auto-restart Uvicorn server when modifying backend files.
* Auto-fix frontend console errors or API payload mismatches autonomously.

---

## 🎯 MODEL ALLOCATION MATRIX

| Phase / Domain | Assigned Model | Key Rationale |
| ----- | ----- | ----- |
| **Phase 1: Architecture & Router** | **Claude Fable 5.2** | Complex system design & robust keyword-to-category mapping logic. |
| **Phase 2: Execution Engine & Mocks** | **Claude Fable 5.2** | Concurrent async execution, clean pipeline abstractions, PIL mocks. |
| **Phase 3: Scorer & Ranking System** | **Claude Opus 5.5 / Fable 5.2** | Math precision for evaluation metrics, EditReward integration & ranking. |
| **Phase 4: Modern Frontend UI/UX** | **Claude Sonnet 3.7 / 3.5** | High-speed, pixel-perfect HTML5, Tailwind CSS, responsive JS engine. |
| **Phase 5: Autonomous E2E & Polish** | **Claude Sonnet 3.7 / 3.5** | Fast agentic feedback loop, self-testing, and Karpathy iterative fixes. |

---

## 📋 PHASES & COPY-PASTE PROMPTS FOR CLAUDE CODE

### Phase 1: Backend Architecture & Task Routing Engine

* **Target Model:** `Claude Fable 5.2`
* **Goal:** Set up FastAPI core architecture, CORS, static routes, and prompt keyword router.

```text
Prompt for Claude Fable 5.2:

Act as a Senior Backend Architect using Claude Fable 5.2. Build the core FastAPI application inside `web_app/app/` in fully autonomous mode:

1. `main.py`: Initialize FastAPI, setup CORS middleware, mount static files (`/static`), and serve Jinja2 templates (`/templates`).
2. `router.py`: Create `TaskRouter.select_models(instruction: str) -> List[str]`. Analyze Vietnamese and English keywords in the prompt to map to 7 model categories (Enhance, Remove, Restore, Add/Replace, Color, Image-Formatting), always including the Shared model list.
3. `schemas.py`: Define Pydantic request/response schemas.

Execute Karpathy Loop: Create unit test scripts to test prompt routing across 20+ sample prompts. Self-correct routing rules until 100% accuracy is reached.

```

### Phase 2: Model Execution & Mock Subsystem

* **Target Model:** `Claude Fable 5.2`
* **Goal:** Async execution engine with PIL transformations for all 22 project models.

```text
Prompt for Claude Fable 5.2:

Act as a Core Backend Engineer using Claude Fable 5.2. Implement `web_app/app/model_runner.py`:

1. Define `ModelRunner.run_models(image_path: str, instruction: str, model_names: List[str]) -> List[Dict]`.
2. Implement mock handlers for all 22 project models using Pillow (PIL) to simulate realistic outputs (sharpening, contrast tweaks, color shifts, background blur, object removal mocks) matching model descriptions.
3. Capture execution time and VRAM estimates to return structured JSON outputs.
4. Abstract class interface for dropping in real model weights seamlessly later.

Run test execution across all categories automatically and fix any PIL processing edge cases.

```

### Phase 3: Scorer Subsystem & Ranking Logic

* **Target Model:** `Claude Opus 5.5` (or `Fable 5.2`)
* **Goal:** Integrate EditReward metric scoring and top-candidate selection.

```text
Prompt for Claude Opus 5.5:

Act as an ML Scoring Specialist using Claude Opus 5.5. Implement `web_app/app/scorer.py`:

1. Implement `ImageScorer.score_candidates(original_img: str, instruction: str, candidates: List[Dict]) -> List[Dict]`.
2. Integrate EditReward calculation (or high-fidelity scoring wrapper) returning gain scores (-1.0 to +1.0).
3. Implement `rank_candidates(candidates: List[Dict]) -> Tuple[Dict, List[Dict]]` returning the winner (`best_candidate`) alongside full sorted rankings.

Test scoring edge cases autonomously to guarantee zero null or invalid scores.

```

### Phase 4: World-Class Frontend UI/UX (Inspired by Midjourney & Leonardo)

* **Target Model:** `Claude Sonnet 3.7` (or `Sonnet 3.5`)
* **Goal:** Build a modern, polished Dark Theme UI with Tailwind CSS & Vanilla JS.

```text
Prompt for Claude Sonnet 3.7:

Act as a Lead Frontend Engineer using Claude Sonnet 3.7. Create `web_app/templates/index.html` and `web_app/static/js/app.js`:

1. Styling & Aesthetic (Inspired by Midjourney & Leonardo AI):
   - Premium Dark Slate theme (`#0f172a`), neon accent borders, glassmorphic cards (`backdrop-blur-md`).
   - Clean typography, smooth transitions, responsive grid layout.
2. Left Column (Input Controls):
   - Drag-and-Drop Image Uploader with instant preview thumbnail.
   - Instruction Textarea with pre-built quick-tag suggestions (e.g., "Làm nét ảnh", "Làm sáng", "Xóa vật thể").
   - Action Button: "Run AI Selection Engine" with animated glow & loading state.
3. Right Column (Output & Benchmark Analytics):
   - Winner Hero Banner: Displays the highest-scoring image, winning model badge (e.g., "VOSR 2.0"), EditReward score gain (+0.110), and processing speed.
   - Interactive Candidate Table/Grid: View all candidate outputs, compare score badges, latency, VRAM, with lightbox image modal zoom.
4. JS Integration: Handle asynchronous upload and API polling smoothly.

```

### Phase 5: Autonomous Integration & Karpathy Self-Improvement Loop

* **Target Model:** `Claude Sonnet 3.7` (or `Sonnet 3.5`)
* **Goal:** End-to-End launch, automated test execution, self-debugging, and final refinement.

```text
Prompt for Claude Sonnet 3.7:

Act as a Full-Stack DevOps Engineer using Claude Sonnet 3.7. Execute complete autonomous setup:

1. Create `requirements_web.txt` (`fastapi`, `uvicorn`, `python-multipart`, `pillow`, `jinja2`).
2. Create `run_web.py` to launch Uvicorn server on `http://127.0.0.1:8000`.
3. KARPATHY IMPROVEMENT LOOP (Execute 100% autonomously):
   - Install all missing requirements.
   - Start the server in background / subprocess.
   - Send test POST requests with sample images and instructions.
   - Check API responses, frontend static file loading, and UI console logs.
   - If any bug occurs (404, 500, broken CSS, layout glitch, JS error), parse the log, fix the code immediately, and re-test.
   - Iterate until the web application runs flawlessly with zero errors and exceptional UI layout quality.
```