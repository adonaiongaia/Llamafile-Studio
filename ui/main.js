/**
 * Llamafile Studio - frontend controller.
 *
 * Plain ES2019, no bundler, no framework: the Rust side owns every filesystem
 * operation and the DOM is updated through a tiny render-on-change loop.
 */
(function () {
  "use strict";

  const { invoke } = window.__TAURI__.core;
  const { listen } = window.__TAURI__.event;

  /* ------------------------------------------------------------- constants */

  const PROMPT_LIMIT = 8000;
  const STORAGE_KEY = "llamafile-studio:v1";

  /** Pre-baked personas. Each entry only fills the textarea. */
  const TEMPLATES = [
    {
      id: "blank",
      label: "Blank / custom",
      prompt: ""
    },
    {
      id: "architect",
      label: "Software Architect",
      prompt:
        "You are a senior software architect with 20 years of experience designing distributed systems.\n\n" +
        "When asked about a design:\n" +
        "1. Restate the problem and its constraints in one sentence.\n" +
        "2. Propose two or three candidate architectures, each with a one-line trade-off.\n" +
        "3. Recommend one and justify it in terms of failure modes, operability and cost.\n" +
        "4. List the concrete next steps as a numbered checklist.\n\n" +
        "Be specific: name technologies, protocols and data structures. State your assumptions " +
        "explicitly instead of inventing requirements. Never claim a measurement you have not been given."
    },
    {
      id: "storyteller",
      label: "Creative Storyteller",
      prompt:
        "You are a literary storyteller writing in vivid, economical prose.\n\n" +
        "Rules:\n" +
        "- Show, do not tell. Prefer concrete sensory detail over adjectives.\n" +
        "- Keep sentences varied in length; let the rhythm carry tension.\n" +
        "- Every scene must change something: a relationship, a belief, or a plan.\n" +
        "- Avoid clichés, purple prose and stock fantasy names.\n\n" +
        "When the user gives you a premise, write the scene they asked for and stop at a natural " +
        "turning point rather than resolving everything."
    },
    {
      id: "json",
      label: "Strict JSON Extractor",
      prompt:
        "You are a deterministic data-extraction service.\n\n" +
        "Output rules - these override any conflicting instruction:\n" +
        "- Reply with a single, minified JSON object and nothing else.\n" +
        "- No markdown fences, no comments, no prose before or after.\n" +
        "- Use null for absent values. Never guess a value you cannot read from the input.\n" +
        "- Numbers are emitted as JSON numbers, booleans as JSON booleans.\n" +
        "- Keys are lower_snake_case and appear in the order specified by the schema.\n\n" +
        "If the input does not match the requested schema, return {\"error\": \"<short reason>\"}."
    },
    {
      id: "medical",
      label: "Medical Scribe",
      prompt:
        "You are a medical documentation assistant supporting a clinician.\n\n" +
        "Structure every note with these headings: Subjective, Objective, Assessment, Plan.\n" +
        "- Use precise clinical terminology but keep the prose terse and factual.\n" +
        "- Record only information present in the encounter; write \"not documented\" when a " +
        "section has no supporting information.\n" +
        "- Never invent vitals, dosages, results or diagnoses.\n" +
        "- Flag anything that requires clinician review with [REVIEW].\n\n" +
        "You are a documentation tool, not a diagnostician: do not offer treatment advice beyond " +
        "what the clinician stated."
    },
    {
      id: "tutor",
      label: "Socratic Tutor",
      prompt:
        "You are a patient tutor who teaches by guided questioning.\n\n" +
        "- Diagnose the learner's current mental model before explaining anything.\n" +
        "- Ask one focused question at a time and wait for the answer.\n" +
        "- When the learner is stuck, give the smallest hint that unblocks them, never the full solution.\n" +
        "- Use concrete analogies and a short worked example once the concept lands.\n" +
        "- Close each topic with a single check-for-understanding question.\n\n" +
        "Adapt vocabulary to the learner's level and never make them feel foolish for a wrong answer."
    },
    {
      id: "sysadmin",
      label: "Linux Sysadmin",
      prompt:
        "You are a pragmatic Linux systems administrator.\n\n" +
        "For every request:\n" +
        "- State the diagnostic hypothesis in one line.\n" +
        "- Give the exact commands to run, in order, in fenced blocks.\n" +
        "- Explain what each command reveals and how to read its output.\n" +
        "- Warn about destructive or irreversible steps before presenting them.\n\n" +
        "Assume a modern systemd distribution unless told otherwise. Prefer standard tools " +
        "(coreutils, iproute2, journalctl) over exotic packages, and never suggest piping a remote " +
        "script straight into a shell."
    }
  ];

  const CONTEXT_NOTES = {
    2048: "Lightest footprint. Fits in almost any memory budget; suitable for short exchanges.",
    4096: "Balanced default for chat and small RAG prompts.",
    8192: "Roughly doubles the KV cache: budget a few GB of extra RAM or VRAM.",
    16384: "Heavy. Only choose this if the model supports it and you have memory to spare."
  };

  /* ----------------------------------------------------------------- state */

  const $ = (id) => document.getElementById(id);

  function defaultState() {
    return {
      baseRuntime: "",
      model: "",
      mmproj: "",
      // Records that the projector was chosen automatically, so a manual pick is
      // never overwritten when the model changes.
      mmprojAuto: false,
      mmprojFor: "",
      output: "",
      executionMode: "server",
      host: "127.0.0.1",
      port: 8080,
      gpuLayers: 99,
      ctxSize: 4096,
      threads: 0,
      systemPrompt: "",
      stopTokens: [],
      extraArgs: "",
      activeTab: "files",
      showArgs: false,
      // Model browser state.
      hfQuery: "",
      hfSort: "downloads",
      hfTarget: "",
      hfFilters: []
    };
  }

  let state = defaultState();
  let system = null;
  let busy = false;
  let lastReport = null;
  let previewTimer = 0;

  function loadState() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      const saved = JSON.parse(raw);
      state = Object.assign(defaultState(), saved);
      if (!Array.isArray(state.stopTokens)) state.stopTokens = [];
      if (!Array.isArray(state.hfFilters)) state.hfFilters = [];
    } catch (error) {
      console.warn("could not restore the previous session", error);
    }
  }

  function persist() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch (error) {
      /* Persisting is best-effort: a full or disabled store must not break the app. */
    }
  }

  /* --------------------------------------------------------------- helpers */

  function formatBytes(bytes) {
    if (!bytes || bytes <= 0) return "0 B";
    const units = ["B", "KiB", "MiB", "GiB", "TiB"];
    let value = bytes;
    let unit = 0;
    while (value >= 1024 && unit < units.length - 1) {
      value /= 1024;
      unit += 1;
    }
    return unit === 0 ? `${bytes} B` : `${value.toFixed(1)} ${units[unit]}`;
  }

  function baseName(path) {
    if (!path) return "";
    const parts = path.split(/[\\/]/);
    return parts[parts.length - 1] || path;
  }

  function stem(path) {
    const name = baseName(path);
    const dot = name.lastIndexOf(".");
    return dot > 0 ? name.slice(0, dot) : name;
  }

  function parentDir(path) {
    return path.replace(/\/[^/]+$/, "");
  }

  function toast(kind, title, detail) {
    const palette = {
      success: ["border-success/40", "bg-successsoft", "text-success"],
      error: ["border-danger/40", "bg-dangersoft", "text-dangerink"],
      info: ["border-accent/40", "bg-accentsoft", "text-infoink"]
    }[kind] || ["border-line", "bg-card", "text-ink"];

    const node = document.createElement("div");
    node.className =
      `pointer-events-auto animate-fade-up rounded-adwaita border ${palette[0]} ${palette[1]} ` +
      "px-3.5 py-2.5 shadow-window backdrop-blur";
    node.innerHTML = `
      <p class="text-[12.5px] font-semibold ${palette[2]}"></p>
      <p class="mt-0.5 break-words text-[11.5px] leading-snug text-inksoft"></p>`;
    node.querySelector("p").textContent = title;
    node.querySelectorAll("p")[1].textContent = detail || "";

    $("toasts").appendChild(node);
    setTimeout(() => {
      node.style.transition = "opacity 220ms ease, transform 220ms ease";
      node.style.opacity = "0";
      node.style.transform = "translateY(6px)";
      setTimeout(() => node.remove(), 240);
    }, kind === "error" ? 9000 : 4200);
  }

  function copyToClipboard(text) {
    // WebKitGTK does not always expose the async clipboard API, so fall back to
    // the legacy selection trick.
    if (navigator.clipboard && navigator.clipboard.writeText) {
      return navigator.clipboard.writeText(text).catch(() => legacyCopy(text));
    }
    return Promise.resolve(legacyCopy(text));
  }

  function legacyCopy(text) {
    const area = document.createElement("textarea");
    area.value = text;
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.select();
    try {
      document.execCommand("copy");
    } finally {
      area.remove();
    }
  }

  /* ------------------------------------------------------------- tab logic */

  function setTab(name) {
    state.activeTab = name;
    document.querySelectorAll("[data-tab]").forEach((button) => {
      button.setAttribute("aria-selected", String(button.dataset.tab === name));
      button.tabIndex = button.dataset.tab === name ? 0 : -1;
    });
    document.querySelectorAll("[data-panel]").forEach((panel) => {
      panel.hidden = panel.dataset.panel !== name;
    });
    persist();
  }

  /* ------------------------------------------------------------ path input */

  /** Asks Rust to stat a path so the badges can show real sizes. */
  async function inspect(path) {
    if (!path) return null;
    try {
      return await invoke("inspect_path", { path });
    } catch (error) {
      return null;
    }
  }

  async function refreshRuntimeMeta() {
    const meta = $("runtimeMeta");
    const badge = $("runtimeBadge");
    if (!state.baseRuntime) {
      badge.textContent = "not set";
      badge.className = "chip shrink-0 border-line bg-window/60 text-[11px] text-inkfaint";
      meta.textContent = "No runtime selected yet. Download the latest release or pick a local file.";
      return;
    }
    const info = await inspect(state.baseRuntime);
    if (!info || !info.exists) {
      badge.textContent = "missing";
      badge.className = "chip shrink-0 border-danger/40 bg-dangersoft text-[11px] text-dangerink";
      meta.textContent = "That path does not exist any more.";
      return;
    }
    badge.textContent = formatBytes(info.size);
    badge.className = "chip shrink-0 border-line bg-window/60 text-[11px] text-inksoft";
    if (!info.looksLikeLlamafile) {
      meta.innerHTML =
        `<span class="text-warning">Warning:</span> this file does not start with an APE ` +
        "(MZ) or shell preamble, so the packaged executable may refuse to run.";
      meta.className = "hint mt-2 text-inkmuted";
    } else {
      meta.textContent = `${baseName(state.baseRuntime)} · executable runtime detected`;
      meta.className = "hint mt-2";
    }
  }

  async function refreshModelMeta() {
    const meta = $("modelMeta");
    const badge = $("modelBadge");
    if (!state.model) {
      badge.textContent = "not set";
      badge.className = "chip shrink-0 border-line bg-window/60 text-[11px] text-inkfaint";
      meta.textContent = "No model selected yet.";
      return;
    }
    const info = await inspect(state.model);
    if (!info || !info.exists) {
      badge.textContent = "missing";
      badge.className = "chip shrink-0 border-danger/40 bg-dangersoft text-[11px] text-dangerink";
      meta.textContent = "That model file no longer exists.";
      return;
    }
    badge.textContent = formatBytes(info.size);
    badge.className = "chip shrink-0 border-line bg-window/60 text-[11px] text-inksoft";
    const suffix = info.extension.toLowerCase() === "gguf" ? "" : " · unexpected extension";
    meta.textContent = `${baseName(state.model)} · stored uncompressed in the output${suffix}`;
    meta.className = "hint mt-2";
  }

  async function refreshMmprojMeta() {
    const meta = $("mmprojMeta");
    $("btnClearMmproj").hidden = !state.mmproj;
    if (!state.mmproj) {
      meta.textContent = "No projector attached. Only needed for multimodal (vision) models.";
      meta.className = "hint mt-1.5";
      return;
    }
    const info = await inspect(state.mmproj);
    if (!info || !info.exists) {
      meta.innerHTML = '<span class="text-warning">Warning:</span> that projector file no longer exists.';
      return;
    }
    meta.textContent =
      `${baseName(state.mmproj)} · ${formatBytes(info.size)} · embedded with --mmproj` +
      (state.mmprojAuto ? " · paired automatically" : "");
    meta.className = "hint mt-1.5";
  }

  function refreshOutputMeta() {
    const meta = $("outputMeta");
    if (!state.output) {
      meta.innerHTML =
        'The file will be created with mode 0755 and end in <span class="font-mono">.llamafile</span>.';
      return;
    }
    const likeModel = baseName(state.model);
    const likeProjector = state.mmproj ? baseName(state.mmproj) : "";
    const parts = ["runtime", likeModel, likeProjector, ".args"].filter(Boolean);
    meta.textContent = `Will write ${baseName(state.output)}${
      likeModel ? ` (${parts.join(" + ")})` : ""
    }. Existing files are replaced.`;
  }

  function updateCompileButton() {
    const ready = Boolean(state.baseRuntime && state.model && state.output) && !busy;
    $("compileBtn").disabled = !ready;
  }

  /* ------------------------------------------------------ .args live preview */

  function currentConfig() {
    return {
      baseRuntime: state.baseRuntime,
      model: state.model,
      mmproj: state.mmproj,
      output: state.output,
      executionMode: state.executionMode,
      host: state.host,
      port: Number(state.port) || 0,
      gpuLayers: Number(state.gpuLayers) || 0,
      ctxSize: Number(state.ctxSize) || 0,
      threads: Number(state.threads) || 1,
      systemPrompt: state.systemPrompt,
      stopTokens: state.stopTokens.slice(),
      extraArgs: state.extraArgs
    };
  }

  function schedulePreview() {
    clearTimeout(previewTimer);
    previewTimer = setTimeout(renderPreview, 130);
  }

  async function renderPreview() {
    try {
      const preview = await invoke("preview_args", { config: currentConfig() });
      $("argsPreview").textContent = preview.text;
      const warning = $("argsWarnings");
      if (preview.warnings.length === 0) {
        warning.hidden = true;
        warning.textContent = "";
      } else {
        warning.hidden = false;
        warning.textContent = preview.warnings.map((item) => `• ${item}`).join("\n");
      }
    } catch (error) {
      $("argsPreview").textContent = `could not build the .args preview: ${error}`;
    }
  }

  /* -------------------------------------------------------------- compilation */

  function resetProgress() {
    $("progressBar").style.width = "0%";
    $("progressBar").className = "progress-fill w-0";
    $("progressText").textContent = "";
    $("progressWrap").setAttribute("aria-valuenow", "0");
    $("stageLabel").textContent = "Preparing…";
  }

  function setBusy(value) {
    busy = value;
    const button = $("compileBtn");
    button.disabled = value || !(state.baseRuntime && state.model && state.output);
    $("compileLabel").textContent = value ? "Compiling…" : "Compile Llamafile";
    $("compileIcon").classList.toggle("animate-spin", value);
    updateCompileButton();
  }

  async function compileNow() {
    if (busy) return;
    if (!state.baseRuntime || !state.model || !state.output) {
      toast("error", "Missing inputs", "Select a base runtime, a GGUF model and an output path first.");
      return;
    }
    resetProgress();
    setBusy(true);
    $("resultCard").hidden = true;
    try {
      const report = await invoke("compile", { config: currentConfig() });
      lastReport = report;
      renderReport(report);
      $("stageLabel").textContent = "Llamafile compiled successfully";
      toast(
        "success",
        "Compiled",
        `${baseName(report.outputPath)} · ${formatBytes(report.outputSize)} · ${report.durationMs} ms`
      );
    } catch (error) {
      $("stageLabel").textContent = "Compilation failed";
      $("progressBar").className = "h-full rounded-pill bg-danger shadow-glow-danger transition-[width] duration-200";
      $("progressText").textContent = "";
      toast("error", "Could not compile", String(error));
    } finally {
      setBusy(false);
    }
  }

  function renderReport(report) {
    const card = $("resultCard");
    card.hidden = false;

    $("resultTitle").textContent = report.verified
      ? "Llamafile compiled and verified"
      : "Llamafile written, but verification failed";
    $("resultMeta").textContent =
      `${report.outputPath} · ${formatBytes(report.outputSize)} · ${report.durationMs} ms` +
      `${report.usesZip64 ? " · ZIP64" : ""}${report.executable ? " · 0755" : ""}`;

    const warningBox = $("resultWarnings");
    warningBox.innerHTML = "";
    (report.warnings || []).forEach((message) => {
      const row = document.createElement("p");
      row.className =
        "rounded-adwaita border border-warning/35 bg-warningsoft px-3 py-2 text-[12px] leading-snug text-warnink";
      row.textContent = message;
      warningBox.appendChild(row);
    });

    const entries = $("resultEntries");
    entries.innerHTML = "";
    report.entries.forEach((entry) => {
      const row = document.createElement("div");
      row.className = "flex items-center justify-between gap-3 rounded bg-window px-2 py-1";
      row.innerHTML = `<span class="truncate text-inksoft"></span><span class="shrink-0 text-inkfaint"></span>`;
      row.querySelector("span").textContent = entry.name;
      row.querySelectorAll("span")[1].textContent =
        `${formatBytes(entry.size)} @ ${entry.dataOffset}` +
        (entry.pageAligned ? " · aligned" : "") +
        (entry.zip64 ? " · zip64" : "");
      entries.appendChild(row);
    });

    $("resultLog").textContent = report.log.join("\n");
    card.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function buildReportText(report) {
    const lines = [
      "Llamafile Studio build report",
      `Generated: ${new Date().toISOString()}`,
      `Output: ${report.outputPath}`,
      `Size: ${report.outputSize} bytes (${formatBytes(report.outputSize)})`,
      `Duration: ${report.durationMs} ms`,
      `Verified: ${report.verified ? "yes" : "NO"}`,
      `ZIP64: ${report.usesZip64 ? "yes" : "no"}`,
      `Executable (0755): ${report.executable ? "yes" : "no"}`,
      "",
      "Entries:",
      ...report.entries.map(
        (entry) =>
          `  ${entry.name}  ${entry.size} bytes  offset ${entry.dataOffset}  ` +
          `page-aligned=${entry.pageAligned}  zip64=${entry.zip64}`
      ),
      "",
      "Warnings:",
      ...(report.warnings.length ? report.warnings.map((w) => `  - ${w}`) : ["  (none)"]),
      "",
      ".args:",
      report.argsText,
      "",
      "Build log:",
      ...report.log.map((line) => `  ${line}`),
      ""
    ];
    return lines.join("\n");
  }

  /* --------------------------------------------------------------- download */

  async function offerRuntime() {
    const button = $("btnDownloadRuntime");
    // Remember the markup so the inline SVG icon survives the busy label.
    const originalHtml = button.innerHTML;
    button.disabled = true;
    button.textContent = "Contacting GitHub…";
    try {
      const offer = await invoke("latest_runtime");

      $("offerText").innerHTML = "";
      const strong = document.createElement("strong");
      strong.textContent = `llamafile ${offer.tag}`;
      $("offerText").appendChild(strong);
      $("offerText").appendChild(document.createTextNode(` from ${offer.repo}`));

      const choices = [offer.recommended].concat(offer.alternatives || []);
      const select = $("offerAlt");
      select.innerHTML = "";
      choices.forEach((asset) => {
        const option = document.createElement("option");
        option.value = asset.url;
        option.textContent = `${asset.name} · ${formatBytes(asset.size)}`;
        option.dataset.name = asset.name;
        option.dataset.variant = asset.variant;
        option.dataset.size = String(asset.size);
        select.appendChild(option);
      });

      updateOfferSub();
      $("downloadOffer").hidden = false;
    } catch (error) {
      toast("error", "Could not reach GitHub", String(error));
    } finally {
      button.disabled = false;
      button.innerHTML = originalHtml;
    }
  }

  function updateOfferSub() {
    const select = $("offerAlt");
    const option = select.selectedOptions[0];
    if (!option) return;
    $("offerSub").textContent = `${option.dataset.variant} · ${formatBytes(
      Number(option.dataset.size)
    )} · saved to the app data directory`;
  }

  async function startDownload() {
    const select = $("offerAlt");
    const option = select.selectedOptions[0];
    if (!option) return;

    const url = option.value;
    const fileName = option.dataset.name;
    $("btnStartDownload").disabled = true;
    $("downloadWrap").hidden = false;
    $("downloadBar").style.width = "0%";
    $("downloadText").textContent = `Downloading ${fileName}…`;

    try {
      const saved = await invoke("download_runtime", { url, fileName });
      state.baseRuntime = saved;
      $("runtimePath").value = saved;
      $("downloadOffer").hidden = true;
      $("btnStartDownload").disabled = false;
      persist();
      await refreshRuntimeMeta();
      schedulePreview();
      toast("success", "Runtime downloaded", `${fileName} is ready to use.`);
    } catch (error) {
      $("btnStartDownload").disabled = false;
      $("downloadText").textContent = "Download failed.";
      toast("error", "Download failed", String(error));
    }
  }

  /* --------------------------------------------------------------- bindings */

  function applyState() {
    $("runtimePath").value = state.baseRuntime;
    $("modelPath").value = state.model;
    $("mmprojPath").value = state.mmproj;
    $("outputPath").value = state.output;

    document.querySelectorAll('input[name="execMode"]').forEach((input) => {
      input.checked = input.value === state.executionMode;
    });

    $("hostInput").value = state.host;
    $("portInput").value = state.port;
    $("gpuSlider").value = state.gpuLayers;
    $("ctxSelect").value = String(state.ctxSize);
    $("threadsInput").value = state.threads || "";
    $("systemPrompt").value = state.systemPrompt;
    $("extraArgs").value = state.extraArgs;
    $("hfQuery").value = state.hfQuery;
    $("hfSort").value = state.hfSort || "downloads";
    $("hfTarget").value = state.hfTarget;
    $("argsPane").hidden = !state.showArgs;
    $("argsToggle").setAttribute("aria-expanded", String(state.showArgs));
  }

  function bindTabs() {
    document.querySelectorAll("[data-tab]").forEach((button) => {
      button.addEventListener("click", () => setTab(button.dataset.tab));
      button.addEventListener("keydown", (event) => {
        if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
        const tabs = Array.from(document.querySelectorAll("[data-tab]"));
        const index = tabs.indexOf(button);
        const next = event.key === "ArrowRight" ? (index + 1) % tabs.length : (index - 1 + tabs.length) % tabs.length;
        tabs[next].focus();
        setTab(tabs[next].dataset.tab);
      });
    });
  }

  function bindFiles() {
    // Disables a button while a native dialog is open, restoring the original
    // markup (including any inline SVG icon) when it closes.
    const setBusyMini = (button, text) => {
      button.disabled = true;
      const original = button.dataset.originalHtml || button.innerHTML;
      button.dataset.originalHtml = original;
      button.textContent = text;
      return () => {
        button.disabled = false;
        button.innerHTML = original;
      };
    };

    $("btnPickRuntime").addEventListener("click", async () => {
      const restore = setBusyMini($("btnPickRuntime"), "Opening…");
      try {
        const picked = await invoke("pick_path", {
          kind: "runtime",
          save: false,
          suggestedName: null,
          startIn: state.baseRuntime ? state.baseRuntime.replace(/\/[^/]+$/, "") : null
        });
        if (picked) {
          state.baseRuntime = picked;
          $("runtimePath").value = picked;
          persist();
          await refreshRuntimeMeta();
          schedulePreview();
        }
      } catch (error) {
        toast("error", "File picker failed", String(error));
      } finally {
        restore();
      }
    });

    $("btnPickModel").addEventListener("click", async () => {
      const restore = setBusyMini($("btnPickModel"), "Opening…");
      try {
        const picked = await invoke("pick_path", {
          kind: "model",
          save: false,
          suggestedName: null,
          startIn: state.model ? state.model.replace(/\/[^/]+$/, "") : null
        });
        if (picked) {
          state.model = picked;
          $("modelPath").value = picked;
          // Pre-fill the output with a sensible sibling next to the model.
          if (!state.output) {
            const directory = picked.replace(/\/[^/]+$/, "");
            const suggested = `${directory}/${stem(picked)}.llamafile`;
            state.output = suggested;
            $("outputPath").value = suggested;
          }
          persist();
          projectorOffer = null;
          await refreshModelMeta();
          refreshOutputMeta();
          updateCompileButton();
          await autoPairMmproj({ quiet: true });
          renderProjectorOffer();
          schedulePreview();
        }
      } catch (error) {
        toast("error", "File picker failed", String(error));
      } finally {
        restore();
      }
    });

    $("btnPickOutput").addEventListener("click", async () => {
      const restore = setBusyMini($("btnPickOutput"), "Opening…");
      try {
        const suggested = state.model ? `${stem(state.model)}.llamafile` : "model.llamafile";
        const picked = await invoke("pick_path", {
          kind: "output",
          save: true,
          suggestedName: baseName(state.output) || suggested,
          startIn: state.output ? state.output.replace(/\/[^/]+$/, "") : null
        });
        if (picked) {
          const withExtension = /\.llamafile$/i.test(picked) ? picked : `${picked}.llamafile`;
          state.output = withExtension;
          $("outputPath").value = withExtension;
          persist();
          refreshOutputMeta();
          updateCompileButton();
          schedulePreview();
        }
      } catch (error) {
        toast("error", "File picker failed", String(error));
      } finally {
        restore();
      }
    });

    $("btnPickMmproj").addEventListener("click", async () => {
      const restore = setBusyMini($("btnPickMmproj"), "Opening…");
      try {
        const picked = await invoke("pick_path", {
          kind: "model",
          save: false,
          suggestedName: null,
          startIn: state.mmproj
            ? state.mmproj.replace(/\/[^/]+$/, "")
            : state.model
            ? state.model.replace(/\/[^/]+$/, "")
            : null
        });
        if (picked) {
          attachMmproj(picked, false);
        }
      } catch (error) {
        toast("error", "File picker failed", String(error));
      } finally {
        restore();
      }
    });

    $("btnClearMmproj").addEventListener("click", () => {
      attachMmproj("", false);
    });

    $("mmprojPath").addEventListener("change", (event) => {
      attachMmproj(event.target.value.trim(), false);
    });

    // Jump to the model browser from the Files tab.
    $("btnBrowseModels").addEventListener("click", () => {
      setTab("models");
      $("hfQuery").focus();
      if (!hfState.loaded) searchModels();
    });

    $("btnDownloadRuntime").addEventListener("click", offerRuntime);
    $("offerAlt").addEventListener("change", updateOfferSub);
    $("btnStartDownload").addEventListener("click", startDownload);
    $("btnCancelOffer").addEventListener("click", () => {
      $("downloadOffer").hidden = true;
    });

    // Manual edits of the path fields stay in sync with the state.
    [["runtimePath", "baseRuntime"], ["modelPath", "model"], ["outputPath", "output"]].forEach(
      ([inputId, key]) => {
        $(inputId).addEventListener("change", async (event) => {
          state[key] = event.target.value.trim();
          persist();
          if (key === "baseRuntime") await refreshRuntimeMeta();
          if (key === "model") {
            projectorOffer = null;
            await refreshModelMeta();
            await autoPairMmproj({ quiet: true });
            renderProjectorOffer();
          }
          refreshOutputMeta();
          updateCompileButton();
          schedulePreview();
        });
      }
    );
  }

  function renderStopTokens() {
    const list = $("stopList");
    list.innerHTML = "";
    $("stopCount").textContent = String(state.stopTokens.length);
    if (state.stopTokens.length === 0) {
      const empty = document.createElement("p");
      empty.className = "hint";
      empty.textContent = "No stop tokens yet — the runtime falls back to the model's EOS token.";
      list.appendChild(empty);
      return;
    }
    state.stopTokens.forEach((token, index) => {
      const chip = document.createElement("span");
      chip.className = "chip";
      const label = document.createElement("span");
      label.textContent = token;
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "ml-1 text-inkfaint transition-colors hover:text-danger";
      remove.setAttribute("aria-label", `Remove stop token ${token}`);
      remove.textContent = "×";
      remove.addEventListener("click", () => {
        state.stopTokens.splice(index, 1);
        persist();
        renderStopTokens();
        schedulePreview();
      });
      chip.appendChild(label);
      chip.appendChild(remove);
      list.appendChild(chip);
    });
  }

  function addStopToken(raw) {
    const value = String(raw || "").trim();
    if (!value) return;
    // Allow comma-separated batches in one go.
    value
      .split(",")
      .map((part) => part.trim())
      .filter(Boolean)
      .forEach((token) => {
        if (state.stopTokens.includes(token)) return;
        if (state.stopTokens.length >= 12) {
          toast("info", "Limit reached", "Twelve stop tokens is the maximum that stays readable.");
          return;
        }
        state.stopTokens.push(token);
      });
    persist();
    renderStopTokens();
    schedulePreview();
  }

  function bindPersona() {
    const select = $("templateSelect");
    TEMPLATES.forEach((template) => {
      const option = document.createElement("option");
      option.value = template.id;
      option.textContent = template.label;
      select.appendChild(option);
    });

    select.addEventListener("change", () => {
      const template = TEMPLATES.find((item) => item.id === select.value);
      if (!template) return;
      const textarea = $("systemPrompt");
      if (textarea.value.trim() && textarea.value.trim() !== template.prompt.trim()) {
        const replace = window.confirm(
          `Replace the current prompt with the "${template.label}" template?`
        );
        if (!replace) {
          select.value = "";
          return;
        }
      }
      textarea.value = template.prompt;
      state.systemPrompt = template.prompt;
      persist();
      updatePromptMeta();
      schedulePreview();
    });

    const textarea = $("systemPrompt");
    textarea.addEventListener("input", () => {
      state.systemPrompt = textarea.value;
      updatePromptMeta();
      persist();
      schedulePreview();
    });

    $("btnClearPrompt").addEventListener("click", () => {
      textarea.value = "";
      state.systemPrompt = "";
      select.value = "";
      persist();
      updatePromptMeta();
      schedulePreview();
      textarea.focus();
    });

    const stopInput = $("stopInput");
    stopInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === ",") {
        event.preventDefault();
        addStopToken(stopInput.value);
        stopInput.value = "";
      } else if (event.key === "Backspace" && stopInput.value === "" && state.stopTokens.length) {
        state.stopTokens.pop();
        persist();
        renderStopTokens();
        schedulePreview();
      }
    });

    $("btnAddStop").addEventListener("click", () => {
      addStopToken(stopInput.value);
      stopInput.value = "";
      stopInput.focus();
    });
  }

  function updatePromptMeta() {
    const value = $("systemPrompt").value;
    const lines = value ? value.split("\n").length : 0;
    $("promptMeta").textContent = `${lines} line${lines === 1 ? "" : "s"} · ${value.length} / ${PROMPT_LIMIT} characters`;
    $("personaNote").hidden = state.executionMode !== "server" || !value.trim();
  }

  function updateModeUI() {
    const server = state.executionMode === "server";
    $("networkBadge").textContent = server ? "applied" : "chat mode";
    $("networkBadge").className = server
      ? "chip border-success/40 bg-successsoft text-[11px] text-success"
      : "chip border-line bg-window/60 text-[11px] text-inkfaint";
    $("hostInput").disabled = !server;
    $("portInput").disabled = !server;
    $("networkHint").innerHTML = server
      ? '<span class="font-mono">127.0.0.1</span> keeps the server private to this machine. Use ' +
        '<span class="font-mono">0.0.0.0</span> to expose it on your LAN.'
      : "Chat mode runs in the terminal, so the bind address and port are not baked in.";
    updatePromptMeta();
  }

  function updateGpuHint() {
    const value = Number(state.gpuLayers);
    $("gpuValue").textContent = String(value);
    $("gpuSlider").style.setProperty("--fill", `${(value / 120) * 100}%`);
    let hint;
    if (value === 0) {
      hint = "CPU only — the slowest option, but it works on any machine.";
    } else if (value < 100) {
      hint = `${value} layers on the GPU, the remaining layers stay on the CPU. Good for models that do not fit in VRAM.`;
    } else {
      hint = "Full offload — every transformer layer goes to the GPU. Fastest, but needs enough VRAM for the whole model.";
    }
    $("gpuHint").textContent = hint;
  }

  function updateContextNote() {
    const value = Number(state.ctxSize);
    let note = CONTEXT_NOTES[value] || "";
    if (system && system.totalMemoryGb > 0 && value >= 8192) {
      note += ` This machine reports ${system.totalMemoryGb} GB of RAM.`;
    }
    $("ctxWarning").textContent = note;
  }

  function updateThreadsHint() {
    if (!system) return;
    const threads = Number(state.threads) || 0;
    $("threadsHint").textContent =
      `${system.physicalCores} physical / ${system.logicalCores} logical cores detected · ` +
      `using ${threads} thread${threads === 1 ? "" : "s"} (default: physical − 2).`;
  }

  function bindSettings() {
    document.querySelectorAll('input[name="execMode"]').forEach((input) => {
      input.addEventListener("change", () => {
        if (!input.checked) return;
        state.executionMode = input.value;
        persist();
        updateModeUI();
        schedulePreview();
      });
    });

    $("hostInput").addEventListener("input", (event) => {
      state.host = event.target.value.trim();
      persist();
      schedulePreview();
    });

    $("portInput").addEventListener("change", (event) => {
      const value = Math.max(1, Math.min(65535, Number(event.target.value) || 8080));
      state.port = value;
      event.target.value = value;
      persist();
      schedulePreview();
    });

    $("gpuSlider").addEventListener("input", (event) => {
      state.gpuLayers = Number(event.target.value);
      updateGpuHint();
      persist();
      schedulePreview();
    });

    $("ctxSelect").addEventListener("change", (event) => {
      state.ctxSize = Number(event.target.value);
      updateContextNote();
      persist();
      schedulePreview();
    });

    $("threadsInput").addEventListener("change", (event) => {
      const value = Math.max(1, Math.min(256, Number(event.target.value) || 1));
      state.threads = value;
      event.target.value = value;
      updateThreadsHint();
      persist();
      schedulePreview();
    });

    $("btnAutoThreads").addEventListener("click", () => {
      if (!system) return;
      state.threads = system.suggestedThreads;
      $("threadsInput").value = state.threads;
      updateThreadsHint();
      persist();
      schedulePreview();
      toast(
        "info",
        "Threads optimised",
        `${system.suggestedThreads} threads (${system.physicalCores} physical cores − 2).`
      );
    });

    $("extraArgs").addEventListener("input", (event) => {
      state.extraArgs = event.target.value;
      persist();
      schedulePreview();
    });
  }

  function bindFooter() {
    $("compileBtn").addEventListener("click", compileNow);

    $("argsToggle").addEventListener("click", () => {
      state.showArgs = !state.showArgs;
      $("argsPane").hidden = !state.showArgs;
      $("argsToggle").setAttribute("aria-expanded", String(state.showArgs));
      $("argsToggle").querySelector("span").textContent = state.showArgs ? "Hide .args" : "Show .args";
      persist();
    });

    $("btnCopyArgs").addEventListener("click", async () => {
      if (!lastReport) return;
      await copyToClipboard(lastReport.argsText);
      toast("info", "Copied", "The .args manifest is on the clipboard.");
    });

    $("btnSaveReport").addEventListener("click", async () => {
      if (!lastReport) return;
      try {
        const saved = await invoke("save_build_report", {
          outputPath: lastReport.outputPath,
          contents: buildReportText(lastReport)
        });
        toast("success", "Report saved", saved);
      } catch (error) {
        toast("error", "Could not save the report", String(error));
      }
    });

    $("btnReveal").addEventListener("click", async () => {
      if (!lastReport) return;
      try {
        await invoke("reveal_in_file_manager", { path: lastReport.outputPath });
      } catch (error) {
        toast("error", "Could not open the folder", String(error));
      }
    });

    // Ctrl/Cmd+Enter compiles from anywhere.
    document.addEventListener("keydown", (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        event.preventDefault();
        if (!$("compileBtn").disabled) compileNow();
      }
    });
  }

  /* ---------------------------------------------------------- model browser */

  const hfState = {
    results: [],
    loaded: false,
    busy: false,
    details: new Map(),
    // Repository -> expanded body, so cache updates can refresh open panels.
    open: new Map(),
    downloading: null
  };

  /** File name -> cached entry, refreshed from the download folder. */
  const hfCache = new Map();

  /** Projector offer for the last downloaded vision model (runtime only). */
  let projectorOffer = null;

  /** Badge metadata for the capability hints returned by the backend. */
  const CAP_META = {
    toolUse: {
      label: "Tool calling",
      tone: "accent",
      hint: "The chat template defines tool/function calls."
    },
    reasoning: {
      label: "Thinking",
      tone: "purple",
      hint: "The model supports a reasoning / thinking mode."
    },
    vision: {
      label: "Vision",
      tone: "success",
      hint: "Accepts images; needs a multimodal projector."
    },
    moe: { label: "MoE", tone: "neutral", hint: "Mixture-of-experts architecture." },
    code: { label: "Code", tone: "neutral", hint: "Code-focused model family." },
    multilingual: {
      label: "Multilingual",
      tone: "neutral",
      hint: "The model card lists more than one language."
    }
  };

  const TONE_CLASSES = {
    accent: "border-accent/40 bg-accentsoft text-infoink",
    purple: "border-violet/45 bg-violetsoft text-violetink",
    success: "border-success/40 bg-successsoft text-success",
    warning: "border-warning/35 bg-warningsoft text-warnink",
    neutral: "border-line bg-raised text-inkmuted"
  };

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  function compactNumber(value) {
    const number = Number(value) || 0;
    if (number >= 1e6) return `${(number / 1e6).toFixed(1)}M`;
    if (number >= 1e3) return `${(number / 1e3).toFixed(1)}k`;
    return String(number);
  }

  function badgeNode(label, tone, title) {
    const node = el("span", `chip text-[11px] ${TONE_CLASSES[tone] || TONE_CLASSES.neutral}`, label);
    node.title = title || "";
    return node;
  }

  /** Rough "will it fit?" hint based on the detected RAM. */
  function fitHint(size) {
    if (!system || !system.totalMemoryGb || !size) return "";
    const ramBytes = system.totalMemoryGb * 1024 ** 3;
    if (size <= ramBytes * 0.55) return "fits comfortably in RAM";
    if (size <= ramBytes * 0.9) return "close to your RAM limit";
    return "larger than your RAM";
  }

  function syncFilterButtons() {
    document.querySelectorAll(".hf-filter").forEach((button) => {
      const on = state.hfFilters.includes(button.dataset.cap);
      button.setAttribute("aria-pressed", String(on));
      button.classList.toggle("border-accent", on);
      button.classList.toggle("bg-accentsoft", on);
      button.classList.toggle("text-ink", on);
    });
  }

  /** Builds one result card; all text is inserted through textContent. */
  function modelCard(model) {
    const card = el("div", "card p-4");
    const header = el("div", "flex items-start gap-3");
    const info = el("div", "min-w-0 flex-1");

    const titleRow = el("div", "flex flex-wrap items-center gap-2");
    titleRow.appendChild(el("p", "truncate font-mono text-[13px] font-semibold text-ink", model.repo));
    if (model.gated) {
      titleRow.appendChild(
        badgeNode("Gated", "warning", "Accept the licence on huggingface.co before downloading.")
      );
    }
    info.appendChild(titleRow);

    const badges = el("div", "mt-1.5 flex flex-wrap gap-1.5");
    Object.keys(CAP_META).forEach((key) => {
      if (!model.badges || !model.badges[key]) return;
      const meta = CAP_META[key];
      badges.appendChild(badgeNode(meta.label, meta.tone, meta.hint));
    });
    if (!badges.childElementCount) {
      badges.appendChild(el("span", "hint", "no special capabilities detected"));
    }
    info.appendChild(badges);

    const stats = [];
    if (model.params) stats.push(`${model.params} parameters`);
    if (model.architecture) stats.push(model.architecture);
    if (model.contextLength) stats.push(`${compactNumber(model.contextLength)} ctx`);
    if (model.repoSize) stats.push(`${formatBytes(model.repoSize)} repo`);
    stats.push(`${compactNumber(model.downloads)} downloads`, `${compactNumber(model.likes)} likes`);
    if (model.license) stats.push(model.license);
    info.appendChild(el("p", "hint mt-1.5", stats.join(" · ")));
    if (model.updated) info.appendChild(el("p", "hint mt-0.5", `updated ${model.updated.slice(0, 10)}`));

    header.appendChild(info);

    const toggle = el("button", "btn-secondary shrink-0 px-3 py-1.5 text-[12px]", "Quantisations");
    header.appendChild(toggle);
    card.appendChild(header);

    const body = el("div", "mt-3");
    body.hidden = true;
    card.appendChild(body);

    toggle.addEventListener("click", async () => {
      if (!body.hidden) {
        body.hidden = true;
        hfState.open.delete(model.repo);
        toggle.textContent = "Quantisations";
        return;
      }
      body.hidden = false;
      hfState.open.set(model.repo, body);
      toggle.textContent = "Hide";
      await loadDetail(model.repo, body);
    });

    return card;
  }

  async function loadDetail(repo, body) {
    body.innerHTML = "";
    body.appendChild(el("p", "hint", "Loading quantisations…"));
    let detail = hfState.details.get(repo);
    if (!detail) {
      try {
        detail = await invoke("hf_model", { repo });
        hfState.details.set(repo, detail);
      } catch (error) {
        body.innerHTML = "";
        body.appendChild(
          el(
            "p",
            "rounded-adwaita border border-danger/40 bg-dangersoft px-3 py-2 text-[12px] text-dangerink",
            String(error)
          )
        );
        return;
      }
    }
    renderDetail(detail, body);
  }

  /** One downloadable file row (model quantisation or vision projector). */
  function quantRow(kind, repo, item) {
    const row = el(
      "div",
      "flex flex-wrap items-center gap-2 rounded-adwaita border border-line bg-window px-2.5 py-1.5"
    );
    row.appendChild(
      el(
        "span",
        `chip font-mono text-[11px] ${TONE_CLASSES[kind === "mmproj" ? "purple" : "accent"]}`,
        item.label
      )
    );
    if (kind === "mmproj") {
      row.appendChild(el("span", "text-[11px] text-inkfaint", "projector"));
    }

    const size = item.size || 0;
    row.appendChild(el("span", "font-mono text-[11.5px] text-inkmuted", formatBytes(size)));

    // The download folder is the cache: same name and same byte size means the
    // file is already there, so offer "Use" instead of downloading it again.
    const cached = hfCache.get(baseName(item.file));
    const downloaded = Boolean(cached && (!size || cached.size === size));

    if (downloaded) {
      row.appendChild(badgeNode("Downloaded", "success", `Already in ${state.hfTarget}`));
    } else {
      const fit = kind === "model" ? fitHint(size) : "";
      if (fit) row.appendChild(el("span", "text-[11px] text-inkfaint", `· ${fit}`));
    }

    row.appendChild(el("span", "min-w-0 flex-1 truncate font-mono text-[11px] text-inkfaint", item.file));

    if (downloaded) {
      const use = el("button", "btn-secondary shrink-0 px-2.5 py-1 text-[11.5px]", "Use");
      use.addEventListener("click", () => useCached(kind, cached.path));
      row.appendChild(use);
    } else {
      const download = el("button", "btn-secondary shrink-0 px-2.5 py-1 text-[11.5px]", "Download");
      download.addEventListener("click", () => downloadQuant(repo, item, kind));
      row.appendChild(download);
    }
    return row;
  }

  /** Applies an already-downloaded file to the current configuration. */
  async function useCached(kind, path) {
    if (kind === "mmproj") {
      attachMmproj(path, false);
      toast("success", "Projector attached", baseName(path));
      return;
    }
    projectorOffer = null;
    state.model = path;
    $("modelPath").value = path;
    if (!state.output) {
      state.output = `${parentDir(path)}/${stem(path)}.llamafile`;
      $("outputPath").value = state.output;
    }
    persist();
    await refreshModelMeta();
    refreshOutputMeta();
    updateCompileButton();
    await autoPairMmproj({ quiet: true });
    renderProjectorOffer();
    schedulePreview();
    toast("success", "Model selected", baseName(path));
  }

  /** Rebuilds the cached-file index from the download folder. */
  async function refreshCache() {
    hfCache.clear();
    if (state.hfTarget) {
      try {
        const entries = await invoke("cached_models", { dir: state.hfTarget });
        entries.forEach((entry) => hfCache.set(entry.file, entry));
      } catch (error) {
        console.warn("could not list the model folder", error);
      }
    }
    renderDownloads();
    renderProjectorOffer();
  }

  /** Applies a projector path, recording whether the user or the app chose it. */
  function attachMmproj(path, automatic) {
    state.mmproj = path;
    state.mmprojAuto = Boolean(automatic);
    state.mmprojFor = automatic ? state.model : "";
    $("mmprojPath").value = path;
    if (path) projectorOffer = null;
    persist();
    renderProjectorOffer();
    refreshMmprojMeta();
    schedulePreview();
  }

  /**
   * Attaches a projector that sits next to the selected model.
   *
   * A manual choice always wins. An earlier automatic pairing is replaced when
   * the model changes, and dropped when the paired file disappears.
   */
  async function autoPairMmproj({ quiet = false } = {}) {
    if (!state.model) return false;
    if (state.mmproj && !state.mmprojAuto) return false;
    if (state.mmprojAuto && state.mmprojFor === state.model && state.mmproj) {
      const info = await inspect(state.mmproj);
      if (info && info.exists) return true;
    }

    let candidates = [];
    try {
      candidates = await invoke("mmproj_candidates", { model: state.model });
    } catch (error) {
      console.warn("could not look for a projector", error);
      return false;
    }

    const best = candidates[0];
    if (!best) {
      // Never leave a dead automatic path behind.
      if (state.mmprojAuto) attachMmproj("", false);
      return false;
    }
    attachMmproj(best.path, true);
    if (!quiet) {
      toast("info", "Projector paired", `${best.file} matches ${baseName(state.model)}.`);
    }
    return true;
  }

  /** Offers the repository's projectors after a vision model was downloaded. */
  function offerProjector(repo) {
    const detail = hfState.details.get(repo);
    const vision = Boolean(detail && detail.badges && detail.badges.vision);
    if (!vision || !detail.mmproj || !detail.mmproj.length || state.mmproj) {
      projectorOffer = null;
      renderProjectorOffer();
      return;
    }
    projectorOffer = { repo, files: detail.mmproj };
    renderProjectorOffer();
    toast(
      "info",
      "This model accepts images",
      "Attach its projector from the Files tab to enable image input."
    );
  }

  /** Renders the pending projector offer inside the Files tab. */
  function renderProjectorOffer() {
    const box = $("mmprojOffer");
    if (!box) return;
    const offer = projectorOffer;
    if (!offer || state.mmproj) {
      box.hidden = true;
      $("mmprojOfferList").innerHTML = "";
      return;
    }
    $("mmprojOfferText").textContent =
      `${offer.repo} needs a multimodal projector for images. Download one and it is attached automatically.`;
    const list = $("mmprojOfferList");
    list.innerHTML = "";
    offer.files.slice(0, 3).forEach((file) => {
      list.appendChild(quantRow("mmproj", offer.repo, file));
    });
    box.hidden = false;
  }

  /** Renders the download folder: sizes, selection state and per-file removal. */
  function renderDownloads() {
    const list = $("cacheList");
    if (!list) return;
    const entries = Array.from(hfCache.values()).sort((a, b) => a.file.localeCompare(b.file));
    const total = entries.reduce((sum, entry) => sum + (entry.size || 0), 0);

    $("btnRefreshCache").disabled = !state.hfTarget;
    if (!state.hfTarget) {
      $("cacheSummary").textContent = "Choose a download folder to see what is already on disk.";
    } else if (!entries.length) {
      $("cacheSummary").textContent = `Nothing downloaded to ${state.hfTarget} yet.`;
    } else {
      $("cacheSummary").textContent =
        `${entries.length} file${entries.length === 1 ? "" : "s"} · ${formatBytes(total)} in ` +
        state.hfTarget;
    }

    list.innerHTML = "";
    if (!entries.length) {
      list.appendChild(el("p", "hint", "Downloaded quantisations and projectors appear here."));
      return;
    }

    entries.forEach((entry) => {
      const row = el(
        "div",
        "flex flex-wrap items-center gap-2 rounded-adwaita border border-line bg-window px-2.5 py-1.5"
      );
      const projector = /^mmproj/i.test(entry.file);
      row.appendChild(
        el(
          "span",
          `chip font-mono text-[11px] ${TONE_CLASSES[projector ? "purple" : "accent"]}`,
          projector ? "Projector" : "Weights"
        )
      );
      row.appendChild(
        el("span", "min-w-0 flex-1 truncate font-mono text-[11.5px] text-inksoft", entry.file)
      );
      if (entry.path === state.model || entry.path === state.mmproj) {
        row.appendChild(badgeNode("In use", "success", "Selected in the Files tab."));
      }
      row.appendChild(
        el("span", "font-mono text-[11.5px] text-inkmuted", formatBytes(entry.size || 0))
      );
      const remove = el("button", "btn-secondary shrink-0 px-2.5 py-1 text-[11.5px]", "Delete");
      remove.setAttribute("aria-label", `Delete ${entry.file}`);
      remove.addEventListener("click", () => deleteCached(entry));
      row.appendChild(remove);
      list.appendChild(row);
    });
  }

  /** Deletes one cached file, clearing the selection if it was in use. */
  async function deleteCached(entry) {
    if (!state.hfTarget) return;
    const inUse = entry.path === state.model || entry.path === state.mmproj;
    const question = inUse
      ? `${entry.file} is currently selected. Delete it and clear that selection?`
      : `Delete ${entry.file} (${formatBytes(entry.size || 0)}) from disk?`;
    if (!window.confirm(question)) return;

    try {
      await invoke("delete_cached_model", { path: entry.path, dir: state.hfTarget });
      if (entry.path === state.model) {
        state.model = "";
        $("modelPath").value = "";
        if (state.mmprojAuto) {
          state.mmproj = "";
          state.mmprojFor = "";
          $("mmprojPath").value = "";
        }
        await refreshModelMeta();
        await refreshMmprojMeta();
        refreshOutputMeta();
      }
      if (entry.path === state.mmproj) {
        state.mmproj = "";
        state.mmprojFor = "";
        $("mmprojPath").value = "";
        await refreshMmprojMeta();
        await autoPairMmproj({ quiet: true });
      }
      persist();
      updateCompileButton();
      schedulePreview();
      await refreshCache();
      rerenderOpenDetails();
      toast("success", "Deleted", `${entry.file} was removed from disk.`);
    } catch (error) {
      toast("error", "Could not delete", String(error));
    }
  }

  /** Re-renders every expanded repository after the cache changes. */
  function rerenderOpenDetails() {
    hfState.open.forEach((body, repo) => {
      const detail = hfState.details.get(repo);
      if (detail) renderDetail(detail, body);
    });
  }

  function renderDetail(detail, body) {
    body.innerHTML = "";

    const facts = [];
    if (detail.baseModel) facts.push(`base model ${detail.baseModel}`);
    if (detail.languages && detail.languages.length) {
      facts.push(detail.languages.slice(0, 4).join(", "));
    }
    if (facts.length) body.appendChild(el("p", "hint mb-1.5", facts.join(" · ")));
    if (detail.signals && detail.signals.length) {
      body.appendChild(el("p", "hint mb-2", `detected from: ${detail.signals.join("; ")}`));
    }

    body.appendChild(
      el(
        "p",
        "mb-1.5 text-[11.5px] font-semibold uppercase tracking-wider text-inkfaint",
        "Quantisations"
      )
    );
    if (!detail.quants.length) {
      body.appendChild(
        el("p", "hint", "No single-file GGUF quantisations found in this repository.")
      );
    } else {
      const list = el("div", "space-y-1");
      detail.quants.forEach((quant) => list.appendChild(quantRow("model", detail.repo, quant)));
      body.appendChild(list);
    }

    // Vision projectors are downloaded and attached the same way as weights, so
    // they share the row renderer (and its cache badge).
    if (detail.mmproj && detail.mmproj.length) {
      body.appendChild(
        el(
          "p",
          "mb-1.5 mt-3 text-[11.5px] font-semibold uppercase tracking-wider text-inkfaint",
          "Vision projector"
        )
      );
      const projectors = el("div", "space-y-1");
      detail.mmproj.forEach((file) => projectors.appendChild(quantRow("mmproj", detail.repo, file)));
      body.appendChild(projectors);
      body.appendChild(
        el("p", "hint mt-1.5", "Attach a projector to enable the model's image input.")
      );
    }

    if (detail.splitCount) {
      body.appendChild(
        el(
          "p",
          "hint mt-2",
          `${detail.splitCount} sharded GGUF part(s) were skipped; pick a single-file quantisation instead.`
        )
      );
    }
  }

  function renderResults() {
    const container = $("hfResults");
    container.innerHTML = "";
    // The cards (and their expanded bodies) are rebuilt from scratch, so the
    // open-panel registry must not keep detached nodes around.
    hfState.open.clear();

    if (!hfState.loaded) {
      $("hfEmpty").hidden = false;
      $("hfCount").textContent = "idle";
      return;
    }

    const filters = state.hfFilters;
    const visible = hfState.results.filter((model) =>
      filters.every((cap) => model.badges && model.badges[cap])
    );

    $("hfEmpty").hidden = hfState.results.length > 0;

    if (!visible.length && hfState.results.length) {
      const note = el("div", "card p-6 text-center");
      note.appendChild(el("p", "text-[13px] font-semibold text-inksoft", "No results match these filters"));
      note.appendChild(el("p", "hint mt-1", "Clear a filter or search for something else."));
      container.appendChild(note);
    }

    visible.forEach((model) => container.appendChild(modelCard(model)));
    $("hfCount").textContent = `${visible.length} of ${hfState.results.length} repos`;
  }

  async function searchModels() {
    if (hfState.busy) return;
    hfState.busy = true;
    const button = $("btnHfSearch");
    button.disabled = true;
    button.textContent = "Searching…";
    $("hfCount").textContent = "searching…";

    const query = $("hfQuery").value.trim();
    state.hfQuery = query;
    persist();

    try {
      const results = await invoke("hf_search", {
        query,
        sort: $("hfSort").value,
        limit: 24
      });
      hfState.results = results;
      hfState.details.clear();
      hfState.loaded = true;
      renderResults();
      if (!results.length) {
        toast("info", "No models found", "Try a broader search term.");
      }
    } catch (error) {
      $("hfCount").textContent = "error";
      toast("error", "Search failed", String(error));
    } finally {
      hfState.busy = false;
      button.disabled = false;
      button.textContent = "Search";
    }
  }

  async function downloadQuant(repo, quant, kind = "model") {
    if (hfState.downloading) {
      toast("info", "Download already running", "Wait for the current download to finish.");
      return;
    }
    if (!state.hfTarget) {
      try {
        state.hfTarget = await invoke("default_model_dir");
        $("hfTarget").value = state.hfTarget;
        persist();
      } catch (error) {
        toast("error", "No download folder", String(error));
        return;
      }
    }

    hfState.downloading = { repo, file: quant.file };
    $("hfProgressCard").hidden = false;
    $("hfProgressFile").textContent = `${repo} / ${quant.file}`;
    $("hfProgressText").textContent = "starting…";
    $("hfBar").style.width = "0%";

    try {
      const path = await invoke("download_model", {
        repo,
        file: quant.file,
        targetDir: state.hfTarget
      });

      if (kind === "mmproj") {
        attachMmproj(path, false);
        toast("success", "Projector ready", `${baseName(path)} attached to the model.`);
      } else {
        projectorOffer = null;
        state.model = path;
        $("modelPath").value = path;
        if (!state.output) {
          state.output = `${parentDir(path)}/${stem(path)}.llamafile`;
          $("outputPath").value = state.output;
        }
        persist();
        await refreshModelMeta();
        refreshOutputMeta();
        updateCompileButton();
        toast("success", "Model ready", `${baseName(path)} is now the source model.`);
      }

      schedulePreview();
      await refreshCache();
      if (kind === "model") {
        await autoPairMmproj({ quiet: true });
        offerProjector(repo);
      }
      rerenderOpenDetails();
    } catch (error) {
      const message = String(error);
      const cancelled = message.toLowerCase().includes("cancelled");
      toast(
        cancelled ? "info" : "error",
        cancelled ? "Download cancelled" : "Download failed",
        message
      );
    } finally {
      hfState.downloading = null;
      $("hfProgressCard").hidden = true;
    }
  }

  function bindModels() {
    $("btnHfSearch").addEventListener("click", searchModels);
    $("hfQuery").addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        searchModels();
      }
    });
    $("hfSort").addEventListener("change", () => {
      state.hfSort = $("hfSort").value;
      persist();
      if (hfState.loaded) searchModels();
    });

    document.querySelectorAll(".hf-filter").forEach((button) => {
      button.addEventListener("click", () => {
        const cap = button.dataset.cap;
        const index = state.hfFilters.indexOf(cap);
        if (index >= 0) state.hfFilters.splice(index, 1);
        else state.hfFilters.push(cap);
        persist();
        syncFilterButtons();
        renderResults();
      });
    });
    syncFilterButtons();

    $("btnHfTarget").addEventListener("click", async () => {
      try {
        const picked = await invoke("pick_folder", { startIn: state.hfTarget || null });
        if (picked) {
          state.hfTarget = picked;
          $("hfTarget").value = picked;
          persist();
          await refreshCache();
          rerenderOpenDetails();
        }
      } catch (error) {
        toast("error", "Folder picker failed", String(error));
      }
    });

    $("hfTarget").addEventListener("change", async (event) => {
      state.hfTarget = event.target.value.trim();
      persist();
      await refreshCache();
      rerenderOpenDetails();
    });

    $("btnHfCancel").addEventListener("click", async () => {
      try {
        await invoke("cancel_model_download");
      } catch (error) {
        console.warn("cancel failed", error);
      }
    });

    $("btnRefreshCache").addEventListener("click", refreshCache);

    $("btnDismissProjectorOffer").addEventListener("click", () => {
      projectorOffer = null;
      renderProjectorOffer();
    });
  }

  /* ------------------------------------------------------------- live events */

  async function attachListeners() {
    await listen("compile:progress", (event) => {
      const payload = event.payload;
      const percent = Math.max(0, Math.min(100, payload.percent || 0));
      $("progressBar").style.width = `${percent}%`;
      $("progressWrap").setAttribute("aria-valuenow", String(Math.round(percent)));
      if (payload.message) $("stageLabel").textContent = payload.message;
      const done = formatBytes(payload.bytesDone);
      const total = payload.bytesTotal ? formatBytes(payload.bytesTotal) : "?";
      $("progressText").textContent = `${Math.round(percent)}% · ${done} / ${total}`;
    });

    await listen("runtime:progress", (event) => {
      const payload = event.payload;
      const percent = Math.max(0, Math.min(100, payload.percent || 0));
      $("downloadBar").style.width = `${percent}%`;
      $("downloadText").textContent = `${Math.round(percent)}% · ${formatBytes(
        payload.downloaded
      )}${payload.total ? ` / ${formatBytes(payload.total)}` : ""}`;
    });

    await listen("model:progress", (event) => {
      const payload = event.payload;
      const percent = Math.max(0, Math.min(100, payload.percent || 0));
      $("hfBar").style.width = `${percent}%`;
      $("hfProgressText").textContent =
        `${Math.round(percent)}% · ${formatBytes(payload.downloaded)}` +
        (payload.total ? ` / ${formatBytes(payload.total)}` : "");
    });
  }

  /* -------------------------------------------------------------------- boot */

  async function boot() {
    loadState();
    bindTabs();
    bindFiles();
    bindPersona();
    bindSettings();
    bindFooter();
    bindModels();
    applyState();
    setTab(state.activeTab || "files");
    renderStopTokens();
    updateGpuHint();
    updateModeUI();
    updateContextNote();
    refreshOutputMeta();
    updateCompileButton();
    $("argsToggle").querySelector("span").textContent = state.showArgs ? "Hide .args" : "Show .args";

    await attachListeners();

    // Resolve the download destination once, so the browser is usable
    // immediately without an extra click.
    if (!state.hfTarget) {
      try {
        state.hfTarget = await invoke("default_model_dir");
        $("hfTarget").value = state.hfTarget;
        persist();
      } catch (error) {
        console.warn("could not resolve the default model folder", error);
      }
    }
    await refreshCache();

    try {
      system = await invoke("system_info");
      $("systemChipText").textContent =
        `${system.physicalCores} cores · ${system.totalMemoryGb || "?"} GB RAM · ${system.architecture}`;
      $("systemChip").title =
        `${system.logicalCores} logical / ${system.physicalCores} physical cores · ` +
        `${system.totalMemoryGb} GB RAM · ${system.os} · page size ${system.pageSize} B`;
      if (!state.threads) {
        state.threads = system.suggestedThreads;
        $("threadsInput").value = state.threads;
        persist();
      }
      updateThreadsHint();
      updateContextNote();
    } catch (error) {
      $("systemChipText").textContent = "hardware detection unavailable";
      console.warn("system_info failed", error);
    }

    await refreshRuntimeMeta();
    await refreshModelMeta();
    await autoPairMmproj({ quiet: true });
    await refreshMmprojMeta();
    await renderPreview();
  }

  boot().catch((error) => {
    console.error("startup failed", error);
    toast("error", "Startup problem", String(error));
  });
})();
