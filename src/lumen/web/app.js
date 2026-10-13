// Lumen web UI: upload -> start -> poll -> render, plus the measurement report. DOM is built with textContent only.
"use strict";

const IMAGE = /\.(jpe?g|png|tiff?)$/i;
const HEIC = /\.(heic|heif)$/i;
const POLL_MS = 1500;
const POLL_LIMIT_MS = 20 * 60 * 1000; // > Lambda's 15 min; the API also marks dead workers failed
const RUNS_KEY = "lumen-runs";
const RUNS_KEPT = 25;
const SVG = "http://www.w3.org/2000/svg";
const $ = (id) => document.getElementById(id);

const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};
const svg = (tag, attrs) => {
  const node = document.createElementNS(SVG, tag);
  Object.entries(attrs || {}).forEach(([k, v]) => node.setAttribute(k, v));
  return node;
};
// Cambridge dataset category codes, in words.
const MOTION = {
  complex: "Complex motion", handheld: "Handheld camera", lolm: "Large object, large motion",
  losm: "Large object, small motion", multiview: "Camera moved (parallax)", nrm: "Cloth, leaves, hair",
  occlusion: "Object hides background", solm: "Small object, large motion", sosm: "Small object, small motion",
};
const motion = (code) => MOTION[code] || code;
const pct = (x, digits = 1) => `${(x * 100).toFixed(digits)} %`;
const fixed = (x, digits) => (typeof x === "number" ? x.toFixed(digits) : "–");

// ---------- views ----------

function showView() {
  const hash = location.hash;
  const view = hash.startsWith("#report") || hash === "#measurements" ? "report" : "process";
  $("view-process").hidden = view !== "process";
  $("view-report").hidden = view !== "report";
  document.querySelectorAll(".sheet-tab[data-view]").forEach((t) => {
    if (t.dataset.view === view) t.setAttribute("aria-current", "page");
    else t.removeAttribute("aria-current");
  });
  if (view === "report") {
    renderRuns();
    loadReport().then(() => {
      const target = hash.startsWith("#report-") && document.getElementById(hash.slice(1));
      if (target) target.scrollIntoView();
      else if (hash === "#report" || hash === "#measurements") scrollTo(0, 0);
    });
  }
}

// ---------- compare: a drawn section line over two images ----------

function makeCompare({ before, after, tagBefore = "Before", tagAfter = "After", altBefore = "", altAfter = "", size, lazy = true }) {
  const node = $("compare-template").content.firstElementChild.cloneNode(true);
  const b = node.querySelector(".compare-before");
  const a = node.querySelector(".compare-after");
  [b, a].forEach((img) => {
    if (lazy) img.loading = "lazy";
    if (size) { img.width = size[0]; img.height = size[1]; }
  });
  if (size) node.style.setProperty("aspect-ratio", `${size[0]} / ${size[1]}`);
  b.src = before;
  b.alt = altBefore;
  node.querySelector(".compare-tag-before").textContent = tagBefore;
  node.querySelector(".compare-tag-after").textContent = tagAfter;
  if (after) {
    a.src = after;
    a.alt = altAfter;
  } else {
    node.classList.add("no-after");
    a.remove();
  }
  node.querySelector(".compare-range").addEventListener("input", (e) => {
    node.style.setProperty("--pos", `${e.target.value}%`);
  });
  return node;
}

// ---------- upload and processing ----------

function setJob(status, detail, busy) {
  $("job").hidden = false;
  $("job").classList.toggle("is-busy", busy);
  $("job-status").textContent = status;
  $("job-detail").textContent = detail || "";
}

function showError(message) {
  $("job").hidden = false;
  $("job").classList.remove("is-busy");
  $("job-error").hidden = false;
  $("job-error").textContent = message;
}

async function api(path, options) {
  const res = await fetch(path, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error || body.detail || `HTTP ${res.status}`);
  return body;
}

async function uploadOne(file, target) {
  let res;
  if (target.method === "PUT") {
    res = await fetch(target.url, { method: "PUT", body: file });
  } else {
    const form = new FormData();
    Object.entries(target.fields).forEach(([k, v]) => form.append(k, v));
    form.append("file", file);
    res = await fetch(target.url, { method: "POST", body: form });
  }
  if (!res.ok) throw new Error(`upload of ${file.name} failed (${res.status})`);
}

// The server accepts [A-Za-z0-9._-] names; phones and Finder copies use spaces and accents.
function safeNames(files) {
  const used = new Set();
  return files.map((f) => {
    const dot = f.name.lastIndexOf(".");
    const ext = f.name.slice(dot).toLowerCase();
    let stem = f.name.slice(0, dot).normalize("NFKD").replace(/[̀-ͯ]/g, "") // strip accents
      .replace(/[^A-Za-z0-9._-]+/g, "_").replace(/^[^A-Za-z0-9]+/, "");
    stem = (stem || "photo").slice(0, 100);
    let name = stem + ext;
    for (let n = 2; used.has(name); n += 1) name = `${stem}_${n}${ext}`;
    used.add(name);
    return name;
  });
}

// Optional steps the person switched off (agent.OPTIONAL_STEPS); merge has no switch to turn off.
const stepsOff = () => [...document.querySelectorAll(".options input[data-step]")].filter((i) => !i.checked).map((i) => i.dataset.step);

async function run(fileList) {
  const all = [...fileList];
  const files = all.filter((f) => IMAGE.test(f.name));
  const heic = all.filter((f) => HEIC.test(f.name)).length;
  $("results").replaceChildren();
  $("job-error").hidden = true;
  $("grouping").hidden = true;
  if (!files.length) {
    return showError(heic ? "HEIC photos are not supported yet: set the camera to JPEG (Most Compatible) and shoot again."
                          : "No JPG, PNG or TIFF files in that selection.");
  }
  try {
    const names = safeNames(files);
    setJob("uploading", `${files.length} photos${heic ? ` (${heic} HEIC skipped)` : ""}`, true);
    const { job, uploads } = await api("/api/jobs", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ files: names, one_bracket: $("one-bracket").checked, off: stepsOff() }),
    });
    let sent = 0;
    for (const [i, file] of files.entries()) {
      await uploadOne(file, uploads[names[i]]);
      sent += 1;
      setJob("uploading", `${sent} of ${files.length} photos`, true);
    }
    await api(`/api/jobs/${job.id}/start`, { method: "POST" });
    await poll(job.id);
  } catch (err) {
    showError(err.message);
  }
}

async function poll(id) {
  const deadline = Date.now() + POLL_LIMIT_MS;
  while (Date.now() < deadline) {
    const job = await api(`/api/jobs/${id}`);
    if (job.status === "done") { saveRun(job); return render(job); }
    if (job.status === "failed") return showError(job.error || "Processing failed.");
    setJob(job.status, "the agent is looking at your photos", true);
    await new Promise((r) => setTimeout(r, POLL_MS));
  }
  return showError("No answer after 20 minutes. Try fewer photos per upload.");
}

// ---------- one result per room ----------

function decisionClass(decision) {
  if (decision.startsWith("drop") || decision.startsWith("change")) return "drop";
  if (decision === "accepted") return "accept";
  if (decision === "rejected") return "review";
  return decision.split(" ")[0];
}

function numbers(result) {
  return Object.entries(result || {})
    .filter(([k, v]) => k !== "frames" && k !== "note" && v !== null && typeof v !== "object")
    .map(([k, v]) => `${k.replaceAll("_", " ")} ${typeof v === "number" ? +v.toFixed(3) : v}`)
    .join(", ");
}

function renderStep(step) {
  const kind = decisionClass(step.decision);
  const li = el("li", `is-${kind}`);
  const head = el("div", "trace-head");
  head.append(el("span", "trace-rule", step.tool === "none" ? "finished" : step.tool.replaceAll("_", " ")), el("span", `chip chip-${kind}`, step.decision));
  li.append(head);
  const nums = numbers(step.result);
  if (nums) li.append(el("p", "trace-nums", nums));
  if (step.result && step.result.note) li.append(el("p", "trace-nums", step.result.note));
  return li;
}

// The numbers a person checks first, pulled from the agent's own trace (last value of each step wins).
function setMetrics(set) {
  const last = (tool) => [...set.steps].reverse().find((s) => s.tool === tool && s.result);
  const rows = [];
  const merge = last("merge_set");
  if (merge && typeof merge.result.ghost_ratio === "number") {
    rows.push(["Moving area removed", pct(merge.result.ghost_ratio)]);
    rows.push(["Clipped after merge", pct(merge.result.clip_ratio)]);
  }
  const win = last("pull_windows");
  if (win && win.decision === "accept") rows.push(["Window clipping", `${pct(win.result.clip_before)} → ${pct(win.result.clip_after)}`]);
  if (win && typeof win.result.foreign_edges === "number") rows.push(["Window edges no frame recorded", pct(win.result.foreign_edges)]);
  const lens = last("correct_lens");
  if (lens) rows.push(["Lens curvature", lens.decision === "accept"
    ? `${fixed(lens.result.curvature_before_px, 2)} → ${fixed(lens.result.curvature_after_px, 2)} px` : "not changed"]);
  const tilt = last("straighten");
  if (tilt && typeof tilt.result.tilt_before_deg === "number") {
    rows.push(["Vertical tilt", `${fixed(tilt.result.tilt_before_deg, 2)}° → ${fixed(tilt.decision === "accept" ? tilt.result.tilt_after_deg : tilt.result.tilt_before_deg, 2)}°`]);
  }
  rows.push(["Frames used", String(set.frames.length)]);
  return rows;
}

// Local output URLs are reused after a re-run, so add a version; presigned S3 URLs must stay untouched.
const bust = (url) => (url && url.startsWith("/") ? `${url}?v=${Date.now()}` : url);

async function sendReview(jobId, set, action, frame, node) {
  const bar = node.querySelector(".review");
  bar.querySelectorAll("button, select").forEach((b) => { b.disabled = true; });
  bar.querySelector(".review-msg").textContent = action === "drop_frame" ? "Re-running the agent…" : "Saving…";
  try {
    const job = await api(`/api/jobs/${jobId}/sets/${set.id}/review`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action, frame }),
    });
    const fresh = job.sets.find((s) => s.id === set.id);
    node.replaceWith(renderSet(fresh, jobId));
    summarise(job);
  } catch (err) {
    bar.querySelector(".review-msg").textContent = err.message;
    bar.querySelectorAll("button, select").forEach((b) => { b.disabled = false; });
  }
}

function reviewBar(jobId, set, node) {
  const bar = node.querySelector(".review");
  if (set.review) {
    bar.hidden = false;
    bar.classList.add("is-done");
    bar.append(el("p", "review-msg",
      `Reviewed by a person: ${set.review.action.replace("_", " ")}${set.review.frame ? ` (${set.review.frame})` : ""}`));
    return;
  }
  if (set.status !== "needs_review") return;
  bar.hidden = false;
  bar.append(el("p", "review-title", "The agent stopped and wants your call"));
  const row = el("div", "review-row");
  const accept = el("button", "btn", "Accept the agent's image");
  accept.type = "button";
  accept.disabled = !set.urls.after;
  accept.title = set.urls.after ? "" : "No image was produced for this room";
  accept.addEventListener("click", () => sendReview(jobId, set, "accept", null, node));
  const reject = el("button", "btn btn-quiet", "Reject");
  reject.type = "button";
  reject.addEventListener("click", () => sendReview(jobId, set, "reject", null, node));
  row.append(accept, reject);
  if (set.frames.length > 2) {
    const pick = el("select", "review-pick");
    pick.setAttribute("aria-label", "Frame to drop");
    set.frames.forEach((f) => pick.append(Object.assign(el("option", null, f), { value: f })));
    const drop = el("button", "btn btn-quiet", "Drop this frame and re-run");
    drop.type = "button";
    drop.addEventListener("click", () => sendReview(jobId, set, "drop_frame", pick.value, node));
    row.append(pick, drop);
  }
  bar.append(row, el("p", "review-msg", ""));
}

function renderSet(set, jobId) {
  const node = $("set-template").content.firstElementChild.cloneNode(true);
  node.querySelector(".set-title").textContent = `Room ${set.id.replace("set", "")}`;
  const status = node.querySelector(".set-status");
  status.textContent = set.status === "needs_review" ? "needs your review" : set.status;
  status.classList.add(`chip-${set.status}`);
  const reason = node.querySelector(".set-reason");
  reason.textContent = set.reason || "";
  reason.classList.toggle("is-calm", set.status !== "needs_review");
  node.querySelector(".set-frames").textContent = `Frames: ${set.frames.join(", ")}`;
  node.classList.toggle("is-review", set.status === "needs_review");
  node.querySelector(".compare-slot").replaceWith(makeCompare({
    before: bust(set.urls.before), after: set.urls.preview ? bust(set.urls.preview) : null,
    tagBefore: "Reference frame", tagAfter: "Lumen", lazy: false,
    altBefore: `Room ${set.id} reference frame before processing`, altAfter: `Room ${set.id} after processing`,
  }));
  if (set.urls.preview) node.querySelector(".download").href = set.urls.after;
  else node.querySelector(".set-links").remove();
  const dl = node.querySelector(".metrics");
  setMetrics(set).forEach(([k, v]) => dl.append(el("dt", null, k), el("dd", null, v)));
  node.querySelector(".trace").append(...set.steps.map(renderStep));
  reviewBar(jobId, set, node);
  return node;
}

function summarise(job) {
  const count = (status) => job.sets.filter((s) => s.status === status).length;
  const rooms = job.sets.length === 1 ? "1 room" : `${job.sets.length} rooms`;
  setJob("done", `${rooms}: ${count("ok")} ready, ${count("needs_review")} need your review` +
    `${count("rejected") ? `, ${count("rejected")} rejected` : ""}. Took ${job.seconds} s.`, false);
}

function render(job) {
  summarise(job);
  $("results").replaceChildren(...job.sets.map((s) => renderSet(s, job.id)));
  const list = $("grouping-list");
  list.replaceChildren(...job.grouping.map((g) =>
    el("li", null, `${job.files[g.left]} → ${job.files[g.right]}: ${g.joined ? "same room" : "new room"} (${g.reason})`)));
  $("grouping").hidden = job.grouping.length === 0;
}

// ---------- runs kept in this browser ----------

function readRuns() {
  try { return JSON.parse(localStorage.getItem(RUNS_KEY) || "[]"); } catch { return []; }
}

function saveRun(job) {
  const tilt = job.sets.flatMap((s) => s.steps.filter((st) => st.tool === "straighten" && st.decision === "accept")
    .map((st) => st.result.tilt_after_deg));
  const run = {
    at: new Date().toISOString(), photos: job.files.length, rooms: job.sets.length,
    ok: job.sets.filter((s) => s.status === "ok").length,
    review: job.sets.filter((s) => s.status === "needs_review").length,
    seconds: job.seconds, worstTilt: tilt.length ? Math.max(...tilt) : null,
  };
  try { localStorage.setItem(RUNS_KEY, JSON.stringify([run, ...readRuns()].slice(0, RUNS_KEPT))); } catch { /* private mode */ }
}

function table(caption, headers, rows, numeric) {
  const wrap = el("div", "table-wrap");
  const t = el("table");
  if (caption) t.append(el("caption", null, caption));
  const thead = el("thead");
  const head = el("tr");
  headers.forEach((h, i) => {
    const th = el("th", numeric[i] ? "num" : null, h);
    th.scope = "col";
    head.append(th);
  });
  thead.append(head);
  const body = el("tbody");
  rows.forEach((r) => {
    const tr = el("tr");
    r.forEach((cell, i) => {
      const td = el("td", numeric[i] ? "num" : null);
      if (cell instanceof Node) td.append(cell); else td.textContent = cell;
      tr.append(td);
    });
    body.append(tr);
  });
  t.append(thead, body);
  wrap.append(t);
  return wrap;
}

function renderRuns() {
  const runs = readRuns();
  const box = $("runs");
  if (!runs.length) {
    box.replaceChildren(el("p", "muted", "No runs yet. Process photos on the first tab and they show up here."));
    return;
  }
  const rows = runs.map((r) => [new Date(r.at).toLocaleString(), String(r.photos), String(r.rooms), String(r.ok),
    String(r.review), r.worstTilt === null ? "–" : `${r.worstTilt.toFixed(2)}°`, `${r.seconds} s`]);
  const clear = el("button", "btn btn-quiet clear-runs", "Clear this list");
  clear.type = "button";
  clear.addEventListener("click", () => { try { localStorage.removeItem(RUNS_KEY); } catch { /* ignore */ } renderRuns(); });
  box.replaceChildren(table(null, ["When", "Photos", "Rooms", "Ready", "Need review", "Worst tilt after", "Time"],
    rows, [false, true, true, true, true, true, true]), clear);
}

// ---------- the measurement report ----------

let reportData = null;
let reportRendered = false;

function fetchReport() {
  if (!reportData) {
    reportData = fetch("/static/report.json").then((res) => {
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json();
    });
    reportData.catch(() => { reportData = null; });
  }
  return reportData;
}

async function loadReport() {
  if (reportRendered) return;
  try {
    renderReport(await fetchReport());
    reportRendered = true;
  } catch (err) {
    $("report").replaceChildren(el("p", "job-error", `The report could not be loaded (${err.message}). Reload the page to try again.`));
  }
}

function block(id, title, ...children) {
  const s = el("section", "report-block");
  s.id = id;
  s.append(el("h2", null, title), ...children);
  return s;
}

// ---------- the cover: the vertical edges OpenCV measured, drawn on the result ----------

async function drawCover() {
  let r;
  try { r = await fetchReport(); } catch { return; }   // the cover photo stands on its own without the lines
  const room = r.gallery.empa.find((e) => e.scene === r.hero.scene);
  const plumb = $("plumb");
  const W = 1000, H = 1000 * (room.size[1] / room.size[0]);
  plumb.setAttribute("viewBox", `0 0 ${W} ${H}`);
  plumb.setAttribute("preserveAspectRatio", "none");
  const lines = r.hero.plumb;
  lines.forEach(([x1, y1, x2, y2], i) => {
    const pts = { x1: x1 * W, y1: y1 * H, x2: x2 * W, y2: y2 * H };
    const delay = `${0.25 + i * 0.05}s`;
    [["seg-halo"], ["seg"]].forEach(([cls]) => {
      const line = svg("line", { class: cls, ...pts, pathLength: 1, "vector-effect": "non-scaling-stroke" });
      line.style.setProperty("animation-delay", delay);
      plumb.append(line);
    });
    [[pts.x1, pts.y1], [pts.x2, pts.y2]].forEach(([cx, cy]) => {
      const tick = svg("circle", { class: "tick", cx, cy, r: 4, "vector-effect": "non-scaling-stroke" });
      tick.style.setProperty("animation-delay", `${1.1 + i * 0.05}s`);
      plumb.append(tick);
    });
  });
  plumb.classList.add("is-drawing");
  $("cover-lines").textContent = String(lines.length);
  $("cover-date").textContent = r.generated || "–";
  // leader: from the callout's right edge to the middle of the longest measured edge (lines are sorted by length)
  const [x1, y1, x2, y2] = lines[0];
  const callout = $("callout");
  callout.hidden = false;
  const fig = plumb.getBoundingClientRect(), box = callout.getBoundingClientRect();
  if (fig.width) {
    const ax = ((box.right - fig.left) / fig.width) * W, ay = ((box.top + box.height / 2 - fig.top) / fig.height) * H;
    const mx = ((x1 + x2) / 2) * W, my = ((y1 + y2) / 2) * H;
    plumb.append(svg("polyline", { class: "leader", points: `${ax},${ay} ${mx - 14},${ay} ${mx},${my}`, "vector-effect": "non-scaling-stroke" }),
      svg("circle", { class: "leader-dot", cx: mx, cy: my, r: 5, "vector-effect": "non-scaling-stroke" }));
  }
  $("cover-tilt").textContent = `${room.tilt_before_deg.toFixed(2)}° → ${room.tilt_after_deg.toFixed(2)}°`;
}

function finding(parts) {
  const p = el("p", "finding");
  parts.forEach((x) => p.append(typeof x === "string" ? document.createTextNode(x) : x));
  return p;
}
const strong = (text) => el("b", null, text);

// Dumbbell: one row per motion type, ochre = ghost removal off, blue = on, against the motion-free merge.
function ghostChart(cats) {
  const W = 680, rowH = 34, left = 200, right = 24, top = 26;
  const H = top + cats.length * rowH + 30;
  const lo = Math.floor(Math.min(...cats.map((c) => c.psnr_off)) / 5) * 5;
  const hi = Math.ceil(Math.max(...cats.map((c) => c.psnr_on)) / 5) * 5;
  const x = (v) => left + ((v - lo) / (hi - lo)) * (W - left - right);
  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": "PSNR against the motion-free merge, per motion type, with ghost removal off and on" });
  for (let v = lo; v <= hi; v += 5) {
    root.append(svg("line", { class: "grid", x1: x(v), x2: x(v), y1: top - 8, y2: H - 26 }));
    const t = svg("text", { class: "axis-label", x: x(v), y: H - 8, "text-anchor": "middle" });
    t.textContent = `${v} dB`;
    root.append(t);
  }
  const tip = el("div", "tip");
  tip.hidden = true;
  document.body.append(tip);
  cats.forEach((c, i) => {
    const y = top + i * rowH + rowH / 2;
    const gain = c.psnr_on - c.psnr_off;
    const g = svg("g", { class: "row", tabindex: "0",
      "aria-label": `${motion(c.name)}: ${c.psnr_off} dB off, ${c.psnr_on} dB on, ${c.scenes} scenes` });
    g.append(svg("rect", { class: "row-hl", x: 0, y: y - rowH / 2, width: W, height: rowH, rx: 4 }));
    const label = svg("text", { class: "cat-label", x: left - 12, y: y + 4, "text-anchor": "end" });
    label.textContent = motion(c.name);
    g.append(label, svg("line", { class: "link", x1: x(c.psnr_off), x2: x(c.psnr_on), y1: y, y2: y }),
      svg("circle", { class: "dot-off", cx: x(c.psnr_off), cy: y, r: 6 }),
      svg("circle", { class: "dot-on", cx: x(c.psnr_on), cy: y, r: 6 }));
    const show = (ev) => {
      tip.textContent = `${motion(c.name)}, ${c.scenes} scenes: ${c.psnr_off} → ${c.psnr_on} dB ` +
        `(${gain >= 0 ? "+" : ""}${gain.toFixed(2)} dB); moving area ${pct(c.flagged)}`;
      tip.hidden = false;
      const r = ev.currentTarget.getBoundingClientRect();
      tip.style.left = `${Math.min(ev.clientX || r.left + 120, innerWidth - 300)}px`;
      tip.style.top = `${(ev.clientY || r.top) + 14}px`;
    };
    g.addEventListener("mousemove", show);
    g.addEventListener("focus", show);
    g.addEventListener("mouseleave", () => { tip.hidden = true; });
    g.addEventListener("blur", () => { tip.hidden = true; });
    root.append(g);
  });
  const fig = el("figure", "chart sheet");
  const legend = el("ul", "legend");
  [["swatch-off", "Ghost removal off"], ["swatch-on", "Ghost removal on (Lumen)"]].forEach(([cls, text]) => {
    const li = el("li");
    li.append(el("span", `swatch ${cls}`), document.createTextNode(text));
    legend.append(li);
  });
  fig.append(legend, root);
  return fig;
}

function titleBlock(cells) {
  const dl = el("dl", "title-block sheet-block");
  cells.forEach(([k, v]) => { const d = el("div"); d.append(el("dt", null, k), el("dd", null, v)); dl.append(d); });
  return dl;
}

function revisions(steps) {
  const ol = el("ol", "trace");
  ol.setAttribute("aria-label", "Revisions: what the agent decided");
  ol.append(...steps.map((s) => renderStep({ tool: s.tool, decision: s.decision, result: null })));
  return ol;
}

function roomSheet(room, i) {
  const sheet = el("article", "room-sheet sheet");
  const visual = el("div");
  visual.append(el("h3", null, `Empa ${room.scene}`), makeCompare({
    before: room.before, after: room.after, tagBefore: "0 EV frame", tagAfter: "Lumen", size: room.size,
    altBefore: `Empa ${room.scene}: the middle exposure straight from the camera`,
    altAfter: `Empa ${room.scene}: Lumen's merged, straightened result`,
  }));
  const side = el("div", "room-side");
  const dl = el("dl", "metrics");
  [["Exposures merged", String(room.frames)],
   ["Vertical tilt", `${room.tilt_before_deg.toFixed(2)}° → ${room.tilt_after_deg.toFixed(2)}°`],
   ["Clipped after merge", pct(room.clip, 2)],
   ["Moving area removed", pct(room.ghost, 2)],
   ["Result", room.status === "ok" ? "finished, no review needed" : "needs review"]]
    .forEach(([k, v]) => dl.append(el("dt", null, k), el("dd", null, v)));
  side.append(dl, el("h4", "trace-title", "Revisions: what the agent decided"), revisions(room.steps));
  if (room.window) {
    const pair = el("div", "window-pair");
    [[room.window.before, "0 EV frame", room.window.blown_before], [room.window.after, "merge", room.window.blown_after]]
      .forEach(([src, label, blown]) => {
        const fig = el("figure");
        const img = el("img");
        Object.assign(img, { src, alt: `Window of ${room.scene}, ${label}`, loading: "lazy", width: 900, height: 600 });
        const cap = el("figcaption");
        cap.append(document.createTextNode(`Window, ${label}: `), el("b", null, `${pct(blown)} blown`));
        fig.append(img, cap);
        pair.append(fig);
      });
    side.append(pair);
  }
  sheet.append(visual, side, titleBlock([["Sheet", `A-2${String(i + 1).padStart(2, "0")}`], ["Scene", `Empa ${room.scene}, ${room.frames} exposures`],
    ["Source", "Empa HDR database, CC BY 4.0"], ["Drawn by", "OpenCV 5.0, full resolution"],
    ["Checked by", room.status === "ok" ? "the agent: ok" : "the agent: needs review"]]));
  return sheet;
}

function ghostSheet(item, i) {
  const review = item.agent !== "ok";
  const sheet = el("article", `ghost-sheet sheet${review ? " is-review" : ""}`);
  sheet.append(el("h3", null, `${motion(item.category)}, scene ${item.scene}`), makeCompare({
    before: item.off, after: item.on, tagBefore: "Ghost removal off", tagAfter: "On", size: item.size,
    altBefore: `Cambridge ${item.scene} merged without ghost removal: moving parts doubled`,
    altAfter: `Cambridge ${item.scene} merged with ghost removal`,
  }));
  const p = el("p");
  p.append(document.createTextNode("Match with the motion-free shot: "), el("b", null, `${item.psnr_off.toFixed(1)} → ${item.psnr_on.toFixed(1)} dB`),
    document.createTextNode(`. Moving area ${pct(item.flagged)}.`));
  sheet.append(p);
  if (review) sheet.append(el("p", "redline-note", `The agent stopped and asked a person: ${item.agent_reason}.`));
  sheet.append(titleBlock([["Sheet", `A-21${i + 1}`], ["Source", "Cambridge deghosting stacks, CC BY 4.0"],
    ["Checked by", review ? "the agent: asked a person" : "the agent: ok"]]));
  return sheet;
}

function renderReport(r) {
  const out = [];
  const g = r.ghost;
  const rooms = r.gallery.empa;
  out.push(block("report-rooms", "Real rooms with bright windows",
    finding(["Three 7-exposure brackets from the Empa HDR database, processed at full resolution: every room finished on its own, and the verticals ended within ",
      strong(`${Math.max(...rooms.map((e) => e.tilt_after_deg)).toFixed(2)}°`), " of plumb."]),
    ...rooms.map((room, i) => roomSheet(room, i)),
    el("p", "source", `Source: ${r.sources.empa}. Window crops are the reference frame and the merge before lens and tilt, so they line up.`)));

  out.push(block("report-ghost", "Moving people and objects, on real photos",
    finding(["On ", strong(`${g.scenes} real brackets`), " where something moved between exposures, ghost removal raised the match with the motion-free shot from ",
      strong(`${g.psnr_off} dB`), " to ", strong(`${g.psnr_on} dB`), ". It helped in ", strong(`${g.better}`), ", changed nothing in ",
      strong(`${g.same}`), " and made ", strong(`${g.worse}`), " worse."]),
    (() => { const grid = el("div", "ghost-grid"); grid.append(...r.gallery.ghost.map((item, i) => ghostSheet(item, i))); return grid; })(),
    ghostChart(g.categories),
    el("p", "source", "PSNR of Lumen's merge against the merge of the same scene shot without motion; higher is closer. "
      + `Hover or focus a row for its numbers. Source: ${r.sources.cambridge}.`),
    table("Per motion type", ["Motion type", "Scenes", "Off (dB)", "On (dB)", "Moving area"],
      g.categories.map((c) => [motion(c.name), String(c.scenes), c.psnr_off.toFixed(2), c.psnr_on.toFixed(2), pct(c.flagged)]),
      [false, true, true, true, true])));

  const status = (s) => el("span", s === "ok" ? "status-ok" : "status-review", s === "ok" ? "ok" : "needs review");
  out.push(block("report-empa-table", "The same rooms, as recorded by the benchmark",
    el("p", "muted", "From docs/traces, written by bench/run.py on the same three brackets."),
    table(null, ["Scene", "Result", "Frames", "Moving area", "Clipped", "Tilt before → after", "Lens"],
      r.empa.map((e) => [e.scene, status(e.status), String(e.frames), pct(e.ghost, 2), pct(e.clip),
        `${e.tilt_before_deg.toFixed(2)}° → ${e.tilt_after_deg.toFixed(2)}°`, e.lens === "skip" ? "not needed" : e.lens]),
      [false, false, true, true, true, true, false])));

  out.push(block("report-aws", "Running on AWS",
    finding(["On Lambda arm64 (Graviton) with 3008 MB, a 7-photo room takes about ", strong(`${r.aws[0].lambda_s} s`),
      " and costs about ", strong(`US$${r.aws[0].usd.toFixed(4)}`), " of compute."]),
    table(null, ["Bracket", "Photos", "Lambda time", "Peak memory", "Compute", "Cost", "Result"],
      r.aws.map((a) => [a.scene, String(a.frames), `${a.lambda_s} s`, `${a.memory_mb} MB`, `${a.gb_s} GB-s`, `$${a.usd.toFixed(5)}`, a.status]),
      [false, true, true, true, true, true, false])));

  out.push(block("report-synthetic", "Synthetic rooms with a known answer",
    el("p", "muted", "Drawn rooms where the true shift, lens distortion, tilt and moving object are known. "
      + "\"Phone\" adds a tone curve, noise and JPEG compression."),
    table(null, ["Measure", "Clean: mean", "Clean: worst", "Phone: mean", "Phone: worst"],
      r.synthetic.map((m) => {
        const f = (v) => (m.unit ? `${v.toFixed(3)} ${m.unit}` : pct(v));
        return [m.label, f(m.clean.mean), f(m.clean.worst), f(m.phone.mean), f(m.phone.worst)];
      }), [false, true, true, true, true])));

  const limits = el("ol", "notes");
  ["Lens self-calibration has not changed a real photo yet: no real room had enough line curvature to gain 5 %.",
   "Grouping relies on capture time. Brackets shot minutes apart need the \"one room\" option.",
   "When the camera moved between shots (parallax), no single warp fits; Lumen stops and asks a person.",
   "On this account Lambda is capped at 3008 MB, so large sets are shrunk to fit before processing."]
    .forEach((t) => limits.append(el("li", null, t)));
  out.push(block("report-notes", "General notes: known limits", limits));
  $("report").replaceChildren(...out);
}

// ---------- wiring ----------

const picker = $("picker");
picker.addEventListener("change", () => run(picker.files));
$("folder-btn").addEventListener("click", () => {
  const dir = el("input");
  dir.type = "file";
  dir.webkitdirectory = true;
  dir.addEventListener("change", () => run(dir.files));
  dir.click();
});
const drop = $("drop");
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, () => drop.classList.add("is-over")));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, () => drop.classList.remove("is-over")));
addEventListener("hashchange", showView);
showView();
drawCover();
