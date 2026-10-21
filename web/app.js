import { FilesetResolver, FaceLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";
import { EFFECTS, drawEffects, drawIcon, faceFrame } from "./effects.js";

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const SHOTS = 5;

// ---------- templates (slot coordinates are in template pixels, same as 1.py) ----------
const box = (x0, y0, x1, y1) => ({ cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, w: x1 - x0, h: y1 - y0, angle: 0 });
const TEMPLATES = [
  { id: "beige4", name: "Best Friend strip", cat: "4", file: "templates/beige4.png", overlay: "templates/beige4_top.png",
    slots: [box(208, 6, 531, 440), box(208, 446, 531, 880), box(208, 886, 531, 1320), box(208, 1326, 531, 1766)] },
  { id: "bw3", name: "Sunway strip", cat: "3", file: "templates/bw3.jpg",
    slots: [box(60, 191, 540, 613), box(60, 635, 540, 1056), box(60, 1079, 540, 1500)] },
  { id: "red3", name: "Vintage letters", cat: "3", file: "templates/red3.png", overlay: "templates/red3_top.png",
    slots: [{ cx: 505, cy: 467, w: 475, h: 386, angle: 3.95 }, { cx: 534, cy: 882, w: 475, h: 386, angle: 3.95 }, { cx: 563, cy: 1297, w: 475, h: 386, angle: 3.95 }] },
  { id: "news2", name: "ECX Week newspaper", cat: "a5", file: "templates/news2.jpg",
    slots: [box(574, 604, 1302, 1325), box(111, 1391, 528, 1880)] },
];
const CAT_LABEL = { "4": "4 photos", "3": "3 photos", a5: "2 photos" };

const imgCache = {};
const loadImg = (src) => imgCache[src] ??= new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = src; });

// ---------- screens ----------
function show(id) {
  document.querySelectorAll(".screen").forEach((s) => s.classList.toggle("active", s.id === id));
  if (id !== "camera") stopLoop();
}
document.addEventListener("click", (e) => {
  const go = e.target.closest("[data-go]")?.dataset.go;
  if (!go) return;
  go === "camera" ? openCamera() : go === "templates" ? openTemplates() : show(go);
});
$("startBtn").onclick = openCamera;

// ---------- state ----------
let shots = [];            // the 5 captured canvases
let tpl = TEMPLATES[0];    // chosen template
let picked = [];           // shot indexes in the order the user tapped them; picked[0] -> spot 1

// ---------- effects picker ----------
const active = new Set();
for (const ef of EFFECTS) {
  const b = document.createElement("button"); b.className = "fx"; b.innerHTML = `<canvas></canvas>${ef.name}`;
  drawIcon(b.querySelector("canvas"), ef);
  b.onclick = () => { active.has(ef.id) ? active.delete(ef.id) : active.add(ef.id); b.classList.toggle("sel", active.has(ef.id)); };
  $("fxGrid").appendChild(b);
}

// ---------- colour filters (applied to the video only, effects keep their colours) ----------
const COLORS = {
  "None": "none",
  "Black & white": "grayscale(1) contrast(1.1)",
  "Sepia": "sepia(.9) contrast(1.05)",
  "Vivid": "saturate(1.7) contrast(1.1)",
  "Vintage": "sepia(.4) contrast(1.1) saturate(1.3) brightness(.95)",
  "Cool": "hue-rotate(185deg) saturate(1.2)",
};
let colorName = "None";
for (const name of Object.keys(COLORS)) {
  const b = document.createElement("button"); b.className = "opt" + (name === colorName ? " sel" : ""); b.textContent = name;
  b.onclick = () => { colorName = name; [...$("colorRow").children].forEach((x) => x.classList.toggle("sel", x === b)); };
  $("colorRow").appendChild(b);
}

// ---------- camera + face tracking ----------
const video = document.createElement("video");
video.playsInline = true; video.muted = true;
const live = $("live"), lctx = live.getContext("2d");
live.width = 960; live.height = 720;
let running = false, shooting = false, landmarker = null, loadingLm = false;
let smooth = [], lastVideoTime = -1;

function renderFrame(ctx, w, h) {
  const vw = video.videoWidth, vh = video.videoHeight, s = Math.max(w / vw, h / vh);
  ctx.save(); ctx.translate(w, 0); ctx.scale(-1, 1); ctx.filter = COLORS[colorName];
  ctx.drawImage(video, (w - vw * s) / 2, (h - vh * s) / 2, vw * s, vh * s);
  ctx.restore();
  if (active.size) drawEffects(ctx, smooth, active);
}

function track() {
  if (!landmarker || !active.size || video.readyState < 2 || video.currentTime === lastVideoTime) return;
  lastVideoTime = video.currentTime;
  const res = landmarker.detectForVideo(video, performance.now());
  const frames = res.faceLandmarks.map((lm) => faceFrame(lm, live.width, live.height, video.videoWidth, video.videoHeight));
  smooth = frames.map((f, i) => {
    const p = smooth[i]; if (!p) return f;
    const mix = (a, b) => (Array.isArray(b) ? b.map((v, j) => mix(a[j], v)) : a + (b - a) * 0.6);
    const o = {}; for (const k in f) o[k] = mix(p[k], f[k]); return o;
  });
  if (!shooting) $("status").textContent = frames.length ? "" : "Looking for your face…";
}

function loop() {
  if (!running) return;
  if (video.readyState >= 2) { track(); renderFrame(lctx, live.width, live.height); }
  requestAnimationFrame(loop);
}
function stopLoop() { running = false; }

async function loadLandmarker() {
  if (landmarker || loadingLm) return; loadingLm = true;
  try {
    const files = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
    const base = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
    const make = (delegate) => FaceLandmarker.createFromOptions(files, { baseOptions: { modelAssetPath: base, delegate }, runningMode: "VIDEO", numFaces: 3 });
    try { landmarker = await make("GPU"); } catch { landmarker = await make("CPU"); }
    $("status").textContent = "";
  } catch (e) {
    console.error(e); $("status").textContent = "Face tracker failed to load (needs internet) — effects won't show.";
  }
  loadingLm = false;
}

async function openCamera() {
  show("camera");
  $("shotInfo").textContent = `${SHOTS} photos`;
  $("shootBtn").textContent = `Start (${SHOTS} photos)`;
  $("status").textContent = "Starting camera…"; $("shootBtn").disabled = true;
  try {
    if (!video.srcObject) {
      video.srcObject = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720, facingMode: "user" }, audio: false });
      await video.play();
    }
  } catch { $("status").textContent = "Camera blocked — allow camera access and reload."; return; }
  running = true; loop();
  $("status").textContent = landmarker ? "" : "Loading face tracker…";
  $("shootBtn").disabled = false;
  loadLandmarker();
}

async function runShoot() {
  if (shooting) return; shooting = true; $("shootBtn").disabled = true;
  const taken = [];
  for (let i = 0; i < SHOTS; i++) {
    $("shotInfo").textContent = `Photo ${i + 1} of ${SHOTS}`;
    for (let n = 3; n >= 1; n--) { $("countdown").textContent = n; await sleep(1000); }
    $("countdown").textContent = "";
    const c = document.createElement("canvas"); c.width = live.width; c.height = live.height;
    renderFrame(c.getContext("2d"), c.width, c.height); taken.push(c);
    const fl = $("flash"); fl.classList.remove("go"); void fl.offsetWidth; fl.classList.add("go");
    await sleep(450);
  }
  shooting = false; $("status").textContent = "";
  shots = taken;
  openTemplates();
}

// ---------- template page ----------
function renderTemplates(cat = "all") {
  const grid = $("tplGrid"); grid.innerHTML = "";
  for (const t of TEMPLATES.filter((t) => cat === "all" || t.cat === cat)) {
    const d = document.createElement("div"); d.className = "tpl";
    d.innerHTML = `<img src="${t.file}" alt="${t.name}"><span>${t.name} · ${CAT_LABEL[t.cat]}</span>`;
    d.onclick = () => { tpl = t; picked = []; openArrange(); };
    grid.appendChild(d);
  }
}
function openTemplates() { show("templates"); }
$("tabs").onclick = (e) => {
  const b = e.target.closest(".tab"); if (!b) return;
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("sel", x === b));
  renderTemplates(b.dataset.cat);
};
renderTemplates();

// ---------- arrange page: choose which photo goes in each slot ----------
function drawCover(ctx, img, w, h) {
  const s = Math.max(w / img.width, h / img.height);
  ctx.drawImage(img, -(img.width * s) / 2, -(img.height * s) / 2, img.width * s, img.height * s);
}

async function compose(canvas) {
  const bg = await loadImg(tpl.file), top = tpl.overlay ? await loadImg(tpl.overlay) : null;
  canvas.width = bg.width; canvas.height = bg.height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(bg, 0, 0);
  tpl.slots.forEach((s, i) => {
    const pad = s.angle ? 4 : 0, w = s.w + pad, h = s.h + pad;
    ctx.save(); ctx.translate(s.cx, s.cy); ctx.rotate((-s.angle * Math.PI) / 180);
    ctx.beginPath(); ctx.rect(-w / 2, -h / 2, w, h); ctx.clip();
    if (picked[i] !== undefined) drawCover(ctx, shots[picked[i]], w, h);
    else {   // empty spot: dark placeholder with its number
      ctx.fillStyle = "#2b2420"; ctx.fillRect(-w / 2, -h / 2, w, h);
      ctx.fillStyle = "rgba(255,255,255,.55)"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.font = `bold ${Math.min(w, h) * 0.4}px Georgia`; ctx.fillText(i + 1, 0, 0);
    }
    ctx.restore();
  });
  if (top) ctx.drawImage(top, 0, 0);
}

function renderPicker() {
  const n = tpl.slots.length, full = picked.length === n;
  $("pickHint").textContent = full ? "All set! Tap a photo again to remove it and change the order."
    : `Tap your photos in the order you want — picked ${picked.length} of ${n}`;
  const row = $("shotRow"); row.innerHTML = "";
  shots.forEach((c, i) => {
    const pos = picked.indexOf(i);
    const b = document.createElement("button"); b.className = "shot" + (pos >= 0 ? " sel" : "");
    const t = document.createElement("canvas"); t.width = 240; t.height = 180; t.getContext("2d").drawImage(c, 0, 0, 240, 180);
    b.appendChild(t);
    if (pos >= 0) { const tag = document.createElement("i"); tag.textContent = pos + 1; b.appendChild(tag); }
    b.onclick = () => {
      if (pos >= 0) picked.splice(pos, 1);        // tap again = remove, later picks shift up
      else if (picked.length < n) picked.push(i);
      refresh();
    };
    row.appendChild(b);
  });
  $("nextBtn").disabled = !full;
  $("downloadBtn").classList.toggle("disabled", !full);
}

async function refresh() {
  renderPicker();
  await compose($("strip"));
  $("downloadBtn").href = $("strip").toDataURL("image/jpeg", 0.95);
  $("downloadBtn").download = `photobooth_${Date.now()}.jpg`;
}

async function openArrange() {
  show("arrange");
  $("saved").textContent = "";
  $("arrangeTitle").textContent = tpl.name;
  await refresh();
}

const payload = () => ({ strip: $("strip").toDataURL("image/jpeg", 0.95), photos: shots.map((p) => p.toDataURL("image/jpeg", 0.95)) });
const post = (url, body) => fetch(url, { method: "POST", body: JSON.stringify(body) }).then(async (r) => {
  const j = await r.json(); if (!r.ok) throw new Error(j.error || "Something went wrong"); return j;
});

// Download + save to ~/photobooth (raw + strips) through the local server
$("downloadBtn").addEventListener("click", () => {
  post("/save", payload()).then((j) => ($("saved").textContent = `Saved to ${j.dir}`)).catch(() => {});
});

// ---------- email page ----------
$("nextBtn").onclick = () => {
  show("email"); $("emailMsg").textContent = ""; $("sendBtn").disabled = false;
  post("/save", payload()).catch(() => {});
  $("emailInput").focus();
};
$("emailForm").onsubmit = async (e) => {
  e.preventDefault();
  $("sendBtn").disabled = true; $("emailMsg").textContent = "Sending…";
  try {
    await post("/send", { email: $("emailInput").value, ...payload() });
    $("emailMsg").textContent = "Sent! Check your inbox 💌";
  } catch (err) {
    $("emailMsg").textContent = err.message; $("sendBtn").disabled = false;
  }
};
$("doneBtn").onclick = () => { $("emailInput").value = ""; shots = []; show("home"); };

$("shootBtn").onclick = runShoot;
