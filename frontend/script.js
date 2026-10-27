/* See2Speak frontend. Talks to the FastAPI backend; all detections, transcripts and
   scores shown here come from real backend responses (nothing is simulated). */
"use strict";

// If you serve this page from somewhere other than the FastAPI server, open it as
// index.html?api=http://localhost:8000
const API = new URLSearchParams(location.search).get("api") || "";

// ---- tunables --------------------------------------------------------------
const STABLE_HITS = 2;        // consecutive detections before an object counts as "locked"
const LOSE_MISSES = 6;        // detection rounds without the target before it is released
const DETECT_FRAME_W = 640;   // width of frames sent for detection
const DETECT_MIN_GAP_MS = 120;
const MAX_RECORD_MS = 4000;   // auto-stop so a forgotten recording can't hang
const MOUTH_FRAME_MS = 100;   // mouth frames sampled during recording
const MOUTH_MAX_FRAMES = 50;

const $ = (id) => document.getElementById(id);
const els = {
  cam: $("cam"), stage: $("stage"), placeholder: $("placeholder"), boxLayer: $("boxLayer"),
  slime: $("slimeWrap"), bubble: $("bubble"), banner: $("banner"),
  btnStart: $("btnStart"), btnRecal: $("btnRecal"), btnFinish: $("btnFinish"),
  btnRestart: $("btnRestart"), btnStop: $("btnStop"),
  btnListen: $("btnListen"), btnRecord: $("btnRecord"),
  targetWord: $("targetWord"), targetHelp: $("targetHelp"), recStatus: $("recStatus"),
  resultBox: $("resultBox"), rTarget: $("rTarget"), rAnswer: $("rAnswer"), rResult: $("rResult"),
  rPoints: $("rPoints"), mouthMsg: $("mouthMsg"),
  hudScore: $("hudScore"), sScore: $("sScore"), sCorrect: $("sCorrect"), sAttempts: $("sAttempts"),
  sProgressText: $("sProgressText"), progressBar: $("progressBar"), progressFill: $("progressFill"),
  wordChips: $("wordChips"), micPill: $("micPill"), micText: $("micText"),
  mDetector: $("mDetector"), mSpeech: $("mSpeech"), mMouth: $("mMouth"), mFps: $("mFps"),
  summary: $("summary"),
};

const S = {
  phase: "off",            // off | starting | scanning | ready | recording | processing
  sessionId: null,
  camStream: null, micStream: null,
  vocab: [], labels: {},   // key -> label
  tracks: new Map(),       // key -> {hits, misses, det}
  target: null, manualKey: null,
  audioPlaying: false,
  models: {},
  stats: { score: 0, attempts: 0, correct: 0, mastered: [], round_target: 6 },
  summaryShown: false,
  fpsMs: [],
};

// ---------------------------------------------------------------- helpers
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
function banner(text, kind = "") {
  els.banner.hidden = !text;
  els.banner.textContent = text || "";
  els.banner.className = "banner " + kind;
}
function setStatus(text) { els.recStatus.textContent = text; }
async function api(path, opts) {
  let res;
  try { res = await fetch(API + path, opts); }
  catch { throw new Error("Can't reach the See2Speak server. Is it running? (uvicorn backend.main:app)"); }
  let data = null;
  try { data = await res.json(); } catch { /* non-JSON error */ }
  if (!res.ok) throw new Error((data && data.detail) || `Server error (${res.status})`);
  return data;
}

// -------------------------------------------------------- slime character
// The original wandering/squishing/blinking animation is untouched (CSS). Below we only
// add event-driven reactions and let the slime stop next to the object being practised.
let reactTimer = null;
function bubbleDefault() {
  if (S.phase === "off") return "Press Start!";
  if (S.target) return `Say: ${S.labels[S.target]}!`;
  return "Show me an object!";
}
function setBubble(t) { els.bubble.textContent = t; }
function react(kind, text, ms = 2200) {
  const w = els.slime;
  w.classList.remove("is-happy", "is-oops", "is-new", "is-listening");
  void w.offsetWidth;                    // restart CSS animations
  w.classList.add("is-" + kind);
  if (text) setBubble(text);
  clearTimeout(reactTimer);
  if (ms) reactTimer = setTimeout(() => { w.classList.remove("is-" + kind); setBubble(bubbleDefault()); }, ms);
}
function parkSlime(x, y) {
  const w = els.slime;
  if (!w.classList.contains("parked")) {
    // freeze at the current animated position first so there's no jump
    w.style.transition = "none";
    w.style.transform = getComputedStyle(w).transform;
    void w.offsetWidth;
    w.classList.add("parked");
    w.style.transition = "";
  }
  w.style.transform = `translate(${x}px, ${y}px)`;
}
function unparkSlime() {
  const w = els.slime;
  if (!w.classList.contains("parked")) return;
  w.style.transform = "translate(33cqw, 50cqh)";   // = first keyframe of the wander animation
  setTimeout(() => {
    if (!S.target) { w.classList.remove("parked"); w.style.transform = ""; }
  }, 1050);
}
function parkNextToTarget() {
  const t = S.tracks.get(S.target);
  if (!t) return;
  const W = els.stage.clientWidth, H = els.stage.clientHeight, b = t.det.box;
  const left = (1 - (b.x + b.w)) * W, right = left + b.w * W, top = b.y * H;
  const sw = 150, sh = 120;
  let x = right + 12 > W - sw ? left - sw - 12 : right + 12;   // right of the box if it fits
  x = Math.max(4, Math.min(W - sw - 4, x));
  const y = Math.max(60, Math.min(H - sh - 40, top + b.h * H - sh));   // bottom-aligned: bubble stays clear of the label
  parkSlime(x, y);
}

// ----------------------------------------------------------- score panel
function renderStats(st) {
  if (!st) return;
  S.stats = st;
  els.sScore.textContent = els.hudScore.textContent = st.score;
  els.sCorrect.textContent = st.correct;
  els.sAttempts.textContent = st.attempts;
  els.sProgressText.textContent = `${st.mastered_count ?? st.mastered.length} / ${st.round_target}`;
  els.progressBar.setAttribute("aria-valuenow", st.mastered.length);
  els.progressBar.setAttribute("aria-valuemax", st.round_target);
  els.progressFill.style.width = (100 * st.mastered.length / st.round_target) + "%";
  els.wordChips.innerHTML = "";
  for (const w of S.vocab) {
    const li = document.createElement("li");
    li.textContent = w.label;
    if (st.mastered.includes(w.key)) { li.className = "done"; li.title = "collected"; }
    els.wordChips.appendChild(li);
  }
}
function resetResult() {
  els.resultBox.hidden = true; els.mouthMsg.hidden = true;
  els.resultBox.className = "result";
}

// ------------------------------------------------------------ button state
function updateButtons() {
  const on = S.phase !== "off" && S.phase !== "starting";
  const busy = S.phase === "processing" || S.phase === "starting";
  els.btnStart.disabled = S.phase !== "off";
  els.btnStop.disabled = !on && !S.camStream;
  els.btnFinish.disabled = !on || busy;
  els.btnRestart.disabled = busy;
  els.btnRecal.disabled = !on || busy || S.phase === "recording";
  els.btnListen.disabled = !(S.target && (S.phase === "ready")) || S.audioPlaying;
  els.btnRecord.disabled = S.phase === "recording" ? false
    : !(S.target && S.phase === "ready") || S.audioPlaying;
  els.btnRecord.classList.toggle("recording", S.phase === "recording");
  els.btnRecord.textContent = S.phase === "recording" ? "⏹ Stop recording" : "🎤 Record";
  els.micPill.classList.toggle("rec", S.phase === "recording");
}

// ------------------------------------------------------------ system health
async function pollHealth() {
  try {
    const h = await api("/api/health");
    S.models = h.models;
    const show = (el, v) => { el.textContent = v === "ready" ? "ready ✓" : v.startsWith("error") ? "⚠ " + v : "loading…"; };
    show(els.mDetector, h.models.detector); show(els.mSpeech, h.models.speech); show(els.mMouth, h.models.mouth);
    const allDone = Object.values(h.models).every((v) => v === "ready" || v.startsWith("error"));
    if (!allDone) setTimeout(pollHealth, 2000);
    else for (const [n, v] of Object.entries(h.models)) {
      if (v.startsWith("error")) banner(`The ${n} model could not load: ${v.slice(7)} See README → Troubleshooting.`, "error");
    }
  } catch (e) {
    banner(e.message, "error");
    setTimeout(pollHealth, 4000);
  }
}

// ------------------------------------------------------------------ camera
function grabFrame(canvas, maxW, quality) {
  const v = els.cam;
  if (!v.videoWidth) return Promise.resolve(null);
  const scale = Math.min(1, maxW / v.videoWidth);
  canvas.width = Math.round(v.videoWidth * scale);
  canvas.height = Math.round(v.videoHeight * scale);
  canvas.getContext("2d").drawImage(v, 0, 0, canvas.width, canvas.height);
  return new Promise((res) => canvas.toBlob(res, "image/jpeg", quality));
}
const detectCanvas = document.createElement("canvas");
const mouthCanvas = document.createElement("canvas");

async function openCamera() {
  try {
    S.camStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
  } catch (e) {
    const msg = e.name === "NotAllowedError" ? "Camera permission was denied. Allow the camera for this site and press Start again."
      : e.name === "NotFoundError" ? "No camera was found on this computer."
      : e.name === "NotReadableError" ? "The camera is being used by another app. Close it and try again."
      : "Could not open the camera: " + e.message;
    throw new Error(msg);
  }
  els.cam.srcObject = S.camStream;
  await new Promise((r) => { if (els.cam.videoWidth) r(); else els.cam.onloadedmetadata = r; });
  els.stage.style.setProperty("--ar", els.cam.videoWidth / els.cam.videoHeight);
  els.stage.style.aspectRatio = `${els.cam.videoWidth} / ${els.cam.videoHeight}`;
  els.placeholder.hidden = true;
}
function releaseDevices() {
  S.camStream?.getTracks().forEach((t) => t.stop()); S.camStream = null;
  S.micStream?.getTracks().forEach((t) => t.stop()); S.micStream = null;
  els.cam.srcObject = null;
  els.placeholder.hidden = false;
  stopReference();
}

// -------------------------------------------------- mouth-baseline calibration
async function calibrate() {
  if (S.models.mouth && S.models.mouth !== "ready") { return; }
  const prev = S.phase;
  S.phase = "starting"; updateButtons();
  els.micText.textContent = "Relax your face & keep your mouth closed…";
  setStatus("Calibrating: relax your face and keep your mouth closed for a moment.");
  await sleep(1200);
  const frames = [];
  for (let i = 0; i < 8; i++) { const f = await grabFrame(mouthCanvas, 480, 0.8); if (f) frames.push(f); await sleep(150); }
  const fd = new FormData();
  fd.append("session_id", S.sessionId);
  frames.forEach((f, i) => fd.append("frames", f, `c${i}.jpg`));
  try {
    const r = await api("/api/calibrate", { method: "POST", body: fd });
    banner(r.ok ? "" : r.message + " The mouth check will use an estimate instead.", r.ok ? "" : "info");
  } catch (e) { banner(e.message, "error"); }
  S.phase = prev === "off" ? "scanning" : prev; updateButtons();
}

// ------------------------------------------------------------------ detection
function updateTracks(dets) {
  const seen = new Set();
  for (const d of dets) {
    if (seen.has(d.key)) continue;            // best box per word (list is sorted by confidence)
    seen.add(d.key);
    const t = S.tracks.get(d.key) || { hits: 0, misses: 0 };
    t.hits++; t.misses = 0; t.det = d;
    S.tracks.set(d.key, t);
  }
  for (const [k, t] of S.tracks) {
    if (!seen.has(k)) { t.misses++; if (t.misses >= LOSE_MISSES) S.tracks.delete(k); }
  }
}
const isStable = (t) => t && t.hits >= STABLE_HITS && t.misses < LOSE_MISSES;

function chooseTarget() {
  if (S.manualKey && !S.tracks.has(S.manualKey)) S.manualKey = null;
  let next = null;
  if (S.manualKey && isStable(S.tracks.get(S.manualKey))) next = S.manualKey;
  else if (S.target && isStable(S.tracks.get(S.target))) next = S.target;       // sticky
  else {
    let best = null;
    for (const [k, t] of S.tracks) if (isStable(t) && t.misses === 0 && (!best || t.det.confidence > best.det.confidence)) best = t;
    next = best ? best.det.key : null;
  }
  if (next !== S.target) setTarget(next);   // fires ONCE per change, not per frame
}

function setTarget(key) {
  S.target = key;
  resetResult();
  if (key) {
    const label = S.labels[key];
    S.phase = "ready";
    els.targetWord.textContent = label;
    els.targetHelp.textContent = "Listen, then press Record and say the word.";
    setStatus(`Ready: say “${label}”.`);
    react("new", `New word: ${label}!`, 2000);
    setTimeout(() => S.target === key && setBubble(bubbleDefault()), 2100);
    parkNextToTarget();
  } else {
    S.phase = S.phase === "ready" ? "scanning" : S.phase;
    els.targetWord.textContent = "—";
    els.targetHelp.textContent = "Hold up a supported object to begin.";
    setStatus("Looking for an object…");
    setBubble(bubbleDefault());
    unparkSlime();
  }
  updateButtons();
}

function renderBoxes() {
  const live = new Set();
  for (const [key, t] of S.tracks) {
    if (!(t.hits >= STABLE_HITS)) continue;
    live.add(key);
    let el = els.boxLayer.querySelector(`[data-key="${key}"]`);
    if (!el) {
      el = document.createElement("div"); el.className = "box"; el.dataset.key = key;
      el.innerHTML = '<div class="tag"></div>';
      el.title = "Click to practise this object";
      el.addEventListener("click", () => {
        if (S.phase === "ready" || S.phase === "scanning") { S.manualKey = key; chooseTarget(); }
      });
      els.boxLayer.appendChild(el);
    }
    const b = t.det.box;                      // video is mirrored on screen -> mirror x
    el.style.left = (1 - (b.x + b.w)) * 100 + "%"; el.style.top = b.y * 100 + "%";
    el.style.width = b.w * 100 + "%"; el.style.height = b.h * 100 + "%";
    el.classList.toggle("selected", key === S.target);
    el.classList.toggle("stale", t.misses > 0);
    const tag = el.firstChild;
    tag.textContent = S.labels[key] || key;
    const small = document.createElement("small");
    small.textContent = Math.round(t.det.confidence * 100) + "%";
    tag.appendChild(small);
  }
  els.boxLayer.querySelectorAll(".box").forEach((el) => { if (!live.has(el.dataset.key)) el.remove(); });
}

async function detectLoop() {
  while (S.camStream) {
    if (S.phase !== "scanning" && S.phase !== "ready") { await sleep(150); continue; }
    if (S.models.detector && S.models.detector !== "ready") {
      if (S.models.detector.startsWith("error")) { await sleep(2000); continue; }
      setStatus("Loading the object detector…"); await sleep(1000); continue;
    }
    const t0 = performance.now();
    try {
      const blob = await grabFrame(detectCanvas, DETECT_FRAME_W, 0.7);
      if (!blob) { await sleep(200); continue; }
      const fd = new FormData(); fd.append("image", blob, "frame.jpg");
      const r = await api("/api/detect", { method: "POST", body: fd });
      if (S.phase !== "scanning" && S.phase !== "ready") continue;   // user started recording meanwhile
      updateTracks(r.detections);
      chooseTarget();
      renderBoxes();
      if (S.target) parkNextToTarget();
      S.fpsMs.push(performance.now() - t0); if (S.fpsMs.length > 10) S.fpsMs.shift();
      const avg = S.fpsMs.reduce((a, b) => a + b, 0) / S.fpsMs.length;
      els.mFps.textContent = `${Math.round(avg)} ms/frame (≈${(1000 / avg).toFixed(1)} fps), model ${r.inference_ms} ms`;
      if (!S.target && !r.detections.length) setStatus("No supported object in view. Try better light or move closer.");
      banner("");
    } catch (e) {
      banner(e.message, "error"); await sleep(1500);
    }
    await sleep(DETECT_MIN_GAP_MS);
  }
}

// ----------------------------------------------------------- reference audio
let currentAudio = null, ttsTimer = null;
function stopReference() {
  if (currentAudio) { currentAudio.pause(); currentAudio = null; }
  if ("speechSynthesis" in window) speechSynthesis.cancel();
  clearTimeout(ttsTimer);
  S.audioPlaying = false;
}
function playFile(url) {
  return new Promise((resolve, reject) => {
    const a = new Audio(url); currentAudio = a;
    a.onended = resolve; a.onerror = () => reject(new Error("no file"));
    a.play().catch(reject);
  });
}
function speak(text) {
  return new Promise((resolve, reject) => {
    if (!("speechSynthesis" in window)) return reject(new Error("This browser has no speech synthesis."));
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.lang = "en-US"; u.rate = 0.8;
    const voice = speechSynthesis.getVoices().find((v) => /^en[-_]/i.test(v.lang));
    if (voice) u.voice = voice;
    u.onend = resolve; u.onerror = (e) => reject(new Error(e.error || "speech error"));
    ttsTimer = setTimeout(resolve, 6000);   // some browsers never fire onend
    speechSynthesis.speak(u);
  });
}
async function playReference() {
  if (!S.target || S.phase !== "ready") return;
  const key = S.target, label = S.labels[key];
  stopReference();
  S.audioPlaying = true; updateButtons();
  setStatus(`🔊 Playing “${label}”…`);
  try {
    // 1) prerecorded file frontend/assets/audio/<key>.mp3 if you add one, 2) browser text-to-speech
    try { await playFile(`assets/audio/${key}.mp3`); }
    catch { await speak(label); }
    setStatus(`Ready: say “${label}”.`);
  } catch (e) {
    setStatus("⚠ I couldn't play the sound: " + e.message + " Check your volume or try another browser.");
  } finally {
    clearTimeout(ttsTimer); S.audioPlaying = false; updateButtons();
  }
}

// -------------------------------------------------------------- recording
let recorder = null, recChunks = [], mouthFrames = [], mouthTimer = null, autoStop = null, countdown = null;

function encodeWav(float32, rate) {
  const buf = new ArrayBuffer(44 + float32.length * 2), v = new DataView(buf);
  const str = (o, s) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  str(0, "RIFF"); v.setUint32(4, 36 + float32.length * 2, true); str(8, "WAVEfmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  str(36, "data"); v.setUint32(40, float32.length * 2, true);
  for (let i = 0; i < float32.length; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buf], { type: "audio/wav" });
}
async function toWav16k(blob) {
  const AC = window.AudioContext || window.webkitAudioContext;
  const ac = new AC();
  let decoded;
  try { decoded = await ac.decodeAudioData(await blob.arrayBuffer()); }
  finally { ac.close(); }
  const off = new OfflineAudioContext(1, Math.max(1, Math.ceil(decoded.duration * 16000)), 16000);
  const src = off.createBufferSource(); src.buffer = decoded; src.connect(off.destination); src.start();
  const rendered = await off.startRendering();
  return encodeWav(rendered.getChannelData(0), 16000);
}

async function startRecording() {
  if (!S.target || S.phase !== "ready") return;
  stopReference();
  try {
    S.micStream ||= await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
  } catch (e) {
    setStatus(e.name === "NotAllowedError"
      ? "⚠ Microphone permission was denied. Allow the microphone for this site, then press Record again."
      : e.name === "NotFoundError" ? "⚠ No microphone was found."
      : "⚠ Could not open the microphone: " + e.message);
    return;
  }
  if (!window.MediaRecorder) { setStatus("⚠ This browser can't record audio. Try Chrome, Edge or Firefox."); return; }
  const label = S.labels[S.target];
  S.attemptId = crypto.randomUUID();
  S.attemptKey = S.target;                       // lock the word for this attempt
  recChunks = []; mouthFrames = [];
  recorder = new MediaRecorder(S.micStream);
  recorder.ondataavailable = (e) => e.data.size && recChunks.push(e.data);
  recorder.onstop = finishRecording;
  S.phase = "recording"; resetResult(); updateButtons();
  els.micText.textContent = "Recording… say the word!";
  react("listening", "I'm listening…", 0);
  let left = MAX_RECORD_MS / 1000;
  setStatus(`🔴 Recording… say “${label}” (auto-stops in ${left}s)`);
  countdown = setInterval(() => { left--; if (left > 0) setStatus(`🔴 Recording… say “${label}” (auto-stops in ${left}s)`); }, 1000);
  recorder.start();
  mouthTimer = setInterval(() => {
    if (mouthFrames.length < MOUTH_MAX_FRAMES) mouthFrames.push(grabFrame(mouthCanvas, 480, 0.8));
  }, MOUTH_FRAME_MS);
  autoStop = setTimeout(stopRecording, MAX_RECORD_MS);
}
function stopRecording() {
  clearTimeout(autoStop); clearInterval(mouthTimer); clearInterval(countdown);
  if (recorder && recorder.state === "recording") recorder.stop();
}

async function finishRecording() {
  S.phase = "processing"; updateButtons();
  els.micText.textContent = "Thinking…";
  setStatus("⏳ Processing your speech…");
  react("listening", "Hmm, let me listen…", 0);
  const key = S.attemptKey, attemptId = S.attemptId;
  try {
    const wav = await toWav16k(new Blob(recChunks, { type: recorder.mimeType }));
    const frames = (await Promise.all(mouthFrames)).filter(Boolean);
    const fd = new FormData();
    fd.append("session_id", S.sessionId); fd.append("attempt_id", attemptId); fd.append("word_key", key);
    fd.append("audio", wav, "speech.wav");
    frames.forEach((f, i) => fd.append("frames", f, `m${i}.jpg`));
    const r = await api("/api/attempt", { method: "POST", body: fd });
    showResult(r, key);
  } catch (e) {
    setStatus("⚠ " + e.message);
    react("oops", "Oops, something went wrong. Try again!", 2500);
  } finally {
    S.phase = S.target ? "ready" : "scanning";
    els.micText.textContent = S.camStream ? "Camera on" : "Camera off";
    updateButtons();
  }
}

function showResult(r, key) {
  renderStats(r.stats);
  if (r.status === "silent" || r.status === "no_words") {
    setStatus("⚠ " + r.message);
    react("oops", "Say it again!", 2500);
    showMouth(r.mouth);
    return;
  }
  els.resultBox.hidden = false;
  els.resultBox.className = "result " + (r.correct ? "good" : "try");
  els.rTarget.textContent = r.target;
  els.rAnswer.textContent = r.transcript || "—";
  els.rResult.textContent = r.correct ? "✓ Correct!" : (r.near_match ? "✗ Close, try again" : "✗ Not quite");
  els.rPoints.textContent = "+" + r.points;
  setStatus(r.message);
  showMouth(r.mouth);
  if (r.correct) {
    react("happy", r.points ? `Great job! +${r.points}` : "Correct again!", 3200);
  } else {
    react("oops", r.near_match ? "So close! Try again!" : "Try again!", 2800);
  }
  if (S.stats.complete && !S.summaryShown) {
    S.summaryShown = true;
    setTimeout(() => finishGame(true), 3000);
  }
}
function showMouth(m) {
  if (!m || !m.message) { els.mouthMsg.hidden = true; return; }
  els.mouthMsg.hidden = false;
  els.mouthMsg.textContent = "👄 " + m.message;
}

// ------------------------------------------------------------ game lifecycle
async function newSession() {
  const s = await api("/api/session", { method: "POST" });
  S.sessionId = s.session_id; S.summaryShown = false; S.manualKey = null;
  S.tracks.clear(); els.boxLayer.innerHTML = "";
  renderStats(s); resetResult(); setTarget(null);
}

async function startGame() {
  if (S.phase !== "off") return;
  S.phase = "starting"; updateButtons();
  banner("");
  try {
    await openCamera();
    await newSession();
    els.micText.textContent = "Camera on";
    S.phase = "scanning";
    await calibrate();
    S.phase = S.target ? "ready" : "scanning";
    setStatus("Looking for an object…"); setBubble(bubbleDefault());
    react("new", "Show me an object!", 2500);
    detectLoop();
  } catch (e) {
    banner(e.message, "error");
    releaseDevices(); S.phase = "off";
    els.micText.textContent = "Camera off";
  }
  updateButtons();
}

async function endSession() {
  if (!S.sessionId) return null;
  try { const st = await api(`/api/session/${S.sessionId}/end`, { method: "POST" }); return st; }
  catch (e) { banner(e.message, "error"); return null; }
}

async function stopAll(summary) {
  stopRecording(); stopReference();
  S.phase = "off";
  const st = await endSession();
  releaseDevices();
  S.tracks.clear(); S.target = null; els.boxLayer.innerHTML = "";
  els.targetWord.textContent = "—"; unparkSlime(); setBubble(bubbleDefault());
  els.micText.textContent = "Camera off"; setStatus("Camera and microphone released.");
  updateButtons();
  if (summary && st) await showSummary(st);
}
const finishGame = (auto) => stopAll(true);

async function restartGame() {
  if (S.phase === "off") { await startGame(); return; }
  stopRecording(); stopReference();
  await endSession();
  S.phase = "scanning";
  try { await newSession(); setBubble(bubbleDefault()); setStatus("New game! Looking for an object…"); }
  catch (e) { banner(e.message, "error"); }
  updateButtons();
}

async function showSummary(st) {
  $("sumScore").textContent = st.score; $("sumCorrect").textContent = st.correct; $("sumAttempts").textContent = st.attempts;
  const names = st.mastered.map((k) => S.labels[k] || k);
  $("sumWords").textContent = names.length
    ? `Words collected (${names.length}/${st.round_target}): ${names.join(", ")}.`
    : "No words collected this time. Keep practising!";
  const ul = $("sumHistory"); ul.innerHTML = "";
  try {
    const h = await api("/api/history");
    for (const s of h.sessions) {
      const li = document.createElement("li");
      li.textContent = `${new Date(s.started * 1000).toLocaleString()}: ${s.score} pts, ${s.correct}/${s.attempts} correct`;
      ul.appendChild(li);
    }
  } catch { /* history is optional */ }
  els.summary.showModal();
  react("happy", st.complete ? "You did it!" : "Good practice!", 4000);
}

// ---------------------------------------------------------------------- init
els.btnStart.onclick = startGame;
els.btnStop.onclick = () => stopAll(false);
els.btnFinish.onclick = () => stopAll(true);
els.btnRestart.onclick = restartGame;
els.btnRecal.onclick = () => S.sessionId && calibrate();
els.btnListen.onclick = playReference;
els.btnRecord.onclick = () => (S.phase === "recording" ? stopRecording() : startRecording());
$("sumAgain").addEventListener("click", () => setTimeout(restartGame, 0));
window.addEventListener("beforeunload", releaseDevices);

(async function init() {
  try {
    const cfg = await api("/api/config");
    S.vocab = cfg.vocabulary; S.labels = Object.fromEntries(cfg.vocabulary.map((w) => [w.key, w.label]));
    renderStats({ score: 0, attempts: 0, correct: 0, mastered: [], round_target: cfg.round_target });
  } catch (e) { banner(e.message, "error"); }
  if ("speechSynthesis" in window) speechSynthesis.getVoices();   // warm up voice list
  setBubble(bubbleDefault());
  updateButtons();
  pollHealth();
})();
