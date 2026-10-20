// Bayes reads your handwriting: the demo page.
//
// The page does five things after every stroke:
//   1. records your strokes on the drawing pad;
//   2. cleans them up the way MNIST was made: fit the digit in a 20x20 box,
//      then place it in a 28x28 square with its center of mass in the middle;
//   3. turns each pixel r = 1, ..., 784 into u_r = 1 (ink) or u_r = 0 (no ink);
//   4. gives each digit c = 0, ..., 9 its log score:
//        log P̂(η = c) + sum over r of [ u_r log θ̂_{c,r} + (1 - u_r) log(1 - θ̂_{c,r}) ];
//   5. applies Bayes' formula: P̂(η = c | ψ = u) = e^(log score of c) / sum over c' of e^(log score of c').
//      These are the bars.
//
// The numbers P̂(η = c) and θ̂_{c,r} come from model.js, written by train_model.py.
// In the code, k is the digit c and
// j = r - 1 is the pixel (JavaScript counts from 0).

// ---------------------------------------------------------------- 1. Settings

const SIZE = 28;     // MNIST images are 28 x 28 pixels
const BOX = 20;      // the digit is resized to fit in a 20 x 20 box
const PEN = 2.6;     // the width of a stroke, in MNIST pixels (MNIST strokes: median 2.9, middle half 2.4-3.5 pixels)
const DETAIL = 8;    // we draw 8 times larger, then average 8 x 8 blocks: smooth gray levels, like MNIST

// ---------------------------------------------------------------- 2. The model

const DIGITS = MODEL.prior.length;                         // 10
const logPrior = MODEL.prior.map(Math.log);                // log P̂(η = c)
const logTheta = MODEL.theta.map(row => row.map(Math.log));                   // log θ̂_{c,r}
const logOneMinusTheta = MODEL.theta.map(row => row.map(t => Math.log(1 - t))); // log(1 - θ̂_{c,r})

// ---------------------------------------------------------------- 3. Drawing on the pad

const pad = document.getElementById("pad");
const padCtx = pad.getContext("2d");
let strokes = [];          // every stroke is a list of points {x, y}, in pad pixels
let current = null;        // the stroke being drawn now

function resizePad() {
  // Sharp lines on high-resolution screens: the canvas gets more pixels than its CSS size
  const ratio = window.devicePixelRatio || 1;
  const size = pad.getBoundingClientRect().width;
  pad.width = pad.height = Math.round(size * ratio);
  padCtx.setTransform(ratio, 0, 0, ratio, 0, 0);
  redrawPad();
}

function padPenWidth() {
  return pad.getBoundingClientRect().width * 0.045;
}

function drawStroke(ctx, points, width) {
  ctx.lineWidth = width;
  ctx.lineCap = ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  // a single point is drawn as a dot: a line of length 0 with round caps
  for (const p of points.length > 1 ? points.slice(1) : points) ctx.lineTo(p.x, p.y);
  ctx.stroke();
}

function redrawPad() {
  const size = pad.getBoundingClientRect().width;
  padCtx.clearRect(0, 0, size, size);
  padCtx.strokeStyle = getComputedStyle(document.body).getPropertyValue("--pen");
  for (const s of strokes) drawStroke(padCtx, s, padPenWidth());
}

function pointFrom(event) {
  const r = pad.getBoundingClientRect();
  return { x: event.clientX - r.left, y: event.clientY - r.top };
}

pad.addEventListener("pointerdown", event => {
  pad.setPointerCapture(event.pointerId);
  current = [pointFrom(event)];
  strokes.push(current);
  redrawPad();
});

pad.addEventListener("pointermove", event => {
  if (!current) return;
  // a pen sends many small moves between two frames; take them all for a smooth line
  const events = event.getCoalescedEvents ? event.getCoalescedEvents() : [event];
  for (const e of events) current.push(pointFrom(e));
  redrawPad();
});

function endStroke() {
  if (!current) return;
  current = null;
  update();               // a stroke is finished: read the drawing again
}
pad.addEventListener("pointerup", endStroke);
pad.addEventListener("pointercancel", endStroke);

function undo() { strokes.pop(); redrawPad(); update(); }
function clearAll() { strokes = []; redrawPad(); update(); }
document.getElementById("undo").addEventListener("click", undo);
document.getElementById("clear").addEventListener("click", clearAll);
document.addEventListener("keydown", event => {
  if (event.key === "z" || event.key === "Backspace") undo();
  if (event.key === "c" || event.key === "Escape") clearAll();
});

// ---------------------------------------------------------------- 4. From strokes to 28 x 28 pixels

const work = document.createElement("canvas");
work.width = work.height = SIZE * DETAIL;
const workCtx = work.getContext("2d", { willReadFrequently: true });

// Draw the strokes into the 28 x 28 square. A pad point p goes to p * scale + (dx, dy).
// Returns 784 gray levels between 0 (no ink) and 1 (full ink), row by row.
function render(allStrokes, scale, dx, dy) {
  workCtx.setTransform(1, 0, 0, 1, 0, 0);
  workCtx.clearRect(0, 0, work.width, work.height);
  workCtx.setTransform(DETAIL, 0, 0, DETAIL, 0, 0);       // from now on, 1 unit = 1 MNIST pixel
  workCtx.strokeStyle = "black";
  for (const s of allStrokes) {
    drawStroke(workCtx, s.map(p => ({ x: p.x * scale + dx, y: p.y * scale + dy })), PEN);
  }
  const alpha = workCtx.getImageData(0, 0, work.width, work.height).data;
  const gray = new Float32Array(SIZE * SIZE);
  for (let row = 0; row < SIZE * DETAIL; row++) {
    for (let col = 0; col < SIZE * DETAIL; col++) {
      const ink = alpha[(row * SIZE * DETAIL + col) * 4 + 3] / 255;   // how much ink covers this small dot
      gray[Math.floor(row / DETAIL) * SIZE + Math.floor(col / DETAIL)] += ink / (DETAIL * DETAIL);
    }
  }
  return gray;
}

function centerOfMass(gray) {
  let total = 0, rowSum = 0, colSum = 0;
  for (let j = 0; j < SIZE * SIZE; j++) {
    total += gray[j];
    rowSum += gray[j] * Math.floor(j / SIZE);
    colSum += gray[j] * (j % SIZE);
  }
  return { row: rowSum / total, col: colSum / total };
}

// The MNIST clean-up. Returns the 784 values u_r (1 = ink, 0 = no ink), or null if nothing is drawn.
function strokesToPixels(allStrokes) {
  const points = allStrokes.flat();
  if (points.length === 0) return null;
  const xs = points.map(p => p.x), ys = points.map(p => p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);

  // a) resize: the digit, with the width of the pen, fits in a 20 x 20 box
  const scale = (BOX - PEN) / Math.max(maxX - minX, maxY - minY, 1);
  // b) first put the middle of the box in the middle of the square...
  let dx = SIZE / 2 - scale * (minX + maxX) / 2;
  let dy = SIZE / 2 - scale * (minY + maxY) / 2;
  let gray = render(allStrokes, scale, dx, dy);
  // c) ...then move the center of mass to the middle (row 14, column 14, as in MNIST) and draw again
  const c = centerOfMass(gray);
  dx += 14 - c.col;
  dy += 14 - c.row;
  gray = render(allStrokes, scale, dx, dy);
  // d) black or white, the same threshold as for the training digits (127 out of 255)
  return Array.from(gray, g => (g > 127 / 255 ? 1 : 0));
}

// ---------------------------------------------------------------- 5. Bayes' formula

function scores(x) {
  const s = [];
  for (let k = 0; k < DIGITS; k++) {
    let sk = logPrior[k];
    for (let j = 0; j < x.length; j++) {
      sk += x[j] === 1 ? logTheta[k][j] : logOneMinusTheta[k][j];
    }
    s.push(sk);
  }
  return s;
}

function posterior(s) {
  const top = Math.max(...s);                   // subtract the largest score: no overflow
  const e = s.map(v => Math.exp(v - top));
  const total = e.reduce((a, b) => a + b, 0);
  return e.map(v => v / total);
}

// Each pixel's vote in the duel between the winner and the runner-up (a and b in the code):
// the pixel's term of the log score of the winner minus its term of the log score of the runner-up.
// Positive: the pixel supports the winner. Negative: it supports the runner-up.
function votes(x, a, b) {
  return x.map((xj, j) => xj === 1
    ? logTheta[a][j] - logTheta[b][j]
    : logOneMinusTheta[a][j] - logOneMinusTheta[b][j]);
}

// ---------------------------------------------------------------- 6. Showing the result

const bars = [...document.querySelectorAll(".bar")];
const sees = document.getElementById("sees").getContext("2d");
const why = document.getElementById("why").getContext("2d");

function drawPixels(ctx, colourOf) {
  // one 28 x 28 picture: pixel r (j = r - 1 in the code) becomes one square (the thin grid lines come from the page's style)
  const cell = ctx.canvas.width / SIZE;
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  for (let j = 0; j < SIZE * SIZE; j++) {
    const colour = colourOf(j);
    if (!colour) continue;
    ctx.fillStyle = colour;
    ctx.fillRect((j % SIZE) * cell, Math.floor(j / SIZE) * cell, cell, cell);
  }
}

// "84% sure", "99.7% sure", "99.998% sure": as many decimals as needed to show the nines.
// Rounding never reaches 100%: with d decimals, 1 - p is at least 10^-(d+2).
function howSure(p) {
  if (1 - p < 1e-6) return "over 99.9999% sure";
  const decimals = Math.min(4, Math.max(0, Math.ceil(-Math.log10(1 - p)) - 2));
  const shown = Math.round(p * 100 * 10 ** decimals) / 10 ** decimals;
  return shown.toFixed(decimals) + "% sure";
}

// The label above a bar: "85%" (rounded like howSure), or ">99%", so that a bar never claims 100%
function barLabel(p) {
  if (p < 0.01) return "";
  return p > 0.99 ? ">99%" : Math.round(p * 100) + "%";
}

function update() {
  const x = strokesToPixels(strokes);
  const css = getComputedStyle(document.body);
  let p, a = null, b = null, v = null;

  if (x === null) {
    p = MODEL.prior;                           // nothing drawn yet: the bars show the prior
  } else {
    p = posterior(scores(x));
    const order = p.map((pk, k) => k).sort((i, j) => p[j] - p[i]);
    [a, b] = order;
    v = votes(x, a, b);
  }

  bars.forEach((bar, k) => {
    bar.style.setProperty("--p", p[k]);
    bar.classList.toggle("winner", k === a);
    bar.querySelector(".value").textContent = barLabel(p[k]);
    bar.setAttribute("aria-label", `${k}: ${(p[k] * 100).toFixed(1)}%`);
  });

  document.body.classList.toggle("empty", x === null);
  document.getElementById("answer").textContent = x === null ? "" : a;
  document.getElementById("sure").textContent = x === null ? "" : howSure(p[a]);
  document.getElementById("winner-name").textContent = a ?? "the winner";
  document.getElementById("runner-name").textContent = b ?? "the runner-up";

  const inkColour = css.getPropertyValue("--ink");
  drawPixels(sees, j => (x && x[j] === 1 ? inkColour : null));

  const biggest = v ? Math.max(...v.map(Math.abs)) : 1;
  drawPixels(why, j => {
    if (!v || Math.abs(v[j]) < 0.02 * biggest) return null;
    const strength = Math.min(1, Math.abs(v[j]) / (0.5 * biggest));
    const rgb = v[j] > 0 ? css.getPropertyValue("--for-winner-rgb") : css.getPropertyValue("--for-runner-rgb");
    return `rgba(${rgb}, ${strength.toFixed(3)})`;
  });
}

window.addEventListener("resize", resizePad);
resizePad();
update();

// For testing from the browser console: read a drawing given as strokes
window.bayesDemo = { strokesToPixels, scores, posterior };
