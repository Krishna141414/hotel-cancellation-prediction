// Overfitting page: train/CV/test gaps, learning + validation curves, boosting replay,
// decision-boundary playground, temporal validation and the leakage ablation.
(async () => {
  "use strict";
  const { data, chart, getChart, css, fmt, escapeHTML, modelColor, alpha, onTheme, selectedModel, modelSwitcher, showError } = window.App;

  let models, final, setup, learning, validation, temporal, ablation, replay, playground;
  try {
    [models, final, setup, learning, validation, temporal, ablation, replay, playground] = await Promise.all(
      ["models", "final", "setup", "learning_curves", "validation_curve", "temporal", "ablation", "replay", "playground"].map(data)
    );
  } catch (error) {
    return showError(document.getElementById("gap-detail"), error);
  }
  const metric = setup.primary_metric;
  const label = setup.primary_label;
  const byKey = Object.fromEntries(models.map((m) => [m.key, m]));
  let selected = selectedModel(models, final.key);

  // ---------------------------------------------------------------- gap chart + detail
  function renderGaps() {
    const series = [
      { name: "Train (CV folds)", get: (m) => m.train[metric], color: css("--primary") },
      { name: "Cross-validation", get: (m) => m.cv[metric].mean, color: css("--marigold") },
      { name: "Test", get: (m) => m.test[metric], color: css("--teal") },
    ];
    chart("gap-chart", {
      type: "bar",
      data: {
        labels: models.map((m) => m.name.replace(" (tuned)", " ✦")),
        datasets: series.map((s) => ({
          label: s.name,
          data: models.map(s.get),
          backgroundColor: models.map((m) => (m.key === selected ? s.color : alpha(s.color, 0.4))),
          borderRadius: 4,
        })),
      },
      options: {
        maintainAspectRatio: false,
        scales: { y: { min: 0, max: 1.05 }, x: { grid: { display: false }, ticks: { font: { size: 10.5 }, maxRotation: 40, minRotation: 0 } } },
        plugins: {
          legend: { position: "bottom" },
          tooltip: { callbacks: { afterBody: (items) => {
            const m = models[items[0].dataIndex];
            return [`Gap (train − CV): ${fmt.num(m.gap)}`, `Diagnosis: ${m.diagnosis.label}`];
          } } },
        },
        onClick: (_, elements) => elements.length && select(models[elements[0].index].key),
      },
    });
    const m = byKey[selected];
    const cvTest = m.cv[metric].mean - m.test[metric];
    document.getElementById("gap-detail").innerHTML = `
      <div class="card-head"><div><h2>${escapeHTML(m.name)}</h2></div>${Widgets.diagnosisChip(m.diagnosis.label)}</div>
      <div class="table-wrap"><table class="kv"><tbody>
        <tr><td>Train (CV folds)</td><td class="num">${fmt.num(m.train[metric])}</td></tr>
        <tr><td>Cross-validation</td><td class="num">${fmt.num(m.cv[metric].mean)} <span class="std">±${fmt.num(m.cv[metric].std)}</span></td></tr>
        <tr><td>Test</td><td class="num">${fmt.num(m.test[metric])}</td></tr>
        <tr><td><strong>Train − CV gap</strong></td><td class="num ${m.gap > setup.overfit_gap ? "up" : "down"}">${fmt.signed(m.gap)}</td></tr>
        <tr><td>CV − test</td><td class="num">${fmt.signed(cvTest)}</td></tr>
      </tbody></table></div>
      <p class="small" style="margin-top:.8rem">${escapeHTML(m.diagnosis.reason)}</p>
      <p class="small muted">${Math.abs(cvTest) < 0.01 ? "CV and test agree within 0.01, so the CV estimate was trustworthy." : "CV and test differ by more than 0.01; check the fold spread."}</p>`;
  }

  // ---------------------------------------------------------------- learning + validation curves
  function bandDatasets(xs, mean, std, color, name) {
    const upper = mean.map((v, i) => Math.min(1, v + std[i]));
    const lower = mean.map((v, i) => v - std[i]);
    return [
      { label: `${name} +1 std`, data: xs.map((x, i) => ({ x, y: upper[i] })), borderWidth: 0, pointRadius: 0, backgroundColor: alpha(color, 0.15), fill: "+1", hidden: false, legend: false },
      { label: `${name} −1 std`, data: xs.map((x, i) => ({ x, y: lower[i] })), borderWidth: 0, pointRadius: 0, fill: false, legend: false },
      { label: name, data: xs.map((x, i) => ({ x, y: mean[i] })), borderColor: color, backgroundColor: color, pointRadius: 4, pointHoverRadius: 6, fill: false },
    ];
  }
  const curveOptions = (xTitle, extra = {}) => ({
    maintainAspectRatio: false,
    parsing: false,
    interaction: { mode: "index", intersect: false },
    scales: { x: { type: "linear", title: { display: true, text: xTitle }, ...extra }, y: { title: { display: true, text: label } } },
    plugins: {
      legend: { position: "bottom", labels: { filter: (item, d) => !d.datasets[item.datasetIndex].label.includes("std") } },
      tooltip: { filter: (item) => !item.dataset.label.includes("std"), callbacks: { label: (c) => ` ${c.dataset.label}: ${fmt.num(c.raw.y)}` } },
    },
  });
  function renderLearning() {
    let key = selected;
    let note = "";
    if (!learning[key]) {
      const tunedKey = `${byKey[key].base}_tuned`;
      if (learning[tunedKey]) key = tunedKey;
      else if (learning[byKey[key].base]) key = byKey[key].base;
      else key = Object.keys(learning).includes(final.key) ? final.key : Object.keys(learning)[0];
      note = `No learning curve for ${byKey[selected].name} (computed for ${Object.keys(learning).map((k) => byKey[k].name).join(", ")}); showing ${byKey[key].name}.`;
    }
    const curve = learning[key];
    const last = curve.cv_mean.length - 1;
    const gapEnd = curve.train_mean[last] - curve.cv_mean[last];
    const gain = curve.cv_mean[last] - curve.cv_mean[last - 1];
    document.getElementById("lc-caption").textContent = `${byKey[key].name}: train and CV ${label} as the training set grows (±1 std band).`;
    document.getElementById("lc-note").textContent = note || `Gap at full size: ${fmt.num(gapEnd)}. Last step added ${fmt.signed(gain)} CV. ${gain > 0.003 ? "Still improving, so more data would help." : "Flat: more of the same data will not help much."}`;
    chart("learning-chart", {
      type: "line",
      data: { datasets: [...bandDatasets(curve.sizes, curve.train_mean, curve.train_std, css("--primary"), "Train"), ...bandDatasets(curve.sizes, curve.cv_mean, curve.cv_std, css("--pink"), "CV")] },
      options: curveOptions("Training rows", { ticks: { callback: (v) => fmt.int(v) } }),
    });
  }
  function renderValidation() {
    const v = validation;
    const bestIndex = v.depths.indexOf(v.best_depth);
    document.getElementById("vc-note").textContent = `Best CV at depth ${v.best_depth} (${fmt.num(v.cv_mean[bestIndex])}). Depth 1–3 underfits; past ~${v.best_depth + 4} the train score keeps rising while CV falls, which is classic overfitting. The untuned tree (no limit) is at the far right.`;
    chart("validation-chart", {
      type: "line",
      data: { datasets: [
        ...bandDatasets(v.depths, v.train_mean, v.train_std, css("--primary"), "Train"),
        ...bandDatasets(v.depths, v.cv_mean, v.cv_std, css("--pink"), "CV"),
        { label: `Best depth (${v.best_depth})`, data: [{ x: v.best_depth, y: 0.3 }, { x: v.best_depth, y: 1 }], borderColor: css("--marigold"), borderDash: [6, 4], borderWidth: 1.5, pointRadius: 0 },
      ] },
      options: curveOptions("max_depth"),
    });
  }

  // ---------------------------------------------------------------- training replay
  const scrub = document.getElementById("replay-scrub");
  const playButton = document.getElementById("replay-play");
  const iterLabel = document.getElementById("replay-iter");
  scrub.max = replay.iterations.length;
  scrub.value = replay.iterations.length;
  let timer = null;
  document.getElementById("replay-caption").textContent = `${replay.model}: ${label} after each boosting iteration (validation = 20% of the training set). Best validation score ${fmt.num(replay.best_score)} at iteration ${replay.best_iteration}; final ${fmt.num(replay.final_score)}.`;
  function renderReplay() {
    const n = Number(scrub.value);
    const xs = replay.iterations;
    chart("replay-chart", {
      type: "line",
      data: { datasets: [
        { label: "Train", data: xs.slice(0, n).map((x, i) => ({ x, y: replay.train[i] })), borderColor: css("--primary"), backgroundColor: css("--primary") },
        { label: "Validation", data: xs.slice(0, n).map((x, i) => ({ x, y: replay.validation[i] })), borderColor: css("--pink"), backgroundColor: alpha(css("--pink"), 0.12), fill: "origin" },
        { label: `Best iteration (${replay.best_iteration})`, data: [{ x: replay.best_iteration, y: Math.min(...replay.validation) - 0.02 }, { x: replay.best_iteration, y: 1 }], borderColor: css("--marigold"), borderDash: [6, 4], borderWidth: 1.5, pointRadius: 0 },
      ] },
      options: {
        maintainAspectRatio: false, parsing: false, animation: false,
        interaction: { mode: "index", intersect: false },
        scales: { x: { type: "linear", min: 0, max: xs[xs.length - 1], title: { display: true, text: "Boosting iteration" } }, y: { min: Math.floor((Math.min(...replay.validation) - 0.02) * 50) / 50, max: Math.min(1, Math.ceil((Math.max(...replay.train) + 0.01) * 50) / 50), title: { display: true, text: label } } },
        plugins: { legend: { position: "bottom" }, tooltip: { callbacks: { label: (c) => ` ${c.dataset.label}: ${fmt.num(c.raw.y)}` } } },
      },
    });
    iterLabel.textContent = `iter ${xs[n - 1]}`;
  }
  function updateReplay() {
    const c = getChart("replay-chart");
    const n = Number(scrub.value);
    c.data.datasets[0].data = replay.iterations.slice(0, n).map((x, i) => ({ x, y: replay.train[i] }));
    c.data.datasets[1].data = replay.iterations.slice(0, n).map((x, i) => ({ x, y: replay.validation[i] }));
    c.update("none");
    iterLabel.textContent = `iter ${replay.iterations[n - 1]}`;
  }
  function stop() {
    clearInterval(timer);
    timer = null;
    playButton.textContent = "▶ Play";
  }
  playButton.addEventListener("click", () => {
    if (timer) return stop();
    if (Number(scrub.value) >= replay.iterations.length) scrub.value = 1;
    playButton.textContent = "❚❚ Pause";
    timer = setInterval(() => {
      scrub.value = Number(scrub.value) + 1;
      updateReplay();
      if (Number(scrub.value) >= replay.iterations.length) stop();
    }, 45);
  });
  scrub.addEventListener("input", () => {
    stop();
    updateReplay();
  });

  // ---------------------------------------------------------------- playground
  const pairSelect = document.getElementById("pg-pair");
  const pgModels = document.getElementById("pg-models");
  const canvas = document.getElementById("pg-canvas");
  pairSelect.innerHTML = playground.pairs.map((p, i) => `<option value="${i}">${p.x} × ${p.y}</option>`).join("");
  let pgModel = "tree_deep";
  pgModels.innerHTML = Object.entries(playground.models).map(([k, name]) => `<button class="pill" type="button" data-key="${k}" aria-pressed="${k === pgModel}">${escapeHTML(name)}</button>`).join("");
  pgModels.addEventListener("click", (e) => {
    const b = e.target.closest("button[data-key]");
    if (!b) return;
    pgModel = b.dataset.key;
    pgModels.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", x === b));
    drawPlayground();
  });
  pairSelect.addEventListener("change", drawPlayground);
  const hexToRgb = (hex) => {
    const n = parseInt(hex.replace("#", ""), 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  };
  function ramp(t) {
    const lo = hexToRgb(css("--teal")), mid = hexToRgb(css("--surface-3")), hi = hexToRgb(css("--pink"));
    const [a, b, u] = t < 0.5 ? [lo, mid, t / 0.5] : [mid, hi, (t - 0.5) / 0.5];
    return a.map((v, i) => Math.round(v + (b[i] - v) * u));
  }
  function drawPlayground() {
    const pair = playground.pairs[Number(pairSelect.value)];
    const result = pair.results[pgModel];
    const ctx = canvas.getContext("2d");
    const W = canvas.width, H = canvas.height, pad = 46;
    const plotW = W - pad - 12, plotH = H - pad - 12;
    const n = pair.size;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = css("--surface");
    ctx.fillRect(0, 0, W, H);
    const cw = plotW / n, ch = plotH / n;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const [r, g, b] = ramp(result.grid[i * n + j] / 100);
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(pad + j * cw, 12 + plotH - (i + 1) * ch, cw + 0.6, ch + 0.6);
      }
    }
    const [x0, x1] = pair.x_range, [y0, y1] = pair.y_range;
    const sx = (x) => pad + ((x - x0) / (x1 - x0)) * plotW;
    const sy = (y) => 12 + plotH - ((y - y0) / (y1 - y0)) * plotH;
    const discreteY = pair.points.every((p) => Number.isInteger(p[1])) && y1 - y0 <= 10;
    const discreteX = pair.points.every((p) => Number.isInteger(p[0])) && x1 - x0 <= 10;
    let seed = 7;
    const jitter = () => ((seed = (seed * 16807) % 2147483647) / 2147483647 - 0.5) * 0.5;
    pair.points.forEach(([x, y, lab]) => {
      if (x < x0 || x > x1 || y < y0 || y > y1) return;
      const px = sx(discreteX ? x + jitter() : x), py = sy(discreteY ? y + jitter() : y);
      ctx.beginPath();
      ctx.arc(px, py, 3.6, 0, Math.PI * 2);
      ctx.fillStyle = lab ? css("--pink") : css("--teal");
      ctx.fill();
      ctx.lineWidth = 1.4;
      ctx.strokeStyle = App.isDark() ? "#0B1020" : "#FFFFFF";
      ctx.stroke();
    });
    ctx.fillStyle = css("--ink-2");
    ctx.font = "600 12px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(pair.x, pad + plotW / 2, H - 8);
    ctx.save();
    ctx.translate(13, 12 + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(pair.y, 0, 0);
    ctx.restore();
    ctx.font = "11px Inter, sans-serif";
    ctx.fillStyle = css("--ink-3");
    for (let k = 0; k <= 4; k++) {
      const xv = x0 + ((x1 - x0) * k) / 4, yv = y0 + ((y1 - y0) * k) / 4;
      ctx.textAlign = "center";
      ctx.fillText(xv.toFixed(xv >= 100 ? 0 : 1), sx(xv), 12 + plotH + 15);
      ctx.textAlign = "right";
      ctx.fillText(yv.toFixed(yv >= 100 ? 0 : 1), pad - 6, sy(yv) + 4);
    }
    document.getElementById("pg-train").textContent = fmt.num(result.train);
    document.getElementById("pg-test").textContent = fmt.num(result.test);
    const gap = result.train - result.test;
    document.getElementById("pg-note").innerHTML =
      gap > 0.15 ? `<span class="chip chip-bad">Overfitting</span> Train beats test by ${fmt.num(gap)}: the jagged islands fit individual training bookings, not a pattern.`
      : pgModel === "logreg" ? `<span class="chip chip-warn">Underfitting risk</span> A single straight boundary can't follow the curved shape of the data.`
      : `<span class="chip chip-good">Balanced</span> Train and test are close (gap ${fmt.num(gap)}).`;
  }

  // ---------------------------------------------------------------- temporal + ablation
  function renderTemporal() {
    const shown = models.filter((m) => m.key !== "dummy" && temporal.metrics[m.key]);
    document.getElementById("temporal-caption").textContent = `Train on arrivals ${temporal.train_period[0]} → ${temporal.train_period[1]}, test on ${temporal.test_period[0]} → ${temporal.test_period[1]}, compared with the random stratified test split.`;
    chart("temporal-chart", {
      type: "bar",
      data: {
        labels: shown.map((m) => m.name.replace(" (tuned)", " ✦")),
        datasets: [
          { label: "Random split (test)", data: shown.map((m) => m.test[metric]), backgroundColor: css("--primary"), borderRadius: 4 },
          { label: "Temporal split (latest period)", data: shown.map((m) => temporal.metrics[m.key][metric]), backgroundColor: css("--pink"), borderRadius: 4 },
        ],
      },
      options: {
        maintainAspectRatio: false,
        scales: { y: { min: 0, max: 1 }, x: { grid: { display: false }, ticks: { font: { size: 10.5 } } } },
        plugins: { legend: { position: "bottom" }, tooltip: { callbacks: { label: (c) => ` ${c.dataset.label}: ${fmt.num(c.raw)}` } } },
      },
    });
    const f = byKey[final.key];
    const ft = temporal.metrics[final.key];
    const drops = shown.map((m) => m.test[metric] - temporal.metrics[m.key][metric]);
    document.getElementById("temporal-detail").innerHTML = `
      <h3>What changed over time</h3>
      <div class="score-pair" style="margin:.6rem 0"><div><b>${fmt.pct(temporal.train_cancel_rate)}</b><span>cancel rate before</span></div><div><b class="up">${fmt.pct(temporal.test_cancel_rate)}</b><span>latest period</span></div></div>
      <div class="table-wrap"><table class="kv"><tbody>
        <tr><td>${escapeHTML(f.name)} · ${label} random</td><td class="num">${fmt.num(f.test[metric])}</td></tr>
        <tr><td>… temporal</td><td class="num">${fmt.num(ft[metric])}</td></tr>
        <tr><td>… ROC-AUC random → temporal</td><td class="num">${fmt.num(f.test.roc_auc)} → ${fmt.num(ft.roc_auc)}</td></tr>
        <tr><td>Average ${label} change, all models</td><td class="num">${fmt.signed(-drops.reduce((a, b) => a + b, 0) / drops.length)}</td></tr>
      </tbody></table></div>
      <p class="small" style="margin-top:.7rem">The latest period (spring–summer 2017) cancels far more than the training period, and its months are under-represented in training. A random split hides this drift. In production the model should be <strong>retrained regularly</strong> and the threshold re-checked each season.</p>`;
  }
  function renderAblation() {
    chart("ablation-chart", {
      type: "bar",
      data: {
        labels: ablation.map((r) => r.variant),
        datasets: [{
          data: ablation.map((r) => r.pr_auc),
          backgroundColor: ablation.map((r) => (r.delta === 0 ? css("--marigold") : r.delta > 0 ? css("--pink") : css("--primary"))),
          borderRadius: 4,
        }],
      },
      options: {
        indexAxis: "y", maintainAspectRatio: false,
        scales: { x: { min: Math.floor((Math.min(...ablation.map((r) => r.pr_auc)) - 0.03) * 20) / 20, max: Math.ceil((Math.max(...ablation.map((r) => r.pr_auc)) + 0.01) * 50) / 50 }, y: { grid: { display: false } } },
        plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => ` CV ${label} ${fmt.num(c.raw, 4)} (${fmt.signed(ablation[c.dataIndex].delta, 4)} vs final)` } } },
      },
    });
  }

  // ---------------------------------------------------------------- wiring
  function select(key) {
    selected = key;
    App.store.set("selectedModel", key);
    document.querySelectorAll("#model-switcher .pill").forEach((b) => b.setAttribute("aria-pressed", b.dataset.key === key));
    renderGaps();
    renderLearning();
  }
  function renderAll() {
    renderGaps();
    renderLearning();
    renderValidation();
    renderReplay();
    drawPlayground();
    renderTemporal();
    renderAblation();
  }
  modelSwitcher(document.getElementById("model-switcher"), models, { selected, onChange: select });
  renderAll();
  onTheme(renderAll);
})();
