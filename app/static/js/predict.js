// Predict page: form built from the cached schema, live predictions (what-if mode), animated
// risk gauge, "why this prediction" bars, what-if curves, threshold slider and cost calculator.
(async () => {
  "use strict";
  const { data, postJSON, chart, getChart, css, fmt, escapeHTML, alpha, onTheme, showError } = window.App;
  const formEl = document.getElementById("booking-form");
  const fieldsEl = document.getElementById("form-fields");

  let schema, final, thresholdInfo, setup;
  try {
    [schema, final, thresholdInfo, setup] = await Promise.all([data("predict"), data("final"), data("threshold"), data("setup")]);
  } catch (error) {
    return showError(fieldsEl, error);
  }
  const tuned = final.threshold;
  let threshold = tuned;
  let last = null; // last prediction response
  let actual = null; // true outcome when a real booking was loaded

  // ---------------------------------------------------------------- form
  const groups = [...new Set(schema.fields.map((f) => f.group))];
  function fieldHTML(f) {
    const help = f.help ? `<span class="help">${escapeHTML(f.help)}</span>` : "";
    if (f.widget === "select") {
      const opts = f.options.map((o) => `<option value="${escapeHTML(o.value)}">${escapeHTML(o.label)}</option>`).join("");
      return `<div class="field"><label for="f-${f.name}">${escapeHTML(f.label)}</label><select id="f-${f.name}" name="${f.name}">${opts}</select>${help}</div>`;
    }
    if (f.widget === "toggle") {
      return `<div class="field"><span class="small" style="font-weight:600;color:var(--ink-2)">${escapeHTML(f.label)}</span><label class="toggle"><input type="checkbox" id="f-${f.name}" name="${f.name}"><span class="track"></span><span>${help ? escapeHTML(f.help) : "yes"}</span></label></div>`;
    }
    if (f.widget === "date") {
      return `<div class="field"><label for="f-${f.name}">${escapeHTML(f.label)}</label><input type="date" id="f-${f.name}" name="${f.name}" min="2015-01-01" max="2030-12-31">${help}</div>`;
    }
    return `<div class="field"><label for="f-${f.name}">${escapeHTML(f.label)} <output id="o-${f.name}"></output></label><input type="range" id="f-${f.name}" name="${f.name}" min="${f.min}" max="${f.max}" step="${f.step}">${help}</div>`;
  }
  fieldsEl.innerHTML = groups
    .map((g) => `<div class="form-group"><h3>${escapeHTML(g)}</h3><div class="fields">${schema.fields.filter((f) => f.group === g).map(fieldHTML).join("")}</div></div>`)
    .join("");

  const paintRange = (input) => {
    const pct = ((input.value - input.min) / (input.max - input.min)) * 100;
    input.style.setProperty("--fill", `${pct}%`);
    const out = document.getElementById(`o-${input.name}`);
    if (out) out.textContent = input.name === "adr" ? `€${input.value}` : input.value;
  };
  function setBooking(booking) {
    schema.fields.forEach((f) => {
      const el = document.getElementById(`f-${f.name}`);
      if (f.widget === "toggle") el.checked = Boolean(booking[f.name]);
      else el.value = booking[f.name];
      if (f.widget === "range") paintRange(el);
    });
  }
  function readBooking() {
    const booking = {};
    schema.fields.forEach((f) => {
      const el = document.getElementById(`f-${f.name}`);
      if (f.widget === "toggle") booking[f.name] = el.checked ? 1 : 0;
      else if (f.widget === "range") booking[f.name] = Number(el.value);
      else booking[f.name] = el.value;
    });
    return booking;
  }

  // ---------------------------------------------------------------- gauge + verdict
  const gaugeFill = document.getElementById("gauge-fill");
  const needle = document.getElementById("gauge-needle");
  const thresholdTick = document.getElementById("gauge-threshold");
  const probEl = document.getElementById("prob-value");
  let shownProb = 0;
  function animateNumber(from, to) {
    const start = performance.now();
    const step = (now) => {
      const t = Math.min(1, (now - start) / 700);
      const v = from + (to - from) * (1 - Math.pow(1 - t, 3));
      probEl.textContent = `${(v * 100).toFixed(1)}%`;
      if (t < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }
  function renderVerdict() {
    thresholdTick.style.transform = `rotate(${(threshold - 0.5) * 180}deg)`;
    document.getElementById("threshold-label").textContent = threshold.toFixed(2);
    if (!last) return;
    const p = last.probability;
    gaugeFill.setAttribute("stroke-dasharray", `${p * 100} 100`);
    needle.style.transform = `rotate(${(p - 0.5) * 180}deg)`;
    animateNumber(shownProb, p);
    shownProb = p;
    const band = p >= threshold ? "high" : p < schema.risk_low ? "low" : "medium";
    const text = { high: "High risk: likely to cancel", medium: "Medium risk: likely to stay", low: "Low risk: likely to stay" }[band];
    const verdict = document.getElementById("verdict");
    verdict.className = `verdict ${band}`;
    verdict.textContent = text;
    const note = document.getElementById("actual-note");
    if (actual === null) note.textContent = "";
    else {
      const predicted = p >= threshold ? 1 : 0;
      note.innerHTML = `Real outcome of this test booking: <strong>${actual ? "canceled" : "not canceled"}</strong> ${predicted === actual ? '<span class="chip chip-good">model was right</span>' : '<span class="chip chip-bad">model was wrong</span>'}`;
    }
  }

  // ---------------------------------------------------------------- explanation + sweeps
  function renderContributions() {
    const list = last.contributions.filter((c) => Math.abs(c.delta) >= 0.002).slice(0, 9);
    const max = Math.max(0.05, ...list.map((c) => Math.abs(c.delta)));
    const el = document.getElementById("contributions");
    if (!list.length) {
      el.innerHTML = '<p class="small muted">This booking looks just like the typical booking: no single field moves the prediction.</p>';
      return;
    }
    el.innerHTML = list.map((c) => {
      const width = (Math.abs(c.delta) / max) * 50;
      const up = c.delta > 0;
      return `<div class="contrib-row" title="${escapeHTML(c.label)}: ${escapeHTML(c.value)} (typical ${escapeHTML(c.typical)})">
        <div class="name">${escapeHTML(c.label)} <small>= ${escapeHTML(c.value)}</small></div>
        <div class="contrib-bar"><i style="left:${up ? 50 : 50 - width}%;width:${width}%;background:${up ? css("--pink") : css("--teal")}"></i></div>
        <div class="delta ${up ? "up" : "down"}">${up ? "+" : "−"}${(Math.abs(c.delta) * 100).toFixed(1)}</div>
      </div>`;
    }).join("") + '<p class="small muted" style="margin:.3rem 0 0">Percentage points. Pink pushes towards "canceled", teal towards "kept". Fields interact, so the bars do not add up exactly.</p>';
  }
  function renderSweeps() {
    Object.entries(last.sweeps).forEach(([field, sweep]) => {
      const points = sweep.x.map((x, i) => ({ x, y: sweep.y[i] }));
      const currentY = last.probability;
      chart(`sweep-${field}`, {
        type: "line",
        data: { datasets: [
          { label: "Probability", data: points, borderColor: css("--primary"), backgroundColor: alpha(css("--primary"), 0.12), fill: "origin", tension: 0.25 },
          { label: "This booking", data: [{ x: sweep.current, y: currentY }], pointRadius: 7, pointHoverRadius: 8, pointBackgroundColor: css("--pink"), pointBorderColor: App.isDark() ? "#0B1020" : "#fff", pointBorderWidth: 2, showLine: false },
          { label: "Threshold", data: [{ x: sweep.x[0], y: threshold }, { x: sweep.x[sweep.x.length - 1], y: threshold }], borderColor: css("--ink-3"), borderDash: [5, 5], borderWidth: 1, pointRadius: 0 },
        ] },
        options: {
          maintainAspectRatio: false, parsing: false, animation: { duration: 300 },
          interaction: { mode: "nearest", intersect: false },
          scales: {
            x: { type: "linear", title: { display: true, text: field === "adr" ? "Daily rate (€)" : "Lead time (days)" } },
            y: { min: 0, max: 1, ticks: { callback: (v) => `${v * 100}%` } },
          },
          plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => ` ${c.dataset.label}: ${fmt.pct(c.raw.y)}` } } },
        },
      });
    });
  }

  // ---------------------------------------------------------------- live prediction
  let pending = null;
  let inflight = 0;
  async function predict() {
    const id = ++inflight;
    try {
      const response = await postJSON("/api/predict", readBooking());
      if (id !== inflight) return; // a newer request was sent
      last = response;
      renderVerdict();
      renderContributions();
      renderSweeps();
    } catch (error) {
      document.getElementById("verdict").className = "verdict medium";
      document.getElementById("verdict").textContent = error.message;
    }
  }
  const schedule = () => {
    clearTimeout(pending);
    pending = setTimeout(predict, 120);
  };
  formEl.addEventListener("input", (event) => {
    if (event.target.type === "range") paintRange(event.target);
    actual = null;
    schedule();
  });
  formEl.addEventListener("submit", (event) => event.preventDefault());
  document.getElementById("reset-btn").addEventListener("click", () => {
    actual = null;
    setBooking(schema.defaults);
    predict();
  });
  document.getElementById("random-btn").addEventListener("click", async () => {
    try {
      const booking = await (await fetch("/api/sample")).json();
      actual = booking.actual;
      setBooking(booking);
      predict();
    } catch (error) {
      showError(document.getElementById("actual-note"), error);
    }
  });

  // ---------------------------------------------------------------- threshold slider
  const slider = document.getElementById("threshold-slider");
  const testTable = final.test_threshold_table;
  const rowAt = (table, t) => table.reduce((best, r) => (Math.abs(r.threshold - t) < Math.abs(best.threshold - t) ? r : best), table[0]);
  function renderThreshold() {
    threshold = Number(slider.value);
    document.getElementById("threshold-out").textContent = threshold.toFixed(2);
    paintRange(slider);
    const r = rowAt(testTable, threshold);
    document.getElementById("th-precision").textContent = fmt.num(r.precision);
    document.getElementById("th-recall").textContent = fmt.num(r.recall);
    document.getElementById("th-f1").textContent = fmt.num(r.f1);
    document.getElementById("threshold-note").innerHTML = `At ${threshold.toFixed(2)}: <strong>${fmt.int(r.tp)}</strong> cancellations caught, <strong>${fmt.int(r.fn)}</strong> missed, <strong>${fmt.int(r.fp)}</strong> false alarms (test set, ${fmt.int(r.tp + r.fp + r.fn + r.tn)} bookings). Tuned threshold ${tuned.toFixed(2)} (${thresholdInfo.strategy === "max_f1" ? "max F1" : "recall target"} on out-of-fold predictions); a recall ≥ ${fmt.pct(thresholdInfo.recall_target, 0)} target would use ${thresholdInfo.recall_target_threshold.toFixed(2)}.`;
    const c = getChart("threshold-chart");
    if (c) {
      c.data.datasets[3].data = [{ x: threshold, y: 0 }, { x: threshold, y: 1 }];
      c.update("none");
    }
    renderVerdict();
    if (last) renderSweeps();
  }
  function thresholdChart() {
    const line = (key, color, name) => ({ label: name, data: testTable.map((r) => ({ x: r.threshold, y: r[key] })), borderColor: color, backgroundColor: color });
    chart("threshold-chart", {
      type: "line",
      data: { datasets: [
        line("precision", css("--primary"), "Precision"),
        line("recall", css("--pink"), "Recall"),
        line("f1", css("--marigold"), "F1"),
        { label: "Current", data: [{ x: threshold, y: 0 }, { x: threshold, y: 1 }], borderColor: css("--teal"), borderWidth: 2, borderDash: [4, 3], pointRadius: 0 },
      ] },
      options: {
        maintainAspectRatio: false, parsing: false,
        interaction: { mode: "index", intersect: false },
        scales: { x: { type: "linear", min: 0.05, max: 0.95, title: { display: true, text: "Threshold" } }, y: { min: 0, max: 1 } },
        plugins: { legend: { position: "bottom" }, tooltip: { filter: (i) => i.dataset.label !== "Current", callbacks: { title: (items) => `threshold ${items[0].raw.x.toFixed(2)}`, label: (c) => ` ${c.dataset.label}: ${fmt.num(c.raw.y)}` } } },
      },
    });
  }
  slider.value = tuned;
  slider.addEventListener("input", renderThreshold);
  document.getElementById("threshold-reset").addEventListener("click", () => {
    slider.value = tuned;
    renderThreshold();
  });

  // ---------------------------------------------------------------- business calculator
  const costFn = document.getElementById("cost-fn");
  const costFp = document.getElementById("cost-fp");
  costFn.value = setup.business_defaults.empty_room_cost;
  costFp.value = setup.business_defaults.overbooking_cost;
  let bestThreshold = tuned;
  const cost = (r, fnCost, fpCost) => r.fn * fnCost + r.fp * fpCost;
  function renderBusiness() {
    const fnCost = Math.max(0, Number(costFn.value) || 0);
    const fpCost = Math.max(0, Number(costFp.value) || 0);
    const oof = thresholdInfo.oof;
    const best = oof.reduce((a, r) => (cost(r, fnCost, fpCost) < cost(a, fnCost, fpCost) ? r : a), oof[0]);
    bestThreshold = best.threshold;
    const testRow = rowAt(testTable, bestThreshold);
    const n = testRow.tp + testRow.fp + testRow.fn + testRow.tn;
    const baseline = (testRow.tp + testRow.fn) * fnCost; // no model: never overbook
    const modelCost = cost(testRow, fnCost, fpCost);
    const per1000 = ((baseline - modelCost) / n) * 1000;
    document.getElementById("best-threshold").textContent = bestThreshold.toFixed(2);
    document.getElementById("best-detail").textContent = `catches ${fmt.pct(testRow.recall, 0)} of cancellations, ${fmt.pct(testRow.precision, 0)} of alarms correct`;
    document.getElementById("savings").textContent = per1000 >= 0 ? fmt.money(per1000) : `−${fmt.money(-per1000)}`;
    document.getElementById("savings-detail").textContent = `Test set: ${fmt.money(modelCost)} in mistakes vs ${fmt.money(baseline)} with no model (${fmt.int(n)} bookings).`;
    chart("cost-chart", {
      type: "line",
      data: { datasets: [
        { label: "Cost of mistakes (test)", data: testTable.map((r) => ({ x: r.threshold, y: cost(r, fnCost, fpCost) })), borderColor: css("--pink"), backgroundColor: alpha(css("--pink"), 0.1), fill: "origin" },
        { label: "No model", data: [{ x: 0.05, y: baseline }, { x: 0.95, y: baseline }], borderColor: css("--ink-3"), borderDash: [5, 5], borderWidth: 1, pointRadius: 0 },
        { label: "Best threshold", data: [{ x: bestThreshold, y: modelCost }], pointRadius: 7, pointBackgroundColor: css("--marigold"), showLine: false },
      ] },
      options: {
        maintainAspectRatio: false, parsing: false, animation: { duration: 250 },
        interaction: { mode: "nearest", intersect: false },
        scales: { x: { type: "linear", min: 0.05, max: 0.95, title: { display: true, text: "Threshold" } }, y: { min: 0, ticks: { callback: (v) => `€${(v / 1000).toFixed(0)}k` } } },
        plugins: { legend: { position: "bottom" }, tooltip: { callbacks: { label: (c) => ` ${c.dataset.label}: ${fmt.money(c.raw.y)}` } } },
      },
    });
  }
  costFn.addEventListener("input", renderBusiness);
  costFp.addEventListener("input", renderBusiness);
  document.getElementById("apply-threshold").addEventListener("click", () => {
    slider.value = bestThreshold;
    renderThreshold();
    slider.scrollIntoView({ behavior: "smooth", block: "center" });
  });

  // ---------------------------------------------------------------- start
  function renderStatic() {
    thresholdChart();
    renderThreshold();
    renderBusiness();
    if (last) {
      renderContributions();
      renderSweeps();
    }
  }
  setBooking(schema.defaults);
  renderStatic();
  predict();
  onTheme(renderStatic);
})();
