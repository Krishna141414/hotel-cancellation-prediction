// Model comparison page: leaderboard, metric charts, radar, sortable table, ROC/PR curves,
// confusion matrix, head-to-head mode, timings and tuning results. One model switcher drives it.
(async () => {
  "use strict";
  const { data, chart, getChart, css, fmt, escapeHTML, modelColor, alpha, onTheme, selectedModel, modelSwitcher, segmented, showError } = window.App;

  let models, final, setup, selection, country, state;
  try {
    [models, final, setup, selection, country] = await Promise.all([data("models"), data("final"), data("setup"), data("selection"), data("country_ablation")]);
  } catch (error) {
    return showError(document.getElementById("leaderboard"), error);
  }
  const metric = setup.primary_metric;
  const labels = setup.metric_labels;
  const metricKeys = setup.metrics;
  const byKey = Object.fromEntries(models.map((m) => [m.key, m]));
  state = { selected: selectedModel(models, final.key), source: "cv" };
  const value = (m, k, source) => (source === "cv" ? m.cv[k].mean : m[source][k]);

  // ---------------------------------------------------------------- leaderboard + why card
  function renderLeaderboard() {
    Widgets.leaderboard(document.getElementById("leaderboard"), models, {
      finalKey: final.key, metric, selected: state.selected, onSelect: select,
    });
  }
  function renderWhy() {
    const m = byKey[final.key];
    const best = byKey[selection.best_cv_key];
    const ranked = models.filter((x) => x.rank !== null).sort((a, b) => a.rank - b.rank);
    const runnerUp = ranked.find((x) => x.base !== m.base) || ranked[1];
    const reasons = [
      `<strong>${best.key === m.key ? "Best" : "Tied for best"} CV ${labels[metric]}</strong>: ${fmt.num(m.cv[metric].mean)} ± ${fmt.num(m.cv[metric].std)}` +
        (best.key === m.key ? " (rank 1)" : ` (within ${selection.rule.match(/[\d.]+/)[0]} of the best)`),
      m.cv[metric].mean >= runnerUp.cv[metric].mean
        ? `<strong>Beats ${escapeHTML(runnerUp.name)}</strong> by ${fmt.num(m.cv[metric].mean - runnerUp.cv[metric].mean, 4)} CV ${labels[metric]}`
        : `<strong>Ties with ${escapeHTML(runnerUp.name)}</strong>: ${fmt.num(runnerUp.cv[metric].mean - m.cv[metric].mean, 4)} lower CV ${labels[metric]}, well inside one fold std (${fmt.num(m.cv[metric].std, 4)})`,
      `<strong>Generalises</strong>: train–CV gap ${fmt.num(m.gap)} (${escapeHTML(m.diagnosis.label.toLowerCase())}); test ${fmt.num(m.test[metric])} matches CV`,
      `<strong>Deployable</strong>: ${fmt.mb(m.size_mb)} file, ${fmt.num(m.predict_ms_per_1k, 0)} ms per 1,000 predictions` +
        (byKey.rf_tuned || byKey.rf ? ` (Random Forest: ${fmt.mb((byKey.rf_tuned || byKey.rf).size_mb)})` : ""),
      `<strong>Explainable</strong>: ${escapeHTML(m.interpretability)}`,
    ];
    document.getElementById("why-card").innerHTML = `
      <span class="badge-selected">★ Why it won</span>
      <h2 style="margin:.6rem 0 .2rem">${escapeHTML(m.name)}</h2>
      <p class="small muted">${escapeHTML(selection.rule)}</p>
      <ul class="why-list">${reasons.map((r) => `<li><span>${r}</span></li>`).join("")}</ul>`;
  }

  // ---------------------------------------------------------------- grouped bars + radar
  function renderMetrics() {
    const shown = metricKeys;
    chart("metrics-chart", {
      type: "bar",
      data: {
        labels: shown.map((k) => labels[k]),
        datasets: models.map((m) => ({
          label: m.name,
          data: shown.map((k) => value(m, k, state.source)),
          backgroundColor: m.key === state.selected ? modelColor(m) : alpha(modelColor(m), 0.35),
          borderColor: css("--surface"),
          borderWidth: 1,
          borderRadius: 4,
        })),
      },
      options: {
        maintainAspectRatio: false,
        scales: { y: { min: 0, max: 1, ticks: { stepSize: 0.2 } }, x: { grid: { display: false } } },
        plugins: {
          legend: { position: "bottom", labels: { font: { size: 11 } } },
          tooltip: { callbacks: { label: (c) => ` ${c.dataset.label}: ${fmt.num(c.raw)}` } },
        },
      },
    });
    const radarModels = [...new Set([state.selected, final.key, "dummy"])].map((k) => byKey[k]);
    chart("radar-chart", {
      type: "radar",
      data: {
        labels: metricKeys.map((k) => labels[k]),
        datasets: radarModels.map((m) => ({
          label: m.name,
          data: metricKeys.map((k) => m.test[k]),
          borderColor: modelColor(m),
          backgroundColor: alpha(modelColor(m), m.key === state.selected ? 0.22 : 0.06),
          borderWidth: m.key === state.selected ? 2.5 : 1.5,
          pointRadius: 3,
          pointBackgroundColor: modelColor(m),
        })),
      },
      options: {
        maintainAspectRatio: false,
        scales: {
          r: {
            min: 0, max: 1, ticks: { stepSize: 0.25, backdropColor: "transparent", color: css("--ink-3") },
            grid: { color: css("--grid") }, angleLines: { color: css("--grid") },
            pointLabels: { color: css("--ink-2"), font: { size: 12, weight: 600 } },
          },
        },
        plugins: { legend: { position: "bottom" } },
      },
    });
  }

  // ---------------------------------------------------------------- sortable table
  const columns = [
    { key: "name", label: "Model", get: (m) => m.name, fmt: (v, m) => `<span class="dot" style="display:inline-block;width:9px;height:9px;border-radius:50%;background:${modelColor(m)};margin-right:.45rem"></span>${escapeHTML(v)}${m.key === final.key ? ' <span class="chip chip-warn">final</span>' : ""}` },
    { key: "rank", label: "Rank", get: (m) => m.rank ?? 99, fmt: (v) => (v === 99 ? "–" : v) },
    ...metricKeys.map((k) => ({ key: `cv_${k}`, label: `CV ${labels[k]}`, get: (m) => m.cv[k].mean, fmt: (v, m) => `${fmt.num(v)} <span class="std">±${fmt.num(m.cv[k].std)}</span>`, best: true })),
    { key: `train_${metric}`, label: `Train ${labels[metric]}`, get: (m) => m.train[metric], fmt: (v) => fmt.num(v) },
    ...metricKeys.map((k) => ({ key: `test_${k}`, label: `Test ${labels[k]}`, get: (m) => m.test[k], fmt: (v) => fmt.num(v), best: true })),
    { key: "gap", label: "Train–CV gap", get: (m) => m.gap, fmt: (v) => fmt.num(v) },
    { key: "diagnosis", label: "Diagnosis", get: (m) => m.diagnosis.label, fmt: (v) => Widgets.diagnosisChip(v) },
    { key: "fit", label: "Fit (s)", get: (m) => m.fit_time_s, fmt: (v) => fmt.num(v, 2) },
    { key: "pred", label: "ms / 1k", get: (m) => m.predict_ms_per_1k, fmt: (v) => fmt.num(v, 1) },
    { key: "size", label: "Size (MB)", get: (m) => m.size_mb, fmt: (v) => fmt.num(v, 2) },
  ];
  let sort = { key: "rank", dir: 1 };
  function renderTable() {
    const table = document.getElementById("comparison-table");
    const rows = [...models].sort((a, b) => {
      const col = columns.find((c) => c.key === sort.key);
      const va = col.get(a), vb = col.get(b);
      return (va > vb ? 1 : va < vb ? -1 : 0) * sort.dir;
    });
    const bests = {};
    columns.filter((c) => c.best).forEach((c) => (bests[c.key] = Math.max(...models.filter((m) => m.key !== "dummy").map(c.get))));
    table.innerHTML =
      `<thead><tr>${columns.map((c) => `<th class="sortable" data-key="${c.key}" ${c.key === sort.key ? `aria-sort="${sort.dir > 0 ? "ascending" : "descending"}"` : 'aria-sort="none"'}>${c.label}</th>`).join("")}</tr></thead>` +
      `<tbody>${rows.map((m) => `<tr class="${m.key === final.key ? "is-final" : ""}">${columns.map((c) => {
        const v = c.get(m);
        const cls = [c.key === "name" ? "" : "num", c.best && v === bests[c.key] ? "best" : ""].join(" ");
        return `<td class="${cls}">${c.fmt(v, m)}</td>`;
      }).join("")}</tr>`).join("")}</tbody>`;
  }
  document.getElementById("comparison-table").addEventListener("click", (event) => {
    const th = event.target.closest("th[data-key]");
    if (!th) return;
    sort = { key: th.dataset.key, dir: sort.key === th.dataset.key ? -sort.dir : th.dataset.key === "name" || th.dataset.key === "rank" ? 1 : -1 };
    renderTable();
  });

  // ---------------------------------------------------------------- with vs without country
  function renderCountry() {
    const w = country.with, wo = country.without;
    const row = (label, a, b) => {
      const diff = a - b;
      return `<tr><td>${label}</td><td class="num">${fmt.num(b)}</td><td class="num">${fmt.num(a)}</td><td class="num ${diff > 0 ? "up" : "down"}">${fmt.signed(diff)}</td></tr>`;
    };
    document.getElementById("country-table").innerHTML = `
      <thead><tr><th>${escapeHTML(country.model)}</th><th>Without country (deployed, ${wo.inputs} inputs)</th><th>With country (${w.inputs} inputs)</th><th>Difference</th></tr></thead>
      <tbody>
        ${row(`CV ${labels[metric]} (training folds)`, w.cv[metric].mean, wo.cv[metric].mean)}
        ${row(`CV ROC-AUC`, w.cv.roc_auc.mean, wo.cv.roc_auc.mean)}
        ${row(`Train ${labels[metric]}`, w.train[metric], wo.train[metric])}
        ${row(`Test ${labels[metric]}`, w.test[metric], wo.test[metric])}
        ${row("Test ROC-AUC", w.test.roc_auc, wo.test.roc_auc)}
        ${row(`Test F1 @ ${country.threshold.toFixed(2)}`, w.test.f1, wo.test.f1)}
        ${row(`Test recall @ ${country.threshold.toFixed(2)}`, w.test.recall, wo.test.recall)}
        ${row(`Test precision @ ${country.threshold.toFixed(2)}`, w.test.precision, wo.test.precision)}
      </tbody>`;
    const e = country.evidence;
    document.getElementById("country-why").innerHTML = `
      <div class="callout"><strong>Why we dropped it.</strong> ${escapeHTML(country.reason)}. Canceled bookings never reach check-in, so if a hotel system fills in or corrects nationality there, the column quietly tells the model who showed up.</div>
      <div class="score-pair"><div><b class="up">${fmt.pct(e.domestic_cancel_rate, 0)}</b><span>Portuguese guests cancel</span></div><div><b>${fmt.pct(e.foreign_cancel_rate, 0)}</b><span>everyone else</span></div></div>
      <p class="small">Portuguese guests are ${fmt.pct(e.domestic_share, 0)} of bookings but ${fmt.pct(e.domestic_share_of_cancellations, 0)} of cancellations. That's a strong pattern, and exactly what a check-in artefact would look like. We can't tell the two apart from the data, so we took the safer option: the model loses ${fmt.num(wo.cv[metric].mean - w.cv[metric].mean)} CV ${labels[metric]}, but every input it uses is known when the booking is made.</p>`;
  }

  // ---------------------------------------------------------------- ROC / PR
  function curveChart(id, kind) {
    const baseRate = byKey.dummy.test.average_precision;
    const datasets = models.map((m) => ({
      label: `${m.name} (${fmt.num(m.test[kind === "roc" ? "roc_auc" : "average_precision"])})`,
      data: m.curves[kind].map(([x, y]) => ({ x, y })),
      borderColor: m.key === state.selected ? modelColor(m) : alpha(modelColor(m), 0.45),
      borderWidth: m.key === state.selected ? 3.5 : 1.5,
      order: m.key === state.selected ? 0 : 1,
      fill: false,
      tension: 0,
    }));
    datasets.push({
      label: kind === "roc" ? "Random guess" : `Base rate (${fmt.pct(baseRate)})`,
      data: kind === "roc" ? [{ x: 0, y: 0 }, { x: 1, y: 1 }] : [{ x: 0, y: baseRate }, { x: 1, y: baseRate }],
      borderColor: css("--ink-3"), borderDash: [5, 5], borderWidth: 1, pointRadius: 0,
    });
    chart(id, {
      type: "line",
      data: { datasets },
      options: {
        maintainAspectRatio: false,
        parsing: false,
        interaction: { mode: "nearest", intersect: false },
        scales: {
          x: { type: "linear", min: 0, max: 1, title: { display: true, text: kind === "roc" ? "False positive rate" : "Recall" } },
          y: { min: 0, max: 1, title: { display: true, text: kind === "roc" ? "True positive rate" : "Precision" } },
        },
        plugins: {
          legend: { position: "bottom", labels: { font: { size: 11 } } },
          tooltip: { callbacks: { label: (c) => ` ${c.dataset.label.split(" (")[0]}: (${fmt.num(c.raw.x, 2)}, ${fmt.num(c.raw.y, 2)})` } },
        },
      },
    });
  }

  // ---------------------------------------------------------------- confusion + detail
  function renderSelected() {
    const m = byKey[state.selected];
    Widgets.confusionMatrix(document.getElementById("confusion"), m.confusion);
    document.getElementById("cm-caption").textContent = `${m.name}, test set at the default 0.50 threshold (the final model's tuned threshold is on the Predict page).`;
    const params = Object.entries(m.params).map(([k, v]) => `<tr><td>${escapeHTML(k)}</td><td class="num">${escapeHTML(v === null ? "None" : v)}</td></tr>`).join("");
    document.getElementById("model-detail").innerHTML = `
      <div class="card-head"><div><h2>${escapeHTML(m.name)}</h2><p>${escapeHTML(m.interpretability)}</p></div>${Widgets.diagnosisChip(m.diagnosis.label)}</div>
      <div class="metric-trio" style="margin-bottom:.8rem">
        <div><b>${fmt.num(m.cv[metric].mean)}</b><span>CV ${labels[metric]}</span></div>
        <div><b>${fmt.num(m.test[metric])}</b><span>Test ${labels[metric]}</span></div>
        <div><b>${fmt.num(m.test.f1)}</b><span>Test F1</span></div>
      </div>
      <p class="small">${escapeHTML(m.diagnosis.reason)}</p>
      <p class="small muted">Preprocessing: <strong>${m.preprocessing === "scaled" ? "log1p + standard scaling + one-hot" : "raw units + one-hot (trees)"}</strong> · trained on ${fmt.int(m.training_rows)} rows${m.training_rows < 60000 ? " (stratified sample: KNN compares every prediction to all stored rows)" : ""}.</p>
      <div class="table-wrap" style="max-height:190px;overflow:auto"><table><thead><tr><th>Hyperparameter</th><th>Value</th></tr></thead><tbody>${params}</tbody></table></div>`;
  }

  // ---------------------------------------------------------------- head-to-head
  const selA = document.getElementById("h2h-a");
  const selB = document.getElementById("h2h-b");
  const options = models.map((m) => `<option value="${m.key}">${escapeHTML(m.name)}</option>`).join("");
  selA.innerHTML = options;
  selB.innerHTML = options;
  selA.value = final.key;
  selB.value = byKey.rf_tuned ? "rf_tuned" : "rf";
  function renderH2H() {
    const a = byKey[selA.value], b = byKey[selB.value];
    chart("h2h-chart", {
      type: "bar",
      data: {
        labels: metricKeys.map((k) => labels[k]),
        datasets: [a, b].map((m, i) => ({
          label: m.name + (selA.value === selB.value && i ? " " : ""),
          data: metricKeys.map((k) => m.test[k]),
          backgroundColor: i === 0 ? css("--primary") : css("--pink"),
          borderRadius: 4,
        })),
      },
      options: {
        indexAxis: "y", maintainAspectRatio: false,
        scales: { x: { min: 0, max: 1 }, y: { grid: { display: false } } },
        plugins: { legend: { position: "bottom" }, tooltip: { callbacks: { label: (c) => ` ${c.dataset.label}: ${fmt.num(c.raw)}` } } },
      },
    });
    const wins = metricKeys.filter((k) => a.test[k] > b.test[k]).length;
    const losses = metricKeys.filter((k) => a.test[k] < b.test[k]).length;
    const row = (label, va, vb, lowerBetter = false) => {
      const aBetter = lowerBetter ? va < vb : va > vb;
      return `<tr><td>${label}</td><td class="num ${aBetter ? "best" : ""}">${va}</td><td class="num ${!aBetter && va !== vb ? "best" : ""}">${vb}</td></tr>`;
    };
    document.getElementById("h2h-summary").innerHTML = `
      <div class="callout"><strong>${escapeHTML(a.name)}</strong> wins <strong>${wins}</strong> of ${metricKeys.length} test metrics, loses ${losses}.</div>
      <div class="table-wrap"><table><thead><tr><th></th><th style="color:var(--primary-ink)">A</th><th style="color:var(--pink-ink)">B</th></tr></thead><tbody>
        ${row("CV " + labels[metric], fmt.num(a.cv[metric].mean), fmt.num(b.cv[metric].mean))}
        ${row("Train–CV gap", fmt.num(a.gap), fmt.num(b.gap), true)}
        ${row("Fit time (s)", fmt.num(a.fit_time_s, 1), fmt.num(b.fit_time_s, 1), true)}
        ${row("Size (MB)", fmt.num(a.size_mb, 2), fmt.num(b.size_mb, 2), true)}
      </tbody></table></div>`;
    Widgets.confusionMatrix(document.getElementById("h2h-cm-a"), a.confusion, { compact: true });
    Widgets.confusionMatrix(document.getElementById("h2h-cm-b"), b.confusion, { compact: true });
    document.getElementById("h2h-cm-a").insertAdjacentHTML("afterbegin", `<h3 style="color:var(--primary-ink)">A · ${escapeHTML(a.name)}</h3>`);
    document.getElementById("h2h-cm-b").insertAdjacentHTML("afterbegin", `<h3 style="color:var(--pink-ink)">B · ${escapeHTML(b.name)}</h3>`);
  }
  selA.addEventListener("change", renderH2H);
  selB.addEventListener("change", renderH2H);

  // ---------------------------------------------------------------- timings
  function costChart(id, getter, unit) {
    const ordered = models.filter((m) => m.key !== "dummy");
    chart(id, {
      type: "bar",
      data: {
        labels: ordered.map((m) => m.name.replace(" (tuned)", " ✦")),
        datasets: [{ data: ordered.map(getter), backgroundColor: ordered.map((m) => (m.key === state.selected ? modelColor(m) : alpha(modelColor(m), 0.55))), borderRadius: 4 }],
      },
      options: {
        indexAxis: "y", maintainAspectRatio: false,
        scales: { x: { type: "logarithmic", ticks: { callback: (v) => (Number.isInteger(Math.log10(v)) ? v : "") } }, y: { grid: { display: false }, ticks: { font: { size: 10.5 } } } },
        plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => ` ${fmt.num(c.raw, 2)} ${unit}` } } },
      },
    });
  }

  // ---------------------------------------------------------------- tuning
  function renderTuning() {
    const container = document.getElementById("tuning");
    const tuned = models.filter((m) => m.search);
    container.innerHTML = tuned.map((m) => `
      <div>
        <h3>${escapeHTML(m.name)} <span class="chip">${m.search.n_iter} settings × ${m.search.folds} folds = ${m.search.fits} fits, ${fmt.num(m.search.search_time_s, 0)} s</span></h3>
        <div class="chart-box short"><canvas id="tune-${m.key}" aria-label="Tuning candidates"></canvas></div>
        <p class="small" style="margin-top:.6rem"><span class="chip chip-warn">★ chosen</span> ${Object.entries(m.search.best_params).map(([k, v]) => `<code>${escapeHTML(k)}=${escapeHTML(v)}</code>`).join(" ")}</p>
        <p class="small muted">Rule: ${escapeHTML(m.search.rule)}. Top CV setting ${fmt.num(m.search.top_cv)}; chosen ${fmt.num(m.cv[metric].mean)} with a smaller train–CV gap.</p>
      </div>`).join("");
    tuned.forEach((m) => {
      chart(`tune-${m.key}`, {
        type: "scatter",
        data: {
          datasets: [{
            label: "Sampled settings",
            data: m.search.candidates.map((c, i) => ({ x: c.cv, y: c.train, i })),
            backgroundColor: m.search.candidates.map((c) => (c.chosen ? css("--marigold-bright") : alpha(modelColor(m), 0.6))),
            borderColor: m.search.candidates.map((c) => (c.chosen ? css("--ink") : "transparent")),
            borderWidth: 1.5,
            pointRadius: m.search.candidates.map((c) => (c.chosen ? 9 : 5)),
            pointHoverRadius: 8,
          }],
        },
        options: {
          maintainAspectRatio: false,
          scales: { x: { title: { display: true, text: `CV ${labels[metric]}` } }, y: { title: { display: true, text: `Train ${labels[metric]}` } } },
          plugins: {
            legend: { display: false },
            tooltip: { callbacks: { label: (c) => {
              const cand = m.search.candidates[c.raw.i];
              return [` CV ${fmt.num(cand.cv)} · train ${fmt.num(cand.train)} · gap ${fmt.num(cand.train - cand.cv)}${cand.chosen ? "  ★ chosen" : c.raw.i === 0 ? "  (top CV)" : ""}`, ...Object.entries(cand.params).map(([k, v]) => `   ${k}: ${v}`)];
            } } },
          },
        },
      });
    });
  }

  // ---------------------------------------------------------------- wiring
  function select(key) {
    state.selected = key;
    App.store.set("selectedModel", key);
    document.querySelectorAll("#model-switcher .pill").forEach((b) => b.setAttribute("aria-pressed", b.dataset.key === key));
    renderSelectedViews();
  }
  function renderSelectedViews() {
    renderLeaderboard();
    renderMetrics();
    curveChart("roc-chart", "roc");
    curveChart("pr-chart", "pr");
    renderSelected();
    costChart("fit-chart", (m) => m.fit_time_s, "s");
    costChart("predict-chart", (m) => m.predict_ms_per_1k, "ms / 1k rows");
    costChart("size-chart", (m) => m.size_mb, "MB");
  }
  function renderAll() {
    renderSelectedViews();
    renderWhy();
    renderTable();
    renderH2H();
    renderTuning();
    renderCountry();
  }
  modelSwitcher(document.getElementById("model-switcher"), models, { selected: state.selected, onChange: select });
  segmented(document.getElementById("metric-source"), (v) => {
    state.source = v;
    renderMetrics();
  });
  renderAll();
  onTheme(renderAll);
  void getChart;
})();
