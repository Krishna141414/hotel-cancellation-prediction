// Explainability page: importances, plain-English insights, an interactive D3 decision tree,
// the "falling bookings" animation and the 3-D PCA booking space (Plotly, lazy-loaded).
(async () => {
  "use strict";
  const { data, chart, css, fmt, escapeHTML, alpha, onTheme, segmented, showError } = window.App;
  const PLOTLY_URL = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js";

  let final, setup, permutation, treeImportance, insights, coefs, tree, falling;
  try {
    [final, setup, permutation, treeImportance, insights, coefs, tree, falling] = await Promise.all(
      ["final", "setup", "permutation", "tree_importance", "insights", "logreg_coefficients", "tree", "falling"].map(data)
    );
  } catch (error) {
    return showError(document.getElementById("insights"), error);
  }
  const label = setup.primary_label;
  const TOP = 15;

  // Teal (0% canceled) -> neutral -> pink (100%), matching the CSS legend ramp.
  const rateColor = (t) => d3.interpolateRgbBasis([css("--teal"), css("--surface-3"), css("--pink")])(t);

  // ---------------------------------------------------------------- importances
  function hbar(id, rows, { valueKey, errorKey, colors, xTitle }) {
    chart(id, {
      type: "bar",
      data: {
        labels: rows.map((r) => r.feature),
        datasets: [{ data: rows.map((r) => r[valueKey]), backgroundColor: colors, borderRadius: 4, barPercentage: 0.8 }],
      },
      options: {
        indexAxis: "y", maintainAspectRatio: false,
        scales: { x: { title: { display: true, text: xTitle } }, y: { grid: { display: false }, ticks: { font: { size: 11 } } } },
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: (c) => ` ${fmt.num(c.raw, 4)}${errorKey ? ` ± ${fmt.num(rows[c.dataIndex][errorKey], 4)}` : ""}` } },
        },
      },
    });
  }
  function renderImportances() {
    const top = permutation.slice(0, TOP);
    document.getElementById("perm-caption").textContent = `${final.name}: drop in test ${label} when each input is shuffled (mean of 5 repeats). Categorical inputs are shuffled as one column.`;
    hbar("perm-chart", top, {
      valueKey: "importance_mean", errorKey: "importance_std",
      colors: top.map((_, i) => (i < 3 ? css("--pink") : alpha(css("--pink"), 0.55))), xTitle: `Mean drop in ${label}`,
    });
    const coefTop = [...coefs].sort((a, b) => b.coef - a.coef);
    chart("coef-chart", {
      type: "bar",
      data: { labels: coefTop.map((c) => c.feature), datasets: [{ data: coefTop.map((c) => c.coef), backgroundColor: coefTop.map((c) => (c.coef > 0 ? css("--pink") : css("--teal"))), borderRadius: 4 }] },
      options: {
        indexAxis: "y", maintainAspectRatio: false,
        scales: { x: { title: { display: true, text: "Weight (log-odds per std / per category)" } }, y: { grid: { display: false }, ticks: { font: { size: 10.5 } } } },
        plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => ` weight ${fmt.signed(c.raw, 3)}` } } },
      },
    });
  }
  const treeKeys = Object.keys(treeImportance);
  let treeKey = treeKeys[treeKeys.length - 1];
  const toggle = document.getElementById("tree-imp-toggle");
  toggle.innerHTML = treeKeys.map((k) => `<button type="button" data-value="${k}" aria-pressed="${k === treeKey}">${k.startsWith("rf") ? "Random Forest" : "Decision Tree"}</button>`).join("");
  segmented(toggle, (v) => {
    treeKey = v;
    renderTreeImportance();
  });
  function renderTreeImportance() {
    const rows = treeImportance[treeKey].slice(0, TOP);
    const color = treeKey.startsWith("rf") ? css("--teal") : css("--marigold");
    hbar("tree-imp-chart", rows, { valueKey: "importance", colors: rows.map(() => color), xTitle: "Importance (sums to 1)" });
  }
  document.getElementById("insights").innerHTML = insights
    .slice(0, 6)
    .map((item, i) => `<div class="insight"><span class="rank">${i + 1}</span><p><b>${escapeHTML(item.feature)}</b><br>${escapeHTML(item.text)}</p></div>`)
    .join("");

  // ---------------------------------------------------------------- interactive tree (D3)
  const svg = d3.select("#tree-svg");
  const NODE_W = 178, NODE_H = 62;
  const zoomLayer = svg.append("g");
  const zoom = d3.zoom().scaleExtent([0.25, 2.5]).on("zoom", (event) => zoomLayer.attr("transform", event.transform));
  svg.call(zoom);
  const root = d3.hierarchy(tree, (d) => d.children);
  // Start with 3 levels visible so the text is readable; the 4th unfolds on click / "Expand all".
  root.each((d) => {
    d._children = null;
    if (d.depth === 2 && d.children) {
      d._children = d.children;
      d.children = null;
    }
  });
  const layout = d3.tree().nodeSize([NODE_W + 18, NODE_H + 58]);
  function fitView(duration = 600) {
    const box = svg.node().getBoundingClientRect();
    const nodes = root.descendants();
    const xs = nodes.map((d) => d.x), ys = nodes.map((d) => d.y);
    const width = Math.max(...xs) - Math.min(...xs) + NODE_W + 40;
    const height = Math.max(...ys) + NODE_H + 40;
    const scale = Math.min(1.1, box.width / width, box.height / height);
    const tx = box.width / 2 - ((Math.max(...xs) + Math.min(...xs)) / 2) * scale;
    svg.transition().duration(duration).call(zoom.transform, d3.zoomIdentity.translate(tx, 20).scale(scale));
  }
  function nodeText(d) {
    const n = d.data;
    const lines = [];
    if (n.question) lines.push({ cls: "q", text: n.question.length > 30 ? `${n.question.slice(0, 29)}…` : n.question });
    else lines.push({ cls: "q", text: `Leaf → ${n.predicted ? "canceled" : "not canceled"}${n.truncated ? " (deeper)" : ""}` });
    lines.push({ cls: "meta", text: `${fmt.int(n.rows)} bookings · ${fmt.pct(n.cancel_rate, 0)} canceled` });
    return lines;
  }
  function updateTree(source, duration = 450) {
    layout(root);
    const nodes = root.descendants();
    const links = root.links();
    const t = svg.transition().duration(duration);
    const link = zoomLayer.selectAll("path.link").data(links, (d) => d.target.data.id);
    const diagonal = (s, d) => `M${s.x},${s.y + NODE_H / 2} C${s.x},${(s.y + d.y) / 2 + NODE_H / 2} ${d.x},${(s.y + d.y) / 2} ${d.x},${d.y - NODE_H / 2}`;
    link.enter().append("path").attr("class", "link")
      .attr("d", () => diagonal(source, source))
      .attr("stroke-width", (d) => 1 + 9 * Math.sqrt(d.target.data.rows / tree.rows))
      .merge(link).transition(t).attr("d", (d) => diagonal(d.source, d.target));
    link.exit().transition(t).attr("d", () => diagonal(source, source)).remove();
    const branch = zoomLayer.selectAll("text.branch").data(links, (d) => d.target.data.id);
    branch.enter().append("text").attr("class", "branch").attr("text-anchor", "middle")
      .text((d) => d.target.data.branch)
      .attr("x", source.x).attr("y", source.y)
      .merge(branch).transition(t)
      .attr("x", (d) => (d.source.x + d.target.x) / 2 + (d.target.x < d.source.x ? -10 : 10))
      .attr("y", (d) => (d.source.y + d.target.y) / 2 + 22);
    branch.exit().remove();
    const node = zoomLayer.selectAll("g.node").data(nodes, (d) => d.data.id);
    const enter = node.enter().append("g").attr("class", "node")
      .attr("transform", `translate(${source.x - NODE_W / 2},${source.y - NODE_H / 2})`)
      .on("click", (_, d) => {
        if (d.children) { d._children = d.children; d.children = null; }
        else if (d._children) { d.children = d._children; d._children = null; }
        updateTree(d);
      });
    enter.append("rect").attr("width", NODE_W).attr("height", NODE_H);
    enter.append("rect").attr("class", "bar-bg").attr("x", 10).attr("y", NODE_H - 13).attr("width", NODE_W - 20).attr("height", 5).attr("rx", 2.5);
    enter.append("rect").attr("class", "bar").attr("x", 10).attr("y", NODE_H - 13).attr("height", 5).attr("rx", 2.5);
    enter.append("title");
    const merged = enter.merge(node);
    merged.classed("collapsed", (d) => !!d._children || !!d.data.truncated);
    merged.select("rect").attr("fill", (d) => alpha(d3.color(rateColor(d.data.cancel_rate)).formatHex(), 0.28)).attr("stroke", (d) => rateColor(d.data.cancel_rate));
    merged.select(".bar-bg").attr("fill", css("--surface-3"));
    merged.select(".bar").attr("fill", css("--pink")).attr("width", (d) => (NODE_W - 20) * d.data.cancel_rate);
    merged.select("title").text((d) => `${d.data.question || "Leaf"}\n${fmt.int(d.data.rows)} training bookings, ${fmt.pct(d.data.cancel_rate)} canceled\nMajority (class-weighted): ${d.data.predicted ? "canceled" : "not canceled"}`);
    merged.selectAll("text.line").remove();
    merged.each(function (d) {
      nodeText(d).forEach((line, i) => {
        d3.select(this).append("text").attr("class", `line ${line.cls}`).attr("x", 10).attr("y", 20 + i * 16).text(line.text);
      });
    });
    merged.transition(t).attr("transform", (d) => `translate(${d.x - NODE_W / 2},${d.y - NODE_H / 2})`);
    node.exit().transition(t).attr("transform", `translate(${source.x - NODE_W / 2},${source.y - NODE_H / 2})`).style("opacity", 0).remove();
  }
  root.x0 = 0; root.y0 = 0;
  updateTree(root, 0);
  fitView(0);
  document.getElementById("tree-reset").addEventListener("click", () => fitView());
  document.getElementById("tree-expand").addEventListener("click", () => {
    root.each((d) => { if (d._children) { d.children = d._children; d._children = null; } });
    updateTree(root);
    setTimeout(() => fitView(), 480);
  });

  // ---------------------------------------------------------------- falling bookings
  const fsvg = d3.select("#falling-svg");
  const fallRoot = d3.hierarchy(falling.tree, (d) => d.children);
  let fallRun = 0;
  function renderFalling() {
    const run = ++fallRun;
    fsvg.selectAll("circle").interrupt();
    fsvg.selectAll("*").remove();
    const box = fsvg.node().getBoundingClientRect();
    const W = Math.max(320, box.width), H = Math.max(360, box.height);
    const compact = W < 640;
    const bucketH = 80;
    d3.tree().size([W - 40, H - bucketH - 70])(fallRoot);
    const nodes = fallRoot.descendants();
    const pos = Object.fromEntries(nodes.map((d) => [d.data.id, { x: d.x + 20, y: d.y + 24 }]));
    fsvg.append("g").selectAll("path").data(fallRoot.links()).join("path").attr("class", "edge")
      .attr("stroke-width", (d) => 1 + 7 * Math.sqrt(d.target.data.rows / falling.tree.rows))
      .attr("d", (d) => `M${pos[d.source.data.id].x},${pos[d.source.data.id].y} L${pos[d.target.data.id].x},${pos[d.target.data.id].y}`);
    const leafY = d3.max(nodes, (d) => pos[d.data.id].y);
    const bucketY = leafY + 36;
    const buckets = [
      { cls: 0, x: 20, label: "Predicted: not canceled", color: css("--teal") },
      { cls: 1, x: W / 2 + 6, label: "Predicted: canceled", color: css("--pink") },
    ];
    const bw = W / 2 - 26;
    const bg = fsvg.append("g").attr("class", "bucket");
    buckets.forEach((b) => {
      bg.append("rect").attr("x", b.x).attr("y", bucketY).attr("width", bw).attr("height", bucketH).attr("fill", alpha(b.color, 0.12)).attr("stroke", b.color);
      bg.append("text").attr("x", b.x + 10).attr("y", bucketY + 18).attr("fill", b.color).text(b.label);
    });
    const ng = fsvg.append("g").selectAll("g").data(nodes).join("g").attr("class", "node")
      .attr("transform", (d) => `translate(${pos[d.data.id].x},${pos[d.data.id].y})`);
    const textFor = (d) => (d.data.question ? d.data.question : d.data.predicted ? "→ canceled" : "→ kept");
    ng.append("rect").attr("x", (d) => -(compact && !d.data.question ? 30 : Math.min(92, textFor(d).length * 3.3 + 10))).attr("y", -11)
      .attr("width", (d) => 2 * (compact && !d.data.question ? 30 : Math.min(92, textFor(d).length * 3.3 + 10))).attr("height", 22)
      .attr("stroke", (d) => (d.data.question ? css("--border") : rateColor(d.data.cancel_rate)));
    ng.append("text").attr("text-anchor", "middle").attr("dy", 4)
      .text((d) => {
        const t = textFor(d);
        const max = compact ? (d.data.question ? 16 : 9) : 28;
        return t.length > max ? `${t.slice(0, max - 1)}…` : t;
      })
      .append("title").text((d) => `${d.data.question || "leaf"} · ${fmt.int(d.data.rows)} training bookings, ${fmt.pct(d.data.cancel_rate)} canceled`);

    const samples = falling.samples;
    document.getElementById("fall-total").textContent = samples.length;
    const counts = { landed: 0, correct: 0, 0: 0, 1: 0 };
    const cols = Math.max(8, Math.floor((bw - 20) / 9));
    const dots = fsvg.append("g");
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const stepMs = reduce ? 0 : 340;
    samples.forEach((s, i) => {
      const path = s.path.map((id) => pos[id]);
      const slot = counts[s.predicted]++;
      const b = buckets[s.predicted];
      const target = { x: b.x + 12 + (slot % cols) * 9, y: bucketY + bucketH - 10 - Math.floor(slot / cols) * 9 };
      const dot = dots.append("circle").attr("r", 3.6).attr("cx", path[0].x).attr("cy", path[0].y - 22)
        .attr("fill", s.label ? css("--pink") : css("--teal")).attr("stroke", App.isDark() ? "#0B1020" : "#fff").attr("stroke-width", 1).attr("opacity", 0);
      let tr = dot.transition().delay(reduce ? 0 : i * 70).duration(reduce ? 0 : 150).attr("opacity", 1);
      path.forEach((p) => {
        tr = tr.transition().duration(stepMs).ease(d3.easeCubicInOut).attr("cx", p.x + (Math.random() - 0.5) * 8).attr("cy", p.y);
      });
      tr.transition().duration(reduce ? 0 : 420).ease(d3.easeBounceOut).attr("cx", target.x).attr("cy", target.y)
        .on("end", () => {
          if (run !== fallRun) return;
          counts.landed += 1;
          if (s.label === s.predicted) counts.correct += 1;
          document.getElementById("fall-landed").textContent = counts.landed;
          document.getElementById("fall-correct").textContent = counts.correct;
          document.getElementById("fall-acc").textContent = fmt.pct(counts.correct / counts.landed);
        });
    });
  }
  document.getElementById("fall-replay").addEventListener("click", renderFalling);
  // Start the animation when it scrolls into view (once).
  const fallObserver = new IntersectionObserver((entries) => {
    if (entries.some((e) => e.isIntersecting)) {
      renderFalling();
      fallObserver.disconnect();
    }
  }, { threshold: 0.3 });
  fallObserver.observe(document.getElementById("falling-svg"));

  // ---------------------------------------------------------------- 3-D PCA (Plotly, lazy)
  let pca = null;
  function plotLayout() {
    const axis = (title) => ({ title: { text: title, font: { size: 11, color: css("--ink-2") } }, gridcolor: css("--grid"), zerolinecolor: css("--border"), color: css("--ink-3"), backgroundcolor: "rgba(0,0,0,0)", showbackground: false });
    return {
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: "Inter, sans-serif", color: css("--ink-2") },
      margin: { l: 0, r: 0, t: 10, b: 0 },
      legend: { orientation: "h", x: 0, y: 1.02, font: { size: 12 } },
      scene: { xaxis: axis("PC1"), yaxis: axis("PC2"), zaxis: axis("PC3"), camera: { eye: { x: 1.5, y: 1.4, z: 0.9 } } },
    };
  }
  function renderPCA() {
    if (!pca || !window.Plotly) return;
    const traces = [0, 1].map((cls) => {
      const pts = pca.points.filter((p) => p[3] === cls);
      return {
        type: "scatter3d", mode: "markers", name: cls ? "Canceled" : "Not canceled",
        x: pts.map((p) => p[0]), y: pts.map((p) => p[1]), z: pts.map((p) => p[2]),
        marker: { size: 2.6, color: cls ? css("--pink") : css("--teal"), opacity: 0.75 },
        hovertemplate: `${cls ? "Canceled" : "Not canceled"}<br>PC1 %{x:.2f}<br>PC2 %{y:.2f}<br>PC3 %{z:.2f}<extra></extra>`,
      };
    });
    const el = document.getElementById("pca-plot");
    el.innerHTML = "";
    window.Plotly.react(el, traces, plotLayout(), { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["toImage"] });
  }
  function loadPCA() {
    const script = document.createElement("script");
    script.src = PLOTLY_URL;
    script.onload = async () => {
      try {
        pca = await data("pca");
        const ev = pca.explained.map((v) => fmt.pct(v, 1)).join(" / ");
        document.getElementById("pca-note").textContent = `Variance explained by PC1 / PC2 / PC3: ${ev}. Strongest loadings: ${pca.loadings.map((l, i) => `PC${i + 1} ${l.join(", ")}`).join(" · ")}. The classes overlap a lot in 3 linear dimensions, which is why non-linear models win.`;
        renderPCA();
      } catch (error) {
        showError(document.getElementById("pca-plot"), error);
      }
    };
    script.onerror = () => showError(document.getElementById("pca-plot"), new Error("Could not load the 3-D chart library (offline?)."));
    document.head.appendChild(script);
  }
  const pcaObserver = new IntersectionObserver((entries) => {
    if (entries.some((e) => e.isIntersecting)) {
      loadPCA();
      pcaObserver.disconnect();
    }
  }, { rootMargin: "300px" });
  pcaObserver.observe(document.getElementById("pca-section"));

  // ---------------------------------------------------------------- wiring
  function renderAll() {
    renderImportances();
    renderTreeImportance();
    updateTree(root, 0);
    renderPCA();
  }
  renderAll();
  onTheme(() => {
    renderAll();
    renderFalling();
  });
  let resizeTimer;
  let lastWidth = window.innerWidth;
  window.addEventListener("resize", () => {
    if (window.innerWidth === lastWidth) return; // ignore mobile address-bar height changes
    lastWidth = window.innerWidth;
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      renderFalling();
      fitView(0);
    }, 250);
  });
})();
