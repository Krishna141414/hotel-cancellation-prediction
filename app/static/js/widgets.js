// Reusable widgets shared by several pages: leaderboard and confusion matrix.
(() => {
  "use strict";
  const { fmt, escapeHTML, modelColor, css } = window.App;

  const diagnosisChip = (label) => {
    const cls = label.includes("verfit") ? "chip-bad" : label === "Underfitting" ? "chip-warn" : label === "Good fit" ? "chip-good" : "";
    return `<span class="chip ${cls}">${escapeHTML(label)}</span>`;
  };

  // Kaggle-style leaderboard ranked by mean CV score (primary metric).
  function leaderboard(container, models, { finalKey, metric, limit, selected, onSelect } = {}) {
    const ranked = models.filter((m) => m.rank !== null).sort((a, b) => a.rank - b.rank);
    const rows = (limit ? ranked.slice(0, limit) : ranked).concat(limit ? [] : models.filter((m) => m.rank === null));
    const best = ranked[0].cv[metric].mean;
    const medal = (rank) => (rank === 1 ? "gold" : rank === 2 ? "silver" : rank === 3 ? "bronze" : "");
    container.innerHTML = rows
      .map((m) => {
        const score = m.cv[metric].mean;
        const width = Math.max(3, (score / best) * 100);
        return `<div class="lb-row ${m.key === finalKey ? "is-final" : ""}" role="button" tabindex="0" data-key="${m.key}" aria-pressed="${m.key === selected}">
          <div class="medal ${medal(m.rank)}">${m.rank ?? "–"}</div>
          <div class="lb-name"><span class="text">${escapeHTML(m.name)}</span>${m.key === finalKey ? '<span class="badge-selected">★ selected</span>' : ""}</div>
          <div class="lb-bar"><i data-width="${width}" style="background:${modelColor(m)}"></i></div>
          <div class="lb-score" title="Mean CV ± std">${fmt.num(score)}<span class="std"> ±${fmt.num(m.cv[metric].std, 3)}</span></div>
          <div class="lb-extra">${diagnosisChip(m.diagnosis.label)}</div>
        </div>`;
      })
      .join("");
    requestAnimationFrame(() =>
      container.querySelectorAll(".lb-bar i").forEach((bar) => {
        bar.style.width = `${bar.dataset.width}%`;
      })
    );
    if (onSelect) {
      const handler = (event) => {
        const row = event.target.closest(".lb-row");
        if (!row || (event.type === "keydown" && event.key !== "Enter" && event.key !== " ")) return;
        event.preventDefault();
        onSelect(row.dataset.key);
      };
      container.onclick = handler;
      container.onkeydown = handler;
    } else {
      container.onclick = () => (window.location.href = "/models");
    }
  }

  // Confusion matrix as an HTML grid: cells tinted by share of all bookings.
  function confusionMatrix(container, matrix, { compact = false } = {}) {
    const [[tn, fp], [fn, tp]] = matrix;
    const total = tn + fp + fn + tp;
    const cell = (value, good, label) => {
      const share = value / total;
      const color = good ? css("--teal") : css("--pink");
      const strength = Math.min(0.85, 0.12 + share * 1.6);
      return `<div class="cell" style="background:${App.alpha(color, strength)};color:${strength > 0.5 ? "#fff" : "var(--ink)"}" title="${label}: ${fmt.int(value)} bookings">
        <b>${fmt.int(value)}</b><span>${fmt.pct(share)}${compact ? "" : ` · ${label}`}</span></div>`;
    };
    container.innerHTML = `<div class="cm">
      <div></div><div class="hdr">Predicted<br>not canceled</div><div class="hdr">Predicted<br>canceled</div>
      <div class="hdr">Actual<br>not canceled</div>${cell(tn, true, "true negative")}${cell(fp, false, "false alarm")}
      <div class="hdr">Actual<br>canceled</div>${cell(fn, false, "missed")}${cell(tp, true, "caught")}
    </div>`;
  }

  window.Widgets = { leaderboard, confusionMatrix, diagnosisChip };
})();
