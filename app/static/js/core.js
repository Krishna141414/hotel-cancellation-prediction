// Shared front-end layer: theme, data fetching, formatting, Chart.js theme, model switcher,
// count-up numbers and reveal-on-scroll. Every page script uses window.App.
(() => {
  "use strict";
  const root = document.documentElement;
  const media = window.matchMedia("(prefers-color-scheme: dark)");
  const themeListeners = [];

  const isDark = () => (root.dataset.theme || (media.matches ? "dark" : "light")) === "dark";
  const css = (name) => getComputedStyle(root).getPropertyValue(name).trim();

  // ------------------------------------------------------------------ storage
  const store = {
    get(key, fallback = null) {
      try {
        const value = localStorage.getItem(key);
        return value === null ? fallback : value;
      } catch (_) {
        return fallback;
      }
    },
    set(key, value) {
      try {
        localStorage.setItem(key, value);
      } catch (_) {
        /* storage unavailable: choice lasts for this page only */
      }
    },
  };

  // ------------------------------------------------------------------ theme
  function announceTheme() {
    applyChartTheme();
    themeListeners.forEach((fn) => fn());
  }
  function onTheme(fn) {
    themeListeners.push(fn);
  }
  document.getElementById("theme-toggle").addEventListener("click", () => {
    const next = isDark() ? "light" : "dark";
    root.dataset.theme = next;
    store.set("theme", next);
    announceTheme();
  });
  media.addEventListener("change", () => {
    if (!root.dataset.theme) announceTheme();
  });

  // ------------------------------------------------------------------ data
  const cache = {};
  function data(section) {
    if (!cache[section]) {
      // ?v=<content hash> lets the browser cache data but never serve it stale after a redeploy.
      cache[section] = fetch(`/api/data/${section}?v=${document.body.dataset.version || ""}`).then((response) => {
        if (!response.ok) throw new Error(`Could not load ${section} (${response.status})`);
        return response.json();
      });
    }
    return cache[section];
  }
  async function postJSON(url, body) {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
    return payload;
  }

  // ------------------------------------------------------------------ formatting
  const fmt = {
    num: (v, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d)),
    pct: (v, d = 1) => (v === null || v === undefined ? "–" : `${(v * 100).toFixed(d)}%`),
    int: (v) => Number(v).toLocaleString("en-US"),
    signed: (v, d = 3) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(d)}`,
    secs: (v) => (v < 1 ? `${(v * 1000).toFixed(0)} ms` : `${v.toFixed(1)} s`),
    mb: (v) => (v < 1 ? `${(v * 1000).toFixed(0)} KB` : `${v.toFixed(1)} MB`),
    money: (v) => `€${Math.round(v).toLocaleString("en-US")}`,
  };
  const escapeHTML = (s) =>
    String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

  // ------------------------------------------------------------------ colours
  const modelColor = (model) => (isDark() ? model.color_dark : model.color);
  function alpha(hex, a) {
    const n = parseInt(hex.replace("#", ""), 16);
    return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
  }
  const classColor = (label) => (label === 1 ? css("--pink") : css("--teal"));

  // ------------------------------------------------------------------ charts
  const charts = {};
  function applyChartTheme() {
    if (!window.Chart) return;
    Chart.defaults.font.family = "Inter, system-ui, sans-serif";
    Chart.defaults.font.size = 12;
    Chart.defaults.color = css("--ink-2");
    Chart.defaults.borderColor = css("--grid");
    Chart.defaults.animation.duration = 650;
    Chart.defaults.plugins.legend.labels.usePointStyle = true;
    Chart.defaults.plugins.legend.labels.boxWidth = 8;
    Chart.defaults.plugins.legend.labels.boxHeight = 8;
    Chart.defaults.plugins.tooltip.backgroundColor = isDark() ? "#0B1020" : "#141A33";
    Chart.defaults.plugins.tooltip.titleColor = "#EEF1FF";
    Chart.defaults.plugins.tooltip.bodyColor = "#D7DCF2";
    Chart.defaults.plugins.tooltip.borderColor = isDark() ? "#262F55" : "#141A33";
    Chart.defaults.plugins.tooltip.borderWidth = 1;
    Chart.defaults.plugins.tooltip.padding = 10;
    Chart.defaults.plugins.tooltip.cornerRadius = 10;
    Chart.defaults.plugins.tooltip.usePointStyle = true;
    Chart.defaults.elements.bar.borderRadius = 4;
    Chart.defaults.elements.line.borderWidth = 2;
    Chart.defaults.elements.point.radius = 0;
    Chart.defaults.elements.point.hoverRadius = 5;
    Chart.defaults.scale.grid.color = css("--grid");
    Chart.defaults.scale.border.display = false;
  }
  function chart(id, config) {
    const canvas = document.getElementById(id);
    if (!canvas) return null;
    if (charts[id]) charts[id].destroy();
    charts[id] = new Chart(canvas, config);
    return charts[id];
  }
  const getChart = (id) => charts[id];

  // ------------------------------------------------------------------ motion
  function countUp(el) {
    const target = parseFloat(el.dataset.count);
    const decimals = parseInt(el.dataset.decimals || "0", 10);
    const suffix = el.dataset.suffix || "";
    const prefix = el.dataset.prefix || "";
    if (Number.isNaN(target)) return;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const show = (v) => {
      el.textContent = `${prefix}${v.toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}${suffix}`;
    };
    if (reduce) return show(target);
    const start = performance.now();
    const duration = 1200;
    const step = (now) => {
      const t = Math.min(1, (now - start) / duration);
      show(target * (1 - Math.pow(1 - t, 3)));
      if (t < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }
  const observer =
    "IntersectionObserver" in window
      ? new IntersectionObserver(
          (entries) => {
            entries.forEach((entry) => {
              if (!entry.isIntersecting) return;
              entry.target.classList.add("is-visible");
              entry.target.querySelectorAll("[data-count]").forEach(countUp);
              observer.unobserve(entry.target);
            });
          },
          { threshold: 0.12 }
        )
      : null;
  function reveal(scope = document) {
    scope.querySelectorAll(".reveal:not(.is-visible)").forEach((el) => {
      if (observer) observer.observe(el);
      else {
        el.classList.add("is-visible");
        el.querySelectorAll("[data-count]").forEach(countUp);
      }
    });
  }

  // ------------------------------------------------------------------ model switcher
  // One selected model shared by every analysis page (kept in localStorage).
  const MODEL_KEY = "selectedModel";
  function selectedModel(models, fallbackKey) {
    const saved = store.get(MODEL_KEY);
    return models.find((m) => m.key === saved) ? saved : fallbackKey;
  }
  function modelSwitcher(container, models, { selected, onChange }) {
    container.innerHTML = models
      .map(
        (m) =>
          `<button class="pill" type="button" data-key="${m.key}" aria-pressed="${m.key === selected}">` +
          `<span class="dot" style="background:${modelColor(m)}"></span>${escapeHTML(m.name)}</button>`
      )
      .join("");
    container.addEventListener("click", (event) => {
      const button = event.target.closest("button[data-key]");
      if (!button) return;
      container.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", b === button));
      store.set(MODEL_KEY, button.dataset.key);
      onChange(button.dataset.key);
    });
    onTheme(() => {
      container.querySelectorAll("button[data-key]").forEach((b) => {
        const model = models.find((m) => m.key === b.dataset.key);
        b.querySelector(".dot").style.background = modelColor(model);
      });
    });
  }
  function segmented(container, onChange) {
    container.addEventListener("click", (event) => {
      const button = event.target.closest("button[data-value]");
      if (!button) return;
      container.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", b === button));
      onChange(button.dataset.value);
    });
  }

  function showError(el, error) {
    if (el) el.innerHTML = `<div class="error-box">${escapeHTML(error.message || error)}</div>`;
    console.error(error);
  }

  applyChartTheme();
  document.addEventListener("DOMContentLoaded", () => reveal());

  window.App = {
    isDark, css, store, onTheme, data, postJSON, fmt, escapeHTML, modelColor, alpha, classColor,
    chart, getChart, countUp, reveal, selectedModel, modelSwitcher, segmented, showError,
  };
})();
