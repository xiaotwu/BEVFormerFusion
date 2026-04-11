(() => {
  let lastRenderedPath = null;

  function formatMetric(value, digits = 4) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
      return "Pending";
    }
    return Number(value).toFixed(digits);
  }

  function formatRuntime(value, suffix = "") {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
      return "Pending";
    }
    return `${Number(value).toFixed(2)}${suffix}`;
  }

  function setText(id, text) {
    const node = document.getElementById(id);
    if (node) {
      node.textContent = text;
    }
  }

  function setHtml(id, html) {
    const node = document.getElementById(id);
    if (node) {
      node.innerHTML = html;
    }
  }

  function buildMainResultsTable(models) {
    const tbody = document.getElementById("main-results-body");
    if (!tbody) {
      return;
    }
    tbody.innerHTML = models.map((model) => `
      <tr>
        <td><strong>${model.label}</strong><br><code>${model.config}</code></td>
        <td>${model.best_iter.toLocaleString()}</td>
        <td>${formatMetric(model.mAP)}</td>
        <td>${formatMetric(model.NDS)}</td>
        <td>${formatRuntime(model.fps, " img/s")}</td>
        <td>${formatRuntime(model.memory_gb, " GB")}</td>
      </tr>
    `).join("");
  }

  function build12GbTable(profileRows) {
    const tbody = document.getElementById("profile-results-body");
    if (!tbody) {
      return;
    }
    tbody.innerHTML = profileRows.map((row) => `
      <tr>
        <td>${row.setting}</td>
        <td>${formatRuntime(row.memory_gb, " GB")}</td>
        <td>${formatRuntime(row.fps, " img/s")}</td>
        <td>${row.mAP === null ? "Pending" : `${formatMetric(row.mAP)} / ${formatMetric(row.NDS)}`}</td>
        <td>${row.notes}</td>
      </tr>
    `).join("");
  }

  function buildHeroMetrics(models, findings) {
    const target = document.getElementById("hero-metrics");
    if (!target || models.length < 2) {
      return;
    }
    const baseline = models.find((model) => model.id === "baseline") || models[0];
    const fusion = models.find((model) => model.id === "fusion") || models[1];
    const cards = [
      {
        value: formatMetric(fusion.mAP),
        label: "Best fused-model mAP"
      },
      {
        value: formatMetric(fusion.NDS),
        label: "Best fused-model NDS"
      },
      {
        value: `+${formatMetric(findings.mAP_gain_abs)}`,
        label: "Absolute mAP gain over local baseline"
      },
      {
        value: `+${formatMetric(findings.NDS_gain_abs)}`,
        label: "Absolute NDS gain over local baseline"
      }
    ];
    target.innerHTML = cards.map((card) => `
      <div class="metric-card">
        <strong>${card.value}</strong>
        <span>${card.label}</span>
      </div>
    `).join("");
    setText(
      "fusion-delta-note",
      `At ${fusion.best_iter.toLocaleString()} iterations, the fused configuration improves mAP by ${formatMetric(findings.mAP_gain_abs)} and NDS by ${formatMetric(findings.NDS_gain_abs)} over the local BEVFormer baseline.`
    );
    setText(
      "hero-runtime-note",
      "Runtime metrics remain pending until a dedicated 12GB GPU profiling run is recorded."
    );
    setText(
      "best-checkpoint-note",
      `${baseline.label}: ${baseline.best_iter.toLocaleString()} iters | ${fusion.label}: ${fusion.best_iter.toLocaleString()} iters`
    );
  }

  function buildInsightCards(findings) {
    const target = document.getElementById("insight-grid");
    if (!target) {
      return;
    }
    const cards = [
      {
        value: formatMetric(findings.baseline_mATE, 4),
        label: "Baseline mATE at best checkpoint"
      },
      {
        value: formatMetric(findings.fusion_mATE, 4),
        label: "Fusion mATE at best checkpoint"
      },
      {
        value: formatMetric(findings.baseline_mAAE, 4),
        label: "Baseline mAAE at best checkpoint"
      },
      {
        value: formatMetric(findings.fusion_mAAE, 4),
        label: "Fusion mAAE at best checkpoint"
      }
    ];
    target.innerHTML = cards.map((card) => `
      <div class="insight-card">
        <strong>${card.value}</strong>
        <span>${card.label}</span>
      </div>
    `).join("");
  }

  function polyline(points) {
    return points.map((point) => `${point.x},${point.y}`).join(" ");
  }

  function scaleCurve(rows, metricKey, width, height, padding) {
    const xMin = Math.min(...rows.map((row) => row.iters));
    const xMax = Math.max(...rows.map((row) => row.iters));
    const yMin = Math.min(...rows.map((row) => row[metricKey]));
    const yMax = Math.max(...rows.map((row) => row[metricKey]));
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;
    return rows.map((row) => ({
      iters: row.iters,
      value: row[metricKey],
      x: padding.left + ((row.iters - xMin) / (xMax - xMin || 1)) * chartWidth,
      y: padding.top + (1 - ((row[metricKey] - yMin) / (yMax - yMin || 1))) * chartHeight
    }));
  }

  function buildCurve(curves) {
    const target = document.getElementById("nds-curve");
    if (!target) {
      return;
    }
    const width = 860;
    const height = 360;
    const padding = { top: 24, right: 24, bottom: 42, left: 56 };
    const baseline = scaleCurve(curves.baseline, "NDS", width, height, padding);
    const fusion = scaleCurve(curves.fusion_encoder_decoder, "NDS", width, height, padding);
    const xTicks = [20000, 40000, 60000, 80000, 100000];
    const yTicks = [0.13, 0.16, 0.19, 0.22, 0.25];
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;
    const xMin = 10000;
    const xMax = 100000;
    const yMin = 0.13;
    const yMax = 0.26;

    const xPos = (tick) => padding.left + ((tick - xMin) / (xMax - xMin)) * chartWidth;
    const yPos = (tick) => padding.top + (1 - ((tick - yMin) / (yMax - yMin))) * chartHeight;

    target.innerHTML = `
      <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="NDS progression for baseline and fused models">
        <rect x="0" y="0" width="${width}" height="${height}" rx="18" fill="#fffdf9"></rect>
        ${yTicks.map((tick) => `
          <g>
            <line x1="${padding.left}" y1="${yPos(tick)}" x2="${width - padding.right}" y2="${yPos(tick)}" stroke="rgba(21,38,59,0.12)" stroke-dasharray="4 6"></line>
            <text x="${padding.left - 10}" y="${yPos(tick) + 4}" text-anchor="end" fill="#5b6778" font-size="12">${tick.toFixed(2)}</text>
          </g>
        `).join("")}
        ${xTicks.map((tick) => `
          <g>
            <line x1="${xPos(tick)}" y1="${padding.top}" x2="${xPos(tick)}" y2="${height - padding.bottom}" stroke="rgba(21,38,59,0.08)"></line>
            <text x="${xPos(tick)}" y="${height - padding.bottom + 18}" text-anchor="middle" fill="#5b6778" font-size="12">${(tick / 1000).toFixed(0)}k</text>
          </g>
        `).join("")}
        <polyline fill="none" stroke="#bb6b2c" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" points="${polyline(baseline)}"></polyline>
        <polyline fill="none" stroke="#136f63" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" points="${polyline(fusion)}"></polyline>
        ${baseline.map((point) => `<circle cx="${point.x}" cy="${point.y}" r="4.5" fill="#bb6b2c"></circle>`).join("")}
        ${fusion.map((point) => `<circle cx="${point.x}" cy="${point.y}" r="4.5" fill="#136f63"></circle>`).join("")}
        <text x="${padding.left}" y="18" fill="#15263b" font-size="14" font-weight="700">Validation NDS across training checkpoints</text>
        <text x="${width / 2}" y="${height - 8}" fill="#5b6778" font-size="12" text-anchor="middle">Iterations</text>
      </svg>
      <div class="legend-row">
        <span><i class="legend-swatch" style="background:#bb6b2c"></i> Local BEVFormer baseline</span>
        <span><i class="legend-swatch" style="background:#136f63"></i> BEVFormerFusion</span>
      </div>
    `;
  }

  function getMetricsUrl() {
    const script = Array.from(document.scripts).find((node) =>
      node.src && node.src.includes("/assets/javascripts/metrics.js")
    );
    if (script) {
      return new URL("../data/metrics.json", script.src);
    }
    return new URL("assets/data/metrics.json", window.location.href);
  }

  async function renderMetrics() {
    const currentPath = window.location.pathname + window.location.search + window.location.hash;
    if (currentPath === lastRenderedPath && !document.getElementById("hero-runtime-note")) {
      return;
    }
    lastRenderedPath = currentPath;

    try {
      const metricsUrl = getMetricsUrl();
      const response = await fetch(metricsUrl, { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`Failed to load ${metricsUrl}`);
      }
      const metrics = await response.json();
      buildHeroMetrics(metrics.published_models, metrics.derived_findings.fusion_vs_baseline);
      buildMainResultsTable(metrics.published_models);
      build12GbTable(metrics.twelve_gb_profile.rows);
      buildInsightCards(metrics.derived_findings.fusion_vs_baseline);
      buildCurve(metrics.curves);
    } catch (error) {
      console.error(error);
      setText("hero-runtime-note", "Metrics could not be loaded. Check assets/data/metrics.json.");
      setHtml("main-results-body", '<tr><td colspan="6">Metrics could not be loaded.</td></tr>');
      setHtml("profile-results-body", '<tr><td colspan="5">Profile status could not be loaded.</td></tr>');
    }
  }

  if (typeof document$ !== "undefined" && document$.subscribe) {
    document$.subscribe(() => {
      renderMetrics();
    });
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", renderMetrics);
  } else {
    renderMetrics();
  }
})();
