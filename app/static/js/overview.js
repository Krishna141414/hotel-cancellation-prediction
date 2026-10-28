// Overview page: class-balance doughnut and the top-3 leaderboard.
(async () => {
  "use strict";
  const { data, chart, css, onTheme, showError } = window.App;
  const board = document.getElementById("mini-leaderboard");
  try {
    const [dataset, models, final, setup] = await Promise.all([
      data("dataset"), data("models"), data("final"), data("setup"),
    ]);
    const render = () => {
      chart("balance-chart", {
        type: "doughnut",
        data: {
          labels: ["Not canceled", "Canceled"],
          datasets: [{
            data: [1 - dataset.train_cancel_rate, dataset.train_cancel_rate],
            backgroundColor: [css("--teal"), css("--pink")],
            borderColor: css("--surface"),
            borderWidth: 3,
            hoverOffset: 6,
          }],
        },
        options: {
          maintainAspectRatio: false,
          cutout: "68%",
          plugins: {
            legend: { position: "bottom" },
            tooltip: { callbacks: { label: (c) => ` ${c.label}: ${(c.raw * 100).toFixed(1)}%` } },
          },
        },
      });
      Widgets.leaderboard(board, models, { finalKey: final.key, metric: setup.primary_metric, limit: 3 });
    };
    render();
    onTheme(render);
  } catch (error) {
    showError(board, error);
  }
})();
