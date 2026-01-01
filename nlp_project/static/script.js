const input = document.getElementById("search");
const results = document.getElementById("results");

// Flow elements (RIGHT PANEL)
const flowQuery = document.getElementById("flow-query");
const flowTokens = document.getElementById("flow-tokens");
const flowOutput = document.getElementById("flow-output");

// -----------------------------
// Autocomplete + Live Flow
// -----------------------------
let debounceTimer;

input.addEventListener("input", () => {
  clearTimeout(debounceTimer);

  debounceTimer = setTimeout(async () => {
    const query = input.value.trim();

    // ---- LIVE FLOW UPDATE ----
    flowQuery.innerHTML = `User Query<br><small>${query || "—"}</small>`;
    flowTokens.innerHTML = `Token IDs → Embeddings<br><small>${query ? query.split(" ").length : 0} tokens</small>`;
    flowOutput.innerHTML = `Top-K Predictions<br><small>—</small>`;

    if (query.length < 3) {
      results.innerHTML = "";
      return;
    }

    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query })
    });

    const data = await res.json();

    // Autocomplete results
    results.innerHTML = "";
    data.suggestions.forEach(s => {
      const li = document.createElement("li");
      li.textContent = s;
      results.appendChild(li);
    });

    // Update final step
    flowOutput.innerHTML =
      `Top-K Predictions<br><small>${data.suggestions.length} suggestions</small>`;

  }, 300);
});

// -----------------------------
// Training Metrics Visualization
// -----------------------------
fetch("/metrics")
  .then(res => res.json())
  .then(data => {
    const ctx = document.getElementById("metricsChart").getContext("2d");

    new Chart(ctx, {
      type: "bar",
      data: {
        labels: ["Epochs", "Batch Size", "Learning Rate"],
        datasets: [{
          data: [data.epochs, data.batch_size, data.learning_rate],
          backgroundColor: ["#4CAF50", "#2196F3", "#FFC107"]
        }]
      },
      options: {
      responsive: false,
      plugins: {
        legend: { display: false }
      },
      scales: {
        x: {
          ticks: {
            autoSkip: false,
            maxRotation: 0,
            minRotation: 0
          }
        },
        y: {
          beginAtZero: true
        }
      }
    }
    });
  });
