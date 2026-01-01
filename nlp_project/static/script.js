const input = document.getElementById("search");
const results = document.getElementById("results");

const flowQuery = document.getElementById("flow-query");
const flowTokens = document.getElementById("flow-tokens");
const flowOutput = document.getElementById("flow-output");

// ==============================
// Autocomplete + Explainability
// ==============================
let debounce;

input.addEventListener("input", () => {
  clearTimeout(debounce);

  debounce = setTimeout(async () => {
    const query = input.value.trim();

    if (query.length < 3) {
      results.innerHTML = "";
      flowQuery.querySelector(".step-desc").innerText = "—";
      flowTokens.querySelector(".step-desc").innerText = "0 tokens";
      flowOutput.querySelector(".step-desc").innerText = "—";
      return;
    }

    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query })
    });

    const data = await res.json();

    // ---------- Pipeline ----------
    flowQuery.querySelector(".step-desc").innerText = query;

    // Clean GPT-2 tokens (remove Ġ)
    flowTokens.querySelector(".step-desc").innerHTML =
      data.tokens
        .map((t, i) => {
          const clean = t.replace("Ġ", " ");
          return i >= data.tokens.length - 2
            ? `<span class="highlight">${clean}</span>`
            : clean;
        })
        .join("");

    flowOutput.querySelector(".step-desc").innerText =
      `${data.suggestions.length} predictions`;

    // ---------- Results ----------
    results.innerHTML = "";

    data.suggestions.forEach(s => {
      const percent = (s.confidence * 100).toFixed(1);

      const li = document.createElement("li");
      li.innerHTML = `
        <div class="suggestion-row">
          <span>${s.text}</span>
          <span class="confidence-label">${percent}%</span>
        </div>
        <div class="bar">
          <div class="fill" style="width:${percent}%"></div>
        </div>
      `;

      results.appendChild(li);
    });

  }, 300);
});
