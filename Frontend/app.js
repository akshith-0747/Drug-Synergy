
"use strict";

/* =========================================
   SYNERGIA — FRONTEND APPLICATION
========================================= */

const $ = (id) => document.getElementById(id);

const API = {
    status: "/api/status",
    options: "/api/options",
    predict: "/api/predict",
    metrics: "/api/metrics",
    chat: "/api/chat",
    login: "/api/auth/login",
    register: "/api/auth/register",
    logout: "/api/auth/logout",
    me: "/api/auth/me",
    history: "/api/history"
};

let chatHistory = [];
let comparisonRows = [];
let aiAvailable = false;
let authMode = "login";
let currentUser = null;
let historyRows = [];

const metricLabels = {
    rmse: "RMSE",
    mae: "MAE",
    r2: "R²",
    spearman: "Spearman",
    auc: "AUC",
    accuracy: "Accuracy",
    f1: "F1"
};

/* UTILITIES */

function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, (char) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;"
    })[char]);
}

function showMessage(element, message, isError = false) {
    if (!element) return;
    element.textContent = message;
    element.style.color = isError ? "#c78383" : "#7e9d91";
}

async function apiRequest(url, options = {}) {
    const response = await fetch(url, {
        credentials: "same-origin",
        ...options,
        headers: {
            ...(options.body ? { "Content-Type": "application/json" } : {}),
            ...(options.headers || {})
        }
    });

    const contentType = response.headers.get("content-type") || "";
    const data = contentType.includes("application/json")
        ? await response.json()
        : await response.text();

    if (!response.ok) {
        const message = typeof data === "object"
            ? data.detail || data.message || "Request failed."
            : data || "Request failed.";
        throw new Error(message);
    }

    return data;
}

/* PAGE NAVIGATION — SWITCHES PAGES, DOES NOT SCROLL */

function navigateTo(pageName) {
    const validPages = [
        "home", "predict", "datasets", "methodology",
        "model-lab", "ai-guide", "history", "login", "about"
    ];

    if (!validPages.includes(pageName)) pageName = "home";

    document.querySelectorAll(".page").forEach((page) => {
        page.classList.toggle("active", page.id === `page-${pageName}`);
    });

    document.querySelectorAll(".nav-link[data-page]").forEach((link) => {
        link.classList.toggle("active", link.dataset.page === pageName);
    });

    const nav = $("main-nav");
    if (nav) nav.classList.remove("open");

    if (pageName === "predict") loadOptions();
    if (pageName === "model-lab") loadMetrics();
    if (pageName === "history") loadHistory();
    if (pageName === "login") updateAuthScreen();

    history.replaceState(null, "", `#${pageName}`);
    window.scrollTo({ top: 0, behavior: "instant" });
}

document.addEventListener("click", (event) => {
    const pageLink = event.target.closest("[data-page]");
    const goButton = event.target.closest("[data-go]");

    if (pageLink) {
        event.preventDefault();
        navigateTo(pageLink.dataset.page);
        return;
    }

    if (goButton) {
        event.preventDefault();
        navigateTo(goButton.dataset.go);
    }
});

window.addEventListener("hashchange", () => {
    const pageName = location.hash.replace("#", "") || "home";
    navigateTo(pageName);
});

$("mobile-menu-button")?.addEventListener("click", () => {
    $("main-nav")?.classList.toggle("open");
});

/* STATUS */

async function loadStatus() {
    try {
        const status = await apiRequest(API.status);
        aiAvailable = Boolean(status.ai_available);

        if ($("ai-status")) {
            $("ai-status").textContent = aiAvailable
                ? "AI Guide is available."
                : "AI Guide is currently unavailable. Check your API configuration.";
        }
    } catch (error) {
        console.warn("Status unavailable:", error.message);
    }
}

/* DRUG AND CELL-LINE OPTIONS */

async function loadOptions() {
    const drugA = $("drug-a");
    const drugB = $("drug-b");
    const cellLine = $("cell-line");

    if (!drugA || !drugB || !cellLine) return;

    if (drugA.dataset.loaded === "true") return;

    try {
        const data = await apiRequest(API.options);

        const drugs = data.drugs || data.drug_options || [];
        const cellLines = data.cell_lines || data.cell_line_options || [];

        drugA.innerHTML = '<option value="">Select first drug</option>';
        drugB.innerHTML = '<option value="">Select second drug</option>';
        cellLine.innerHTML = '<option value="">Select cancer cell line</option>';

        drugs.forEach((drug) => {
            const name = typeof drug === "string" ? drug : drug.name || drug.label;
            if (!name) return;

            drugA.add(new Option(name, name));
            drugB.add(new Option(name, name));
        });

        cellLines.forEach((cell) => {
            const name = typeof cell === "string" ? cell : cell.name || cell.label;
            if (!name) return;

            cellLine.add(new Option(name, name));
        });

        drugA.dataset.loaded = "true";
        drugB.dataset.loaded = "true";
        cellLine.dataset.loaded = "true";
    } catch (error) {
        drugA.innerHTML = '<option value="">Unable to load drugs</option>';
        drugB.innerHTML = '<option value="">Unable to load drugs</option>';
        cellLine.innerHTML = '<option value="">Unable to load cell lines</option>';
        console.error("Options error:", error);
    }
}

/* PREDICTION */

function normalizePrediction(data) {
    return {
        score: data.predicted_s_score ?? data.s_score ?? data.prediction ?? data.score,
        synergistic: data.synergistic ?? data.predicted_synergistic,
        feature: data.feature_set ?? data.feature ?? "auto",
        model: data.model ?? "Default",
        implementation: data.implementation ?? "—",
        message: data.message ?? ""
    };
}

function renderPrediction(rawData) {
    const result = $("prediction-result");
    const status = $("result-status");
    const data = normalizePrediction(rawData);

    const score = Number(data.score);
    const scoreText = Number.isFinite(score) ? score.toFixed(4) : "N/A";

    const isSynergistic = data.synergistic === true ||
        (data.synergistic === undefined && Number.isFinite(score) && score >= 10);

    status.textContent = "Completed";

    result.className = "prediction-result-content";
    result.innerHTML = `
        <div class="prediction-score">
            <strong>${escapeHtml(scoreText)}</strong>
            <span>Predicted S-score</span>
        </div>

        <div class="prediction-label">
            ${isSynergistic ? "✧ Predicted synergistic combination" : "Predicted non-synergistic combination"}
        </div>

        <div class="prediction-detail-grid">
            <div class="prediction-detail">
                <small>Feature set</small>
                <strong>${escapeHtml(data.feature)}</strong>
            </div>
            <div class="prediction-detail">
                <small>Model</small>
                <strong>${escapeHtml(data.model)}</strong>
            </div>
            <div class="prediction-detail">
                <small>Implementation</small>
                <strong>${escapeHtml(data.implementation)}</strong>
            </div>
            <div class="prediction-detail">
                <small>Prediction status</small>
                <strong>Successful</strong>
            </div>
        </div>

        ${data.message ? `<p>${escapeHtml(data.message)}</p>` : ""}
    `;

    if (currentUser) loadHistory();
}

$("predict-form")?.addEventListener("submit", async (event) => {
    event.preventDefault();

    const drugA = $("drug-a").value;
    const drugB = $("drug-b").value;
    const cellLine = $("cell-line").value;

    if (!drugA || !drugB || !cellLine) {
        alert("Please select both drugs and a cell line.");
        return;
    }

    if (drugA === drugB) {
        alert("Please select two different drugs.");
        return;
    }

    const button = $("predict-button");
    const result = $("prediction-result");
    const status = $("result-status");

    button.disabled = true;
    button.innerHTML = "Analysing combination...";
    status.textContent = "Running";

    result.className = "result-placeholder";
    result.innerHTML = `
        <div class="placeholder-icon">✧</div>
        <h3>Analysing your combination</h3>
        <p>Please wait while the prediction model processes your input.</p>
    `;

    try {
        const payload = {
            drug_a: drugA,
            drug_b: drugB,
            cell_line: cellLine,
            feature_set: $("feature-set").value,
            model: $("model-select").value || null
        };

        const data = await apiRequest(API.predict, {
            method: "POST",
            body: JSON.stringify(payload)
        });

        renderPrediction(data);
    } catch (error) {
        status.textContent = "Error";
        result.className = "result-placeholder";
        result.innerHTML = `
            <div class="placeholder-icon">!</div>
            <h3>Prediction failed</h3>
            <p>${escapeHtml(error.message)}</p>
        `;
    } finally {
        button.disabled = false;
        button.innerHTML = "<span>Run prediction</span><span>→</span>";
    }
});

/* MODEL METRICS */

function metricValue(row, key) {
    const aliases = {
        rmse: ["rmse", "RMSE"],
        mae: ["mae", "MAE"],
        r2: ["r2", "R2", "r_squared", "R²"],
        spearman: ["spearman", "Spearman"],
        auc: ["auc", "AUC"],
        accuracy: ["accuracy", "Accuracy"],
        f1: ["f1", "F1"]
    };

    for (const alias of aliases[key] || [key]) {
        if (row[alias] !== undefined && row[alias] !== null && row[alias] !== "") {
            return Number(row[alias]);
        }
    }

    return null;
}

function renderMetrics(rows) {
    const body = $("metrics-body");
    if (!body) return;

    if (!rows.length) {
        body.innerHTML = '<tr><td colspan="9" class="empty-cell">No model metrics are available.</td></tr>';
        return;
    }

    body.innerHTML = rows.map((row) => {
        const feature = row.feature_set ?? row.feature ?? "—";
        const model = row.model ?? row.model_name ?? "—";

        return `
            <tr>
                <td>${escapeHtml(feature)}</td>
                <td>${escapeHtml(model)}</td>
                <td>${formatMetric(metricValue(row, "rmse"))}</td>
                <td>${formatMetric(metricValue(row, "mae"))}</td>
                <td>${formatMetric(metricValue(row, "r2"))}</td>
                <td>${formatMetric(metricValue(row, "spearman"))}</td>
                <td>${formatMetric(metricValue(row, "auc"))}</td>
                <td>${formatMetric(metricValue(row, "accuracy"))}</td>
                <td>${formatMetric(metricValue(row, "f1"))}</td>
            </tr>
        `;
    }).join("");

    renderComparisonGraph();
    renderFeatureSummary();
}

function formatMetric(value) {
    return value === null || !Number.isFinite(value) ? "—" : value.toFixed(4);
}

async function loadMetrics() {
    const body = $("metrics-body");
    if (body) body.innerHTML = '<tr><td colspan="9" class="empty-cell">Loading metrics...</td></tr>';

    try {
        const data = await apiRequest(API.metrics);
        comparisonRows = Array.isArray(data) ? data : data.results || data.metrics || [];
        renderMetrics(comparisonRows);

        if ($("metrics-updated")) {
            $("metrics-updated").textContent =
                `Loaded ${comparisonRows.length} model result(s).`;
        }
    } catch (error) {
        if (body) {
            body.innerHTML = `<tr><td colspan="9" class="empty-cell">${escapeHtml(error.message)}</td></tr>`;
        }
        if ($("metrics-updated")) $("metrics-updated").textContent = "Unable to load metrics.";
    }
}

function renderComparisonGraph() {
    const container = $("model-comparison-graphs");
    const selectedMetric = $("graph-metric")?.value || "r2";

    if (!container) return;

    const usable = comparisonRows.map((row) => ({
        name: `${row.feature_set ?? row.feature ?? "Feature"} / ${row.model ?? row.model_name ?? "Model"}`,
        value: metricValue(row, selectedMetric)
    })).filter((item) => item.value !== null && Number.isFinite(item.value));

    if (!usable.length) {
        container.innerHTML = '<p class="empty-cell">No data available for this metric.</p>';
        return;
    }

    const values = usable.map((item) => item.value);
    const maxAbs = Math.max(...values.map(Math.abs), 0.0001);

    container.innerHTML = `
        <div class="comparison-chart">
            ${usable.map((item) => {
                const width = Math.max(2, Math.abs(item.value) / maxAbs * 100);
                return `
                    <div class="chart-row">
                        <span class="chart-label">${escapeHtml(item.name)}</span>
                        <div class="chart-track">
                            <div class="chart-bar" style="width:${width}%"></div>
                        </div>
                        <span class="chart-value">${item.value.toFixed(4)}</span>
                    </div>
                `;
            }).join("")}
        </div>
    `;
}

function renderFeatureSummary() {
    const container = $("feature-summary-content");
    if (!container) return;

    const features = ["structure", "ds", "drs"];

    container.innerHTML = features.map((feature) => {
        const count = comparisonRows.filter((row) =>
            String(row.feature_set ?? row.feature ?? "").toLowerCase() === feature
        ).length;

        const descriptions = {
            structure: "Chemical structure representation",
            ds: "Drug perturbation signatures",
            drs: "Drug resistance signatures"
        };

        return `
            <div class="feature-summary-card">
                <strong>${feature.toUpperCase()}</strong>
                <span>${descriptions[feature]}</span>
                <span>${count} result(s) in comparison</span>
            </div>
        `;
    }).join("");
}

$("refresh-results")?.addEventListener("click", loadMetrics);
$("graph-metric")?.addEventListener("change", renderComparisonGraph);

/* AI GUIDE */

function appendChatMessage(message, sender = "assistant") {
    const container = $("chat-messages");
    if (!container) return;

    const wrapper = document.createElement("div");
    wrapper.className = `chat-message ${sender === "user" ? "user-message" : "assistant-message"}`;

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = sender === "user" ? "You" : "✧";

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";
    bubble.textContent = message;

    wrapper.append(avatar, bubble);
    container.appendChild(wrapper);
    container.scrollTop = container.scrollHeight;
}

async function sendChatMessage(message) {
    if (!message.trim()) return;

    appendChatMessage(message, "user");
    chatHistory.push({ role: "user", content: message });

    const sendButton = $("chat-send");
    if (sendButton) {
        sendButton.disabled = true;
        sendButton.textContent = "Thinking...";
    }

    try {
        const data = await apiRequest(API.chat, {
            method: "POST",
            body: JSON.stringify({
                messages: chatHistory.slice(-10),
                message: message
            })
        });

        const answer = data.response || data.reply || data.answer || data.message ||
            "I couldn't generate a response.";

        appendChatMessage(answer, "assistant");
        chatHistory.push({ role: "assistant", content: answer });
        showMessage($("ai-status"), "Response received.");
    } catch (error) {
        appendChatMessage(`Sorry, I couldn't answer that. ${error.message}`, "assistant");
        showMessage($("ai-status"), "AI request failed.", true);
    } finally {
        if (sendButton) {
            sendButton.disabled = false;
            sendButton.textContent = "Send ↑";
        }
    }
}

$("chat-form")?.addEventListener("submit", (event) => {
    event.preventDefault();
    const input = $("chat-input");
    const message = input.value.trim();
    if (!message) return;
    input.value = "";
    sendChatMessage(message);
});

document.querySelectorAll(".suggestion").forEach((button) => {
    button.addEventListener("click", () => {
        sendChatMessage(button.dataset.question || button.textContent);
    });
});

/* AUTHENTICATION */

function setAuthMode(mode) {
    authMode = mode;

    document.querySelectorAll("[data-auth-mode]").forEach((button) => {
        button.classList.toggle("active", button.dataset.authMode === mode);
    });

    $("auth-title").textContent = mode === "login" ? "Welcome back!" : "Create your account";
    $("auth-description").textContent = mode === "login"
        ? "Enter your details to sign in."
        : "Create an account to save your prediction history.";

    $("auth-submit").textContent = mode === "login" ? "Sign in →" : "Create account →";
    $("auth-password").autocomplete = mode === "login" ? "current-password" : "new-password";
    $("auth-switch").innerHTML = mode === "login"
        ? 'New here? <button type="button" data-auth-mode="register">Create an account</button>'
        : 'Already registered? <button type="button" data-auth-mode="login">Sign in</button>';

    $("auth-message").textContent = "";
}

document.addEventListener("click", (event) => {
    const modeButton = event.target.closest("[data-auth-mode]");
    if (modeButton) setAuthMode(modeButton.dataset.authMode);
});

function updateAuthScreen() {
    if (currentUser) {
        $("auth-title").textContent = "You're signed in!";
        $("auth-description").textContent = currentUser.email || "Your account is active.";
        $("auth-form").classList.add("hidden");
        $("auth-switch").innerHTML =
            '<button type="button" data-go="history">View your prediction history →</button>';
    } else {
        $("auth-form").classList.remove("hidden");
        setAuthMode(authMode);
    }
}

async function checkCurrentUser() {
    try {
        const data = await apiRequest(API.me);
        currentUser = data.user || data;
        if (data.authenticated === false) currentUser = null;
    } catch {
        currentUser = null;
    }

    updateAccountNavigation();
}

function updateAccountNavigation() {
    const loginLink = $("login-nav");
    const logoutButton = $("logout-button");

    if (currentUser) {
        if (loginLink) {
            loginLink.textContent = "Account";
            loginLink.dataset.page = "login";
        }
        logoutButton?.classList.remove("hidden");
    } else {
        if (loginLink) {
            loginLink.textContent = "Login";
            loginLink.dataset.page = "login";
        }
        logoutButton?.classList.add("hidden");
    }
}

$("auth-form")?.addEventListener("submit", async (event) => {
    event.preventDefault();

    const email = $("auth-email").value.trim();
    const password = $("auth-password").value;

    const endpoint = authMode === "login" ? API.login : API.register;

    $("auth-submit").disabled = true;
    $("auth-submit").textContent = "Please wait...";

    try {
        const data = await apiRequest(endpoint, {
            method: "POST",
            body: JSON.stringify({ email, password })
        });

        currentUser = data.user || { email };
        updateAccountNavigation();
        showMessage($("auth-message"), data.message || "Signed in successfully.");

        $("auth-form").reset();
        navigateTo("history");
    } catch (error) {
        showMessage($("auth-message"), error.message, true);
    } finally {
        $("auth-submit").disabled = false;
        $("auth-submit").textContent = authMode === "login" ? "Sign in →" : "Create account →";
    }
});

$("logout-button")?.addEventListener("click", async () => {
    try {
        await apiRequest(API.logout, { method: "POST" });
    } catch (error) {
        console.warn("Logout request:", error.message);
    }

    currentUser = null;
    historyRows = [];
    updateAccountNavigation();
    navigateTo("home");
});

/* HISTORY */

async function loadHistory() {
    const container = $("history-list");
    if (!container) return;

    if (!currentUser) {
        container.innerHTML = `
            <div class="panel empty-history">
                <div class="placeholder-icon">◷</div>
                <h3>Sign in to view your history</h3>
                <p>Your saved predictions are connected to your account.</p>
                <button class="button button-primary" data-go="login">Login →</button>
            </div>
        `;
        return;
    }

    container.innerHTML = '<div class="panel empty-history">Loading prediction history...</div>';

    try {
        const data = await apiRequest(API.history);
        historyRows = Array.isArray(data) ? data : data.history || data.records || [];
        renderHistory(historyRows);
    } catch (error) {
        container.innerHTML = `
            <div class="panel empty-history">
                <h3>Unable to load history</h3>
                <p>${escapeHtml(error.message)}</p>
            </div>
        `;
    }
}

function renderHistory(rows) {
    const container = $("history-list");
    if (!container) return;

    if (!rows.length) {
        container.innerHTML = `
            <div class="panel empty-history">
                <div class="placeholder-icon">◷</div>
                <h3>No predictions yet</h3>
                <p>Your saved predictions will appear here.</p>
                <button class="button button-primary" data-go="predict">Make a prediction</button>
            </div>
        `;
        return;
    }

    container.innerHTML = rows.map((row) => {
        const id = row.id ?? row.record_id;
        const drugA = row.drug_a ?? row.drug1 ?? "Drug A";
        const drugB = row.drug_b ?? row.drug2 ?? "Drug B";
        const cell = row.cell_line ?? "—";
        const score = row.predicted_s_score ?? row.s_score ?? row.score ?? "—";
        const feature = row.feature_set ?? "—";
        const model = row.model ?? "—";
        const dateValue = row.timestamp ?? row.created_at ?? row.date;
        const date = dateValue ? new Date(dateValue).toLocaleString() : "Date unavailable";

        return `
            <article class="history-card">
                <div>
                    <h3>${escapeHtml(drugA)} + ${escapeHtml(drugB)}</h3>
                    <p>Cell line: ${escapeHtml(cell)} · ${escapeHtml(feature)} · ${escapeHtml(model)}</p>
                    <p>${escapeHtml(date)}</p>
                </div>
                <div class="history-actions">
                    <div class="history-score">${escapeHtml(score)}</div>
                    <button class="small-action" data-delete-history="${escapeHtml(id)}">Delete</button>
                </div>
            </article>
        `;
    }).join("");
}

$("history-search")?.addEventListener("input", (event) => {
    const query = event.target.value.toLowerCase();

    const filtered = historyRows.filter((row) => {
        const text = [
            row.drug_a, row.drug_b, row.drug1, row.drug2,
            row.cell_line, row.model, row.feature_set
        ].join(" ").toLowerCase();

        return text.includes(query);
    });

    renderHistory(filtered);
});

$("refresh-history")?.addEventListener("click", loadHistory);

document.addEventListener("click", async (event) => {
    const deleteButton = event.target.closest("[data-delete-history]");
    if (!deleteButton) return;

    const id = deleteButton.dataset.deleteHistory;
    if (!id || id === "undefined") return;

    if (!confirm("Delete this prediction from your history?")) return;

    try {
        await apiRequest(`${API.history}/${encodeURIComponent(id)}`, {
            method: "DELETE"
        });
        await loadHistory();
    } catch (error) {
        alert(`Unable to delete history: ${error.message}`);
    }
});

/* INITIALIZE */

async function initializeApp() {
    await Promise.all([
        loadStatus(),
        checkCurrentUser()
    ]);

    const initialPage = location.hash.replace("#", "") || "home";
    navigateTo(initialPage);
}

document.addEventListener("DOMContentLoaded", initializeApp);
