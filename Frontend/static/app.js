
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
    train: "/api/train",
    authState: "/api/auth/state",
    login: "/api/auth/login",
    signup: "/api/auth/signup",
    logout: "/api/auth/logout",
    reviews: "/api/reviews"
};

let chatHistory = [];
let comparisonRows = [];
let aiAvailable = false;
let historyRows = [];
let trainingPollTimer = null;
let accountState = { authenticated: false, free_predictions_left: 2 };
let reviewPredictions = [];
let pendingReviewPredictionId = null;

const HISTORY_STORAGE_KEY = "synergia.prediction-history.v1";

const metricLabels = {
    mse: "MSE",
    rmse: "RMSE",
    clip3_mse: "MSE · 3σ clipped",
    clip3_rmse: "RMSE · 3σ clipped",
    r2: "R²",
    clip3_r2: "R² · 3σ clipped",
    pearson: "Pearson",
    spearman: "Spearman",
    clip3_pearson: "Pearson · 3σ clipped",
    clip3_spearman: "Spearman · 3σ clipped",
    auc_s_ge_10: "AUC · S ≥ 10",
    balanced_accuracy_s_ge_10: "Balanced accuracy",
    f1_s_ge_10: "F1 · S ≥ 10"
};

const featureOrder = ["structure", "structure_pubchem", "ds", "drs", "paper_drs", "paper_drs_attention", "paper_drs_unseen_pair", "multilevel"];

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

function buildDnaHelix() {
    const host = $("dna-helix");
    if (!host || host.dataset.ready === "true") return;

    const rotor = document.createElement("div");
    rotor.className = "dna-rotor";
    const count = 27;
    const radius = 88;
    const verticalStep = 24;
    const strandPoints = [[], []];

    for (let index = 0; index < count; index += 1) {
        const angle = index * 36 * Math.PI / 180;
        const y = (index - (count - 1) / 2) * verticalStep;
        const points = [
            { x: Math.cos(angle) * radius, y, z: Math.sin(angle) * radius },
            { x: Math.cos(angle + Math.PI) * radius, y, z: Math.sin(angle + Math.PI) * radius }
        ];
        points.forEach((point, strandIndex) => strandPoints[strandIndex].push(point));

        const rung = document.createElement("span");
        rung.className = "dna-rung-3d";
        rung.style.setProperty("--dna-y", `${y.toFixed(2)}px`);
        rung.style.setProperty("--dna-angle", `${(index * 36).toFixed(1)}deg`);
        rung.style.setProperty("--dna-diameter", `${radius * 2}px`);
        rotor.append(rung);
    }

    function cross(a, b) {
        return [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]
        ];
    }

    function addStrandSegment(start, end, strandIndex) {
        const delta = [end.x - start.x, end.y - start.y, end.z - start.z];
        const length = Math.hypot(...delta);
        const direction = delta.map((value) => value / length);
        const reference = [0, 1, 0];
        const side = cross(direction, reference);
        const sideLength = Math.hypot(...side);
        const normalizedSide = side.map((value) => value / sideLength);
        const up = cross(normalizedSide, direction);
        const matrix = [
            direction[0], direction[1], direction[2], 0,
            normalizedSide[0], normalizedSide[1], normalizedSide[2], 0,
            up[0], up[1], up[2], 0,
            start.x, start.y, start.z, 1
        ];
        const segment = document.createElement("span");
        segment.className = `dna-strand-segment strand-${strandIndex + 1}`;
        segment.style.width = `${length.toFixed(2)}px`;
        segment.style.transform = `matrix3d(${matrix.map((value) => value.toFixed(5)).join(",")})`;
        rotor.append(segment);
    }

    strandPoints.forEach((points, strandIndex) => {
        for (let index = 0; index < points.length - 1; index += 1) {
            addStrandSegment(points[index], points[index + 1], strandIndex);
        }
    });

    strandPoints.forEach((points, strandIndex) => {
        points.forEach((point) => {
            const node = document.createElement("span");
            node.className = `dna-node strand-${strandIndex + 1}`;
            node.style.left = `calc(50% + ${point.x.toFixed(2)}px)`;
            node.style.top = `calc(50% + ${point.y.toFixed(2)}px)`;
            node.style.transform = `translate(-50%, -50%) translateZ(${point.z.toFixed(2)}px)`;
            rotor.append(node);
        });
    });

    host.append(rotor);
    host.dataset.ready = "true";
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
        "model-lab", "ai-guide", "history", "review", "about"
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
    if (pageName === "history") loadHistory();
    if (pageName === "review") loadReviews();
    if (pageName === "model-lab") {
        loadMetrics();
        syncTrainingStatus();
    }

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

/* ACCOUNT ACCESS */

async function refreshAccountState() {
    try {
        accountState = await apiRequest(API.authState);
        const summary = $("account-summary");
        const counter = $("free-counter");
        const openButton = $("auth-open");
        const logoutButton = $("auth-logout");

        if (accountState.authenticated) {
            if (summary) summary.textContent = `Welcome, ${accountState.username}`;
            if (counter) counter.textContent = `${formatCount(accountState.predictions_used || 0)} saved predictions`;
            openButton?.classList.add("hidden");
            logoutButton?.classList.remove("hidden");
        } else {
            const remaining = Number(accountState.free_predictions_left ?? 0);
            if (summary) summary.textContent = `${remaining} free ${remaining === 1 ? "prediction" : "predictions"} left`;
            if (counter) counter.textContent = remaining > 0
                ? "Sign in after your first two predictions"
                : "Log in to keep predicting";
            openButton?.classList.remove("hidden");
            logoutButton?.classList.add("hidden");
        }
    } catch (error) {
        console.warn("Could not load account status:", error.message);
        if ($("free-counter")) $("free-counter").textContent = "Account status unavailable";
    }
}

function setAuthMode(mode) {
    const signingUp = mode === "signup";
    $("login-form")?.classList.toggle("hidden", signingUp);
    $("signup-form")?.classList.toggle("hidden", !signingUp);
    $("auth-login-tab")?.classList.toggle("active", !signingUp);
    $("auth-signup-tab")?.classList.toggle("active", signingUp);
    $("auth-login-tab")?.setAttribute("aria-selected", String(!signingUp));
    $("auth-signup-tab")?.setAttribute("aria-selected", String(signingUp));
    if ($("auth-status")) $("auth-status").textContent = "";
}

function openAuthModal(message = "Create a free account to keep predicting and save your research activity.") {
    const backdrop = $("auth-backdrop");
    if (!backdrop) return;
    backdrop.classList.remove("hidden");
    $("auth-message").textContent = message;
    setAuthMode("login");
    window.setTimeout(() => $("login-identifier")?.focus(), 30);
}

function closeAuthModal() {
    $("auth-backdrop")?.classList.add("hidden");
}

$("auth-open")?.addEventListener("click", () => openAuthModal());
$("auth-close")?.addEventListener("click", closeAuthModal);
$("auth-backdrop")?.addEventListener("click", (event) => {
    if (event.target.id === "auth-backdrop") closeAuthModal();
});
$("auth-login-tab")?.addEventListener("click", () => setAuthMode("login"));
$("auth-signup-tab")?.addEventListener("click", () => setAuthMode("signup"));
document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeAuthModal();
});

$("login-form")?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const submit = event.currentTarget.querySelector("button[type='submit']");
    submit.disabled = true;
    showMessage($("auth-status"), "Signing in…");
    try {
        const form = new FormData(event.currentTarget);
        const result = await apiRequest(API.login, {
            method: "POST",
            body: JSON.stringify({ identifier: form.get("identifier"), password: form.get("password") })
        });
        closeAuthModal();
        await refreshAccountState();
        if ($("page-review")?.classList.contains("active")) await loadReviews();
        showMessage($("result-status"), `Welcome back, ${result.username}.`);
    } catch (error) {
        showMessage($("auth-status"), error.message, true);
    } finally {
        submit.disabled = false;
    }
});

$("signup-form")?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const submit = event.currentTarget.querySelector("button[type='submit']");
    submit.disabled = true;
    showMessage($("auth-status"), "Creating your account…");
    try {
        const form = new FormData(event.currentTarget);
        const result = await apiRequest(API.signup, {
            method: "POST",
            body: JSON.stringify({
                email: form.get("email"),
                username: form.get("username"),
                password: form.get("password")
            })
        });
        closeAuthModal();
        await refreshAccountState();
        if ($("page-review")?.classList.contains("active")) await loadReviews();
        showMessage($("result-status"), `Your account is ready, ${result.username}.`);
    } catch (error) {
        showMessage($("auth-status"), error.message, true);
    } finally {
        submit.disabled = false;
    }
});

$("auth-logout")?.addEventListener("click", async () => {
    try {
        await apiRequest(API.logout, { method: "POST" });
        await refreshAccountState();
        showMessage($("result-status"), "You have logged out.");
    } catch (error) {
        alert("Unable to log out: " + error.message);
    }
});

/* STATUS */

async function loadStatus() {
    try {
        const status = await apiRequest(API.status);
        aiAvailable = Boolean(status.ai_available);
        syncPredictionModelOptions(status);

        if ($("ai-status")) {
            $("ai-status").textContent = aiAvailable
                ? "AI Guide is available."
                : "AI Guide is currently unavailable. Check your API configuration.";
        }
    } catch (error) {
        console.warn("Status unavailable:", error.message);
    }
}

function syncPredictionModelOptions(status) {
    const select = $("model-select");
    if (!select) return;
    const option = new Option(
        "Automatic · best validated model for each view",
        "auto",
        true,
        true
    );
    select.replaceChildren(option);
    select.value = "auto";
}

/* DRUG AND CELL-LINE OPTIONS */

async function loadOptions() {
    const drugA = $("drug-a");
    const drugB = $("drug-b");
    const drugOptions = $("drug-options");
    const cellLine = $("cell-line");
    const cellLineOptions = $("cell-line-options");

    if (!drugA || !drugB || !drugOptions || !cellLine || !cellLineOptions) return;

    if (drugOptions.dataset.loaded === "true" && cellLineOptions.dataset.loaded === "true") return;

    try {
        const data = await apiRequest(API.options);

        const drugs = data.drugs || data.drug_options || [];
        const cellLines = data.cell_lines || data.cell_line_options || [];

        drugOptions.innerHTML = "";
        cellLineOptions.innerHTML = "";

        drugs.forEach((drug) => {
            const name = typeof drug === "string" ? drug : drug.name || drug.label;
            if (!name) return;

            drugOptions.append(new Option(name, name));
        });

        cellLines.forEach((cell) => {
            const name = typeof cell === "string" ? cell : cell.name || cell.label;
            if (!name) return;

            cellLineOptions.append(new Option(name, name));
        });

        drugOptions.dataset.loaded = "true";
        cellLineOptions.dataset.loaded = "true";
    } catch (error) {
        drugOptions.innerHTML = "";
        cellLineOptions.innerHTML = "";
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
        deploymentStatus: data.deployment_status ?? "unknown",
        highestScore: data.highest_predicted_s_score ?? null,
        highestScoreFeature: data.highest_score_feature_set ?? null,
        bestValidatedScore: data.best_validated_s_score ?? null,
        bestValidatedFeature: data.best_validated_feature_set ?? null,
        bestValidatedModel: data.best_validated_model ?? null,
        target: data.target ?? "DrugComb S score",
        datasetLabel: data.dataset_label ?? "",
        trainingRows: data.training_rows ?? null,
        trainingCellLines: data.training_cell_lines ?? null,
        cellLineSeenInTraining: data.cell_line_seen_in_training ?? null,
        implementation: data.implementation ?? "—",
        modelVariant: data.model_variant ?? null,
        predictionId: data.prediction_id ?? null,
        freePredictionsLeft: data.free_predictions_left ?? null,
        message: data.selection_strategy ?? data.message ?? "",
        featureScores: Array.isArray(data.feature_scores) ? data.feature_scores : []
    };
}

function renderPrediction(rawData, predictionInput) {
    const result = $("prediction-result");
    const status = $("result-status");
    const data = normalizePrediction(rawData);
    const historySaved = savePredictionToHistory(data, predictionInput);

    const score = Number(data.score);
    const scoreText = Number.isFinite(score) ? score.toFixed(4) : "N/A";
    const scoreLabel = data.featureScores.length > 1
        ? "Highest predicted S_sum score"
        : (String(data.target).includes("S_sum") ? "Predicted S_sum score" : "Predicted S-score");
    const highestScore = Number(data.highestScore);
    const highestScoreNote = data.featureScores.length > 1 && Number.isFinite(highestScore)
        ? `Highest estimate · ${featureLabel(data.highestScoreFeature)}`
        : "";
    const trainingRowsText = Number.isFinite(Number(data.trainingRows))
        ? formatCount(Number(data.trainingRows))
        : "—";
    const trainingCellLinesText = data.trainingCellLines !== null && Number.isFinite(Number(data.trainingCellLines))
        ? formatCount(Number(data.trainingCellLines))
        : "—";
    const cellLineSeenText = data.cellLineSeenInTraining === true
        ? "Yes"
        : data.cellLineSeenInTraining === false
            ? "No · extrapolative"
            : "Not recorded";

    const isSynergistic = data.synergistic === true ||
        (data.synergistic === undefined && Number.isFinite(score) && score >= 10);
    const viewClassifications = new Set(data.featureScores
        .filter((item) => item.prediction_status !== "unavailable" && Number.isFinite(Number(item.predicted_s_score)))
        .map((item) => Number(item.predicted_s_score) >= 10));
    const viewsDisagree = viewClassifications.size > 1;
    const comparisonRowsHtml = data.featureScores.length > 1
        ? `
            <div class="prediction-view-comparison">
                <strong>Structure, DS, and DRS comparison</strong>
                <p class="comparison-method-note">The main score is the largest available S-score across the three views. “Best validated” marks the view with the strongest pair-held-out validation and cell-line support; a larger estimate does not by itself mean a more accurate model.</p>
                ${data.featureScores.map((item) => {
                    const viewScore = Number(item.predicted_s_score);
                    const available = item.prediction_status !== "unavailable" && Number.isFinite(viewScore);
                    const modelText = item.model ? ` · ${modelLabel(item.model)}` : "";
                    const variantText = item.model_variant ? ` · ${modelVariantLabel(item.model_variant)}` : "";
                    const titleReason = item.reason ?? item.model_selection_reason;
                    const title = titleReason ? ` title="${escapeHtml(titleReason)}"` : "";
                    const statusNote = available ? "S-score prediction" : escapeHtml(item.reason || "This view could not produce a score for the entered inputs.");
                    return `<div class="prediction-view-row${item.is_best ? " is-best" : ""}"${title}>
                        <div class="prediction-view-copy">
                            <span>${escapeHtml(featureLabel(item.feature_set))}${escapeHtml(modelText)}${escapeHtml(variantText)}</span>
                            <small>${statusNote}</small>
                        </div>
                        <div class="prediction-view-badges">
                            ${item.is_best ? '<span class="best-view-badge">BEST VALIDATED</span>' : ""}
                            ${item.is_highest_score ? '<span class="highest-view-badge">HIGHEST S-SCORE</span>' : ""}
                        </div>
                        <b>${available ? viewScore.toFixed(2) : "Unavailable"}</b>
                    </div>`;
                }).join("")}
            </div>
        `
        : "";

    status.textContent = "Completed";

    result.className = "prediction-result-content";
    result.innerHTML = `
        <div class="prediction-score">
            <strong>${escapeHtml(scoreText)}</strong>
            <span>${escapeHtml(scoreLabel)}</span>
            ${highestScoreNote ? `<small class="prediction-score-note">${escapeHtml(highestScoreNote)}</small>` : ""}
        </div>

        <div class="prediction-label">
            ${isSynergistic ? "✧ Predicted synergistic combination" : "Predicted non-synergistic combination"}
        </div>
        ${viewsDisagree ? '<p class="prediction-warning" role="status">The available feature views disagree on the S_sum ≥ 10 threshold. The main score and synergy label follow the highest available estimate; compare the individual estimates and validation results below.</p>' : ""}

        <div class="prediction-detail-grid">
            <div class="prediction-detail">
                <small>Highest-score feature view</small>
                <strong>${escapeHtml(data.feature)}</strong>
            </div>
            ${data.bestValidatedFeature ? `<div class="prediction-detail"><small>Best-validated view</small><strong>${escapeHtml(featureLabel(data.bestValidatedFeature))}${Number.isFinite(Number(data.bestValidatedScore)) ? ` · S-score ${Number(data.bestValidatedScore).toFixed(2)}` : ""}</strong></div>` : ""}
            <div class="prediction-detail">
                <small>Model</small>
                <strong>${escapeHtml(modelLabel(data.model))} · ${escapeHtml(data.deploymentStatus === "production" ? "active" : data.deploymentStatus === "candidate" ? "candidate" : data.deploymentStatus)}</strong>
            </div>
            <div class="prediction-detail">
                <small>Training target</small>
                <strong>${escapeHtml(data.target)}</strong>
            </div>
            <div class="prediction-detail">
                <small>Training rows for this view</small>
                <strong>${escapeHtml(trainingRowsText)}</strong>
            </div>
            <div class="prediction-detail">
                <small>Training cell lines</small>
                <strong>${escapeHtml(trainingCellLinesText)}</strong>
            </div>
            <div class="prediction-detail">
                <small>Selected line in training</small>
                <strong>${escapeHtml(cellLineSeenText)}</strong>
            </div>
            ${data.datasetLabel ? `<div class="prediction-detail"><small>Dataset</small><strong>${escapeHtml(data.datasetLabel)}</strong></div>` : ""}
            ${data.modelVariant ? `<div class="prediction-detail"><small>Model route</small><strong>${escapeHtml(modelVariantLabel(data.modelVariant))}</strong></div>` : ""}
            <div class="prediction-detail">
                <small>Implementation</small>
                <strong>${escapeHtml(data.implementation)}</strong>
            </div>
            <div class="prediction-detail">
                <small>Prediction status</small>
                <strong>Successful</strong>
            </div>
        </div>

        ${data.cellLineSeenInTraining === false ? '<p class="prediction-warning" role="status">This cell line was not represented in the selected feature view’s training rows. Treat this estimate as an extrapolation.</p>' : ""}
        ${data.deploymentStatus === "candidate" ? '<p class="prediction-warning" role="status">This is a full-data comparison candidate. Its pair-held-out validation score is research evidence, not proof of performance on a new dataset.</p>' : ""}
        ${data.message ? `<p>${escapeHtml(data.message)}</p>` : ""}
        ${comparisonRowsHtml}
        ${data.predictionId ? `<button class="button button-secondary review-result-button" type="button" data-review-prediction="${escapeHtml(data.predictionId)}">Review this prediction</button>` : ""}
    `;

    const historyNote = document.createElement("p");
    historyNote.className = "table-status";
    historyNote.setAttribute("role", "status");
    historyNote.textContent = historySaved
        ? "Saved in this browser's history."
        : "Prediction succeeded, but browser history could not be saved.";
    result.append(historyNote);
}

function normalizeDrugInput(value) {
    return String(value || "").trim().toUpperCase().replace(/[^A-Z0-9]/g, "");
}

$("predict-form")?.addEventListener("submit", async (event) => {
    event.preventDefault();

    const drugA = $("drug-a").value;
    const drugB = $("drug-b").value;
    const cellLine = $("cell-line").value;

    if (!drugA || !drugB || !cellLine) {
        alert("Enter or choose both drug names and a cell line.");
        return;
    }

    if (normalizeDrugInput(drugA) === normalizeDrugInput(drugB)) {
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

        renderPrediction(data, payload);
        await refreshAccountState();
        if (!accountState.authenticated && Number(data.free_predictions_left) === 0) {
            openAuthModal("You’ve used your 2 free predictions. Log in or create a free account to run the next one.");
        }
    } catch (error) {
        status.textContent = "Error";
        result.className = "result-placeholder";
        result.innerHTML = `
            <div class="placeholder-icon">!</div>
            <h3>Prediction failed</h3>
            <p>${escapeHtml(error.message)}</p>
            ${error.message.toLowerCase().includes("free predictions") ? '<button class="button button-primary" type="button" data-open-auth>Log in or sign up</button>' : ""}
        `;
        if (error.message.toLowerCase().includes("free predictions")) {
            openAuthModal("Your 2 free predictions have been used. Log in or create a free account to continue.");
        }
    } finally {
        button.disabled = false;
        button.innerHTML = "<span>Run prediction</span><span>→</span>";
    }
});

/* MODEL METRICS */

function metricValue(row, key) {
    const aliases = {
        mse: ["mse", "MSE"],
        rmse: ["rmse", "RMSE"],
        r2: ["r2", "R2", "r_squared", "R²"],
        pearson: ["pearson", "Pearson"],
        spearman: ["spearman", "Spearman"],
        auc_s_ge_10: ["auc_s_ge_10", "auc", "AUC"],
        balanced_accuracy_s_ge_10: ["balanced_accuracy_s_ge_10", "balanced_accuracy"],
        f1_s_ge_10: ["f1_s_ge_10", "f1", "F1"]
    };

    for (const alias of aliases[key] || [key]) {
        if (row[alias] !== undefined && row[alias] !== null && row[alias] !== "") {
            const value = Number(row[alias]);
            return Number.isFinite(value) ? value : null;
        }
    }

    return null;
}

function featureLabel(value) {
    const labels = {
        structure: "Structure",
        ds: "DS",
        drs: "DRS",
        paper_drs: "Author DRS + multi-omics",
        structure_pubchem: "Structure · expanded coverage",
        paper_drs_attention: "Author features · row split",
        paper_drs_unseen_pair: "Author features · unseen pairs",
        multilevel: "Multilevel"
    };
    const key = String(value || "").toLowerCase();
    return labels[key] || value || "Unknown";
}

function modelLabel(value) {
    const labels = {
        lasso: "LASSO",
        random_forest: "Random Forest",
        adaboost: "AdaBoost",
        xgboost: "XGBoost",
        xgboost_tuned: "XGBoost · tuned (experimental)",
        xgboost_regularized: "XGBoost · regularized (experimental)",
        xgboost_pairwise: "XGBoost · pairwise (experimental)",
        synergyx: "SynergyX",
        synergyx_attention: "SynergyX-inspired attention",
        synergyx_gdsc_reference: "SynergyX · GDSC-only reference",
        synergyx_prism_fallback: "SynergyX · PRISM fallback (deployed)",
        synergyx_prism_new_coverage: "SynergyX · newly covered rows",
        xgboost_pair_routed: "XGBoost · pair-routed",
        xgboost_expanded_global: "XGBoost · expanded global",
        stacked_random_forest: "Stacked Random Forest"
    };
    const key = String(value || "").toLowerCase();
    return labels[key] || String(value || "Unknown").replaceAll("_", " ");
}

function modelVariantLabel(value) {
    const labels = {
        structure_legacy_coverage: "Existing structure coverage",
        structure_pubchem_expanded: "PubChem-expanded structures"
    };
    return labels[String(value || "").toLowerCase()] || String(value || "").replaceAll("_", " ");
}

function formatCount(value) {
    if (value === undefined || value === null || value === "") return "—";
    const count = Number(value);
    return Number.isFinite(count) ? Math.round(count).toLocaleString() : "—";
}

function datasetLabel(row) {
    const label = String(row?.dataset_label || "").trim();
    return label || "Unlabeled evaluation";
}

function activeMetricRows() {
    const selected = $("graph-dataset")?.value || "all";
    return comparisonRows.filter((row) => selected === "all" || datasetLabel(row) === selected);
}

function refreshDatasetFilter() {
    const select = $("graph-dataset");
    if (!select) return;
    const previous = select.value || "all";
    const labels = [...new Set(comparisonRows.map(datasetLabel))].sort((a, b) => a.localeCompare(b));
    const preferred = labels.find((label) => /full-data all-model comparison/i.test(label));
    select.innerHTML = '<option value="all">All datasets</option>' + labels.map((label) =>
        `<option value="${escapeHtml(label)}">${escapeHtml(label)}</option>`
    ).join("");
    select.value = previous !== "all" && labels.includes(previous)
        ? previous
        : preferred || "all";
}

function renderMetrics(rows) {
    const body = $("metrics-body");
    if (!body) return;

    rows = activeMetricRows();

    if (!rows.length) {
        body.innerHTML = '<tr><td colspan="12" class="empty-cell">No model metrics are available for this dataset.</td></tr>';
        if ($("metrics-count")) $("metrics-count").textContent = "0 comparisons";
        renderComparisonGraph();
        renderFeatureSummary();
        return;
    }

    const coverageByFeature = new Map();
    rows.forEach((row) => {
        const feature = String(row.feature_set ?? row.feature ?? "unknown").toLowerCase();
        const key = `${datasetLabel(row)}|${feature}`;
        if (row.rows_evaluated === undefined || row.rows_evaluated === null || row.rows_evaluated === "") return;
        const count = Number(row.rows_evaluated);
        if (!Number.isFinite(count)) return;
        if (!coverageByFeature.has(key)) coverageByFeature.set(key, new Set());
        coverageByFeature.get(key).add(count);
    });

    const bestByFeature = new Map();
    rows.forEach((row) => {
        const feature = String(row.feature_set ?? row.feature ?? "unknown").toLowerCase();
        const key = `${datasetLabel(row)}|${feature}`;
        const model = String(row.model ?? row.model_name ?? "").toLowerCase();
        if (["xgboost_pairwise", "xgboost_regularized", "xgboost_tuned"].includes(model)) return;
        if ((coverageByFeature.get(key)?.size || 0) > 1) return;
        const score = metricValue(row, "rmse");
        const current = bestByFeature.get(key);
        if (score !== null && (!current || score < current.score)) {
            bestByFeature.set(key, { score, model: row.model ?? row.model_name });
        }
    });

    const sortedRows = [...rows].sort((left, right) => {
        const datasetDifference = datasetLabel(left).localeCompare(datasetLabel(right));
        if (datasetDifference) return datasetDifference;
        const leftFeature = String(left.feature_set ?? left.feature ?? "unknown").toLowerCase();
        const rightFeature = String(right.feature_set ?? right.feature ?? "unknown").toLowerCase();
        const featureDifference = featureOrder.indexOf(leftFeature) - featureOrder.indexOf(rightFeature);
        if (featureDifference) return featureDifference;
        return (metricValue(left, "rmse") ?? Infinity) - (metricValue(right, "rmse") ?? Infinity);
    });

    const formatCell = (row, key) => {
        const value = metricValue(row, key);
        if (value === null) return '<span class="metric-missing">—</span>';
        const low = Number(row[`${key}_ci95_low`]);
        const high = Number(row[`${key}_ci95_high`]);
        const title = Number.isFinite(low) && Number.isFinite(high)
            ? ` title="Approximate fold-based interval ${low.toFixed(3)} to ${high.toFixed(3)}; folds are not independent"`
            : "";
        return `<span class="metric-number"${title}>${formatMetric(value)}</span>`;
    };

    body.innerHTML = sortedRows.map((row) => {
        const featureKey = String(row.feature_set ?? row.feature ?? "unknown").toLowerCase();
        const modelKey = row.model ?? row.model_name ?? "unknown";
        const best = bestByFeature.get(`${datasetLabel(row)}|${featureKey}`);
        const isBest = best?.model === modelKey;
        const isUnstable = String(row.evaluation_status || "").toLowerCase() === "unstable";
        const featureClass = featureKey.replace(/[^a-z0-9-]/g, "");

        return `
            <tr>
                <td>${escapeHtml(datasetLabel(row))}</td>
                <td><span class="feature-tag feature-${escapeHtml(featureClass)}">${escapeHtml(featureLabel(featureKey))}</span></td>
                <td><span class="model-name">${escapeHtml(modelLabel(modelKey))}</span>${isBest ? '<span class="best-badge">Best RMSE</span>' : ""}${isUnstable ? '<span class="unstable-badge" title="This L1 fit had extreme held-out error and is excluded from prediction choices.">Unstable fit</span>' : ""}</td>
                <td>${formatCell(row, "mse")}</td>
                <td class="metric-primary">${formatCell(row, "rmse")}</td>
                <td>${formatCell(row, "clip3_rmse")}</td>
                <td>${formatCell(row, "r2")}</td>
                <td>${formatCell(row, "spearman")}</td>
                <td>${formatCell(row, "auc_s_ge_10")}</td>
                <td>${formatCell(row, "balanced_accuracy_s_ge_10")}</td>
                <td>${formatCell(row, "f1_s_ge_10")}</td>
                <td title="${escapeHtml(row.unique_pairs_evaluated ? `${formatCount(row.unique_pairs_evaluated)} drug pairs` : "")}">${escapeHtml(row.folds_evaluated ?? row.folds ?? "—")} · ${escapeHtml(formatCount(row.rows_evaluated))}</td>
            </tr>
        `;
    }).join("");

    if ($("metrics-count")) {
        const models = new Set(rows.map((row) => row.model ?? row.model_name).filter(Boolean));
        const views = new Set(rows.map((row) => row.feature_set ?? row.feature).filter(Boolean));
        const datasets = new Set(rows.map(datasetLabel));
        $("metrics-count").textContent = `${rows.length} comparisons · ${models.size} models · ${views.size} views · ${datasets.size} datasets`;
    }

    renderComparisonGraph();
    renderFeatureSummary();
}

function formatMetric(value) {
    return value === null || !Number.isFinite(value) ? "—" : value.toFixed(3);
}

async function loadMetrics() {
    const body = $("metrics-body");
    if (body) body.innerHTML = '<tr><td colspan="12" class="empty-cell">Loading metrics...</td></tr>';

    try {
        const data = await apiRequest(API.metrics);
        comparisonRows = Array.isArray(data)
            ? data
            : data.rows || data.results || data.metrics || [];
        refreshDatasetFilter();
        renderMetrics(comparisonRows);

        if ($("metrics-updated")) {
            $("metrics-updated").textContent = comparisonRows.length
                ? "Showing saved evaluations; fold count, validation split, and feature coverage vary by dataset. Each row shows its own fold and row counts."
                : data.message || "No model evaluation results are available yet.";
        }
    } catch (error) {
        if (body) {
            body.innerHTML = `<tr><td colspan="12" class="empty-cell">${escapeHtml(error.message)}</td></tr>`;
        }
        if ($("metrics-updated")) $("metrics-updated").textContent = "Unable to load metrics.";
    }
}

function setTrainingStatus(message, isRunning = false) {
    const button = $("start-quick-evaluation");
    const status = $("training-status");

    if (button) {
        button.disabled = isRunning;
        button.textContent = isRunning ? "Evaluation running..." : "Run quick comparison";
    }
    if (status) status.textContent = message;
}

async function pollTrainingStatus() {
    if (trainingPollTimer !== null) {
        window.clearTimeout(trainingPollTimer);
        trainingPollTimer = null;
    }

    try {
        const data = await apiRequest(API.status);
        const training = data.training || {};
        const isRunning = ["queued", "running"].includes(training.status);
        const message = training.status === "idle"
            ? "Quick run: 2,000 rows, 3 pair-grouped folds, 1 repeat, five model families and a structure-only XGBoost variant when available."
            : training.message || "Checking evaluation status.";

        setTrainingStatus(message, isRunning);

        if (isRunning) {
            trainingPollTimer = window.setTimeout(pollTrainingStatus, 1500);
        } else if (training.status === "complete") {
            await loadMetrics();
        }
    } catch (error) {
        setTrainingStatus("Unable to read evaluation status: " + error.message);
    }
}

async function syncTrainingStatus() {
    try {
        const data = await apiRequest(API.status);
        const training = data.training || {};
        const isRunning = ["queued", "running"].includes(training.status);
        const message = training.status === "idle"
            ? "Quick run: 2,000 rows, 3 pair-grouped folds, 1 repeat, five model families and a structure-only XGBoost variant when available."
            : training.message || "Ready to start an evaluation.";

        setTrainingStatus(message, isRunning);
        if (isRunning && trainingPollTimer === null) {
            trainingPollTimer = window.setTimeout(pollTrainingStatus, 1500);
        }
    } catch (error) {
        setTrainingStatus("Unable to read evaluation status: " + error.message);
    }
}

async function startQuickEvaluation() {
    const button = $("start-quick-evaluation");
    if (!button || button.disabled) return;

    setTrainingStatus("Starting quick model comparison...", true);
    try {
        const response = await apiRequest(API.train, {
            method: "POST",
            body: JSON.stringify({
                feature_sets: ["structure", "ds", "drs"],
                models: ["lasso", "random_forest", "adaboost", "xgboost", "xgboost_pairwise", "synergyx"],
                max_rows: 2000,
                folds: 3,
                repeats: 1,
                synergyx_epochs: 40
            })
        });
        setTrainingStatus(response.message || "Quick model comparison started.", true);
        await pollTrainingStatus();
    } catch (error) {
        setTrainingStatus(error.message);
    }
}

$("start-quick-evaluation")?.addEventListener("click", startQuickEvaluation);

function renderComparisonGraph() {
    const container = $("model-comparison-graphs");
    const selectedFeature = $("graph-feature")?.value || "all";

    if (!container) return;

    const modelOrder = ["lasso", "random_forest", "adaboost", "xgboost", "synergyx"];
    const featureViews = selectedFeature === "all" ? ["ds", "drs", "structure"] : [selectedFeature];
    const normalizeKey = (value) => String(value || "").trim().toLowerCase().replace(/[\s-]+/g, "_");
    const candidates = activeMetricRows().filter((row) =>
        modelOrder.includes(normalizeKey(row.model ?? row.model_name))
    );
    const metricConfigs = [
        { key: "rmse", title: "RMSE", direction: "Lower is better", scale: "log" },
        { key: "mse", title: "MSE", direction: "Lower is better", scale: "log" },
        { key: "r2", title: "R²", direction: "Higher is better", scale: "symlog" }
    ];

    const showNumber = (value) => {
        const absolute = Math.abs(value);
        return absolute >= 1e6 || (absolute > 0 && absolute < 0.001)
            ? value.toExponential(2)
            : value.toFixed(3);
    };

    const renderMetricCard = ({ key, title, direction, scale: requestedScale }) => {
        const rows = featureViews.flatMap((feature) => modelOrder.map((model) => {
            const source = candidates.find((row) =>
                normalizeKey(row.feature_set ?? row.feature) === feature
                && normalizeKey(row.model ?? row.model_name) === model
            );
            return {
                feature,
                model,
                rows: source ? Number(source.rows_evaluated) : NaN,
                value: source ? metricValue(source, key) : null
            };
        }));
        const values = rows.map((item) => item.value).filter(Number.isFinite);
        const valueCount = values.length;
        const titleText = `${title} · ${direction}`;
        const missingNote = key === "mse" && selectedFeature === "all"
            ? '<p class="metric-data-note">MSE was not saved for the DS and DRS runs; those model rows are shown as unavailable.</p>'
            : "";

        if (!rows.length || !valueCount) {
            return `<section class="model-metric-card"><div class="model-metric-card-heading"><h3>${titleText}</h3><span>${valueCount} values</span></div>${missingNote}<p class="empty-cell">No saved ${title} values are available for this dataset and feature view.</p></section>`;
        }

        let scaleMode = requestedScale;
        if (scaleMode === "log" && values.some((value) => value <= 0)) scaleMode = "symlog";
        const transform = scaleMode === "log"
            ? (value) => Math.log10(value)
            : (value) => Math.sign(value) * Math.log10(1 + Math.abs(value));
        const inverse = scaleMode === "log"
            ? (value) => 10 ** value
            : (value) => Math.sign(value) * (10 ** Math.abs(value) - 1);
        const transformed = values.map(transform);
        let minValue = Math.min(...transformed);
        let maxValue = Math.max(...transformed);
        if (minValue === maxValue) maxValue = minValue + 1;

        const width = 1200;
        const labelWidth = 280;
        const valueColumn = 150;
        const plotLeft = labelWidth;
        const plotRight = width - valueColumn;
        const plotWidth = plotRight - plotLeft;
        const rowTop = 43;
        const rowHeight = 32;
        const height = rowTop + rows.length * rowHeight + 20;
        const scaleX = (value) => plotLeft + ((value - minValue) / (maxValue - minValue)) * plotWidth;
        const ticks = Array.from({ length: 5 }, (_, index) =>
            minValue + ((maxValue - minValue) * index) / 4
        );
        const grid = ticks.map((tick) => {
            const x = scaleX(tick);
            return `
                <line class="chart-gridline" x1="${x}" y1="27" x2="${x}" y2="${height - 6}" />
                <text class="chart-tick" x="${x}" y="18" text-anchor="middle">${escapeHtml(showNumber(inverse(tick)))}</text>
            `;
        }).join("");
        const zeroX = scaleMode === "log" ? null : scaleX(transform(0));
        const bars = rows.map((item, index) => {
            const y = rowTop + index * rowHeight;
            const featureClass = item.feature.replace(/[^a-z0-9-]/g, "");
            const label = `${featureLabel(item.feature)} · ${modelLabel(item.model)}`;
            const valueX = item.value === null ? null : scaleX(transform(item.value));
            const rectX = valueX === null ? 0 : (zeroX === null ? plotLeft : Math.min(valueX, zeroX));
            const rectWidth = valueX === null ? 0 : Math.max(1.5, Math.abs(valueX - (zeroX === null ? plotLeft : zeroX)));
            const valueText = item.value === null ? "Not recorded" : showNumber(item.value);
            const tooltip = item.value === null
                ? `${label}: metric not recorded in the saved evaluation`
                : `${label}: ${item.value}; ${formatCount(item.rows)} usable rows`;
            return `
                <g class="chart-entry">
                    <title>${escapeHtml(tooltip)}</title>
                    <text class="chart-row-label" x="10" y="${y + 17}">${escapeHtml(label)}</text>
                    ${valueX === null ? "" : `<rect class="chart-bar bar-${escapeHtml(featureClass)}" x="${rectX}" y="${y + 5}" width="${rectWidth}" height="16" rx="6" />`}
                    <text class="chart-row-value${item.value === null ? " chart-value-missing" : ""}" x="${width - 12}" y="${y + 18}" text-anchor="end">${escapeHtml(valueText)}</text>
                </g>
            `;
        }).join("");
        const scaleText = scaleMode === "log" ? "Log scale · values shown at right" : "Symmetric log scale · values shown at right";

        return `
            <section class="model-metric-card">
                <div class="model-metric-card-heading">
                    <div><h3>${titleText}</h3><p>${escapeHtml(datasetLabel(candidates[0] || {}))} · ${valueCount} of ${rows.length} values recorded</p></div>
                    <span class="metric-scale-label">${scaleText}</span>
                </div>
                ${missingNote}
                <div class="comparison-chart-scroll">
                    <svg class="comparison-chart-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${title} comparison for LASSO, Random Forest, AdaBoost, XGBoost, and SynergyX">
                        ${grid}
                        ${zeroX === null ? "" : `<line class="chart-zero-line" x1="${zeroX}" y1="26" x2="${zeroX}" y2="${height - 6}" />`}
                        ${bars}
                    </svg>
                </div>
                <div class="chart-legend">
                    ${featureViews.map((feature) => `<span><i class="legend-dot dot-${feature}"></i>${escapeHtml(featureLabel(feature))}</span>`).join("")}
                </div>
            </section>
        `;
    };

    if (!candidates.length) {
        container.innerHTML = '<p class="empty-cell">No saved comparison rows are available for these five model families in this dataset.</p>';
        return;
    }

    container.innerHTML = `<div class="model-metric-grid">${metricConfigs.map(renderMetricCard).join("")}</div>`;
}

function renderFeatureSummary() {
    const container = $("feature-summary-content");
    if (!container) return;

    const visibleRows = activeMetricRows();

    const groups = [...new Map(visibleRows.map((row) => {
        const feature = String(row.feature_set ?? row.feature ?? "").toLowerCase();
        return [`${datasetLabel(row)}|${feature}`, { dataset: datasetLabel(row), feature }];
    })).values()].filter((group) => featureOrder.includes(group.feature));

    if (!groups.length) {
        container.innerHTML = '<div class="feature-summary-card"><strong>No comparison loaded</strong><span>Refresh results or run an evaluation to populate this view.</span></div>';
        return;
    }

    container.innerHTML = groups.map(({ dataset, feature }) => {
        const rows = visibleRows.filter((row) =>
            datasetLabel(row) === dataset
            && String(row.feature_set ?? row.feature ?? "").toLowerCase() === feature
        );
        const coverageCounts = [...new Set(rows.map((row) => Number(row.rows_evaluated)).filter(Number.isFinite))];
        const deployableRows = rows.filter((row) =>
            !(["xgboost_pairwise", "xgboost_regularized", "xgboost_tuned"].includes(String(row.model ?? row.model_name ?? "").toLowerCase()))
        );
        const best = deployableRows.filter((row) => metricValue(row, "rmse") !== null)
            .sort((left, right) => metricValue(left, "rmse") - metricValue(right, "rmse"))[0];
        const modelCount = new Set(rows.map((row) => row.model ?? row.model_name).filter(Boolean)).size;
        const folds = best?.folds_evaluated ?? best?.folds ?? "—";
        const bestScore = best ? formatMetric(metricValue(best, "rmse")) : "—";

        if (coverageCounts.length > 1) {
            return `
                <div class="feature-summary-card feature-card-${feature}">
                    <span class="summary-kicker">${escapeHtml(dataset)} · ${escapeHtml(featureLabel(feature))}</span>
                    <strong>Coverage varies by model</strong>
                    <span class="summary-score">${coverageCounts.map(formatCount).join(" / ")}<small> rows</small></span>
                    <span>${modelCount} models · compare results within matching coverage</span>
                </div>
            `;
        }

        return `
                <div class="feature-summary-card feature-card-${feature}">
                    <span class="summary-kicker">${escapeHtml(dataset)} · ${escapeHtml(featureLabel(feature))}</span>
                <strong>${escapeHtml(best ? modelLabel(best.model ?? best.model_name) : "No RMSE")}</strong>
                <span class="summary-score">${bestScore}<small> RMSE</small></span>
                <span>${modelCount} model${modelCount === 1 ? "" : "s"} · ${folds} folds</span>
            </div>
        `;
    }).join("");
}

$("refresh-results")?.addEventListener("click", loadMetrics);
$("graph-feature")?.addEventListener("change", renderComparisonGraph);
$("graph-dataset")?.addEventListener("change", () => renderMetrics(comparisonRows));

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

/* PREDICTION REVIEWS */

function updateReviewSelection() {
    const selected = reviewPredictions.find(
        (item) => String(item.id) === String($("review-prediction")?.value || "")
    );
    const preview = $("review-selected-preview");
    if (!preview) return;
    preview.textContent = selected
        ? `${selected.drug_a} + ${selected.drug_b} · ${selected.cell_line} · score ${Number.isFinite(Number(selected.predicted_score)) ? Number(selected.predicted_score).toFixed(2) : "not recorded"}`
        : "Select one of your recent predictions to continue.";
}

function renderReviewPredictions() {
    const select = $("review-prediction");
    if (!select) return;
    const selectedBefore = select.value;
    if (!reviewPredictions.length) {
        select.innerHTML = '<option value="">No predictions available yet</option>';
        select.disabled = true;
        $("review-submit")?.setAttribute("disabled", "disabled");
        updateReviewSelection();
        return;
    }
    select.disabled = false;
    $("review-submit")?.removeAttribute("disabled");
    select.innerHTML = '<option value="">Choose a prediction…</option>' + reviewPredictions.map((item) => {
        const score = Number.isFinite(Number(item.predicted_score)) ? Number(item.predicted_score).toFixed(2) : "—";
        const label = `${item.drug_a} + ${item.drug_b} · ${item.cell_line} · score ${score}`;
        return `<option value="${escapeHtml(item.id)}">${escapeHtml(label)}</option>`;
    }).join("");
    if (selectedBefore && reviewPredictions.some((item) => String(item.id) === selectedBefore)) {
        select.value = selectedBefore;
    }
    if (pendingReviewPredictionId && reviewPredictions.some((item) => String(item.id) === pendingReviewPredictionId)) {
        select.value = pendingReviewPredictionId;
        pendingReviewPredictionId = null;
    }
    updateReviewSelection();
}

function renderReviewCards(rows) {
    const container = $("review-list");
    if (!container) return;
    if (!rows.length) {
        container.innerHTML = '<div class="review-empty"><span>✦</span><p>No reviews submitted yet. Your feedback will appear here.</p></div>';
        return;
    }
    container.innerHTML = rows.map((item) => {
        const outcome = item.outcome === "agrees"
            ? "Agreed with experimental result"
            : item.outcome === "differs"
                ? "Differed from experimental result"
                : "Not experimentally checked";
        const date = item.created_at ? new Date(item.created_at).toLocaleDateString() : "Date unavailable";
        return `<article class="review-card">
            <div class="review-card-top"><strong>${escapeHtml(item.drug_a)} + ${escapeHtml(item.drug_b)}</strong><span>${escapeHtml(date)}</span></div>
            <p>${escapeHtml(item.cell_line)} · predicted score ${Number.isFinite(Number(item.predicted_score)) ? Number(item.predicted_score).toFixed(2) : "—"}</p>
            <div class="review-card-rating"><span aria-label="${escapeHtml(item.rating)} out of 5">${"★".repeat(Number(item.rating))}${"☆".repeat(5 - Number(item.rating))}</span><b>${escapeHtml(outcome)}</b></div>
            ${item.comment ? `<p class="review-card-comment">${escapeHtml(item.comment)}</p>` : ""}
        </article>`;
    }).join("");
}

async function loadReviews() {
    try {
        const data = await apiRequest(API.reviews);
        reviewPredictions = Array.isArray(data.predictions) ? data.predictions : [];
        renderReviewPredictions();
        renderReviewCards(Array.isArray(data.reviews) ? data.reviews : []);
        if ($("review-status")) $("review-status").textContent = "";
    } catch (error) {
        if ($("review-list")) $("review-list").innerHTML = `<div class="review-empty"><p>${escapeHtml(error.message)}</p></div>`;
        showMessage($("review-status"), error.message, true);
    }
}

$("review-prediction")?.addEventListener("change", updateReviewSelection);
$("refresh-reviews")?.addEventListener("click", loadReviews);
$("review-form")?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const submit = $("review-submit");
    submit.disabled = true;
    showMessage($("review-status"), "Saving your review…");
    try {
        const result = await apiRequest(API.reviews, {
            method: "POST",
            body: JSON.stringify({
                prediction_id: $("review-prediction").value,
                rating: Number($("review-rating").value),
                outcome: $("review-outcome").value,
                comment: $("review-comment").value
            })
        });
        showMessage($("review-status"), result.message || "Review saved.");
        $("review-comment").value = "";
        await loadReviews();
    } catch (error) {
        showMessage($("review-status"), error.message, true);
    } finally {
        submit.disabled = false;
    }
});

document.addEventListener("click", (event) => {
    const reviewButton = event.target.closest("[data-review-prediction]");
    if (reviewButton) {
        pendingReviewPredictionId = reviewButton.dataset.reviewPrediction;
        navigateTo("review");
        return;
    }
    if (event.target.closest("[data-open-auth]")) openAuthModal();
});

function readPredictionHistory() {
    try {
        const rows = JSON.parse(window.localStorage.getItem(HISTORY_STORAGE_KEY) || "[]");
        return Array.isArray(rows)
            ? rows.filter((row) => row && typeof row === "object")
            : [];
    } catch (error) {
        console.warn("Could not read browser history:", error.message);
        return [];
    }
}

function savePredictionToHistory(data, predictionInput) {
    const record = {
        id: globalThis.crypto?.randomUUID?.()
            || String(Date.now()) + "-" + Math.random().toString(36).slice(2),
        drug_a: predictionInput.drug_a,
        drug_b: predictionInput.drug_b,
        cell_line: predictionInput.cell_line,
        predicted_s_score: Number.isFinite(Number(data.score)) ? Number(data.score) : null,
        feature_set: data.feature,
        model: data.model,
        implementation: data.implementation,
        timestamp: new Date().toISOString()
    };

    try {
        historyRows = [record, ...readPredictionHistory()].slice(0, 250);
        window.localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(historyRows));
        return true;
    } catch (error) {
        console.warn("Could not save browser history:", error.message);
        return false;
    }
}

/* HISTORY */
function loadHistory() {
    historyRows = readPredictionHistory();
    const query = $("history-search")?.value.trim().toLowerCase() || "";
    const visibleRows = query
        ? historyRows.filter((row) => {
            const text = [
                row.drug_a, row.drug_b, row.drug1, row.drug2,
                row.cell_line, row.model, row.feature_set
            ].join(" ").toLowerCase();
            return text.includes(query);
        })
        : historyRows;
    renderHistory(visibleRows);
}

function renderHistory(rows) {
    const container = $("history-list");
    if (!container) return;

    if (!rows.length) {
        container.innerHTML = `
            <div class="panel empty-history">
                <div class="placeholder-icon">◷</div>
                <h3>No predictions yet</h3>
                <p>Successful predictions are saved in this browser profile only.</p>
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
    const query = event.target.value.trim().toLowerCase();

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
        const remaining = readPredictionHistory().filter(
            (row) => String(row.id ?? row.record_id ?? "") !== id
        );
        window.localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(remaining));
        await loadHistory();
    } catch (error) {
        alert("Unable to delete history: " + error.message);
    }
});

/* INITIALIZE */

async function initializeApp() {
    buildDnaHelix();
    await loadStatus();
    await refreshAccountState();
    await loadReviews();

    const initialPage = location.hash.replace("#", "") || "home";
    navigateTo(initialPage);

    const visual = $("hero-visual");
    if (visual) {
        visual.addEventListener("pointermove", (event) => {
            const bounds = visual.getBoundingClientRect();
            const x = (event.clientX - bounds.left) / bounds.width;
            const y = (event.clientY - bounds.top) / bounds.height;
            visual.style.setProperty("--tilt-x", `${(0.5 - y) * 9}deg`);
            visual.style.setProperty("--tilt-y", `${(x - 0.5) * 12}deg`);
            visual.style.setProperty("--glow-x", `${x * 100}%`);
            visual.style.setProperty("--glow-y", `${y * 100}%`);
        });
        visual.addEventListener("pointerleave", () => {
            visual.style.setProperty("--tilt-x", "0deg");
            visual.style.setProperty("--tilt-y", "0deg");
            visual.style.setProperty("--glow-x", "50%");
            visual.style.setProperty("--glow-y", "50%");
        });
    }
}

document.addEventListener("DOMContentLoaded", initializeApp);
