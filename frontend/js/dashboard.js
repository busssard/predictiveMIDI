import { LossChart } from './loss-chart.js';

// --- API ---
const API = {
    base: '/api',
    async get(path) {
        const r = await fetch(this.base + path);
        return r.json();
    },
    async post(path, body = {}) {
        const r = await fetch(this.base + path, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        return r.json();
    },

    status: ()     => API.get('/training/status/'),
    runs:   ()     => API.get('/training/runs/'),
    corpus: ()     => API.get('/corpus/stats/'),
    metrics: (id)  => API.get('/training/metrics/?run_id=' + id),
    start:  (cfg)  => API.post('/training/start/', cfg),
    stop:   ()     => API.post('/training/stop/'),
};

// --- Field docs ---
const FIELD_DOCS = {
    pos_weight: 'Upweights active (non-zero) notes at input/output boundaries. Counteracts the ~96% silence in piano rolls. Default 20 = active notes matter 20\u00d7 more than silence.',
    lambda_sparse: 'L1 sparsity penalty on representations: <code>r -= \u03bb\u00b7sign(r)</code>. Pushes inactive neurons toward zero. Start with 0.01.',
};

// --- DOM ---
const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);

// --- State ---
let isRunning = false;
let selectedRunId = null;
let chart = null;
let selectedLayoutId = null;
let selectedLayoutData = null;
let pollTimer = null;

// --- Init ---
async function init() {
    chart = new LossChart($('#loss-canvas'));
    chart.resize();
    window.addEventListener('resize', () => {
        chart.resize();
        if (selectedRunId) loadRunMetrics(selectedRunId);
    });

    $('#btn-train').addEventListener('click', handleTrainButton);
    $('#cfg-layout').addEventListener('change', handleLayoutChange);

    await Promise.all([
        loadCorpus(),
        loadRuns(),
        checkStatus(),
        loadLayouts(),
    ]);
}

// --- Corpus ---
async function loadCorpus() {
    try {
        const data = await API.corpus();
        let html = `<span class="val">${data.total_songs}</span> songs`;
        if (data.by_dataset) {
            const parts = Object.entries(data.by_dataset)
                .map(([k, v]) => `${k}: ${v}`)
                .join(' / ');
            html += `<div class="sub">${parts}</div>`;
        }
        if (data.vocabulary) {
            const cats = Object.keys(data.vocabulary).join(', ');
            html += `<div class="sub">${data.num_instruments} categories: ${cats}</div>`;
        }
        $('#corpus-info').innerHTML = html;
    } catch {
        $('#corpus-info').innerHTML = '<span class="err">index not found</span>';
    }
}

// --- Status ---
async function checkStatus() {
    try {
        const st = await API.status();
        isRunning = st.running;
        updateStatusUI(st);
    } catch {
        isRunning = false;
        updateStatusUI({ running: false });
    }
}

function updateStatusUI(st) {
    const dot = $('#status-dot');
    const label = $('#status-label');
    const btn = $('#btn-train');

    if (st.running) {
        dot.className = 'dot running';
        label.textContent = 'TRAINING';
        label.className = 'status-text running';
        btn.textContent = 'STOP';
        btn.className = 'btn-train running';
        setConfigDisabled(true);
        startPolling();
    } else {
        dot.className = 'dot idle';
        label.textContent = 'IDLE';
        label.className = 'status-text';
        btn.textContent = 'START TRAINING';
        btn.className = 'btn-train';
        setConfigDisabled(false);
        stopPolling();
    }

    if (st.error) {
        $('#status-error').textContent = st.error;
        $('#status-error').style.display = 'block';
    } else {
        $('#status-error').style.display = 'none';
    }

    // Training progress stats
    if (st.step != null) {
        $('#stat-step').textContent = `${st.step}/${st.num_steps || '?'}`;
    }
    if (st.step_time != null) {
        $('#stat-step-time').textContent = `${st.step_time}s/step`;
    }
    if (st.latest_error != null) {
        $('#stat-error').textContent = st.latest_error.toFixed(6);
    } else if (st.run_id != null) {
        // Try to show from run data
    }
    if (st.active_error != null) {
        $('#stat-active-error').textContent = st.active_error.toFixed(6);
    } else {
        $('#stat-active-error').textContent = '-';
    }
    if (st.f1 != null) {
        $('#stat-f1').textContent = st.f1.toFixed(3);
        $('#stat-prec').textContent = (st.precision || 0).toFixed(3);
        $('#stat-recall').textContent = (st.recall || 0).toFixed(3);
    }
    if (st.current_phase != null) {
        $('#stat-phase').textContent = st.current_phase;
    }
}

function startPolling() {
    if (pollTimer) return;
    pollTimer = setInterval(async () => {
        try {
            const st = await API.status();
            isRunning = st.running;
            updateStatusUI(st);
            // Update chart and runs table with latest data
            if (st.run_id) {
                await loadRunMetrics(st.run_id);
            }
            await loadRuns();
            if (!st.running) {
                stopPolling();
            }
        } catch { /* ignore */ }
    }, 3000);
}

function stopPolling() {
    if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
    }
}

function setConfigDisabled(disabled) {
    $$('.config-input').forEach(el => el.disabled = disabled);
}

// --- Training control ---
async function handleTrainButton() {
    const btn = $('#btn-train');
    btn.disabled = true;

    try {
        if (isRunning) {
            await API.stop();
            // Wait a moment then refresh
            setTimeout(async () => {
                await checkStatus();
                await loadRuns();
                btn.disabled = false;
            }, 1500);
        } else {
            const config = readConfig();
            await API.start(config);
            await checkStatus();
            btn.disabled = false;
        }
    } catch (e) {
        btn.disabled = false;
        console.error('Training action failed:', e);
    }
}

function readConfig() {
    const cfg = {
        relaxation_steps: parseInt($('#cfg-relax').value),
        batch_size:       parseInt($('#cfg-batch').value),
        num_steps:        parseInt($('#cfg-steps').value),
        checkpoint_every: parseInt($('#cfg-checkpoint').value),
        fs:               parseFloat($('#cfg-fs').value),
        activation:       $('#cfg-activation').value,
        lr:               parseFloat($('#cfg-lr').value),
        lr_w:             parseFloat($('#cfg-lr-w').value),
        pos_weight:       parseFloat($('#cfg-pos-weight').value),
        lambda_sparse:    parseFloat($('#cfg-lambda-sparse').value),
        spike_boost:      parseFloat($('#cfg-spike-boost').value),
        state_momentum:   parseFloat($('#cfg-state-momentum').value),
        asl_gamma_neg:    parseFloat($('#cfg-asl-gamma-neg').value),
        asl_margin:       parseFloat($('#cfg-asl-margin').value),
        lr_amplification: parseFloat($('#cfg-lr-amplification').value),
    };

    if (selectedLayoutId && selectedLayoutData) {
        cfg.layout_id = selectedLayoutId;
        cfg.layout_json = selectedLayoutData;
    } else {
        cfg.grid_width = parseInt($('#cfg-grid-width').value);
        cfg.grid_height = parseInt($('#cfg-grid-height').value);
        cfg.connectivity = $('#cfg-connectivity').value;
    }

    return cfg;
}

// --- Layouts ---
async function loadLayouts() {
    try {
        const data = await API.get('/training/layouts/');
        const sel = $('#cfg-layout');
        // Keep the manual option
        sel.innerHTML = '<option value="">-- manual config --</option>';
        for (const l of (data.layouts || [])) {
            const opt = document.createElement('option');
            opt.value = l.id;
            opt.textContent = l.name;
            sel.appendChild(opt);
        }
    } catch {
        // No layouts available, that's fine
    }
}

async function handleLayoutChange() {
    const id = $('#cfg-layout').value;
    const manualFields = $('#manual-grid-fields');
    const summary = $('#layout-summary');
    if (!id) {
        selectedLayoutId = null;
        selectedLayoutData = null;
        manualFields.classList.remove('hidden');
        summary.style.display = 'none';
        return;
    }
    try {
        const data = await API.get(`/training/layouts/${id}/`);
        selectedLayoutId = data.id;
        selectedLayoutData = data.layout_json;
        manualFields.classList.add('hidden');
        summary.style.display = '';
        // Summarize the layout
        const blocks = selectedLayoutData.blocks || [];
        const conns = selectedLayoutData.connections || [];
        const parts = blocks.map(b => {
            const h = (b.heights || []).join('\u2192');
            return `${b.name || b.role} [${b.columns}col, ${h}${b.connectivity ? ', ' + b.connectivity : ''}]`;
        });
        summary.innerHTML = parts.join(' \u2014 ') +
            `<br>${conns.length} connection${conns.length !== 1 ? 's' : ''}`;
    } catch (e) {
        console.error('Failed to load layout:', e);
        selectedLayoutId = null;
        selectedLayoutData = null;
        manualFields.classList.remove('hidden');
        summary.style.display = 'none';
    }
}

// --- Runs ---
async function loadRuns() {
    try {
        const data = await API.runs();
        const tbody = $('#runs-body');
        tbody.innerHTML = '';
        for (const run of data.runs) {
            const tr = document.createElement('tr');
            tr.className = run.status === 'running' ? 'run-active' : '';
            tr.innerHTML = `
                <td>#${run.id}</td>
                <td class="status-${run.status}">${run.status}</td>
                <td>${run.total_steps}</td>
                <td>${run.latest_error != null ? run.latest_error.toFixed(6) : '-'}</td>
                <td>P${run.current_phase}</td>
                <td>${formatTime(run.started_at)}</td>
            `;
            tr.addEventListener('click', () => selectRun(run.id));
            tbody.appendChild(tr);
        }
    } catch {
        $('#runs-body').innerHTML = '<tr><td colspan="6">no runs</td></tr>';
    }
}

async function selectRun(runId) {
    selectedRunId = runId;
    // Highlight
    $$('#runs-body tr').forEach(tr => tr.classList.remove('selected'));
    const rows = $$('#runs-body tr');
    for (const row of rows) {
        if (row.querySelector('td')?.textContent === '#' + runId) {
            row.classList.add('selected');
        }
    }
    await loadRunMetrics(runId);
}

async function loadRunMetrics(runId) {
    try {
        const data = await API.metrics(runId);
        chart.resize();
        chart.draw(data.metrics);
        $('#chart-title').textContent = `LOSS — RUN #${runId} (${data.total_steps} steps, P${data.current_phase})`;
    } catch {
        chart.draw([]);
        $('#chart-title').textContent = 'LOSS';
    }
}

// --- Helpers ---
function formatTime(iso) {
    if (!iso) return '-';
    const d = new Date(iso);
    return d.toLocaleString('en-GB', {
        month: 'short', day: 'numeric',
        hour: '2-digit', minute: '2-digit',
    });
}

// Boot
init();
