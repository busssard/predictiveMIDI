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

// --- DOM ---
const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);

// --- State ---
let isRunning = false;
let selectedRunId = null;
let chart = null;

// --- Init ---
async function init() {
    chart = new LossChart($('#loss-canvas'));
    chart.resize();
    window.addEventListener('resize', () => {
        chart.resize();
        if (selectedRunId) loadRunMetrics(selectedRunId);
    });

    $('#btn-train').addEventListener('click', handleTrainButton);

    await Promise.all([
        loadCorpus(),
        loadRuns(),
        checkStatus(),
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
    } else {
        dot.className = 'dot idle';
        label.textContent = 'IDLE';
        label.className = 'status-text';
        btn.textContent = 'START TRAINING';
        btn.className = 'btn-train';
        setConfigDisabled(false);
    }

    if (st.error) {
        $('#status-error').textContent = st.error;
        $('#status-error').style.display = 'block';
    } else {
        $('#status-error').style.display = 'none';
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
    return {
        grid_size:        parseInt($('#cfg-grid-size').value),
        relaxation_steps: parseInt($('#cfg-relax').value),
        batch_size:       parseInt($('#cfg-batch').value),
        num_steps:        parseInt($('#cfg-steps').value),
        checkpoint_every: parseInt($('#cfg-checkpoint').value),
        fs:               parseFloat($('#cfg-fs').value),
        activation:       $('#cfg-activation').value,
    };
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
