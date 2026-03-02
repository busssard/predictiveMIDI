import { loadModel, createFloatTexture, createColumnTexture, createFramebuffer } from './model-loader.js';
import { GridCompute } from './grid-compute.js';
import { GridRenderer } from './grid-renderer.js';

// --- DOM ---
const $ = s => document.querySelector(s);

// --- State ---
let gl, compute, renderer;
let model = null;
let stateTextures = [null, null];
let stateFramebuffers = [null, null];
let clampTexture = null;
let currentIdx = 0;

let midiData = null;        // {instruments: [{name, category, roll: T×128}], total_ticks, fs}
let targetIdx = 0;           // index into midiData.instruments
let playing = false;
let currentTick = 0;
let relaxationSteps = 128;
let lastTickTime = 0;
const FS = 8.0;
const TICK_MS = 1000 / FS;

// Audio
let audioCtx = null;
let activeOscillators = {};
let inputOscillators = {};
let playInputAudioEnabled = false;
let playOutputAudioEnabled = true;

// Output history for piano roll visualization
let outputHistory = [];
const MAX_OUTPUT_HISTORY = 64;

// --- Init ---
async function init() {
    // Populate model list first — this works even if WebGL fails
    await loadModelList();

    const canvas = $('#grid');
    gl = canvas.getContext('webgl', { preserveDrawingBuffer: true });
    if (!gl) {
        setStatus('WebGL not supported');
        return;
    }

    gl.getExtension('OES_texture_float');
    gl.getExtension('WEBGL_color_buffer_float') || gl.getExtension('EXT_color_buffer_float');

    // Load shaders
    try {
        const bust = '?v=' + Date.now();
        const [vertResp, stepResp, renderResp] = await Promise.all([
            fetch('/static/shaders/passthrough.vert' + bust),
            fetch('/static/shaders/pc-step.frag' + bust),
            fetch('/static/shaders/render-grid.frag' + bust),
        ]);
        if (!vertResp.ok || !stepResp.ok || !renderResp.ok) {
            setStatus('Failed to load shaders (' +
                [vertResp, stepResp, renderResp].map(r => r.status).join(', ') + ')');
            return;
        }
        const [vertSrc, stepFragSrc, renderFragSrc] = await Promise.all([
            vertResp.text(), stepResp.text(), renderResp.text(),
        ]);
        compute = new GridCompute(gl, vertSrc, stepFragSrc);
        renderer = new GridRenderer(gl, vertSrc, renderFragSrc);
    } catch (e) {
        setStatus('Shader error: ' + e.message);
        console.error('Shader init failed:', e);
        return;
    }

    // Load default model
    await loadSelectedModel();

    // Load demo MIDI data so the user can play immediately
    loadDemoMidi();

    // Wire UI
    $('#midi-file').addEventListener('change', handleMidiUpload);
    $('#btn-play').addEventListener('click', startPlayback);
    $('#btn-stop').addEventListener('click', stopPlayback);
    $('#relax-steps').addEventListener('input', (e) => {
        relaxationSteps = parseInt(e.target.value);
        $('#relax-val').textContent = relaxationSteps;
    });
    $('#model-select').addEventListener('change', loadSelectedModel);
    $('#chk-input-audio').addEventListener('change', (e) => {
        playInputAudioEnabled = e.target.checked;
        if (!playInputAudioEnabled) stopInputAudio();
    });
    $('#chk-output-audio').addEventListener('change', (e) => {
        playOutputAudioEnabled = e.target.checked;
        if (!playOutputAudioEnabled) stopOutputAudio();
    });
}

function setStatus(msg) {
    $('#model-status').innerHTML = msg;
}

function resizeCanvas() {
    if (!model) return;
    const canvas = $('#grid');
    const w = model.config.grid_width;
    const h = model.config.grid_height;
    // Scale so the longest dimension fills 512px, cells stay square
    const cellPx = Math.floor(512 / Math.max(w, h));
    canvas.width = w * cellPx;
    canvas.height = h * cellPx;
}

// --- Demo MIDI ---
function loadDemoMidi() {
    // Generate a simple piano + bass pattern so the page is immediately usable
    const totalTicks = 64; // 8 seconds at fs=8
    const pianoRoll = [];
    const bassRoll = [];

    // C major scale pattern for piano (ascending then descending)
    const pianoNotes = [60, 62, 64, 65, 67, 69, 71, 72, 71, 69, 67, 65, 64, 62, 60, 60];
    // Simple bass: root notes, 2 ticks each
    const bassNotes = [48, 48, 43, 43, 45, 45, 48, 48]; // C, G, A, C

    for (let t = 0; t < totalTicks; t++) {
        const pRow = new Array(128).fill(0);
        const bRow = new Array(128).fill(0);

        // Piano: one note per 4 ticks
        const pIdx = Math.floor(t / 4) % pianoNotes.length;
        pRow[pianoNotes[pIdx]] = 0.8;

        // Bass: one note per 8 ticks
        const bIdx = Math.floor(t / 8) % bassNotes.length;
        bRow[bassNotes[bIdx]] = 0.7;

        pianoRoll.push(pRow);
        bassRoll.push(bRow);
    }

    midiData = {
        instruments: [
            { name: 'Piano', category: 'piano', program: 0, is_drum: false, roll: pianoRoll },
            { name: 'Bass', category: 'bass', program: 32, is_drum: false, roll: bassRoll },
        ],
        total_ticks: totalTicks,
        fs: FS,
    };

    // Target = first instrument (piano), input = the rest (bass)
    targetIdx = 0;
    populateInstrumentList();
    $('#midi-status').innerHTML = '<span class="val">Demo loaded</span> — upload MIDI to replace';
    $('#total-ticks').textContent = totalTicks;
    updatePlayButton();
}

function populateInstrumentList() {
    const list = $('#instrument-list');
    list.innerHTML = '';
    midiData.instruments.forEach((inst, i) => {
        const li = document.createElement('li');
        li.innerHTML = `<span class="inst-radio"></span>${inst.name} (${inst.category})`;
        if (i === targetIdx) li.classList.add('selected');
        li.addEventListener('click', () => selectInstrument(i));
        list.appendChild(li);
    });
}

// --- Model loading ---
async function loadModelList() {
    const select = $('#model-select');
    try {
        const resp = await fetch('/api/training/models/');
        if (!resp.ok) throw new Error('API ' + resp.status);
        const data = await resp.json();
        const models = data.models || [];
        if (models.length === 0) {
            select.innerHTML = '<option value="latest">latest</option>';
            return;
        }
        select.innerHTML = '';
        for (const m of models) {
            const opt = document.createElement('option');
            opt.value = m.name;
            opt.textContent = m.name;
            select.appendChild(opt);
        }
        console.log('Model list loaded:', models.map(m => m.name));
    } catch (e) {
        console.error('Failed to load model list:', e);
        if (select.options.length === 0) {
            select.innerHTML = '<option value="latest">latest</option>';
        }
    }
}

async function loadSelectedModel() {
    const name = $('#model-select').value;
    const baseUrl = name === 'latest' ? '/static/model' : `/static/model/${name}`;
    setStatus('Loading ' + name + '...');
    console.log('Loading model from:', baseUrl);

    try {
        model = await loadModel(gl, baseUrl);
        resizeCanvas();
        setupPingPong();

        const w = model.config.grid_width;
        const h = model.config.grid_height;
        const numInst = model.config.num_instruments || 0;
        setStatus(`<span class="green">Loaded</span> ${name} (${w}x${h}, ${numInst} inst)`);
        console.log('Model loaded:', model.config);

        updatePlayButton();
        renderCurrentState();
    } catch (e) {
        model = null;
        setStatus(`<span class="accent">Failed:</span> ${e.message}`);
        console.error('Model load failed:', e);
    }
}

function setupPingPong() {
    const w = model.config.grid_width;
    const h = model.config.grid_height;
    const emptyData = new Float32Array(w * h * 4);
    stateTextures[0] = model.state;
    stateTextures[1] = createFloatTexture(gl, w, h, emptyData);
    stateFramebuffers[0] = createFramebuffer(gl, stateTextures[0]);
    stateFramebuffers[1] = createFramebuffer(gl, stateTextures[1]);
    currentIdx = 0;

    const clampData = new Float32Array(h * 4);
    clampTexture = createColumnTexture(gl, h, clampData);
}

function resetGridState() {
    if (!model || !model.stateData) return;
    const w = model.config.grid_width;
    const h = model.config.grid_height;
    gl.bindTexture(gl.TEXTURE_2D, stateTextures[0]);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, w, h, gl.RGBA, gl.FLOAT, model.stateData);
    currentIdx = 0;
}

function renderCurrentState() {
    if (!model || !renderer) return;
    const canvas = $('#grid');
    const w = model.config.grid_width;
    const h = model.config.grid_height;
    renderer.render(stateTextures[currentIdx], model.weights, w, h, canvas.width, canvas.height);
}

// --- MIDI handling ---
async function handleMidiUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    $('#midi-status').textContent = 'Parsing...';
    const formData = new FormData();
    formData.append('file', file);

    try {
        const resp = await fetch('/api/corpus/midi-to-roll/', { method: 'POST', body: formData });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ error: 'Upload failed (' + resp.status + ')' }));
            throw new Error(err.error || 'Upload failed');
        }
        midiData = await resp.json();
        targetIdx = 0;
        populateInstrumentList();

        $('#midi-status').innerHTML = `<span class="green">${midiData.instruments.length}</span> instruments, <span class="val">${midiData.total_ticks}</span> ticks`;
        $('#total-ticks').textContent = midiData.total_ticks;
        updatePlayButton();
    } catch (err) {
        $('#midi-status').innerHTML = `<span class="accent">${err.message}</span>`;
        console.error('MIDI upload failed:', err);
    }
}

function selectInstrument(idx) {
    targetIdx = idx;
    const items = $('#instrument-list').querySelectorAll('li');
    items.forEach((li, i) => li.classList.toggle('selected', i === idx));
}

function updatePlayButton() {
    const canPlay = model && midiData && midiData.instruments.length >= 2;
    $('#btn-play').disabled = !canPlay;
}

// --- Clamp data preparation ---
function prepareClampData(tick) {
    if (!midiData || !model) return;
    const h = model.config.grid_height;
    const numInst = model.config.num_instruments || 0;
    const condStart = Math.floor((h - numInst) / 2);

    const data = new Float32Array(h * 4);

    // Mix all non-target instruments for this tick
    for (let i = 0; i < midiData.instruments.length; i++) {
        if (i === targetIdx) continue;
        const roll = midiData.instruments[i].roll;
        if (tick < roll.length) {
            for (let p = 0; p < 128 && p < h; p++) {
                data[p * 4] = Math.max(data[p * 4], roll[tick][p]);
            }
        }
    }

    // Overlay conditioning one-hot
    const vocab = model.config.vocabulary || {};
    const targetCat = midiData.instruments[targetIdx].category;
    const catIdx = vocab[targetCat];
    if (catIdx !== undefined && condStart + catIdx < h) {
        for (let j = 0; j < numInst && condStart + j < h; j++) {
            data[(condStart + j) * 4] = 0.0;
        }
        data[(condStart + catIdx) * 4] = 1.0;
    }

    gl.bindTexture(gl.TEXTURE_2D, clampTexture);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 1, h, gl.RGBA, gl.FLOAT, data);
}

// --- Inference loop ---
function startPlayback() {
    if (playing || !model || !midiData) return;

    playing = true;
    currentTick = 0;
    outputHistory = [];
    lastTickTime = performance.now();

    resetGridState();
    initAudio();

    $('#btn-play').disabled = true;
    $('#btn-stop').disabled = false;
    $('#play-state').textContent = 'playing';
    $('#play-state').className = 'val green';

    requestAnimationFrame(inferenceLoop);
}

function stopPlayback() {
    playing = false;
    $('#btn-play').disabled = false;
    $('#btn-stop').disabled = true;
    $('#play-state').textContent = 'stopped';
    $('#play-state').className = 'val';
    stopAllAudio();
}

function inferenceLoop(timestamp) {
    if (!playing) return;

    if (timestamp - lastTickTime >= TICK_MS) {
        lastTickTime = timestamp;

        if (currentTick >= midiData.total_ticks) {
            stopPlayback();
            return;
        }

        prepareClampData(currentTick);

        const gw = model.config.grid_width;
        const gh = model.config.grid_height;
        for (let i = 0; i < relaxationSteps; i++) {
            const src = currentIdx;
            const dst = 1 - currentIdx;
            compute.step(
                stateTextures, stateFramebuffers,
                model.weights, model.params,
                src, dst, gw, gh,
                clampTexture
            );
            currentIdx = dst;
        }

        const output = readOutputColumn(gw, gh);
        outputHistory.push(output);
        if (outputHistory.length > MAX_OUTPUT_HISTORY) outputHistory.shift();

        if (playInputAudioEnabled) playInputMidi(currentTick);
        else stopInputAudio();

        if (playOutputAudioEnabled) playOutputAudio(output);
        else stopOutputAudio();

        renderCurrentState();
        drawOutputRoll();

        $('#tick-count').textContent = currentTick;
        currentTick++;
    }

    requestAnimationFrame(inferenceLoop);
}

function readOutputColumn(gridWidth, gridHeight) {
    const fb = stateFramebuffers[currentIdx];
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);

    const pixels = new Float32Array(gridHeight * 4);
    gl.readPixels(gridWidth - 1, 0, 1, gridHeight, gl.RGBA, gl.FLOAT, pixels);

    const output = new Float32Array(128);
    for (let row = 0; row < Math.min(gridHeight, 128); row++) {
        output[row] = pixels[row * 4];
    }
    return output;
}

// --- Output piano roll ---
function drawOutputRoll() {
    const canvas = $('#output-roll');
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;

    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, w, h);

    if (outputHistory.length === 0) return;

    const tickWidth = Math.max(1, Math.floor(w / MAX_OUTPUT_HISTORY));
    const pitchHeight = h / 128;

    for (let t = 0; t < outputHistory.length; t++) {
        const output = outputHistory[t];
        const x = (t / MAX_OUTPUT_HISTORY) * w;

        for (let p = 0; p < 128; p++) {
            const v = Math.abs(output[p]);
            if (v > 0.05) {
                const brightness = Math.min(1, v);
                const r = Math.floor(255 * brightness);
                const g = Math.floor(80 * brightness);
                const b = Math.floor(20 * brightness);
                ctx.fillStyle = `rgb(${r},${g},${b})`;
                const y = h - (p + 1) * pitchHeight;
                ctx.fillRect(x, y, tickWidth, Math.max(1, pitchHeight));
            }
        }
    }
}

// --- Audio ---
function initAudio() {
    if (!audioCtx) {
        audioCtx = new AudioContext();
    }
    if (audioCtx.state === 'suspended') {
        audioCtx.resume();
    }
    activeOscillators = {};
    inputOscillators = {};
}

function stopOscillatorSet(set) {
    for (const key of Object.keys(set)) {
        const { osc, gain } = set[key];
        gain.gain.linearRampToValueAtTime(0, audioCtx.currentTime + 0.05);
        osc.stop(audioCtx.currentTime + 0.1);
        delete set[key];
    }
}

function stopOutputAudio() { stopOscillatorSet(activeOscillators); }
function stopInputAudio() { stopOscillatorSet(inputOscillators); }

function stopAllAudio() {
    stopOutputAudio();
    stopInputAudio();
}

function playInputMidi(tick) {
    if (!audioCtx || !midiData) return;

    const threshold = 0.15;
    const activePitches = new Set();

    // Mix all non-target instruments at this tick
    for (let i = 0; i < midiData.instruments.length; i++) {
        if (i === targetIdx) continue;
        const roll = midiData.instruments[i].roll;
        if (tick < roll.length) {
            for (let p = 0; p < 128; p++) {
                if (roll[tick][p] > threshold) activePitches.add(p);
            }
        }
    }

    // Stop oscillators for pitches no longer active
    for (const key of Object.keys(inputOscillators)) {
        if (!activePitches.has(parseInt(key))) {
            const { osc, gain } = inputOscillators[key];
            gain.gain.linearRampToValueAtTime(0, audioCtx.currentTime + 0.05);
            osc.stop(audioCtx.currentTime + 0.1);
            delete inputOscillators[key];
        }
    }

    // Start oscillators for new active pitches (sine wave, lower volume)
    for (const p of activePitches) {
        if (!inputOscillators[p]) {
            const freq = 440 * Math.pow(2, (p - 69) / 12);
            const osc = audioCtx.createOscillator();
            const gain = audioCtx.createGain();
            osc.type = 'sine';
            osc.frequency.value = freq;
            gain.gain.value = 0;
            gain.gain.linearRampToValueAtTime(0.06, audioCtx.currentTime + 0.02);
            osc.connect(gain);
            gain.connect(audioCtx.destination);
            osc.start();
            inputOscillators[p] = { osc, gain };
        }
    }
}

function playOutputAudio(output) {
    if (!audioCtx || !playOutputAudioEnabled) return;

    const threshold = 0.15;
    const activePitches = new Set();

    for (let p = 0; p < 128; p++) {
        if (Math.abs(output[p]) > threshold) {
            activePitches.add(p);
        }
    }

    for (const key of Object.keys(activeOscillators)) {
        if (!activePitches.has(parseInt(key))) {
            const { osc, gain } = activeOscillators[key];
            gain.gain.linearRampToValueAtTime(0, audioCtx.currentTime + 0.05);
            osc.stop(audioCtx.currentTime + 0.1);
            delete activeOscillators[key];
        }
    }

    for (const p of activePitches) {
        if (!activeOscillators[p]) {
            const freq = 440 * Math.pow(2, (p - 69) / 12);
            const osc = audioCtx.createOscillator();
            const gain = audioCtx.createGain();
            osc.type = 'triangle';
            osc.frequency.value = freq;
            gain.gain.value = 0;
            gain.gain.linearRampToValueAtTime(
                0.1 * Math.min(1, Math.abs(output[p])),
                audioCtx.currentTime + 0.02
            );
            osc.connect(gain);
            gain.connect(audioCtx.destination);
            osc.start();
            activeOscillators[p] = { osc, gain };
        }
    }
}

// --- Boot ---
init().catch(e => {
    console.error('Init failed:', e);
    const el = document.querySelector('#model-status');
    if (el) el.innerHTML = `<span class="accent">Init error:</span> ${e.message}`;
});
