import { loadModel, createFloatTexture, createFramebuffer } from './model-loader.js';

const canvas = document.getElementById('grid');
const gl = canvas.getContext('webgl', { preserveDrawingBuffer: true });

if (!gl) {
    document.body.innerHTML = '<h1>WebGL not supported</h1>';
    throw new Error('No WebGL');
}

// Will be initialized when model loads
let model = null;
let stateTextures = [null, null]; // ping-pong pair
let stateFramebuffers = [null, null];
let currentIdx = 0;

async function init() {
    // Load model from server
    try {
        model = await loadModel(gl, '/static/model');
        console.log('Model loaded:', model.config);
        populateInstruments(model.config.vocabulary);
        setupPingPong(gl, model);
    } catch (e) {
        console.log('No model loaded yet. Waiting for training...');
    }
}

function populateInstruments(vocabulary) {
    const select = document.getElementById('instrument-select');
    select.innerHTML = '';
    for (const [name, idx] of Object.entries(vocabulary)) {
        const opt = document.createElement('option');
        opt.value = idx;
        opt.textContent = name;
        select.appendChild(opt);
    }
}

function setupPingPong(gl, model) {
    const size = model.config.grid_size;
    // Create two state textures for ping-pong
    const emptyData = new Float32Array(size * size * 4);
    stateTextures[0] = model.state;
    stateTextures[1] = createFloatTexture(gl, size, emptyData);
    stateFramebuffers[0] = createFramebuffer(gl, stateTextures[0]);
    stateFramebuffers[1] = createFramebuffer(gl, stateTextures[1]);
}

init();
