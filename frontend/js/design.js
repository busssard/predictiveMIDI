import { DesignCanvas } from './design-canvas.js';
import { BlockModel, resetBlockIdCounter } from './design-blocks.js';

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
    async put(path, body = {}) {
        const r = await fetch(this.base + path, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        return r.json();
    },
    async del(path) {
        await fetch(this.base + path, { method: 'DELETE' });
    },
};

// --- DOM ---
const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);

// --- State ---
let layouts = [];
let currentLayout = null;
let currentLayoutId = null;
let canvas = null;
let selectedBlock = null;
let undoStack = [];
let redoStack = [];
let autoSaveTimer = null;

// --- Init ---
async function init() {
    canvas = new DesignCanvas($('#design-canvas'));
    canvas.resize();
    window.addEventListener('resize', () => canvas.resize());

    // Canvas callbacks
    canvas.onBlockSelect = block => {
        selectedBlock = block;
        updatePropsPanel();
    };
    canvas.onBlockAdd = (x, y) => addBlock(x, y);
    canvas.onBlockDelete = block => deleteBlock(block);
    canvas.onConnectionAdd = (fromBlock, fromEdge, toBlock, toEdge) => {
        addConnection(fromBlock, fromEdge, toBlock, toEdge);
    };
    canvas.onConnectionDelete = conn => deleteConnection(conn);
    canvas.onChange = () => scheduleAutoSave();

    // Tool buttons
    $$('.tool-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            $$('.tool-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            canvas.tool = btn.dataset.tool;
        });
    });

    // Action buttons
    $('#btn-save').addEventListener('click', saveLayout);
    $('#btn-export').addEventListener('click', exportJSON);
    $('#btn-import').addEventListener('click', () => $('#import-file').click());
    $('#import-file').addEventListener('change', importJSON);
    $('#btn-new-layout').addEventListener('click', newLayout);

    // Property panel inputs
    setupPropsListeners();

    // Keyboard shortcuts
    window.addEventListener('keydown', e => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' ||
            e.target.tagName === 'TEXTAREA') return;

        if (e.ctrlKey && e.key === 's') {
            e.preventDefault();
            saveLayout();
        }
        if (e.ctrlKey && e.key === 'z') {
            e.preventDefault();
            undo();
        }
        if (e.ctrlKey && e.key === 'y') {
            e.preventDefault();
            redo();
        }
        if (e.ctrlKey && e.key === 'd' && selectedBlock) {
            e.preventDefault();
            duplicateBlock(selectedBlock);
        }
    });

    // Load layouts
    await loadLayouts();

    // Try restoring from localStorage
    const saved = localStorage.getItem('pcjam_design_layout');
    if (saved) {
        try {
            loadLayoutData(JSON.parse(saved));
        } catch (e) {
            console.warn('Failed to restore layout from localStorage:', e);
        }
    }

    // If no layout loaded, create a default one
    if (!currentLayout) {
        createDefaultLayout();
    }
}

// --- Layout data ---

function createDefaultLayout() {
    currentLayout = {
        name: 'untitled',
        version: 1,
        blocks: [],
        connections: [],
        input_blocks: [],
        output_blocks: [],
    };
    resetBlockIdCounter();

    // Create a default encoder-decoder pair
    const encoder = new BlockModel({
        x: 100, y: 80,
        columns: 4,
        heights: [128, 100, 80, 64],
        connectivity: 'fc',
        role: 'encoder',
        name: 'encoder',
    });
    const decoder = new BlockModel({
        x: 300, y: 80,
        columns: 4,
        heights: [64, 80, 100, 128],
        connectivity: 'fc',
        role: 'decoder',
        name: 'decoder',
    });

    currentLayout.blocks = [encoder.toJSON(), decoder.toJSON()];
    currentLayout.connections = [{
        from_block: encoder.id,
        from_edge: 'right',
        to_block: decoder.id,
        to_edge: 'left',
        type: 'fc',
    }];
    currentLayout.input_blocks = [encoder.id];
    currentLayout.output_blocks = [decoder.id];

    syncToCanvas();
    updateLayoutSelector();
}

function loadLayoutData(data) {
    currentLayout = data;
    resetBlockIdCounter();
    syncToCanvas();
    updateLayoutSelector();
}

function syncToCanvas() {
    canvas.blocks = (currentLayout.blocks || []).map(b => BlockModel.fromJSON(b));
    canvas.connections = currentLayout.connections || [];
    selectedBlock = null;
    updatePropsPanel();
}

function syncFromCanvas() {
    if (!currentLayout) return;
    currentLayout.blocks = canvas.blocks.map(b => b.toJSON());
    currentLayout.connections = [...canvas.connections];
}

function pushUndo() {
    syncFromCanvas();
    undoStack.push(JSON.stringify(currentLayout));
    if (undoStack.length > 50) undoStack.shift();
    redoStack = [];
}

function undo() {
    if (undoStack.length === 0) return;
    syncFromCanvas();
    redoStack.push(JSON.stringify(currentLayout));
    const prev = JSON.parse(undoStack.pop());
    loadLayoutData(prev);
}

function redo() {
    if (redoStack.length === 0) return;
    syncFromCanvas();
    undoStack.push(JSON.stringify(currentLayout));
    const next = JSON.parse(redoStack.pop());
    loadLayoutData(next);
}

// --- Block operations ---

function addBlock(x, y) {
    pushUndo();
    const block = new BlockModel({
        x: Math.round(x),
        y: Math.round(y),
        columns: 4,
        heights: [128, 128, 128, 128],
        connectivity: 'fc',
        role: 'encoder',
        name: `block_${canvas.blocks.length + 1}`,
    });
    canvas.blocks.push(block);
    selectedBlock = block;
    canvas.setSelected(block);
    updatePropsPanel();
    scheduleAutoSave();
}

function deleteBlock(block) {
    pushUndo();
    canvas.blocks = canvas.blocks.filter(b => b !== block);
    canvas.connections = canvas.connections.filter(
        c => c.from_block !== block.id && c.to_block !== block.id);
    if (selectedBlock === block) {
        selectedBlock = null;
        updatePropsPanel();
    }
    scheduleAutoSave();
}

function duplicateBlock(block) {
    pushUndo();
    const dup = block.clone();
    canvas.blocks.push(dup);
    selectedBlock = dup;
    canvas.setSelected(dup);
    updatePropsPanel();
    scheduleAutoSave();
}

function addConnection(fromBlock, fromEdge, toBlock, toEdge) {
    if (fromBlock === toBlock) return;
    pushUndo();
    canvas.connections.push({
        from_block: fromBlock.id,
        from_edge: fromEdge,
        to_block: toBlock.id,
        to_edge: toEdge,
        type: fromBlock.connectivity,
    });
    scheduleAutoSave();
}

function deleteConnection(conn) {
    pushUndo();
    canvas.connections = canvas.connections.filter(c => c !== conn);
    scheduleAutoSave();
}

// --- Properties panel ---

function updatePropsPanel() {
    const panel = $('#props-panel');
    if (!selectedBlock) {
        panel.classList.add('hidden');
        return;
    }
    panel.classList.remove('hidden');

    $('#prop-name').value = selectedBlock.name;
    $('#prop-columns').value = selectedBlock.columns;
    $('#prop-heights').value = selectedBlock.heights.join(', ');
    $('#prop-connectivity').value = selectedBlock.connectivity;
    $('#prop-algorithm').value = selectedBlock.algorithm;
    $('#prop-role').value = selectedBlock.role;
    $('#prop-input-edge').value = selectedBlock.input_edge;
    $('#prop-output-edge').value = selectedBlock.output_edge;
}

function setupPropsListeners() {
    const update = (id, field, parse) => {
        $(id).addEventListener('change', () => {
            if (!selectedBlock) return;
            pushUndo();
            selectedBlock[field] = parse ? parse($(id).value) : $(id).value;
            scheduleAutoSave();
        });
    };

    update('#prop-name', 'name');
    update('#prop-connectivity', 'connectivity');
    update('#prop-algorithm', 'algorithm');
    update('#prop-role', 'role');
    update('#prop-input-edge', 'input_edge');
    update('#prop-output-edge', 'output_edge');

    $('#prop-columns').addEventListener('change', () => {
        if (!selectedBlock) return;
        pushUndo();
        const newCols = parseInt($('#prop-columns').value) || 4;
        const oldHeights = selectedBlock.heights;
        const newHeights = [];
        for (let i = 0; i < newCols; i++) {
            const t = newCols > 1 ? i / (newCols - 1) : 0;
            const srcIdx = t * (oldHeights.length - 1);
            const lo = Math.floor(srcIdx);
            const hi = Math.min(lo + 1, oldHeights.length - 1);
            const frac = srcIdx - lo;
            newHeights.push(Math.round(oldHeights[lo] * (1 - frac) + oldHeights[hi] * frac));
        }
        selectedBlock.columns = newCols;
        selectedBlock.heights = newHeights;
        updatePropsPanel();
        scheduleAutoSave();
    });

    $('#prop-heights').addEventListener('change', () => {
        if (!selectedBlock) return;
        pushUndo();
        const parts = $('#prop-heights').value.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
        if (parts.length > 0) {
            selectedBlock.heights = parts;
            selectedBlock.columns = parts.length;
            updatePropsPanel();
            scheduleAutoSave();
        }
    });
}

// --- Layout management ---

async function loadLayouts() {
    try {
        const data = await API.get('/training/layouts/');
        layouts = data.layouts || [];
        updateLayoutSelector();
    } catch {
        layouts = [];
    }
}

function updateLayoutSelector() {
    const sel = $('#layout-select');
    sel.innerHTML = '<option value="">-- Select Layout --</option>';
    for (const l of layouts) {
        const opt = document.createElement('option');
        opt.value = l.id;
        opt.textContent = l.name;
        if (currentLayoutId === l.id) opt.selected = true;
        sel.appendChild(opt);
    }

    sel.onchange = async () => {
        const id = sel.value;
        if (!id) return;
        try {
            const data = await API.get(`/training/layouts/${id}/`);
            currentLayoutId = data.id;
            loadLayoutData(data.layout_json);
        } catch (e) {
            console.error('Failed to load layout:', e);
        }
    };
}

async function saveLayout() {
    syncFromCanvas();
    const name = currentLayout.name || 'untitled';
    try {
        const data = await API.post('/training/layouts/', {
            name,
            layout_json: currentLayout,
        });
        currentLayoutId = data.id;
        await loadLayouts();
        showToast('Saved');
    } catch (e) {
        console.error('Save failed:', e);
        showToast('Save failed');
    }
}

function newLayout() {
    currentLayoutId = null;
    createDefaultLayout();
    showToast('New layout');
}

function exportJSON() {
    syncFromCanvas();
    const json = JSON.stringify(currentLayout, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = (currentLayout.name || 'layout') + '.json';
    a.click();
    URL.revokeObjectURL(url);
}

function importJSON(e) {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (evt) => {
        try {
            const data = JSON.parse(evt.target.result);
            loadLayoutData(data);
            showToast('Imported');
        } catch (err) {
            showToast('Invalid JSON');
        }
    };
    reader.readAsText(file);
    e.target.value = '';
}

// --- Auto-save to localStorage ---

function scheduleAutoSave() {
    if (autoSaveTimer) clearTimeout(autoSaveTimer);
    autoSaveTimer = setTimeout(() => {
        syncFromCanvas();
        if (currentLayout) {
            localStorage.setItem('pcjam_design_layout',
                                 JSON.stringify(currentLayout));
        }
    }, 1000);
}

// --- Toast ---

function showToast(msg) {
    const toast = $('#toast');
    toast.textContent = msg;
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), 2000);
}

// Boot
init();
