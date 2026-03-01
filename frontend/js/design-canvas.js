/**
 * Canvas 2D rendering engine for the network designer.
 * Handles zoom/pan, block rendering, connections, hit testing.
 */

class Camera {
    constructor() {
        this.x = 0;
        this.y = 0;
        this.zoom = 1.0;
    }
    screenToWorld(sx, sy) {
        return {
            x: sx / this.zoom + this.x,
            y: sy / this.zoom + this.y,
        };
    }
    worldToScreen(wx, wy) {
        return {
            x: (wx - this.x) * this.zoom,
            y: (wy - this.y) * this.zoom,
        };
    }
    apply(ctx) {
        ctx.setTransform(this.zoom, 0, 0, this.zoom,
                         -this.x * this.zoom, -this.y * this.zoom);
    }
}

// Connection type colors
const CONN_COLORS = {
    neighbor: '#555',
    fc: '#4488cc',
    fc_double: '#8844cc',
};

// Role colors for block header
const ROLE_COLORS = {
    input: '#00ff41',
    encoder: '#4488cc',
    bottleneck: '#cc8844',
    decoder: '#cc4488',
    output: '#ff2200',
};

export class DesignCanvas {
    constructor(canvasEl) {
        this.el = canvasEl;
        this.ctx = canvasEl.getContext('2d');
        this.camera = new Camera();
        this.blocks = [];
        this.connections = [];

        // Interaction state
        this._dragging = null;   // {block, offsetX, offsetY}
        this._panning = false;
        this._panStart = null;
        this._hovered = null;
        this._selected = null;
        this._cornerDrag = null; // {block, cornerIdx, startY, startH}
        this._connectFrom = null; // {block, edge}
        this._connectPreview = null; // {x, y}

        // Tool: 'select' | 'add' | 'connect' | 'delete'
        this.tool = 'select';

        // Callbacks
        this.onBlockSelect = null;
        this.onBlockMove = null;
        this.onBlockAdd = null;
        this.onBlockDelete = null;
        this.onConnectionAdd = null;
        this.onConnectionDelete = null;
        this.onCornerResize = null;
        this.onChange = null;

        this._setupEvents();
        this._animFrame = null;
        this._render();
    }

    destroy() {
        if (this._animFrame) cancelAnimationFrame(this._animFrame);
    }

    // --- Events ---

    _setupEvents() {
        const el = this.el;
        el.addEventListener('mousedown', e => this._onMouseDown(e));
        el.addEventListener('mousemove', e => this._onMouseMove(e));
        el.addEventListener('mouseup', e => this._onMouseUp(e));
        el.addEventListener('wheel', e => this._onWheel(e), { passive: false });
        el.addEventListener('contextmenu', e => e.preventDefault());

        // Keyboard
        window.addEventListener('keydown', e => this._onKeyDown(e));
    }

    _getCanvasPos(e) {
        const rect = this.el.getBoundingClientRect();
        return { sx: e.clientX - rect.left, sy: e.clientY - rect.top };
    }

    _onMouseDown(e) {
        const { sx, sy } = this._getCanvasPos(e);
        const { x: wx, y: wy } = this.camera.screenToWorld(sx, sy);

        // Middle click or space+click = pan
        if (e.button === 1) {
            this._panning = true;
            this._panStart = { sx, sy, cx: this.camera.x, cy: this.camera.y };
            return;
        }

        if (e.button !== 0) return;

        if (this.tool === 'add') {
            if (this.onBlockAdd) this.onBlockAdd(wx, wy);
            return;
        }

        if (this.tool === 'delete') {
            // Check connections first
            const conn = this._hitTestConnection(wx, wy);
            if (conn) {
                if (this.onConnectionDelete) this.onConnectionDelete(conn);
                return;
            }
            const block = this._hitTestBlock(wx, wy);
            if (block) {
                if (this.onBlockDelete) this.onBlockDelete(block);
                return;
            }
            return;
        }

        if (this.tool === 'connect') {
            const block = this._hitTestBlock(wx, wy);
            if (block) {
                const edge = this._hitTestEdge(block, wx, wy);
                if (!this._connectFrom) {
                    this._connectFrom = { block, edge: edge || 'right' };
                } else {
                    const toEdge = edge || 'left';
                    if (this.onConnectionAdd) {
                        this.onConnectionAdd(
                            this._connectFrom.block, this._connectFrom.edge,
                            block, toEdge);
                    }
                    this._connectFrom = null;
                    this._connectPreview = null;
                }
            }
            return;
        }

        // Select tool
        // Check corner handles first
        const corner = this._hitTestCorner(wx, wy);
        if (corner) {
            this._cornerDrag = {
                block: corner.block,
                cornerIdx: corner.cornerIdx,
                startY: wy,
                startH: corner.cornerIdx === 0
                    ? corner.block.heights[0]
                    : corner.block.heights[corner.block.columns - 1],
            };
            return;
        }

        const block = this._hitTestBlock(wx, wy);
        if (block) {
            this._selected = block;
            this._dragging = {
                block,
                offsetX: wx - block.x,
                offsetY: wy - block.y,
            };
            if (this.onBlockSelect) this.onBlockSelect(block);
        } else {
            this._selected = null;
            if (this.onBlockSelect) this.onBlockSelect(null);
        }
    }

    _onMouseMove(e) {
        const { sx, sy } = this._getCanvasPos(e);
        const { x: wx, y: wy } = this.camera.screenToWorld(sx, sy);

        if (this._panning && this._panStart) {
            const dx = (sx - this._panStart.sx) / this.camera.zoom;
            const dy = (sy - this._panStart.sy) / this.camera.zoom;
            this.camera.x = this._panStart.cx - dx;
            this.camera.y = this._panStart.cy - dy;
            return;
        }

        if (this._cornerDrag) {
            const delta = (wy - this._cornerDrag.startY) / 0.6;
            this._cornerDrag.block.resizeCorner(
                this._cornerDrag.cornerIdx, delta);
            this._cornerDrag.startY = wy;
            if (this.onCornerResize) this.onCornerResize(this._cornerDrag.block);
            if (this.onChange) this.onChange();
            return;
        }

        if (this._dragging) {
            const block = this._dragging.block;
            block.x = Math.round(wx - this._dragging.offsetX);
            block.y = Math.round(wy - this._dragging.offsetY);
            if (this.onBlockMove) this.onBlockMove(block, block.x, block.y);
            if (this.onChange) this.onChange();
            return;
        }

        if (this.tool === 'connect' && this._connectFrom) {
            this._connectPreview = { x: wx, y: wy };
            return;
        }

        // Hover detection
        this._hovered = this._hitTestBlock(wx, wy);
    }

    _onMouseUp(e) {
        if (this._panning) {
            this._panning = false;
            this._panStart = null;
        }
        if (this._dragging) {
            this._dragging = null;
        }
        if (this._cornerDrag) {
            this._cornerDrag = null;
        }
    }

    _onWheel(e) {
        e.preventDefault();
        const { sx, sy } = this._getCanvasPos(e);
        const { x: wx, y: wy } = this.camera.screenToWorld(sx, sy);

        const factor = e.deltaY > 0 ? 0.9 : 1.1;
        const newZoom = Math.min(20, Math.max(0.1, this.camera.zoom * factor));

        // Zoom centered on cursor
        this.camera.x = wx - sx / newZoom;
        this.camera.y = wy - sy / newZoom;
        this.camera.zoom = newZoom;
    }

    _onKeyDown(e) {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' ||
            e.target.tagName === 'TEXTAREA') return;

        if ((e.key === 'Delete' || e.key === 'Backspace') && this._selected) {
            if (this.onBlockDelete) this.onBlockDelete(this._selected);
            this._selected = null;
            e.preventDefault();
        }
        if (e.key === '+' || e.key === '=') {
            this.camera.zoom = Math.min(20, this.camera.zoom * 1.2);
        }
        if (e.key === '-') {
            this.camera.zoom = Math.max(0.1, this.camera.zoom / 1.2);
        }
        if (e.key === '0') {
            this.camera.x = 0;
            this.camera.y = 0;
            this.camera.zoom = 1.0;
        }
    }

    // --- Hit testing ---

    _hitTestBlock(wx, wy) {
        // Reverse order = topmost first
        for (let i = this.blocks.length - 1; i >= 0; i--) {
            const b = this.blocks[i];
            if (wx >= b.x && wx <= b.x + b.width &&
                wy >= b.y && wy <= b.y + b.height) {
                return b;
            }
        }
        return null;
    }

    _hitTestCorner(wx, wy) {
        const r = 6 / this.camera.zoom;
        for (let i = this.blocks.length - 1; i >= 0; i--) {
            const b = this.blocks[i];
            // Top-left corner
            if (Math.abs(wx - b.x) < r && Math.abs(wy - b.y) < r) {
                return { block: b, cornerIdx: 0 };
            }
            // Top-right corner
            if (Math.abs(wx - (b.x + b.width)) < r && Math.abs(wy - b.y) < r) {
                return { block: b, cornerIdx: 1 };
            }
        }
        return null;
    }

    _hitTestEdge(block, wx, wy) {
        const margin = 15 / this.camera.zoom;
        if (wx < block.x + margin) return 'left';
        if (wx > block.x + block.width - margin) return 'right';
        if (wy < block.y + margin) return 'top';
        if (wy > block.y + block.height - margin) return 'bottom';
        return null;
    }

    _hitTestConnection(wx, wy) {
        const thresh = 8 / this.camera.zoom;
        for (const conn of this.connections) {
            const from = this._getBlockById(conn.from_block);
            const to = this._getBlockById(conn.to_block);
            if (!from || !to) continue;
            const p1 = from.getEdgePosition(conn.from_edge);
            const p2 = to.getEdgePosition(conn.to_edge);
            // Distance to line segment
            const d = this._pointToSegDist(wx, wy, p1.x, p1.y, p2.x, p2.y);
            if (d < thresh) return conn;
        }
        return null;
    }

    _pointToSegDist(px, py, x1, y1, x2, y2) {
        const dx = x2 - x1, dy = y2 - y1;
        const len2 = dx * dx + dy * dy;
        if (len2 === 0) return Math.hypot(px - x1, py - y1);
        let t = ((px - x1) * dx + (py - y1) * dy) / len2;
        t = Math.max(0, Math.min(1, t));
        return Math.hypot(px - (x1 + t * dx), py - (y1 + t * dy));
    }

    _getBlockById(id) {
        return this.blocks.find(b => b.id === id) || null;
    }

    // --- Rendering ---

    _render() {
        const ctx = this.ctx;
        const w = this.el.width;
        const h = this.el.height;

        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.fillStyle = '#0a0a0a';
        ctx.fillRect(0, 0, w, h);

        this.camera.apply(ctx);

        // Grid background
        this._drawGrid(ctx);

        // Connections
        for (const conn of this.connections) {
            this._drawConnection(ctx, conn);
        }

        // Connection preview
        if (this._connectFrom && this._connectPreview) {
            const from = this._connectFrom.block.getEdgePosition(this._connectFrom.edge);
            ctx.beginPath();
            ctx.moveTo(from.x, from.y);
            ctx.lineTo(this._connectPreview.x, this._connectPreview.y);
            ctx.strokeStyle = '#ff220088';
            ctx.lineWidth = 2 / this.camera.zoom;
            ctx.setLineDash([5 / this.camera.zoom, 5 / this.camera.zoom]);
            ctx.stroke();
            ctx.setLineDash([]);
        }

        // Blocks
        for (const block of this.blocks) {
            this._drawBlock(ctx, block);
        }

        this._animFrame = requestAnimationFrame(() => this._render());
    }

    _drawGrid(ctx) {
        const step = 50;
        const cam = this.camera;
        const startX = Math.floor(cam.x / step) * step;
        const startY = Math.floor(cam.y / step) * step;
        const endX = cam.x + this.el.width / cam.zoom;
        const endY = cam.y + this.el.height / cam.zoom;

        ctx.beginPath();
        ctx.strokeStyle = '#111';
        ctx.lineWidth = 0.5 / cam.zoom;
        for (let x = startX; x < endX; x += step) {
            ctx.moveTo(x, startY);
            ctx.lineTo(x, endY);
        }
        for (let y = startY; y < endY; y += step) {
            ctx.moveTo(startX, y);
            ctx.lineTo(endX, y);
        }
        ctx.stroke();
    }

    _drawBlock(ctx, block) {
        const isSelected = block === this._selected;
        const isHovered = block === this._hovered;
        const bw = block.width;
        const bh = block.height;
        const colW = 12;
        const padX = 10;
        const padY = 20;

        // Block background with tapered shape
        ctx.beginPath();
        const maxH = Math.max(...block.heights);
        // Draw tapered polygon
        const topY = block.y;
        const bottomBase = block.y + bh;
        ctx.moveTo(block.x, topY);
        ctx.lineTo(block.x + bw, topY);
        ctx.lineTo(block.x + bw, bottomBase);
        ctx.lineTo(block.x, bottomBase);
        ctx.closePath();
        ctx.fillStyle = '#0e0e0e';
        ctx.fill();

        // Border
        ctx.strokeStyle = isSelected ? '#ff2200' : isHovered ? '#2a2a2a' : '#1a1a1a';
        ctx.lineWidth = isSelected ? 2 / this.camera.zoom : 1 / this.camera.zoom;
        ctx.stroke();

        // Column separators and neurons
        for (let c = 0; c < block.columns; c++) {
            const colX = block.x + padX + c * colW;
            const colH = block.heights[c];
            const neuronScale = 0.6;
            const drawH = colH * neuronScale;
            const colY = block.y + padY;

            // Column separator line
            if (c > 0) {
                ctx.beginPath();
                ctx.moveTo(colX - 1, block.y + padY - 2);
                ctx.lineTo(colX - 1, block.y + padY + Math.max(...block.heights) * neuronScale + 2);
                ctx.strokeStyle = '#1a1a1a';
                ctx.lineWidth = 0.5 / this.camera.zoom;
                ctx.stroke();
            }

            // Neuron pixels at high zoom
            if (this.camera.zoom > 2) {
                const pixSize = Math.max(1, 2 / this.camera.zoom);
                const neuronsVisible = Math.min(colH, Math.floor(drawH / pixSize));
                const roleColor = ROLE_COLORS[block.role] || '#999';
                for (let n = 0; n < neuronsVisible; n++) {
                    const ny = colY + n * (drawH / colH);
                    ctx.fillStyle = roleColor + '44';
                    ctx.fillRect(colX, ny, colW - 2, pixSize);
                }
            } else {
                // Gradient fill for column
                const roleColor = ROLE_COLORS[block.role] || '#999';
                ctx.fillStyle = roleColor + '18';
                ctx.fillRect(colX, colY, colW - 2, drawH);
            }

            // Height label at bottom of column
            if (this.camera.zoom > 0.6) {
                ctx.fillStyle = '#555';
                ctx.font = `${9 / this.camera.zoom}px monospace`;
                ctx.textAlign = 'center';
                ctx.fillText(colH.toString(), colX + (colW - 2) / 2,
                             colY + drawH + 10 / this.camera.zoom);
            }
        }

        // Block label
        ctx.fillStyle = isSelected ? '#eee' : '#999';
        ctx.font = `bold ${10 / this.camera.zoom}px monospace`;
        ctx.textAlign = 'left';
        ctx.fillText(block.name, block.x + 4, block.y + 12 / this.camera.zoom);

        // Role badge
        const roleColor = ROLE_COLORS[block.role] || '#999';
        ctx.fillStyle = roleColor;
        ctx.font = `${8 / this.camera.zoom}px monospace`;
        ctx.textAlign = 'right';
        ctx.fillText(block.role, block.x + bw - 4, block.y + 12 / this.camera.zoom);

        // Edge highlights (input=green, output=red)
        if (isHovered || isSelected) {
            const inPos = block.getEdgePosition(block.input_edge);
            const outPos = block.getEdgePosition(block.output_edge);
            ctx.beginPath();
            ctx.arc(inPos.x, inPos.y, 4 / this.camera.zoom, 0, Math.PI * 2);
            ctx.fillStyle = '#00ff41';
            ctx.fill();
            ctx.beginPath();
            ctx.arc(outPos.x, outPos.y, 4 / this.camera.zoom, 0, Math.PI * 2);
            ctx.fillStyle = '#ff2200';
            ctx.fill();
        }

        // Corner handles
        if (isSelected) {
            const handleR = 4 / this.camera.zoom;
            ctx.fillStyle = '#ff2200';
            ctx.fillRect(block.x - handleR, block.y - handleR, handleR * 2, handleR * 2);
            ctx.fillRect(block.x + bw - handleR, block.y - handleR, handleR * 2, handleR * 2);
        }
    }

    _drawConnection(ctx, conn) {
        const from = this._getBlockById(conn.from_block);
        const to = this._getBlockById(conn.to_block);
        if (!from || !to) return;

        const p1 = from.getEdgePosition(conn.from_edge);
        const p2 = to.getEdgePosition(conn.to_edge);

        // Quadratic bezier control point
        const mx = (p1.x + p2.x) / 2;
        const my = (p1.y + p2.y) / 2;
        const dx = p2.x - p1.x;
        const cpx = mx;
        const cpy = my - Math.abs(dx) * 0.2;

        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.quadraticCurveTo(cpx, cpy, p2.x, p2.y);
        ctx.strokeStyle = CONN_COLORS[conn.type] || '#555';
        ctx.lineWidth = 2 / this.camera.zoom;
        ctx.stroke();

        // Arrow head
        const angle = Math.atan2(p2.y - cpy, p2.x - cpx);
        const aLen = 8 / this.camera.zoom;
        ctx.beginPath();
        ctx.moveTo(p2.x, p2.y);
        ctx.lineTo(p2.x - aLen * Math.cos(angle - 0.3),
                   p2.y - aLen * Math.sin(angle - 0.3));
        ctx.lineTo(p2.x - aLen * Math.cos(angle + 0.3),
                   p2.y - aLen * Math.sin(angle + 0.3));
        ctx.closePath();
        ctx.fillStyle = CONN_COLORS[conn.type] || '#555';
        ctx.fill();
    }

    resize() {
        const rect = this.el.parentElement.getBoundingClientRect();
        this.el.width = rect.width;
        this.el.height = rect.height;
    }

    setSelected(block) {
        this._selected = block;
    }
}
