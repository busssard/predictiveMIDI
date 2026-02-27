/**
 * Canvas-based loss chart — raw red line on dark grid.
 * Log10 Y axis, phase transition markers.
 */
export class LossChart {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.pad = { top: 16, right: 16, bottom: 28, left: 56 };
        this.colors = {
            bg: '#0a0a0a',
            grid: '#1a1a1a',
            tick: '#666',
            line: '#ff2200',
            phase: '#444',
            phaseText: '#888',
        };
    }

    resize() {
        const rect = this.canvas.parentElement.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        this.canvas.width = rect.width * dpr;
        this.canvas.height = Math.max(180, rect.height) * dpr;
        this.canvas.style.width = rect.width + 'px';
        this.canvas.style.height = Math.max(180, rect.height) + 'px';
        this.ctx.scale(dpr, dpr);
        this._w = rect.width;
        this._h = Math.max(180, rect.height);
    }

    draw(metrics) {
        const ctx = this.ctx;
        const w = this._w || this.canvas.width;
        const h = this._h || this.canvas.height;

        ctx.setTransform(window.devicePixelRatio || 1, 0, 0, window.devicePixelRatio || 1, 0, 0);
        ctx.fillStyle = this.colors.bg;
        ctx.fillRect(0, 0, w, h);

        if (!metrics || metrics.length < 2) {
            ctx.fillStyle = '#333';
            ctx.font = '11px JetBrains Mono, monospace';
            ctx.fillText('no data', w / 2 - 20, h / 2);
            return;
        }

        const steps = metrics.map(m => m.step);
        const errors = metrics.map(m => m.avg_error).map(e => Math.max(e, 1e-8));
        const minStep = steps[0];
        const maxStep = steps[steps.length - 1];
        const logMin = Math.floor(Math.log10(Math.min(...errors)));
        const logMax = Math.ceil(Math.log10(Math.max(...errors)));

        const pw = w - this.pad.left - this.pad.right;
        const ph = h - this.pad.top - this.pad.bottom;
        const xMap = s => this.pad.left + (s - minStep) / (maxStep - minStep || 1) * pw;
        const yMap = e => {
            const le = Math.log10(Math.max(e, 1e-8));
            return this.pad.top + (1 - (le - logMin) / (logMax - logMin || 1)) * ph;
        };

        // Grid lines
        ctx.strokeStyle = this.colors.grid;
        ctx.lineWidth = 0.5;
        ctx.font = '9px JetBrains Mono, monospace';
        ctx.fillStyle = this.colors.tick;

        for (let p = logMin; p <= logMax; p++) {
            const y = yMap(Math.pow(10, p));
            ctx.beginPath();
            ctx.moveTo(this.pad.left, y);
            ctx.lineTo(this.pad.left + pw, y);
            ctx.stroke();
            ctx.fillText('1e' + p, 4, y + 3);
        }

        const si = this._niceInterval(maxStep - minStep);
        for (let s = Math.ceil(minStep / si) * si; s <= maxStep; s += si) {
            const x = xMap(s);
            ctx.beginPath();
            ctx.moveTo(x, this.pad.top);
            ctx.lineTo(x, this.pad.top + ph);
            ctx.stroke();
            ctx.fillText(String(Math.round(s)), x - 8, h - 4);
        }

        // Phase markers
        let lastPhase = metrics[0].phase;
        for (let i = 1; i < metrics.length; i++) {
            if (metrics[i].phase !== lastPhase) {
                const x = xMap(metrics[i].step);
                ctx.strokeStyle = this.colors.phase;
                ctx.lineWidth = 1;
                ctx.setLineDash([4, 4]);
                ctx.beginPath();
                ctx.moveTo(x, this.pad.top);
                ctx.lineTo(x, this.pad.top + ph);
                ctx.stroke();
                ctx.setLineDash([]);
                ctx.fillStyle = this.colors.phaseText;
                ctx.font = '8px JetBrains Mono, monospace';
                ctx.fillText('P' + metrics[i].phase, x + 3, this.pad.top + 10);
                lastPhase = metrics[i].phase;
            }
        }

        // Loss line
        ctx.beginPath();
        ctx.strokeStyle = this.colors.line;
        ctx.lineWidth = 1.5;
        for (let i = 0; i < metrics.length; i++) {
            const x = xMap(metrics[i].step);
            const y = yMap(metrics[i].avg_error);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
    }

    _niceInterval(range) {
        if (range <= 0) return 1;
        const rough = range / 5;
        const mag = Math.pow(10, Math.floor(Math.log10(rough)));
        const norm = rough / mag;
        if (norm < 1.5) return mag;
        if (norm < 3.5) return 2 * mag;
        if (norm < 7.5) return 5 * mag;
        return 10 * mag;
    }
}
