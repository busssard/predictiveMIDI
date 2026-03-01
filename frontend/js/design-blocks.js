/**
 * Block data model and manipulation logic for the network designer.
 */

let _nextId = 1;

export class BlockModel {
    constructor({id, x, y, columns, heights, connectivity, algorithm, role, name,
                 input_edge, output_edge, rotation} = {}) {
        this.id = id || `block_${_nextId++}`;
        this.x = x || 0;
        this.y = y || 0;
        this.rotation = rotation || 0;
        this.columns = columns || 4;
        this.heights = heights || new Array(this.columns).fill(128);
        this.connectivity = connectivity || 'fc';
        this.algorithm = algorithm || 'pc';
        this.role = role || 'encoder';
        this.name = name || this.id;
        this.input_edge = input_edge || 'left';
        this.output_edge = output_edge || 'right';
    }

    /** Pixel dimensions at default scale. */
    get width() {
        return this.columns * 12 + 20;
    }

    get height() {
        return Math.max(...this.heights) * 0.6 + 30;
    }

    /** Get (x, y) midpoint of an edge in world coordinates. */
    getEdgePosition(edge) {
        const cx = this.x + this.width / 2;
        const cy = this.y + this.height / 2;
        switch (edge) {
            case 'left':   return { x: this.x, y: cy };
            case 'right':  return { x: this.x + this.width, y: cy };
            case 'top':    return { x: cx, y: this.y };
            case 'bottom': return { x: cx, y: this.y + this.height };
        }
        return { x: cx, y: cy };
    }

    /** Resize a corner by adjusting that column's height and interpolating. */
    resizeCorner(cornerIdx, deltaH) {
        const idx = cornerIdx === 0 ? 0 : this.columns - 1;
        this.heights[idx] = Math.max(4, Math.round(this.heights[idx] + deltaH));
        // Linearly interpolate between first and last
        if (this.columns > 1) {
            const h0 = this.heights[0];
            const hN = this.heights[this.columns - 1];
            for (let i = 1; i < this.columns - 1; i++) {
                const t = i / (this.columns - 1);
                this.heights[i] = Math.round(h0 * (1 - t) + hN * t);
            }
        }
    }

    getBounds() {
        return {
            x: this.x,
            y: this.y,
            width: this.width,
            height: this.height,
        };
    }

    clone() {
        return new BlockModel({
            x: this.x + 30,
            y: this.y + 30,
            columns: this.columns,
            heights: [...this.heights],
            connectivity: this.connectivity,
            algorithm: this.algorithm,
            role: this.role,
            name: this.name + '_copy',
            input_edge: this.input_edge,
            output_edge: this.output_edge,
        });
    }

    toJSON() {
        return {
            id: this.id,
            x: this.x,
            y: this.y,
            rotation: this.rotation,
            columns: this.columns,
            heights: this.heights,
            connectivity: this.connectivity,
            algorithm: this.algorithm,
            role: this.role,
            name: this.name,
            input_edge: this.input_edge,
            output_edge: this.output_edge,
        };
    }

    static fromJSON(data) {
        const b = new BlockModel(data);
        // Ensure id counter stays ahead
        const num = parseInt((data.id || '').replace('block_', ''), 10);
        if (!isNaN(num) && num >= _nextId) _nextId = num + 1;
        return b;
    }
}

export function resetBlockIdCounter() {
    _nextId = 1;
}
