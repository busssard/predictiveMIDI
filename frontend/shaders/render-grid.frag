precision highp float;

uniform sampler2D u_state;    // r, e, s, h
uniform sampler2D u_weights;  // w_left, w_right, w_up, w_down
uniform vec2 u_resolution;    // grid size
uniform vec3 u_color_pos;     // color for positive error (over-predicting)
uniform vec3 u_color_neg;     // color for negative error (under-predicting)

varying vec2 v_uv;

void main() {
    vec2 pixel = 1.0 / u_resolution;
    vec2 cell = v_uv * u_resolution;
    vec2 cellCenter = floor(cell) + 0.5;
    vec2 cellFrac = fract(cell);

    // Distance from cell center (for gap rendering)
    vec2 distFromCenter = abs(cellFrac - 0.5);

    // Get this cell's state
    vec2 centerUV = cellCenter / u_resolution;
    vec4 state = texture2D(u_state, centerUV);
    vec4 weights = texture2D(u_weights, centerUV);

    // Column index for this cell
    float col = floor(cell.x);

    // For input (col 0) and output (last col): show representation channel
    // For interior columns: show error channel
    bool isClampedCol = (col < 0.5) || (col > u_resolution.x - 1.5);
    float value = isClampedCol ? state.r : state.g;
    float absValue = abs(value);

    // Color: clamped columns use green/cyan, interior uses error colors
    vec3 valueColor;
    if (isClampedCol) {
        valueColor = value > 0.0
            ? vec3(0.0, 1.0, 0.6)   // green for positive (active input)
            : vec3(0.0, 0.4, 0.6);  // dark cyan for negative
    } else {
        valueColor = value > 0.0 ? u_color_pos : u_color_neg;
    }

    // Alpha from value magnitude (clamped)
    float alpha = clamp(absValue * 5.0, 0.05, 1.0);

    // Gap blending: check if we're in the gap region between cells
    float gapSize = 0.12; // fraction of cell that is gap
    float inGapX = smoothstep(0.5 - gapSize, 0.5, distFromCenter.x);
    float inGapY = smoothstep(0.5 - gapSize, 0.5, distFromCenter.y);
    float inGap = max(inGapX, inGapY);

    // Connection strength determines gap fill
    // If in horizontal gap, use w_right; if vertical gap, use w_up
    float connectionStrength = 0.0;
    if (inGapX > inGapY) {
        connectionStrength = cellFrac.x > 0.5
            ? abs(weights.g) // w_right
            : abs(weights.r); // w_left
    } else {
        connectionStrength = cellFrac.y > 0.5
            ? abs(weights.b) // w_up
            : abs(weights.a); // w_down
    }
    connectionStrength = clamp(connectionStrength * 2.0, 0.0, 1.0);

    // In gap: blend toward black based on connection strength
    // Strong connection = color bleeds through; weak = black gap
    float gapDarkness = inGap * (1.0 - connectionStrength);

    vec3 finalColor = valueColor * alpha * (1.0 - gapDarkness);

    gl_FragColor = vec4(finalColor, 1.0);
}
