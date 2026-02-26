precision highp float;

uniform sampler2D u_state;    // r, e, s, h
uniform sampler2D u_weights;  // w_left, w_right, w_up, w_down
uniform sampler2D u_params;   // alpha, beta, lr, bias
uniform vec2 u_resolution;    // grid size (128.0, 128.0)

varying vec2 v_uv;

void main() {
    vec2 pixel = 1.0 / u_resolution;
    vec4 state = texture2D(u_state, v_uv);
    vec4 weights = texture2D(u_weights, v_uv);
    vec4 params = texture2D(u_params, v_uv);

    float r = state.r;
    float e = state.g;
    float s = state.b;
    float h = state.a;

    float w_left  = weights.r;
    float w_right = weights.g;
    float w_up    = weights.b;
    float w_down  = weights.a;

    float alpha = params.r;
    float beta  = params.g;
    float lr    = params.b;
    float bias  = params.a;

    // Activation
    float r_act = tanh(r);

    // Get neighbor representations (zero at boundaries via clamp-to-edge)
    float r_left  = tanh(texture2D(u_state, v_uv + vec2(-pixel.x, 0.0)).r);
    float r_right = tanh(texture2D(u_state, v_uv + vec2( pixel.x, 0.0)).r);
    float r_up    = tanh(texture2D(u_state, v_uv + vec2(0.0,  pixel.y)).r);
    float r_down  = tanh(texture2D(u_state, v_uv + vec2(0.0, -pixel.y)).r);

    // Zero out boundary lookups (clamp-to-edge repeats edge values, we want 0)
    if (v_uv.x - pixel.x < 0.0) r_left = 0.0;
    if (v_uv.x + pixel.x > 1.0) r_right = 0.0;
    if (v_uv.y + pixel.y > 1.0) r_up = 0.0;
    if (v_uv.y - pixel.y < 0.0) r_down = 0.0;

    // Incoming predictions
    float pred = w_left * r_left + w_right * r_right + w_up * r_up + w_down * r_down;

    // Error
    float new_e = r - pred - bias;

    // Neighbor errors
    float e_left  = texture2D(u_state, v_uv + vec2(-pixel.x, 0.0)).g;
    float e_right = texture2D(u_state, v_uv + vec2( pixel.x, 0.0)).g;
    float e_up    = texture2D(u_state, v_uv + vec2(0.0,  pixel.y)).g;
    float e_down  = texture2D(u_state, v_uv + vec2(0.0, -pixel.y)).g;

    if (v_uv.x - pixel.x < 0.0) e_left = 0.0;
    if (v_uv.x + pixel.x > 1.0) e_right = 0.0;
    if (v_uv.y + pixel.y > 1.0) e_up = 0.0;
    if (v_uv.y - pixel.y < 0.0) e_down = 0.0;

    float neighbor_err = e_left * w_left + e_right * w_right + e_up * w_up + e_down * w_down;

    // Temporal updates
    float new_s = alpha * s + (1.0 - alpha) * r;
    float new_h = beta * tanh(h);

    // Representation update
    float new_r = r + lr * (-new_e + neighbor_err + new_h + new_s - r);

    gl_FragColor = vec4(new_r, new_e, new_s, new_h);
}
