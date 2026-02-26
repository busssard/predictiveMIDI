/**
 * Load exported PC grid model into WebGL textures.
 *
 * Expects files: config.json, state.bin, weights.bin, params.bin
 * in the given base URL.
 */
export async function loadModel(gl, baseUrl) {
    const configResp = await fetch(`${baseUrl}/config.json`);
    const config = await configResp.json();

    const size = config.grid_size;

    const [stateData, weightsData, paramsData] = await Promise.all([
        fetch(`${baseUrl}/state.bin`).then(r => r.arrayBuffer()),
        fetch(`${baseUrl}/weights.bin`).then(r => r.arrayBuffer()),
        fetch(`${baseUrl}/params.bin`).then(r => r.arrayBuffer()),
    ]);

    return {
        config,
        state: createFloatTexture(gl, size, new Float32Array(stateData)),
        weights: createFloatTexture(gl, size, new Float32Array(weightsData)),
        params: createFloatTexture(gl, size, new Float32Array(paramsData)),
    };
}

/**
 * Create a size x size RGBA float texture from flat float32 data.
 */
export function createFloatTexture(gl, size, data) {
    const ext = gl.getExtension('OES_texture_float');
    if (!ext) throw new Error('OES_texture_float not supported');

    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(
        gl.TEXTURE_2D, 0, gl.RGBA,
        size, size, 0,
        gl.RGBA, gl.FLOAT, data
    );
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return tex;
}

/**
 * Create a framebuffer with a float texture attachment (for ping-pong).
 */
export function createFramebuffer(gl, texture) {
    const fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(
        gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0,
        gl.TEXTURE_2D, texture, 0
    );
    return fb;
}
