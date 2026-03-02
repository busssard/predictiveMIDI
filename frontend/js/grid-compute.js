/**
 * Manages the WebGL ping-pong compute pipeline for PC inference.
 */
export class GridCompute {
    constructor(gl, vertSource, fragSource) {
        this.gl = gl;
        this.program = this._createProgram(vertSource, fragSource);

        // Attribute: full-screen quad
        const quad = new Float32Array([-1,-1, 1,-1, -1,1, 1,1]);
        this.quadBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, quad, gl.STATIC_DRAW);

        // Uniform locations
        this.u_state = gl.getUniformLocation(this.program, 'u_state');
        this.u_weights = gl.getUniformLocation(this.program, 'u_weights');
        this.u_params = gl.getUniformLocation(this.program, 'u_params');
        this.u_resolution = gl.getUniformLocation(this.program, 'u_resolution');
        this.u_clamp_tex = gl.getUniformLocation(this.program, 'u_clamp_tex');
        this.u_do_clamp = gl.getUniformLocation(this.program, 'u_do_clamp');
    }

    /**
     * Run one relaxation step: read from stateTextures[srcIdx],
     * write to stateFramebuffers[dstIdx].
     *
     * @param {WebGLTexture|null} clampTexture - 1×H float texture for column-0 clamping
     */
    step(stateTextures, stateFramebuffers, weightsTexture, paramsTexture,
         srcIdx, dstIdx, gridWidth, gridHeight, clampTexture = null) {
        const gl = this.gl;

        gl.useProgram(this.program);
        gl.bindFramebuffer(gl.FRAMEBUFFER, stateFramebuffers[dstIdx]);
        gl.viewport(0, 0, gridWidth, gridHeight);

        // Bind textures
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, stateTextures[srcIdx]);
        gl.uniform1i(this.u_state, 0);

        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, weightsTexture);
        gl.uniform1i(this.u_weights, 1);

        gl.activeTexture(gl.TEXTURE2);
        gl.bindTexture(gl.TEXTURE_2D, paramsTexture);
        gl.uniform1i(this.u_params, 2);

        // Clamp texture on unit 3
        gl.activeTexture(gl.TEXTURE3);
        if (clampTexture) {
            gl.bindTexture(gl.TEXTURE_2D, clampTexture);
            gl.uniform1i(this.u_clamp_tex, 3);
            gl.uniform1f(this.u_do_clamp, 1.0);
        } else {
            gl.bindTexture(gl.TEXTURE_2D, null);
            gl.uniform1f(this.u_do_clamp, 0.0);
        }

        gl.uniform2f(this.u_resolution, gridWidth, gridHeight);

        // Draw full-screen quad
        const a_position = gl.getAttribLocation(this.program, 'a_position');
        gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
        gl.enableVertexAttribArray(a_position);
        gl.vertexAttribPointer(a_position, 2, gl.FLOAT, false, 0, 0);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }

    _createProgram(vertSource, fragSource) {
        const gl = this.gl;
        const vs = this._compile(gl.VERTEX_SHADER, vertSource);
        const fs = this._compile(gl.FRAGMENT_SHADER, fragSource);
        const prog = gl.createProgram();
        gl.attachShader(prog, vs);
        gl.attachShader(prog, fs);
        gl.linkProgram(prog);
        if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
            throw new Error('Link error: ' + gl.getProgramInfoLog(prog));
        }
        return prog;
    }

    _compile(type, source) {
        const gl = this.gl;
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            throw new Error('Compile error: ' + gl.getShaderInfoLog(shader));
        }
        return shader;
    }
}
