/**
 * Handles MIDI input (live and file playback) and output via Web MIDI API.
 */
export class MidiIO {
    constructor() {
        this.midiAccess = null;
        this.inputDevice = null;
        this.outputDevice = null;
        // Current state: velocity per pitch (0-127 -> 0.0-1.0)
        this.inputState = new Float32Array(128);
        this.outputState = new Float32Array(128);
        this._onNoteCallback = null;
    }

    async init() {
        if (!navigator.requestMIDIAccess) {
            console.warn('Web MIDI API not available');
            return;
        }
        this.midiAccess = await navigator.requestMIDIAccess();
        return this.getInputDevices();
    }

    getInputDevices() {
        if (!this.midiAccess) return [];
        const devices = [];
        for (const input of this.midiAccess.inputs.values()) {
            devices.push({ id: input.id, name: input.name });
        }
        return devices;
    }

    getOutputDevices() {
        if (!this.midiAccess) return [];
        const devices = [];
        for (const output of this.midiAccess.outputs.values()) {
            devices.push({ id: output.id, name: output.name });
        }
        return devices;
    }

    connectInput(deviceId) {
        if (this.inputDevice) {
            this.inputDevice.onmidimessage = null;
        }
        this.inputDevice = this.midiAccess.inputs.get(deviceId);
        if (this.inputDevice) {
            this.inputDevice.onmidimessage = (msg) => this._handleMessage(msg);
        }
    }

    connectOutput(deviceId) {
        this.outputDevice = this.midiAccess.outputs.get(deviceId);
    }

    _handleMessage(msg) {
        const [status, note, velocity] = msg.data;
        const command = status & 0xf0;
        if (command === 0x90 && velocity > 0) {
            // Note on
            this.inputState[note] = velocity / 127.0;
        } else if (command === 0x80 || (command === 0x90 && velocity === 0)) {
            // Note off
            this.inputState[note] = 0.0;
        }
        if (this._onNoteCallback) {
            this._onNoteCallback(note, this.inputState[note]);
        }
    }

    onNote(callback) {
        this._onNoteCallback = callback;
    }

    /**
     * Send the grid's output column as MIDI note events.
     * outputValues: Float32Array(128) with velocities 0.0-1.0
     */
    sendOutput(outputValues) {
        if (!this.outputDevice) return;
        for (let pitch = 0; pitch < 128; pitch++) {
            const vel = Math.round(outputValues[pitch] * 127);
            const prevVel = Math.round(this.outputState[pitch] * 127);

            if (vel > 0 && prevVel === 0) {
                // Note on
                this.outputDevice.send([0x90, pitch, vel]);
            } else if (vel === 0 && prevVel > 0) {
                // Note off
                this.outputDevice.send([0x80, pitch, 0]);
            }
        }
        this.outputState.set(outputValues);
    }

    /**
     * Load and play back a MIDI file through inputState.
     * Returns a controller object with stop() method.
     */
    playFile(arrayBuffer, fs, onTick) {
        // Minimal MIDI file parser — we parse note events and schedule them
        // For a proper implementation, use a library like Midi.js or tone.js
        // This is a placeholder that will be replaced with proper parsing
        let stopped = false;
        return {
            stop() { stopped = true; }
        };
    }

    getInputState() {
        return this.inputState;
    }
}
