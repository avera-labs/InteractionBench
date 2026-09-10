// Use opus-recorder's encoderWorker (same as the browser, with embedded wasm) to encode PCM into OGG/Opus.
// The server / sphn only accepts opus-recorder's exact format; sphn's own encoder and ffmpeg are both incompatible.
// Usage: node opus_encode.js <in_f32_raw> <out_ogg>   —— input = raw float32 LE (mono 24k)
const fs = require('fs');
const path = require('path');

// ---- Web Worker global shim (lets the browser worker run bare in node) ----
global.self = global;
global.importScripts = () => {};
global.close = () => {};
const pages = [];
global.postMessage = (msg) => {
  if (msg && msg.message === 'page' && msg.page) {
    const p = msg.page;
    pages.push(Buffer.from(p.buffer, p.byteOffset, p.byteLength));
  }
};

// Load the worker: it attaches onmessage to global and instantiates the wasm internally
require(path.join(__dirname, 'dashboard', 'static', 'encoderWorker.min.js'));

const inRaw = fs.readFileSync(process.argv[2]);
const f32 = new Float32Array(inRaw.buffer, inRaw.byteOffset, Math.floor(inRaw.byteLength / 4));

// Match the opus-recorder config in tts.html item by item
const cfg = {
  command: 'init',
  encoderSampleRate: 24000, originalSampleRate: 24000, bufferLength: 4096,
  numberOfChannels: 1, encoderFrameSize: 20, encoderApplication: 2049,
  streamPages: true, maxFramesPerPage: 2, encoderComplexity: 0, resampleQuality: 3,
};
global.onmessage({ data: cfg });
global.onmessage({ data: { command: 'getHeaderPages' } });

const CHUNK = parseInt(process.argv[4] || '128', 10); // samples fed to the encoder per call (browser AudioWorklet uses 128-sample quanta)
for (let i = 0; i < f32.length; i += CHUNK) {
  const slice = f32.subarray(i, Math.min(i + CHUNK, f32.length));
  global.onmessage({ data: { command: 'encode', buffers: [slice] } });
}
global.onmessage({ data: { command: 'done' } });

fs.writeFileSync(process.argv[3], Buffer.concat(pages));
process.stderr.write(`opus_encode: ${f32.length} samples → ${pages.length} pages, ${Buffer.concat(pages).length} bytes\n`);
