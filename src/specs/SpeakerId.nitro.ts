import type { HybridObject } from 'react-native-nitro-modules';

/**
 * On-device speaker identification via ECAPA-TDNN.
 *
 * Contract: takes 16 kHz mono PCM16 audio (matches Anvil's speaker windows),
 * returns a 192-d L2-normalized Float32Array embedding. Same shape as the
 * SpeechBrain cloud model — swap this in for `getVoiceEmbeddingFromBuffer`
 * in your streamer and nothing else changes.
 *
 * Model runs on ONNX Runtime with Core ML delegate on iOS and NNAPI on
 * Android. Inference is 15-40 ms on iPhone 12+ Neural Engine, 40-80 ms
 * on mid-range Android. No network. No cost per call.
 */
export interface SpeakerId extends HybridObject<{
  ios: 'swift';
  android: 'kotlin';
}> {
  /**
   * Load the ONNX model from a file path. Call this once at app startup
   * (or lazily on first meeting). Idempotent — a second call with the same
   * path is a no-op. A different path swaps the model.
   *
   * Recommended: bundle the .onnx in your app assets and copy to the
   * documents dir on first launch, or download from your CDN. The model
   * is ~20 MB INT8-quantized.
   */
  loadModel(modelPath: string): Promise<void>;

  /**
   * Whether the model is loaded and ready. Cheap — just checks the native
   * handle. Use before calling embed() to fail fast.
   */
  readonly isLoaded: boolean;

  /**
   * Free the model handle and release its memory. Call when the user logs
   * out or the app goes to the background for a long time. Idempotent.
   */
  unloadModel(): void;

  /**
   * Embed a PCM16 mono 16 kHz audio buffer. Returns 192 Float32s,
   * L2-normalized so cosine similarity is a dot product.
   *
   * Buffer format:
   *   - ArrayBuffer of int16 little-endian samples
   *   - 16000 samples per second
   *   - single channel
   *   - typical length: 0.5s to 30s of audio
   *
   * Anvil's SpeakerWindow.buffer matches this exactly.
   */
  embed(pcm: ArrayBuffer, sampleRate: number): Promise<ArrayBuffer>;

  /**
   * Cosine similarity between two L2-normalized embeddings.
   * Just a dot product — provided in native for perf when comparing
   * against many enrolled voiceprints, but pure JS is fine for < 100.
   *
   * Both buffers must be Float32Array-backed and the same length.
   */
  cosine(a: ArrayBuffer, b: ArrayBuffer): number;
}
