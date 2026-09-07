import { NitroModules } from 'react-native-nitro-modules';
import type { SpeakerId as SpeakerIdSpec } from './specs/SpeakerId.nitro';

/**
 * Nitro HybridObject singleton. Every call goes through the same native handle
 * — the ONNX runtime session is instantiated once inside the native side and
 * kept warm for the process lifetime.
 */
const HybridSpeakerId =
  NitroModules.createHybridObject<SpeakerIdSpec>('SpeakerId');

/**
 * High-level facade with convenience wrappers. Same object as HybridSpeakerId
 * but with typed helpers for the common cases so callers don't hand-cast
 * ArrayBuffers on every use.
 */
export const SpeakerId = {
  /**
   * Load the ECAPA ONNX model. Call once at startup or lazily on first use.
   */
  async loadModel(modelPath: string): Promise<void> {
    return HybridSpeakerId.loadModel(modelPath);
  },

  /**
   * True after loadModel has succeeded and unloadModel has not been called.
   */
  get isLoaded(): boolean {
    return HybridSpeakerId.isLoaded;
  },

  unloadModel(): void {
    HybridSpeakerId.unloadModel();
  },

  /**
   * Embed one PCM16 audio window. Returns a Float32Array (192 values,
   * L2-normalized). This is what you match against enrolled voiceprints.
   *
   * Pass Anvil's SpeakerWindow.buffer straight through — it's already the
   * right shape (int16 mono, 16 kHz).
   */
  async embed(
    pcm: ArrayBuffer,
    sampleRate: number = 16000
  ): Promise<Float32Array> {
    const raw = await HybridSpeakerId.embed(pcm, sampleRate);
    return new Float32Array(raw);
  },

  /**
   * Cosine similarity for two embeddings. Both inputs must be Float32Array
   * or ArrayBuffer-backed Float32 data, same length, L2-normalized.
   */
  cosine(a: Float32Array | ArrayBuffer, b: Float32Array | ArrayBuffer): number {
    const abuf = (a instanceof Float32Array ? a.buffer : a) as ArrayBuffer;
    const bbuf = (b instanceof Float32Array ? b.buffer : b) as ArrayBuffer;
    return HybridSpeakerId.cosine(abuf, bbuf);
  },
};

export type { SpeakerId as SpeakerIdSpec } from './specs/SpeakerId.nitro';
