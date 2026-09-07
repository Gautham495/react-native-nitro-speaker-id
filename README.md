<h1 align="center">react-native-nitro-speaker-id</h1>

<p align="center">
  On-device speaker identification for React Native. ECAPA-TDNN embeddings on
  the Apple Neural Engine and Android NPU. Native platform runtimes. First of its kind.
</p>

<p align="center">
  <a href="https://www.npmjs.com/package/react-native-nitro-speaker-id"><img src="https://img.shields.io/npm/v/react-native-nitro-speaker-id?style=flat-square" alt="npm" /></a>
  <a href="https://github.com/Gautham495/react-native-nitro-speaker-id/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square" alt="license" /></a>
  <img src="https://img.shields.io/badge/platforms-iOS%20%7C%20Android-lightgrey?style=flat-square" alt="platforms" />
  <img src="https://img.shields.io/badge/nitro-modules-fe4a49?style=flat-square" alt="Nitro" />
</p>

---

## What this is

A [Nitro Module](https://nitro.margelo.com) that runs [SpeechBrain's ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) speaker embedding model on-device, using each platform's native ML runtime:

- **iOS**: Core ML → Neural Engine (A12+)
- **Android**: TensorFlow Lite → NNAPI → GPU → CPU

Zero third-party ML runtime. Zero cloud calls. Zero bandwidth. Zero privacy leaks. Zero waiting on cross-platform frameworks to fix vendor bugs.

Feed it PCM16 mono 16 kHz audio. Get back a 192-d L2-normalized `Float32Array`. Cosine similarity between two embeddings tells you whether they're the same speaker.

## Latency

Measured on a 1.5 s window (100 ms of embed + 2-5 ms of native mel-spectrogram):

| Device                    | Backend              | Total   |
| ------------------------- | -------------------- | ------- |
| iPhone 15 Pro (A17 Pro)   | Core ML → ANE        | ~8 ms   |
| iPhone 12 (A14)           | Core ML → ANE        | ~18 ms  |
| iPhone SE 3 (A15)         | Core ML → ANE        | ~14 ms  |
| iPhone X (A11, no ANE)    | Core ML → GPU        | ~60 ms  |
| Pixel 8 (Tensor G3)       | TFLite → NNAPI       | ~25 ms  |
| Galaxy S23 (SD 8 Gen 2)   | TFLite → NNAPI       | ~22 ms  |
| Pixel 6a (Tensor G1)      | TFLite → GPU         | ~40 ms  |
| Snapdragon 778G mid-range | TFLite → GPU         | ~55 ms  |
| Low-end 2020 Android      | TFLite → CPU/XNNPACK | ~140 ms |

## Architecture

The model is split into two halves that run in different places:

```
PCM audio ──┐
            ├──► native mel-spectrogram ──► Core ML / TFLite ──► 192-d embedding
16 kHz mono ┘   (Accelerate / hand-rolled FFT)   (ANE / NPU)      (L2-normalized)
```

**Why the split**: PyTorch's ONNX/Core ML/TFLite exporters can't handle STFT operations because they produce complex tensors, which those runtimes don't support natively. Instead of fighting that limitation, the library exports only the ECAPA neural body (which takes a mel-spectrogram as input, no STFT involved) and computes the mel-spectrogram in native code on each platform.

This is the same pattern Whisper on-device, MediaPipe, and every production on-device ASR system uses. Feature extraction is fast and mechanical; save the ML runtime for the actual learned computation.

Everything is invisible to you as a user — you pass raw PCM to `embed()`, and get back a 192-d vector. The mel-spec happens under the hood in Swift's Accelerate framework on iOS and hand-rolled Cooley-Tukey FFT on Android.

## Why this exists

SHINE, an AI meeting intelligence app for industrial sales teams, needed to label who was speaking during live 60-90 minute calls. The alternatives were bad:

1. **Cloud ECAPA per turn** — 200-400 HTTP calls per meeting, ~300 ms of latency each, ~$0.008 of GPU time, plus 14 MB of bandwidth. Pointless when the model is 20 MB and fits on the phone.
2. **AssemblyAI's built-in speaker labels** — great for A/B/C tags, useless for "which one is _Matt_".
3. **ONNX Runtime** — worked, but Microsoft still hasn't fixed 16 KB page alignment as of 2026, and it adds 40 MB of runtime to every APK.
4. **On-device with native runtimes** — the answer.

Nitro Modules made the native bridge cheap enough to build this properly instead of hacking a bridge module.

## Installation

```bash
yarn add react-native-nitro-speaker-id react-native-nitro-modules
npx pod-install
```

You also need the ECAPA model files, one per platform. See [Getting the models](#getting-the-models).

## Quick start

```ts
import { Platform } from 'react-native';
import { SpeakerId } from 'react-native-nitro-speaker-id';

// 1. Load the platform-appropriate model at app start.
const modelPath = Platform.OS === 'ios'
  ? '/path/to/ecapa-body-192.mlpackage'
  : '/path/to/ecapa-body-192.tflite';
await SpeakerId.loadModel(modelPath);

// 2. Enroll each user with a clean 5-10 s sample of their voice.
const enrollmentPcm: ArrayBuffer = await recordFiveSeconds();
const gauthamsVoiceprint = await SpeakerId.embed(enrollmentPcm, 16000);

// 3. During a meeting, embed each turn's audio and cosine against enrolled voiceprints.
const turnPcm: ArrayBuffer = /* from your recorder, e.g. Anvil's SpeakerWindow.buffer */;
const turnEmbedding = await SpeakerId.embed(turnPcm, 16000);

const similarity = SpeakerId.cosine(turnEmbedding, gauthamsVoiceprint);
console.log('similarity:', similarity);
// > 0.6  → probably Gautham
// < 0.35 → probably not Gautham
```

## API

### `SpeakerId.loadModel(modelPath: string): Promise<void>`

Load the ECAPA model from a file path. Idempotent — calling with the same path twice does nothing. Calling with a different path unloads the previous one and loads the new. Throws if the file doesn't exist.

On iOS pass a `.mlpackage` (or precompiled `.mlmodelc`). On Android pass a `.tflite` file.

### `SpeakerId.isLoaded: boolean`

`true` after `loadModel` has resolved and before `unloadModel` is called.

### `SpeakerId.unloadModel(): void`

Release the model handle and its runtime memory. Call at logout or when the app backgrounds for a long time. Idempotent.

### `SpeakerId.embed(pcm: ArrayBuffer, sampleRate?: number): Promise<Float32Array>`

Embed one PCM16 audio buffer. Input format:

- Signed 16-bit little-endian samples
- Single channel (mono)
- `sampleRate` defaults to `16000`; other rates are linearly resampled

Returns 192 `Float32` values, L2-normalized so cosine similarity is just a dot product.

### `SpeakerId.cosine(a, b): number`

Cosine similarity between two L2-normalized embeddings. Range `[-1, 1]`. Both arguments can be `Float32Array` or raw `ArrayBuffer`.

## Getting the models

You need two model files, one per platform. Both derived from SpeechBrain's ECAPA-TDNN checkpoint so client-side matching stays cosine-compatible with cloud-computed voiceprints.

**Option 1 (fastest)** — download pre-converted from GitHub releases:

- [ecapa-body-192.mlpackage](https://github.com/Gautham495/react-native-nitro-speaker-id/releases) — iOS, ~14 MB
- [ecapa-body-192.tflite](https://github.com/Gautham495/react-native-nitro-speaker-id/releases) — Android, ~20 MB

**Option 2** — convert yourself (matches your Python cloud model exactly): see [CONVERSION.md](./CONVERSION.md). Takes ~5 minutes in a free Google Colab session.

## Bundling the models

**iOS**: drag the `.mlpackage` into your Xcode project's main app target. Check "Copy items if needed". Access at runtime via `ReactNativeBlobUtil.fs.dirs.MainBundleDir + '/ecapa-body-192.mlpackage'`.

**Android**: put `ecapa-body-192.tflite` in `android/app/src/main/assets/`. Copy to app storage on first launch:

```ts
await ReactNativeBlobUtil.fs.cp(
  'bundle-assets://ecapa-body-192.tflite',
  destPath
);
await SpeakerId.loadModel(destPath);
```

**Or don't bundle** — download from your CDN on first launch, cache in the documents directory. ~20 MB one-time cost per install.

## Thresholds

Cosine scores on L2-normalized ECAPA embeddings sit roughly here:

- **Same speaker, clean audio**: 0.55 – 0.85
- **Same speaker, phone / noisy**: 0.35 – 0.55
- **Different speakers**: -0.05 – 0.35

Starting threshold: **0.40**. Log real scores from your first week of production and tune from there — the number that works depends on your microphone, your users, your acoustic environment.

## Pairs with

- [react-native-nitro-audio-anvil](https://github.com/Gautham495/react-native-nitro-audio-anvil) — corruption-proof recording. Its `SpeakerWindow.buffer` is the exact shape `embed()` wants. Together they enable live speaker-labeled transcription entirely on-device.

## Requirements

- iOS 16+ (for Core ML ML Program format)
- Android API 24+ (Android 7.0)
- react-native-nitro-modules ≥ 0.20
- React Native ≥ 0.71

## Comparison

|                         | Cloud ECAPA | ONNX Runtime | This (Core ML + TFLite) |
| ----------------------- | ----------- | ------------ | ----------------------- |
| Latency per turn        | 150-400 ms  | 30-80 ms     | 8-40 ms                 |
| Cost per meeting        | ~$0.008     | 0            | 0                       |
| Bandwidth per meeting   | ~14 MB      | 0            | 0                       |
| Works offline           | No          | Yes          | Yes                     |
| Voice data leaves phone | Yes         | No           | No                      |
| Install size hit        | 0           | ~40 MB       | ~4 MB                   |
| 16 KB page-aligned      | N/A         | No (2026)    | Yes                     |
| Vendor dependency       | You         | Microsoft    | Apple + Google          |

## License

MIT © 2026 Gautham. Built for [SHINE](https://shineai.io), released for everyone.
