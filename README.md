<a href="https://gauthamvijay.com">
  <picture>
    <img alt="react-native-nitro-speaker-id" src="./docs/img/banner.png" />
  </picture>
</a>

# react-native-nitro-speaker-id

**React Native Nitro Module** for **on-device speaker identification** — ECAPA-TDNN embeddings running natively via ExecuTorch on **Apple Neural Engine** (iOS) and **Android CPU**.

---

> [!NOTE]
>
> - This library was built for my production app SHINE, an AI meeting intelligence platform, where we needed to label who was speaking during live sales calls — without shipping audio to the cloud, without per-turn latency, and without vendor lock-in.
> - It runs [SpeechBrain's ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) speaker embedding model on-device via [**ExecuTorch**](https://pytorch.org/executorch), PyTorch's official on-device runtime:
>   - **iOS** — ExecuTorch → Core ML (fp16) → Neural Engine / GPU / CPU
>   - **Android** — ExecuTorch → XNNPACK → CPU
>
> **What you get out of the box:**
>
> - True on-device inference — zero cloud calls, zero bandwidth, zero privacy leaks
> - 192-dimensional L2-normalized embeddings, cosine-comparable
> - **Feature extraction is inside the model graph** — no hand-rolled DSP, no mel-spec drift between cloud and device
> - **Cosine parity with cloud SpeechBrain: 0.998+** (verified by the model exporter)
> - Enroll voice samples once, match new recordings against enrolled voiceprints
> - Latency: 2-30 ms per 3s window on flagship devices, 40-140 ms on mid-range Android
>
> **What this library does NOT do** (by design):
>
> - Handle recording — pair with [react-native-nitro-audio-anvil](https://github.com/Gautham495/react-native-nitro-audio-anvil) for corruption-proof audio capture
> - Provide the ECAPA model file — download the pre-converted `.pte` files from [HuggingFace](https://huggingface.co/mlboydaisuke/ECAPA-TDNN-Speaker-ExecuTorch) and bundle/host them yourself
> - Voice activity detection — feed it audio windows where someone is actually speaking
> - Cloud speaker verification — this is 100% local; if you need a cloud fallback, that's your backend
>
> If you need on-device speaker identification for React Native — meeting transcription, voice-locked features, speaker diarization — this library gives you production-grade inference at native speed, backed by the same weights your cloud pipeline uses.

---

## 📦 Installation

```bash
npm install react-native-nitro-speaker-id react-native-nitro-modules
```

### iOS

Add **ExecuTorch** as a Swift Package Dependency in Xcode:

1. `File → Add Package Dependencies`
2. Paste `https://github.com/pytorch/executorch`
3. Set branch to `swiftpm-1.3.0` (or the latest tagged `swiftpm-X.Y.Z` release)
4. Link these products to your app target:
   - `executorch` (core runtime)
   - `backend_coreml` (Core ML backend)
   - `backend_xnnpack` (CPU fallback)
   - `kernels_portable` (required kernel set)

Then:

```bash
cd ios && bundle exec pod install
```

### Android

Add ExecuTorch to your app's `android/build.gradle`:

```gradle
dependencies {
    implementation "org.pytorch:executorch-android:0.6.0"
}
```

That's it. No separate FFT library, no LiteRT, no JitPack. ExecuTorch bundles XNNPACK internally.

> [!IMPORTANT]
>
> - **iOS**: Fully tested and production-ready ✅
>   - ExecuTorch runtime, Core ML backend, Neural Engine acceleration on A12+
>   - Zero hand-rolled DSP — mel-spec pipeline is inside the model graph
> - **Android**: Fully tested and production-ready ✅
>   - ExecuTorch runtime with XNNPACK backend for CPU inference
>   - Zero hand-rolled DSP — same graph as iOS
>   - 16 KB page-aligned — Google Play submission-ready
> - Tested on React Native 0.85.3 and above. PRs welcome for lower versions.

---

## 🎥 Demo

<table>
  <tr>
    <th align="center">🍏 iOS Demo</th>
    <th align="center">🤖 Android Demo</th>
  </tr>
  <tr>
    <td align="center">
    <img src="./docs/img/iOS.png" width="300" alt="iOS demo" />
    </td>
     <td align="center">
    <img src="./docs/img/android.png" width="300" alt="Android demo" />
    </td>
  </tr>
</table>

---

> [!NOTE]
>
> The ECAPA-TDNN model files are not bundled with the library. Both are available pre-converted on HuggingFace:
>
> - iOS: [`speaker_ecapa_coreml_all.pte`](https://huggingface.co/mlboydaisuke/ECAPA-TDNN-Speaker-ExecuTorch/blob/main/speaker_ecapa_coreml_all.pte) (~42 MB, Core ML fp16)
> - Android: [`speaker_ecapa_xnnpack_fp32.pte`](https://huggingface.co/mlboydaisuke/ECAPA-TDNN-Speaker-ExecuTorch/blob/main/speaker_ecapa_xnnpack_fp32.pte) (~84 MB, XNNPACK fp32)
>
> Pre-hosted for testing on my personal Cloudflare R2 bucket:
>
> - iOS: `https://ml-models-bucket.gauthamvijay.com/ecapa-speaker-ios.pte`
> - Android: `https://ml-models-bucket.gauthamvijay.com/ecapa-speaker-android.pte`
>
> ⚠️ **My personal bucket for demo purposes only.** For anything you're actually shipping, download the files from HuggingFace and host them yourself, or bundle them in your app.

---

## 🧠 Overview

| Feature                            | Implementation                             |
| ---------------------------------- | ------------------------------------------ |
| On-device speaker embeddings       | ECAPA-TDNN (SpeechBrain)                   |
| Runtime                            | ExecuTorch (PyTorch's official on-device)  |
| iOS backend                        | Core ML → Neural Engine / GPU / CPU (fp16) |
| Android backend                    | XNNPACK → CPU (fp32)                       |
| Feature extraction                 | **Baked into the model graph**             |
| Model format                       | `.pte` (ExecuTorch)                        |
| Input                              | PCM16 mono 16 kHz, 3-second window         |
| Output                             | 192-d Float32, L2-normalized               |
| Similarity metric                  | Cosine (dot product on L2-normed vectors)  |
| Cosine parity vs cloud SpeechBrain | **0.998+** (verified)                      |
| 16 KB page alignment               | ✅ Both platforms                          |
| Cloud dependency                   | ❌ None (fully offline)                    |

---

## 📈 Latency

Measured on a 3-second audio window:

| Device                      | Backend           | Total   |
| --------------------------- | ----------------- | ------- |
| Apple Silicon Mac (M3)      | Core ML → ANE     | ~2 ms   |
| iPhone 15 Pro (A17 Pro)     | Core ML → ANE     | ~8 ms   |
| iPhone 12 (A14)             | Core ML → ANE     | ~20 ms  |
| iPhone SE 3 (A15)           | Core ML → ANE     | ~15 ms  |
| iPhone X (A11, no ANE)      | Core ML → GPU/CPU | ~60 ms  |
| Galaxy S23 (SD 8 Gen 2)     | XNNPACK → CPU     | ~30 ms  |
| Pixel 8 (Tensor G3)         | XNNPACK → CPU     | ~40 ms  |
| Pixel 6a (Tensor G1)        | XNNPACK → CPU     | ~60 ms  |
| Vivo mid-range (SD 680/720) | XNNPACK → CPU     | ~100 ms |
| Low-end 2020 Android        | XNNPACK → CPU     | ~140 ms |

---

## ⚙️ Basic Usage

```tsx
import { Platform } from 'react-native';
import { SpeakerId } from 'react-native-nitro-speaker-id';

// 1. Load the platform-appropriate .pte model once at app start.
const modelPath = Platform.OS === 'ios'
  ? '/path/to/ecapa-speaker-ios.pte'
  : '/path/to/ecapa-speaker-android.pte';

await SpeakerId.loadModel(modelPath);

// 2. Enroll each speaker. The model expects 3 seconds of audio; shorter
//    buffers get zero-padded internally.
const enrollmentPcm: ArrayBuffer = await recordCleanSample(); // ~3s clean sample
const gauthamsVoiceprint = await SpeakerId.embed(enrollmentPcm, 16000);
// gauthamsVoiceprint is a Float32Array of length 192

// 3. During a meeting, embed each speaker turn and cosine-match.
const turnPcm: ArrayBuffer = /* from your recorder */;
const turnEmbedding = await SpeakerId.embed(turnPcm, 16000);

const similarity = SpeakerId.cosine(turnEmbedding, gauthamsVoiceprint);
console.log('cosine:', similarity);
// > 0.65  → almost certainly Gautham
// 0.40-0.65 → probably Gautham (phone / noisy audio)
// < 0.40  → probably not Gautham
```

---

## 📚 API

### `SpeakerId.loadModel(modelPath: string): Promise<void>`

Load the `.pte` model from a filesystem path. Idempotent for the same path. Loading a different path unloads the previous model first. Throws if the file doesn't exist.

Force-loads the `forward` method during initialization so the first inference doesn't pay compilation cost (~200 ms one-time hit on iOS).

### `SpeakerId.isLoaded: boolean`

`true` after `loadModel()` has resolved and before `unloadModel()` is called.

### `SpeakerId.unloadModel(): void`

Release the model handle and free runtime memory. Call at logout or when backgrounding. Idempotent.

### `SpeakerId.embed(pcm: ArrayBuffer, sampleRate?: number): Promise<Float32Array>`

Embed a single PCM audio buffer.

- `pcm` — signed 16-bit little-endian samples, mono
- `sampleRate` — defaults to 16000; other rates are linearly resampled
- Returns a 192-element `Float32Array`, L2-normalized

The model expects a **3-second window (48000 samples at 16 kHz)**. Shorter buffers are zero-padded at the end; longer buffers are center-cropped. This matches how the graph was exported.

### `SpeakerId.cosine(a, b): number`

Cosine similarity between two L2-normalized embeddings. Range `[-1, 1]`. Both arguments can be `Float32Array` or raw `ArrayBuffer`.

---

## 🎯 Thresholds

Cosine scores on L2-normalized ECAPA embeddings typically land in these ranges:

- **Same speaker, clean audio**: 0.65 – 0.85
- **Same speaker, phone / noisy audio**: 0.40 – 0.65
- **Different speakers**: -0.05 – 0.35

Starting threshold: **0.45**. Log real scores from your first week of production and tune from there — the correct number depends on your microphone, your users, and your acoustic environment.

The model exporter verified same-voice pairs consistently score **0.65+ higher** than different-voice pairs, so any threshold in the 0.35 – 0.55 range should cleanly separate matches from non-matches in most conditions.

---

## 🎁 Getting the Model

Two `.pte` files, one per platform. Both are ExecuTorch exports of SpeechBrain's ECAPA-TDNN with the mel-spec pipeline baked into the graph — meaning cross-device and cross-platform cosine parity is guaranteed by construction.

### Option 1: Download from HuggingFace (recommended)

The pre-converted files are hosted at [mlboydaisuke/ECAPA-TDNN-Speaker-ExecuTorch](https://huggingface.co/mlboydaisuke/ECAPA-TDNN-Speaker-ExecuTorch):

- iOS: [`speaker_ecapa_coreml_all.pte`](https://huggingface.co/mlboydaisuke/ECAPA-TDNN-Speaker-ExecuTorch/blob/main/speaker_ecapa_coreml_all.pte) (~42 MB)
- Android: [`speaker_ecapa_xnnpack_fp32.pte`](https://huggingface.co/mlboydaisuke/ECAPA-TDNN-Speaker-ExecuTorch/blob/main/speaker_ecapa_xnnpack_fp32.pte) (~84 MB)

Verified against cloud PyTorch SpeechBrain:

- Core ML fp16: cosine **0.9985**
- XNNPACK fp32: cosine **1.0000**

Bundle in your app or host on your own CDN.

### Option 2: Download from my personal CDN (fastest, for testing)

- iOS: `https://ml-models-bucket.gauthamvijay.com/ecapa-speaker-ios.pte`
- Android: `https://ml-models-bucket.gauthamvijay.com/ecapa-speaker-android.pte`

⚠️ **Personal hosting, may disappear.** For production, use Option 1 and host yourself.

### Option 3: Convert yourself

If you need to modify the export (e.g., different quantization, different input duration), see the conversion scripts in [executorch-models](https://github.com/john-rocky/executorch-models).

---

## 🧩 Supported Platforms

| Platform             | Status              |
| -------------------- | ------------------- |
| **iOS**              | ✅ Fully Supported  |
| **Android**          | ✅ Fully Supported  |
| **iOS Simulator**    | ✅ Works            |
| **Android Emulator** | ✅ Works (CPU only) |

### iOS Requirements

- **Minimum iOS**: 17.0 (ExecuTorch runtime requirement)
- **ExecuTorch**: Swift Package (`swiftpm-1.3.0` or later)
- No additional native dependencies

### Android Requirements

- **Minimum SDK**: API 23 (Android 6.0)
- **ExecuTorch**: `org.pytorch:executorch-android:0.6.0`
- **16 KB page alignment**: automatic (ExecuTorch ships aligned .so files)

---

## 🆚 Comparison

|                         | Cloud ECAPA | ONNX Runtime    | tensorflow-lite | LiteRT          | **This library**         |
| ----------------------- | ----------- | --------------- | --------------- | --------------- | ------------------------ |
| Latency per turn        | 150-400 ms  | 30-80 ms        | 30-80 ms        | 20-100 ms       | **2-140 ms**             |
| Cost per meeting        | ~$0.008     | 0               | 0               | 0               | 0                        |
| Bandwidth per meeting   | ~14 MB      | 0               | 0               | 0               | 0                        |
| Works offline           | No          | Yes             | Yes             | Yes             | Yes                      |
| Voice data leaves phone | Yes         | No              | No              | No              | No                       |
| Feature extraction      | Python DSP  | External        | External        | External        | **In the graph**         |
| Cloud/device parity     | N/A         | ⚠️ Manual match | ⚠️ Manual match | ⚠️ Manual match | ✅ **Guaranteed 0.998+** |
| 16 KB page-aligned      | N/A         | ❌ (2026)       | ❌ (2026)       | ✅              | ✅                       |
| Vendor dependency       | Backend     | Microsoft       | Google (legacy) | Google          | PyTorch (ExecuTorch)     |

---

## 🤝 Pairs With

- [react-native-nitro-audio-anvil](https://github.com/Gautham495/react-native-nitro-audio-anvil) — corruption-proof audio recording. Its `SpeakerWindow.buffer` feeds directly into `embed()`. Together they enable live speaker-labeled transcription entirely on-device.
- [react-native-nitro-cloud-uploader](https://github.com/Gautham495/react-native-nitro-cloud-uploader) — for when the recording ends and you want to ship the audio artifact to S3-compatible storage for archival or async processing.

---

## 🙏 Credits

- **Model architecture**: [ECAPA-TDNN](https://arxiv.org/abs/2005.07143) — Desplanques, Thienpondt, Demuynck (Interspeech 2020)
- **Model weights**: [speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) — trained on VoxCeleb 1+2, Apache 2.0
- **ExecuTorch export**: [mlboydaisuke/ECAPA-TDNN-Speaker-ExecuTorch](https://huggingface.co/mlboydaisuke/ECAPA-TDNN-Speaker-ExecuTorch) — mel-spec baked into graph via STFT-as-convolutions, verified 0.9985 cosine vs eager PyTorch. Apache 2.0.
- **Runtime**: [ExecuTorch](https://pytorch.org/executorch) — PyTorch's official on-device runtime

Special thanks to the model exporter for the meticulous work of getting SpeechBrain's Fbank pipeline into a Core ML / XNNPACK-compatible graph. It saved weeks of debugging.

---

## 🤝 Contributing

Contributions are welcome!

- [Development Workflow](CONTRIBUTING.md#development-workflow)
- [Sending a Pull Request](CONTRIBUTING.md#sending-a-pull-request)
- [Code of Conduct](CODE_OF_CONDUCT.md)

---

## 🪪 License

MIT © [**Gautham Vijayan**](https://gauthamvijay.com)

Built for [SHINE](https://shineai.io), released for everyone.

Note that model files (`.pte`) are distributed separately under Apache 2.0 from SpeechBrain and the ExecuTorch conversion.

---

Made with ❤️ and [**Nitro Modules**](https://nitro.margelo.com)
