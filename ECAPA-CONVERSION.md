# Model Conversion Guide

How to convert SpeechBrain's ECAPA-TDNN speaker verification model into the Core ML and TFLite files that `react-native-nitro-speaker-id` loads on device.

---

## What this produces

Two model files, one per platform:

- `ecapa-body-192.mlpackage` — iOS Core ML, FP16 quantized, **~14 MB**
- `ecapa-body-192.tflite` — Android TFLite/LiteRT, unquantized FP32, **~83 MB**

Both take an **80-dimensional mel-spectrogram** as input and output a **192-dimensional L2-normalized speaker embedding**. Cosine similarity between two embeddings tells you whether they're the same speaker.

The `-body` suffix reflects the architecture: only the neural network body is exported. Feature extraction (audio → mel-spectrogram) happens in native code on each platform (Apple's `vDSP` on iOS, JTransforms FFT on Android).

---

## Why is the Android file so much bigger?

Real talk: **FP16 quantization for `litert-torch` output is currently broken.**

- Google's own [`litert-torch` issue #875](https://github.com/google-ai-edge/litert-torch/issues/875) is open and unresolved. Setting `model.half()` before conversion fails with op legalization errors. Passing `_ai_edge_converter_flags` with FP16 settings produces mixed FP32/INT8 output instead of clean FP16.
- Re-quantizing the exported `.tflite` afterwards doesn't work either — `tf.lite.TFLiteConverter` needs a TF SavedModel as input, not a `.tflite` file. The intermediate SavedModel that `litert-torch` uses internally isn't exposed as a stable API.
- INT8 quantization via PT2E works, but requires a **representative dataset** of 100+ real audio samples for calibration, and can regress the speaker embedding accuracy in ways that only show up in production.

So the library ships FP32 TFLite at ~83 MB. It's annoying on cellular downloads but it works reliably on every device, and matches the cloud SpeechBrain model exactly.

**If you need smaller on Android**: the practical fix is to bundle the model in your app instead of downloading it. ~83 MB inside an AAB doesn't hurt install size much (Play compresses on transport), and users only pay for it once. The example app demonstrates both a bundled path and a CDN download path.

---

## Why the split architecture

The full ECAPA pipeline is `audio → STFT → mel-spectrogram → ECAPA body → embedding`. Exporting the full pipeline **doesn't work** — every attempt (Core ML, ONNX, TFLite) dies on the STFT step with `RuntimeError: view_as_real is only supported for complex tensors` or similar. PyTorch's STFT produces complex tensors, and none of these mobile runtimes handle complex tensors natively.

Instead, the library ships:

1. **A neural-body-only model** that takes a mel-spectrogram (not raw audio) as input — exports cleanly to Core ML and TFLite because there's no STFT in the graph.
2. **Native mel-spectrogram code** on each platform — Apple's Accelerate/vDSP on iOS, JTransforms FFT on Android — which computes the mel-spec from raw PCM in 2-5 ms.

This is exactly how Whisper on-device, MediaPipe, and every production on-device speech system works: feature extraction runs natively, only the learned model runs in the ML runtime.

Invisible to you as a user of the library — you pass raw PCM to `SpeakerId.embed()`, and get back a 192-d vector. The mel-spec happens under the hood.

---

## Why two files (not one universal runtime)

Cross-platform runtimes like ONNX Runtime work, but come with real costs:

- ~30-40 MB install-size hit per platform
- 16 KB page-alignment problems on Android 15+ (Microsoft still hasn't fixed this in ONNX Runtime as of 2026)
- Slower than native runtimes because they translate ops instead of executing them directly
- Extra vendor dependency between you and shipping

Native platform runtimes solve all of that. **Core ML** on iOS is maintained by Apple, ships with the OS, and gets Neural Engine acceleration on A12+ chips for free. **LiteRT** on Android (Google's replacement for the deprecated `tensorflow-lite` package) is 16 KB clean by design and ships as a tiny AAR. Both run the model at max hardware speed.

The tradeoff: two conversion pipelines instead of one. This script handles both.

---

## Fastest path: Google Colab

**Recommended for most people.** Free, no local setup, one-click download.

1. Open [colab.research.google.com](https://colab.research.google.com), create a new notebook.
2. Runtime → **Change runtime type → Python 3.11** (SpeechBrain and `litert-torch` don't fully support 3.13 as of 2026).
3. Runtime → **Restart session** (Colab needs to reload after the version change).
4. Paste this into a cell and run:

```python
!pip install -q speechbrain coremltools litert-torch tensorflow
```

5. Wait for `Successfully installed …`. Then paste the entire contents of `ecapa-conversion.py` (below or in the repo) into a new cell and run it.
6. Both files download automatically to your Mac's Downloads folder.

Total time: ~5 minutes on Colab's default runtime, including the SpeechBrain download.

---

## Local setup (Python 3.11)

```bash
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install speechbrain coremltools litert-torch tensorflow torch

python ecapa-conversion.py
```

Output files land in the current directory.

---

## What the script does, step by step

1. **Load SpeechBrain's ECAPA-TDNN checkpoint** — the same model behind [`speechbrain/spkrec-ecapa-voxceleb`](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) on Hugging Face. Pretrained speaker verification model with 0.69% EER on VoxCeleb1.

2. **Extract only `mods.embedding_model`** — the ECAPA neural body. Wrap it in a small module that mean-pools across time frames and L2-normalizes.

3. **Trace the wrapped model** with `torch.jit.trace` on a dummy mel-spectrogram input `(1, 150, 80)`.

4. **Convert to Core ML** with `coremltools.convert()`. Uses the newer ML Program format (`convert_to='mlprogram'`) required for Neural Engine dispatch on A12+. FP16 quantization halves the Core ML file size with no measurable accuracy loss on ECAPA.

5. **Convert to TFLite** with `litert_torch.convert()`. Unquantized FP32 — see the section above for why.

6. **Sanity check** both exports against the reference PyTorch model. Cosine similarity should be > 0.99 (essentially identical outputs).

7. **Zip the `.mlpackage`** for HTTP transfer. Core ML's `.mlpackage` is a directory, not a single file, so it can't be served over plain HTTP. The zip wrapper is what gets uploaded to your CDN.

---

## Expected sanity-check output

```
Sanity check…
  ✓ Core ML: cosine=0.9999 (should be > 0.99)
  ✓ TFLite:  cosine=1.0000 (should be > 0.99)
```

If cosine is < 0.99 on Core ML, drop FP16 quantization and try again (see "Common issues"). If cosine is < 0.99 on TFLite, that's unusual — file an issue.

---

## Common issues

### `ModuleNotFoundError: No module named 'speechbrain'`

You're on Python 3.13 (Colab's default). Switch to 3.11 via Runtime → Change runtime type, then restart the session. `litert-torch` and some SpeechBrain dependencies aren't 3.13-compatible as of 2026.

### `RuntimeError: view_as_real is only supported for complex tensors`

You're trying to export the **full** SpeechBrain model, not just `mods.embedding_model`. This is exactly the STFT problem this script exists to avoid — make sure you're extracting the embedding body only, matching what `ecapa-conversion.py` does.

### Core ML cosine < 0.99 after export

FP16 quantization occasionally shifts activations enough to matter. Remove:

```python
compute_precision=ct.precision.FLOAT16,
```

File goes from ~14 MB to ~28 MB but restores full precision.

### `AttributeError: module 'litert_torch' has no attribute 'convert'`

Google renamed `ai-edge-torch` to `litert-torch`. If you have the old package installed:

```bash
pip uninstall ai-edge-torch
pip install litert-torch
```

### `NoClassDefFoundError` for JTransforms at runtime (Android)

The library uses JTransforms for the mel-spectrogram FFT on Android. Make sure your app's `android/build.gradle` includes the JitPack repository:

```gradle
allprojects {
    repositories {
        google()
        mavenCentral()
        maven { url 'https://jitpack.io' }
    }
}
```

---

## Bundling the models in your React Native app

**iOS**: drag `ecapa-body-192.mlpackage` into your Xcode project's main target. Check "Copy items if needed". Access at runtime:

```ts
const modelPath = `${ReactNativeBlobUtil.fs.dirs.MainBundleDir}/ecapa-body-192.mlpackage`;
await SpeakerId.loadModel(modelPath);
```

**Android**: put `ecapa-body-192.tflite` in `android/app/src/main/assets/`. Copy to app storage on first launch:

```ts
const modelPath = `${ReactNativeBlobUtil.fs.dirs.DocumentDir}/ecapa-body-192.tflite`;
await ReactNativeBlobUtil.fs.cp(
  'bundle-assets://ecapa-body-192.tflite',
  modelPath
);
await SpeakerId.loadModel(modelPath);
```

**Or download at runtime** — host on your own CDN, fetch on first launch. iOS requires unzipping the `.mlpackage.zip` after download (see the example app). Android's `.tflite` is a single file, no unpacking needed.

**Since Android's `.tflite` is ~83 MB**, bundling in-app is usually the better tradeoff — the file compresses somewhat inside an AAB, and users don't pay a first-launch download cost. Do CDN downloads if you specifically need to keep your APK small at any cost.

---

## Matching cloud voiceprints

If your backend already stores voiceprints computed with the SpeechBrain Python model, these on-device embeddings will cosine-match them at the same thresholds. Same weights, same architecture, same L2-normalization. Only the front-end changes — cloud uses SpeechBrain's Python mel-spec, device uses native platform mel-spec, and both produce feature-equivalent 80-dim mel-spectrograms.

Expect cosine similarities of 0.95+ between an on-device embedding and a cloud embedding of the same audio. Lower than that means the native mel-spec has drifted from SpeechBrain's Fbank defaults — check `n_fft`, `hop_length`, `n_mels`, and the log formula in the native Swift/Kotlin code.

---

## Credits

- **Model**: [speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) — Apache 2.0
- **ECAPA-TDNN paper**: Desplanques et al., Interspeech 2020
- **Conversion pipeline**: SpeechBrain + `coremltools` (Apple) + `litert-torch` (Google)
- **Native mel-spectrogram**: Apple `vDSP` (iOS), [JTransforms](https://github.com/wendykierp/JTransforms) (Android)
