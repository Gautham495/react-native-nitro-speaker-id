# Model Conversion Guide

How to convert SpeechBrain's ECAPA-TDNN speaker verification model to Core ML and TFLite for use with `react-native-nitro-speaker-id`.

## What this produces

Two model files, one per platform:

- `ecapa-body-192.mlpackage` — iOS, Core ML, ~14 MB (FP16 quantized)
- `ecapa-body-192.tflite` — Android, TFLite, ~20 MB (unquantized)

Both take an 80-dim mel-spectrogram as input, output a 192-d L2-normalized speaker embedding. Cosine similarity between two embeddings tells you whether they're the same speaker.

## Why two files (and not one universal model)

Cross-platform runtimes like ONNX Runtime work but come with real costs:

- 30-40 MB install-size hit per platform
- 16 KB page-alignment problems on Android 15+ (Microsoft still hasn't fixed this in ORT as of 2026)
- Slower than native runtimes because they translate ops instead of running them directly
- Extra vendor dependency between you and shipping

Native platform runtimes (Core ML on iOS, TensorFlow Lite on Android) solve all of that. They're maintained by Apple and Google respectively, ship with the OS or as tiny AAR/framework additions, run models at max hardware speed (Neural Engine, NNAPI), and don't fight page alignment because they're 16 KB clean by design.

The tradeoff: two conversion pipelines instead of one. This script handles both.

## Why the mel-spectrogram runs in native code, not the model

Full end-to-end audio → embedding conversion **would fail**. Here's why:

The ECAPA pipeline is `audio → STFT → mel-spectrogram → ECAPA body → embedding`. PyTorch's STFT operation outputs **complex tensors**, and ONNX/Core ML/TFLite exporters don't support them. So every attempt to export the full model dies with `RuntimeError: view_as_real is only supported for complex tensors` or similar.

The fix is architectural: **export only the ECAPA body** (which takes a mel-spectrogram as input, no STFT involved), and **compute the mel-spectrogram in native code on the phone**. Native mel-spec is ~150 lines of standard signal processing that runs in 2-5 ms on modern hardware. This is exactly how Whisper on-device, MediaPipe, and every production on-device speech system works — feature extraction is native, only the learned model runs in the ML runtime.

`convert-ecapa.py` does step 1 (export the ECAPA body). The Swift and Kotlin sides of `react-native-nitro-speaker-id` handle step 2 (native mel-spectrogram matching SpeechBrain's Fbank defaults).

## Fastest path: Google Colab

**Recommended for most people.** Free, no local setup, one-click download.

1. Open [colab.research.google.com](https://colab.research.google.com), create a new notebook
2. Runtime → **Change runtime type → Python 3.11** (SpeechBrain doesn't fully support 3.13 as of 2026)
3. Runtime → Restart session (Colab needs to reload after the version change)
4. Paste this into a single cell:

```python
!pip install -q speechbrain coremltools litert-torch tensorflow
```

5. Wait for `Successfully installed …`. Then in a new cell paste the entire contents of `convert-ecapa.py` and run.
6. Both files download automatically to your Mac's Downloads folder.

Total time: ~5 minutes on Colab's default runtime, including the SpeechBrain download.

## Local setup (Python 3.11)

```bash
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install speechbrain coremltools litert-torch tensorflow torch

python convert-ecapa.py
```

Output files land in the current directory.

## What the script does, step by step

1. **Load SpeechBrain's ECAPA-TDNN checkpoint** — the same model backing `speechbrain/spkrec-ecapa-voxceleb` on Hugging Face. This is the pretrained speaker verification model with a state-of-the-art 0.69% EER on VoxCeleb1.

2. **Extract only `mods.embedding_model`** — the ECAPA body. Wrap it in a small module that adds mean-pooling across time frames and L2 normalization, so the exported model outputs cosine-ready embeddings.

3. **Trace the wrapped model** with `torch.jit.trace` on a dummy mel-spectrogram input `(1, 150, 80)` — batch of one, 150 time frames (~1.5 s of audio at 10 ms hop), 80 mel bins.

4. **Convert to Core ML** with `coremltools.convert()`. Uses the newer ML Program format (`convert_to='mlprogram'`) required for Neural Engine dispatch on A12+ devices. FP16 quantization halves the file size with no measurable accuracy loss on ECAPA.

5. **Convert to TFLite** with `litert_torch.convert()` — Google's official PyTorch → TFLite pipeline (formerly `ai-edge-torch`).

6. **Sanity check both outputs** — runs the reference PyTorch model and each exported model on the same dummy input, computes relative difference and cosine similarity. Cosine should be > 0.99 (essentially identical). If it's not, drop FP16 quantization (see notes below).

## Sanity check output should look like

```
Sanity check…
  Core ML: rel_diff=0.0032 (should be < 0.05), cosine=0.9999 (should be > 0.99)
  TFLite:  rel_diff=0.0018 (should be < 0.05), cosine=1.0000 (should be > 0.99)
```

If numbers are off, see the "Common issues" section below.

## Common issues

### `ModuleNotFoundError: No module named 'speechbrain'`

You're on Python 3.13 in Colab (the default). Follow the Colab steps above to switch to Python 3.11. Restart the session after switching or the environment stays cached.

### `RuntimeError: view_as_real is only supported for complex tensors`

You're trying to export the **full** SpeechBrain model, not just `mods.embedding_model`. This is exactly the STFT problem this script exists to avoid. Make sure you're extracting the embedding body only, matching what `convert-ecapa.py` does.

### Cosine < 0.99 after export

FP16 quantization occasionally shifts things enough to matter. Fix — drop it:

For Core ML, delete this line:

```python
compute_precision=ct.precision.FLOAT16,
```

For TFLite, add explicit unquantized mode by not passing an `optimizations` list.

Doubles the model size (Core ML goes from ~14 MB to ~28 MB) but restores full precision.

### `AttributeError: module 'litert_torch' has no attribute 'convert'`

Google renamed `ai-edge-torch` to `litert-torch`. If you have the old package installed:

```bash
pip uninstall ai-edge-torch
pip install litert-torch
```

## Adding INT8 quantization for smaller TFLite

The Colab conversion produces an unquantized TFLite (~20 MB). If you want it smaller, add this after the `litert_torch.convert` call:

```python
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model('...')  # path to intermediate
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]  # or full INT8 with a representative dataset
tflite = converter.convert()
```

Full INT8 quantization requires a **representative dataset** (100+ real audio samples) to calibrate the activation ranges. If you don't have real audio handy, stick with FP16 quantization (2x smaller than FP32, no accuracy loss on ECAPA).

## Bundling the models in your React Native app

**iOS**: drag `ecapa-body-192.mlpackage` into your Xcode project's main target. Check "Copy items if needed". Access at runtime via `NSBundle.mainBundle().pathForResource("ecapa-body-192", ofType: "mlpackage")` — or from JS via `ReactNativeBlobUtil.fs.dirs.MainBundleDir + '/ecapa-body-192.mlpackage'`.

**Android**: put `ecapa-body-192.tflite` in `android/app/src/main/assets/`. Copy to app storage on first launch via `ReactNativeBlobUtil.fs.cp('bundle-assets://ecapa-body-192.tflite', destPath)`.

**Or CDN it** — host the files on Cloudflare R2, S3, or wherever, and download on first launch. ~20 MB one-time cost, cache forever.

## Matching cloud voiceprints

If your backend already stores voiceprints computed with the SpeechBrain cloud model, these on-device embeddings will cosine-match them at the same thresholds. Same weights, same architecture, same L2-normalization. Only the front-end changes — cloud uses SpeechBrain's Python mel-spec, device uses native platform mel-spec, and both produce feature-equivalent 80-dim mel-spectrograms.

Expect cosine similarities of 0.95+ between an on-device embedding and a cloud embedding of the same audio. Any lower means the native mel-spec has drifted from SpeechBrain's Fbank defaults — check `n_fft`, `hop_length`, `n_mels`, and the log mode (`log(1 + 10*mel)`) in the native code.

## Credits

Model: [speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) — Apache 2.0

ECAPA-TDNN paper: Desplanques et al., Interspeech 2020

Conversion pipeline: SpeechBrain + coremltools (Apple) + litert-torch (Google)
