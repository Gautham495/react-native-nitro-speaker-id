"""
Convert SpeechBrain's ECAPA-TDNN speaker verification model to Core ML and TFLite
for use with `react-native-nitro-speaker-id`.

Output:
    ecapa-body-192.mlpackage       (iOS Core ML, ~14 MB with FP16)
    ecapa-body-192.mlpackage.zip   (zipped for CDN/HTTP transfer)
    ecapa-body-192.tflite          (Android LiteRT, ~83 MB, unquantized)

Both models take an 80-dim mel-spectrogram as input, output a 192-d
L2-normalized speaker embedding. Audio → mel-spectrogram is done in native
code on each platform (Apple vDSP on iOS, JTransforms on Android).

Sizes as of Sep 2026:
    - Core ML uses FP16 by default — comes out ~14 MB. Nice.
    - TFLite via litert-torch has NO working FP16 path (Google's own
      issue #875 in litert-torch is open, unresolved). INT8 requires a
      representative dataset and can regress speaker embedding accuracy.
      So Android ships FP32 at ~83 MB. Not great, but shippable.

Usage:
    In Google Colab (recommended — no local setup):
        Runtime → Change runtime type → Python 3.11
        Runtime → Restart session
        Paste this script, run.

    Locally:
        pip install speechbrain coremltools litert-torch torch
        python ecapa-conversion.py
"""

import os
import shutil

import torch
from speechbrain.inference.speaker import EncoderClassifier


# ─── 1. Load SpeechBrain's pretrained ECAPA ─────────────────────────────

print("Loading SpeechBrain ECAPA-TDNN checkpoint…")
classifier = EncoderClassifier.from_hparams(
    source='speechbrain/spkrec-ecapa-voxceleb',
    savedir='/tmp/spkrec-ecapa-voxceleb',
    run_opts={'device': 'cpu'},
)


# ─── 2. Extract only the ECAPA body (skip STFT + mel-spec) ────────────
#
# Full SpeechBrain pipeline: audio → STFT → mel → ECAPA body → embedding.
# STFT outputs complex tensors, which no mobile ML runtime supports natively.
# We export only the ECAPA body (which takes a mel-spectrogram as input),
# and compute the mel-spec in native code on each platform. Same pattern
# Whisper on-device and MediaPipe use.

ecapa_body = classifier.mods.embedding_model


class BodyOnly(torch.nn.Module):
    """
    Wraps the ECAPA body with mean pooling + L2 normalization so the exported
    model's output is directly cosine-comparable — no post-processing needed.

    Input:  mel-spectrogram [batch, time_frames, 80]
    Output: embedding       [batch, 192], L2-normalized
    """

    def __init__(self, body):
        super().__init__()
        self.body = body

    def forward(self, mel):
        emb = self.body(mel)          # [batch, time, 192]
        emb = emb.mean(dim=1)         # collapse time dimension
        norm = torch.linalg.norm(emb, dim=-1, keepdim=True) + 1e-9
        return emb / norm


model = BodyOnly(ecapa_body).eval()
# Dummy: batch of one, 150 time frames (~1.5 s @ 10 ms hop), 80 mel bins.
dummy_mel = torch.randn(1, 150, 80)


# ─── 3. Export to Core ML (.mlpackage), FP16 ───────────────────────────

print("\nExporting to Core ML…")
try:
    import coremltools as ct

    traced = torch.jit.trace(model, dummy_mel)
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name='mel_spectrogram',
                shape=(1, ct.RangeDim(50, 3000), 80),
            )
        ],
        outputs=[ct.TensorType(name='embedding')],
        convert_to='mlprogram',
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS16,
    )
    mlmodel.save('ecapa-body-192.mlpackage')
    print("✓ Core ML: ecapa-body-192.mlpackage")
except ImportError:
    print("✗ Skipping Core ML — install `coremltools`")
except Exception as e:
    print(f"✗ Core ML export failed: {e}")


# ─── 4. Export to TFLite (.tflite), unquantized ────────────────────────
#
# NOTE: FP16 quantization for litert-torch output is broken as of 2026.
# See github.com/google-ai-edge/litert-torch/issues/875.
# We ship the unquantized ~83 MB file. Cellular download is annoying but
# it works reliably on every device.

print("\nExporting to TFLite…")
try:
    import litert_torch
    edge_model = litert_torch.convert(model, (dummy_mel,))
    edge_model.export('ecapa-body-192.tflite')
    print("✓ TFLite: ecapa-body-192.tflite")
except ImportError:
    print("✗ Skipping TFLite — install `litert-torch`")
except Exception as e:
    print(f"✗ TFLite export failed: {e}")


# ─── 5. Sanity check both exports ──────────────────────────────────────

print("\nSanity check against reference PyTorch…")
import numpy as np

with torch.no_grad():
    reference = model(dummy_mel).numpy().flatten()

if os.path.exists('ecapa-body-192.mlpackage'):
    try:
        import coremltools as ct
        ml = ct.models.MLModel('ecapa-body-192.mlpackage')
        out = ml.predict({'mel_spectrogram': dummy_mel.numpy()})['embedding'].flatten()
        cosine = float(np.dot(out, reference))
        marker = "✓" if cosine > 0.99 else "⚠"
        print(f"  {marker} Core ML: cosine={cosine:.4f} (should be > 0.99)")
    except Exception as e:
        print(f"  ✗ Core ML sanity check failed: {e}")

if os.path.exists('ecapa-body-192.tflite'):
    try:
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path='ecapa-body-192.tflite')
        input_details = interp.get_input_details()
        interp.resize_tensor_input(input_details[0]['index'], [1, 150, 80])
        interp.allocate_tensors()
        interp.set_tensor(input_details[0]['index'], dummy_mel.numpy().astype(np.float32))
        interp.invoke()
        out = interp.get_tensor(interp.get_output_details()[0]['index']).flatten()
        cosine = float(np.dot(out, reference))
        marker = "✓" if cosine > 0.99 else "⚠"
        print(f"  {marker} TFLite:  cosine={cosine:.4f} (should be > 0.99)")
    except ImportError:
        print("  ↷ TFLite sanity check skipped — install `tensorflow`")
    except Exception as e:
        print(f"  ✗ TFLite sanity check failed: {e}")


# ─── 6. Zip the .mlpackage for HTTP transfer ───────────────────────────
#
# Core ML's .mlpackage is a directory, not a file. HTTP servers can't
# serve directories, so ship a zip that the client unpacks after download.

if os.path.exists('ecapa-body-192.mlpackage'):
    print("\nZipping .mlpackage for CDN upload…")
    if os.path.exists('ecapa-body-192.mlpackage.zip'):
        os.remove('ecapa-body-192.mlpackage.zip')
    shutil.make_archive('ecapa-body-192.mlpackage', 'zip', '.', 'ecapa-body-192.mlpackage')
    print(f"✓ ecapa-body-192.mlpackage.zip")


# ─── 7. Report ─────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("FILES")
print("=" * 60)
for path in ['ecapa-body-192.mlpackage', 'ecapa-body-192.mlpackage.zip',
             'ecapa-body-192.tflite']:
    if os.path.exists(path):
        if os.path.isdir(path):
            size = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, _, fs in os.walk(path) for f in fs
            )
            print(f"  {path:40s} {size / 1e6:>7.1f} MB (directory)")
        else:
            print(f"  {path:40s} {os.path.getsize(path) / 1e6:>7.1f} MB")


# ─── 8. Colab download ─────────────────────────────────────────────────

try:
    from google.colab import files
    print("\nDownloading to your machine…")
    for path in ['ecapa-body-192.mlpackage.zip', 'ecapa-body-192.tflite']:
        if os.path.exists(path):
            files.download(path)
except ImportError:
    print(f"\nFiles are in: {os.getcwd()}")
    print("Upload the .mlpackage.zip and .tflite to your CDN or bundle in your app.")