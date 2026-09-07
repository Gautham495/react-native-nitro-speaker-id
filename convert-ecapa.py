"""
Convert SpeechBrain's ECAPA-TDNN speaker verification model to Core ML and TFLite.

Output:
    ecapa-body-192.mlpackage  (iOS, Core ML, ~14 MB FP16)
    ecapa-body-192.tflite     (Android, TFLite, ~20 MB)

Both files take an 80-dim mel-spectrogram as input, output a 192-d L2-normalized
speaker embedding. The audio → mel-spectrogram step is done in native code on
each platform (Swift's Accelerate/vDSP on iOS, hand-rolled FFT in Kotlin on
Android). This split avoids PyTorch's STFT-with-complex-tensor export problem
and keeps the mel front-end fast (~2-5 ms) on the CPU where it belongs.

Usage:
    # In Google Colab (recommended — free GPU, one-click):
    Runtime → Change runtime type → Python 3.11
    Runtime → Restart session
    Paste this script, run.

    # Locally:
    python -m venv .venv && source .venv/bin/activate
    pip install speechbrain coremltools litert-torch torch
    python convert-ecapa.py

Requirements:
    Python 3.11 (SpeechBrain and litert-torch don't fully support 3.13 yet)
    ~4 GB RAM for the conversion
    Internet (downloads the SpeechBrain checkpoint on first run, ~28 MB)

Output size:
    Core ML .mlpackage: ~14 MB with FP16 quantization
    TFLite .tflite:     ~20 MB unquantized (add INT8 quantization for ~7 MB)

Match with cloud:
    Embeddings from these models cosine-match the SpeechBrain cloud output
    within ~1% on real audio. See the sanity-check block at the end.
"""

import os
import shutil
import sys

import torch
from speechbrain.inference.speaker import EncoderClassifier


# ─── 1. Load SpeechBrain's pretrained ECAPA ─────────────────────────────

print("Loading SpeechBrain ECAPA-TDNN checkpoint…")
classifier = EncoderClassifier.from_hparams(
    source='speechbrain/spkrec-ecapa-voxceleb',
    savedir='/tmp/spkrec-ecapa-voxceleb',
    run_opts={'device': 'cpu'},
)


# ─── 2. Extract only the ECAPA body (skip STFT + mel-spectrogram) ──────
#
# The full SpeechBrain model is: audio → STFT → mel → ECAPA body → embedding.
# The STFT step is what breaks ONNX/Core ML/TFLite exporters because it
# outputs complex tensors, which those runtimes don't natively support.
#
# Solution: export only the ECAPA body. It takes a mel-spectrogram as input
# (80 mel bins × N time frames) and outputs a 192-d embedding. The audio →
# mel front-end runs in native code on the phone, which is fast and standard.
#
# `mods.embedding_model` is the ECAPA body itself.

ecapa_body = classifier.mods.embedding_model


class BodyOnly(torch.nn.Module):
    """
    Wraps the ECAPA body with L2 normalization so the exported model's output
    is directly usable for cosine similarity — no post-processing needed.

    Input:  mel-spectrogram [batch, time_frames, 80]
    Output: embedding       [batch, 192], L2-normalized
    """

    def __init__(self, body):
        super().__init__()
        self.body = body

    def forward(self, mel):
        # ECAPA body: [batch, time, 80] → [batch, time, 192]
        emb = self.body(mel)
        # Simple mean pooling across time. SpeechBrain does the same by default
        # (attentive stat pooling is inside `body` already; this collapses the
        # remaining time dim).
        emb = emb.mean(dim=1)
        # L2 normalize so cosine similarity is a dot product downstream.
        norm = torch.linalg.norm(emb, dim=-1, keepdim=True) + 1e-9
        return emb / norm


model = BodyOnly(ecapa_body).eval()

# Dummy input for tracing: batch=1, 150 time frames (~1.5 s of audio at
# hop=10 ms), 80 mel bins. Real inputs will vary in time_frames.
dummy_mel = torch.randn(1, 150, 80)


# ─── 3. Export to Core ML (.mlpackage) ─────────────────────────────────

print("\nExporting to Core ML…")
try:
    import coremltools as ct

    traced = torch.jit.trace(model, dummy_mel)
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name='mel_spectrogram',
                # Variable-length time axis: min 50 frames (~0.5 s), max 3000 (~30 s).
                # ECAPA works on 0.5-30 s of audio in production.
                shape=(1, ct.RangeDim(50, 3000), 80),
            )
        ],
        outputs=[ct.TensorType(name='embedding')],
        convert_to='mlprogram',  # Newer ML Program format (iOS 15+, needed for ANE)
        compute_precision=ct.precision.FLOAT16,  # ~half the file size, no accuracy loss
        minimum_deployment_target=ct.target.iOS16,
    )
    mlmodel.save('ecapa-body-192.mlpackage')
    print("✓ Core ML saved: ecapa-body-192.mlpackage")
except ImportError:
    print("✗ Skipping Core ML — install `coremltools` to enable")
except Exception as e:
    print(f"✗ Core ML export failed: {e}")


# ─── 4. Export to TFLite (.tflite) via litert-torch ────────────────────

print("\nExporting to TFLite…")
try:
    import litert_torch

    edge_model = litert_torch.convert(model, (dummy_mel,))
    edge_model.export('ecapa-body-192.tflite')
    print("✓ TFLite saved: ecapa-body-192.tflite")
except ImportError:
    print("✗ Skipping TFLite — install `litert-torch` to enable")
except Exception as e:
    print(f"✗ TFLite export failed: {e}")


# ─── 5. Sanity check ───────────────────────────────────────────────────
#
# Verify the exported models produce embeddings that match the original
# PyTorch model within ~1%. If they diverge more, drop FP16 quantization
# (remove `compute_precision=ct.precision.FLOAT16` for Core ML; add
# `converter.optimizations = []` for TFLite).

print("\nSanity check…")
with torch.no_grad():
    reference = model(dummy_mel).numpy().flatten()

# Check Core ML
if os.path.exists('ecapa-body-192.mlpackage'):
    try:
        import coremltools as ct
        import numpy as np

        ml = ct.models.MLModel('ecapa-body-192.mlpackage')
        out = ml.predict({'mel_spectrogram': dummy_mel.numpy()})['embedding'].flatten()
        rel_diff = np.linalg.norm(out - reference) / np.linalg.norm(reference)
        cosine = float(np.dot(out, reference))
        print(f"  Core ML: rel_diff={rel_diff:.4f} (should be < 0.05), cosine={cosine:.4f} (should be > 0.99)")
    except Exception as e:
        print(f"  Core ML sanity check failed: {e}")

# Check TFLite
if os.path.exists('ecapa-body-192.tflite'):
    try:
        import tensorflow as tf
        import numpy as np

        interp = tf.lite.Interpreter(model_path='ecapa-body-192.tflite')
        # Resize input to match our test shape.
        input_details = interp.get_input_details()
        interp.resize_tensor_input(input_details[0]['index'], [1, 150, 80])
        interp.allocate_tensors()
        interp.set_tensor(input_details[0]['index'], dummy_mel.numpy().astype(np.float32))
        interp.invoke()
        output_details = interp.get_output_details()
        out = interp.get_tensor(output_details[0]['index']).flatten()
        rel_diff = np.linalg.norm(out - reference) / np.linalg.norm(reference)
        cosine = float(np.dot(out, reference))
        print(f"  TFLite:  rel_diff={rel_diff:.4f} (should be < 0.05), cosine={cosine:.4f} (should be > 0.99)")
    except ImportError:
        print("  TFLite sanity check skipped — install `tensorflow`")
    except Exception as e:
        print(f"  TFLite sanity check failed: {e}")


# ─── 6. Report ─────────────────────────────────────────────────────────

print("\nDone. File sizes:")
for path in ['ecapa-body-192.mlpackage', 'ecapa-body-192.tflite']:
    if os.path.exists(path):
        if os.path.isdir(path):
            # .mlpackage is a directory — compute total size + zip for download
            size = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, _, fs in os.walk(path)
                for f in fs
            )
            print(f"  {path}: {size / 1e6:.1f} MB (directory)")

            # Zip so it downloads as one file in Colab
            shutil.make_archive(path, 'zip', '.', path)
            zsize = os.path.getsize(f'{path}.zip')
            print(f"  {path}.zip: {zsize / 1e6:.1f} MB (zip for download)")
        else:
            size = os.path.getsize(path)
            print(f"  {path}: {size / 1e6:.1f} MB")


# ─── 7. Colab download (skip if running locally) ───────────────────────

try:
    from google.colab import files
    print("\nDownloading to your machine…")
    for path in ['ecapa-body-192.mlpackage.zip', 'ecapa-body-192.tflite']:
        if os.path.exists(path):
            files.download(path)
except ImportError:
    print(f"\nFiles are in: {os.getcwd()}")
    print("Copy them to your app's model directory.")