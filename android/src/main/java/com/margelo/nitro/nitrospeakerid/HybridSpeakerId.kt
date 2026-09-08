package com.margelo.nitro.nitrospeakerid

import android.util.Log
import com.facebook.proguard.annotations.DoNotStrip
import com.margelo.nitro.core.ArrayBuffer
import com.margelo.nitro.core.Promise
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executors
import kotlin.math.sqrt

/**
 * On-device ECAPA-TDNN speaker embedding for Android via ExecuTorch.
 *
 * Uses mlboydaisuke/ECAPA-TDNN-Speaker-ExecuTorch — a repackaging of
 * SpeechBrain's spkrec-ecapa-voxceleb where the ENTIRE feature pipeline
 * (STFT → mel filterbank → per-utterance mean subtraction → ECAPA-TDNN)
 * is baked into the model graph. Verified cosine similarity of 1.000
 * against cloud SpeechBrain's EncoderClassifier.encode_batch on the
 * same audio. Same-voice pairs land 0.65+ above different-voice pairs.
 *
 * This eliminates the entire class of "our mel-spec differs from cloud's"
 * bugs that plagued the hand-rolled pipeline — window function, FFT size,
 * centering, log formula, sentence-mean normalization all now live inside
 * the .pte where nothing external can drift them.
 *
 * Model input:  Float32 tensor [1, 48000] — raw 16 kHz mono PCM in [-1, 1],
 *                                            exactly 3.0 seconds
 * Model output: Float32 tensor [1, 192]   — L2-normalized ECAPA embedding
 *
 * Backend: XNNPACK fp32 (CPU-only). ExecuTorch's XNNPACK backend is the
 * most portable option — runs on every Android arm64 device without a
 * device allowlist, and matches the cloud output at cosine 1.0.
 *
 * The 3-second window is fixed — shorter audio is zero-padded, longer is
 * center-cropped. This matches how the model was exported and what its
 * per-utterance mean subtraction expects to see.
 */
@DoNotStrip
class HybridSpeakerId : HybridSpeakerIdSpec() {

    companion object {
        // Fixed input contract from the exported .pte model.
        private const val TARGET_SAMPLE_RATE = 16000
        private const val TARGET_SAMPLE_COUNT = 48000  // 3 seconds at 16 kHz
        private const val EMBEDDING_DIM = 192
        private const val TAG = "SpeakerId"
    }

    // Guard concurrent embed() with a single-thread executor. ExecuTorch's
    // Module is documented as not thread-safe; serializing keeps native
    // memory usage predictable under load in busy meetings.
    private val sessionExecutor = Executors.newSingleThreadExecutor()

    private var module: Module? = null
    private var currentPath: String? = null

    override val isLoaded: Boolean
        get() = module != null

    override fun loadModel(modelPath: String): Promise<Unit> {
        return Promise.async {
            synchronized(this) {
                if (currentPath == modelPath && module != null) return@async
                val file = File(modelPath.removePrefix("file://"))
                if (!file.exists()) {
                    throw RuntimeException("Model file not found: ${file.absolutePath}")
                }

                closeModel()

                // Module.load returns a loaded module; XNNPACK backend
                // handles execution on CPU.
                module = Module.load(file.absolutePath)
                currentPath = modelPath
                Log.i(TAG, "ExecuTorch model loaded (XNNPACK): ${file.absolutePath}")
            }
        }
    }

    override fun unloadModel() {
        synchronized(this) {
            closeModel()
        }
    }

    private fun closeModel() {
        module?.destroy()
        module = null
        currentPath = null
    }

    override fun embed(pcm: ArrayBuffer, sampleRate: Double): Promise<ArrayBuffer> {
        // Snapshot PCM to Kotlin floats on the calling thread — the
        // ArrayBuffer's lifetime is scoped to this turn, don't race the executor.
        val floats = pcmToFloats(pcm, sampleRate)

        return Promise.async {
            val currentModule = module
                ?: throw RuntimeException("Model not loaded — call loadModel() first")

            val submit = sessionExecutor.submit<ArrayBuffer> {
                // Model expects exactly 48000 samples. Anvil emits 1.5s
                // windows (24000 samples at 16 kHz), so zero-pad up to 3s.
                val fixedPcm = padOrCropToFixedLength(floats, TARGET_SAMPLE_COUNT)

                // Build [1, 48000] float32 input tensor.
                val inputTensor = Tensor.fromBlob(
                    fixedPcm,
                    longArrayOf(1L, TARGET_SAMPLE_COUNT.toLong())
                )

                // Forward pass. Model does STFT, mel, mean subtraction,
                // ECAPA body internally.
                val outputs = currentModule.forward(EValue.from(inputTensor))
                    ?: throw RuntimeException("Model returned null output")
                if (outputs.isEmpty()) {
                    throw RuntimeException("Model returned no output tensors")
                }

                val outputTensor = outputs[0].toTensor()
                val embedding = outputTensor.dataAsFloatArray

                if (embedding.size != EMBEDDING_DIM) {
                    throw RuntimeException(
                        "Expected $EMBEDDING_DIM-d embedding, got ${embedding.size}"
                    )
                }

                // Model already L2-normalizes internally; re-norm as
                // insurance against fp32 accumulation drift.
                val normalized = l2Normalize(embedding)
                floatArrayToArrayBuffer(normalized)
            }
            submit.get()
        }
    }

    override fun cosine(a: ArrayBuffer, b: ArrayBuffer): Double {
        val aFloats = arrayBufferToFloatArray(a)
        val bFloats = arrayBufferToFloatArray(b)
        val n = minOf(aFloats.size, bFloats.size)
        var dot = 0.0
        for (i in 0 until n) {
            dot += aFloats[i].toDouble() * bFloats[i].toDouble()
        }
        return dot
    }

    // ── PCM helpers ────────────────────────────────────────────────────

    private fun pcmToFloats(pcm: ArrayBuffer, sampleRate: Double): FloatArray {
        val byteBuffer = pcm.getBuffer(false).order(ByteOrder.LITTLE_ENDIAN)
        val sampleCount = byteBuffer.remaining() / 2
        val samples = FloatArray(sampleCount)
        for (i in 0 until sampleCount) {
            samples[i] = byteBuffer.short.toFloat() / 32768f
        }

        // Linear resampling only when input isn't 16 kHz. In practice
        // Anvil already emits at 16 kHz, so this is a fallback for
        // third-party recorders.
        val target = TARGET_SAMPLE_RATE.toDouble()
        if (kotlin.math.abs(sampleRate - target) < 1.0 || sampleCount == 0) {
            return samples
        }
        val ratio = sampleRate / target
        val outCount = (sampleCount / ratio).toInt()
        val out = FloatArray(outCount)
        for (i in 0 until outCount) {
            val srcPos = i * ratio
            val lo = srcPos.toInt()
            val hi = minOf(lo + 1, sampleCount - 1)
            val frac = (srcPos - lo).toFloat()
            out[i] = samples[lo] * (1 - frac) + samples[hi] * frac
        }
        return out
    }

    /**
     * Pad-or-crop to exactly `target` samples.
     *
     * Shorter → zero-pad at the end. The model's per-utterance mean
     * subtraction treats silence as approximately mean, so trailing
     * zeros contribute little to the embedding.
     *
     * Longer → center crop. Speaker identity is stable across a voice
     * sample, and the middle is usually the most articulated portion.
     */
    private fun padOrCropToFixedLength(input: FloatArray, target: Int): FloatArray {
        if (input.size == target) return input

        if (input.size > target) {
            val start = (input.size - target) / 2
            val out = FloatArray(target)
            System.arraycopy(input, start, out, 0, target)
            return out
        }

        val out = FloatArray(target)
        System.arraycopy(input, 0, out, 0, input.size)
        return out
    }

    private fun l2Normalize(v: FloatArray): FloatArray {
        var sum = 0f
        for (x in v) sum += x * x
        val norm = sqrt(sum) + 1e-9f
        return FloatArray(v.size) { v[it] / norm }
    }

    private fun floatArrayToArrayBuffer(floats: FloatArray): ArrayBuffer {
        val bytes = ByteBuffer.allocateDirect(floats.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        for (f in floats) bytes.putFloat(f)
        bytes.flip()
        return ArrayBuffer.copy(bytes)
    }

    private fun arrayBufferToFloatArray(buf: ArrayBuffer): FloatArray {
        val bytes = buf.getBuffer(false).order(ByteOrder.LITTLE_ENDIAN)
        val count = bytes.remaining() / 4
        val out = FloatArray(count)
        for (i in 0 until count) out[i] = bytes.float
        return out
    }
}