package com.margelo.nitro.nitrospeakerid

import android.util.Log
import com.facebook.proguard.annotations.DoNotStrip
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.margelo.nitro.core.ArrayBuffer
import com.margelo.nitro.core.Promise
import org.jtransforms.fft.FloatFFT_1D
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executors
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.log10
import kotlin.math.ln
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * On-device ECAPA-TDNN speaker embedding for Android, backed by LiteRT.
 *
 * Architecture: raw PCM → mel-spectrogram (native, JTransforms FFT) →
 * ECAPA body (LiteRT) → 192-d L2-normalized embedding.
 *
 * The mel-spectrogram front-end is ported from the working iOS version
 * (which uses Apple's vDSP), swapping vDSP for JTransforms' FloatFFT_1D.
 * JTransforms is a well-tested FFT library that produces numerically
 * correct results — safer than hand-rolling Cooley-Tukey which is
 * one-sign-flip-away from garbage embeddings.
 *
 * Currently CPU-only. GPU delegate crashed on Vivo/mid-range devices.
 * CPU is universal and fast enough (~50-140 ms per 1.5 s window on
 * mid-range, ~15-30 ms on flagship).
 */
@DoNotStrip
class HybridSpeakerId : HybridSpeakerIdSpec() {

    private val sessionExecutor = Executors.newSingleThreadExecutor()

    private var model: CompiledModel? = null
    private var currentPath: String? = null

    // Params match SpeechBrain's Fbank defaults — same as iOS's Swift version.
    private val melExtractor = MelSpectrogram(
        sampleRate = 16000,
        nFft = 400,        // 25 ms window
        hopLength = 160,   // 10 ms hop
        nMels = 80         // ECAPA input dim
    )

    override val isLoaded: Boolean
        get() = model != null

    override fun loadModel(modelPath: String): Promise<Unit> {
        return Promise.async {
            synchronized(this) {
                if (currentPath == modelPath && model != null) return@async
                if (!File(modelPath).exists()) {
                    throw RuntimeException("Model file not found: $modelPath")
                }
                closeModel()
                model = CompiledModel.create(
                    modelPath,
                    CompiledModel.Options(Accelerator.CPU)
                )
                currentPath = modelPath
                Log.i("SpeakerId", "LiteRT model loaded (CPU)")
            }
        }
    }

    override fun unloadModel() {
        synchronized(this) { closeModel() }
    }

    private fun closeModel() {
        model?.close()
        model = null
        currentPath = null
    }

    override fun embed(pcm: ArrayBuffer, sampleRate: Double): Promise<ArrayBuffer> {
        val floats = pcmToFloats(pcm, sampleRate)

        return Promise.async {
            val currentModel = model
                ?: throw RuntimeException("Model not loaded — call loadModel() first")

            val submit = sessionExecutor.submit<ArrayBuffer> {
                // 1. audio → mel-spec [timeFrames * 80]
                val mel = melExtractor.compute(floats)
                if (mel.isEmpty()) {
                    throw RuntimeException("Audio too short for mel-spectrogram (need >= 25 ms)")
                }

                // 2. Pad/center-crop to model's fixed input shape (1, 150, 80).
                val targetFrames = 150
                val nMels = 80
                val actualFrames = mel.size / nMels
                val fixedMel = FloatArray(targetFrames * nMels)

                if (actualFrames >= targetFrames) {
                    val startFrame = (actualFrames - targetFrames) / 2
                    System.arraycopy(mel, startFrame * nMels, fixedMel, 0, targetFrames * nMels)
                } else {
                    System.arraycopy(mel, 0, fixedMel, 0, mel.size)
                }

                // 3. LiteRT forward pass.
                val inputBuffers = currentModel.createInputBuffers()
                val outputBuffers = currentModel.createOutputBuffers()

                try {
                    inputBuffers[0].writeFloat(fixedMel)
                    currentModel.run(inputBuffers, outputBuffers)
                    val embedding = outputBuffers[0].readFloat()
                    val normalized = l2Normalize(embedding)
                    floatArrayToArrayBuffer(normalized)
                } finally {
                    inputBuffers.forEach { it.close() }
                    outputBuffers.forEach { it.close() }
                }
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

    // ── Buffer helpers ──────────────────────────────────────────────

    private fun pcmToFloats(pcm: ArrayBuffer, sampleRate: Double): FloatArray {
        val byteBuffer = pcm.getBuffer(false).order(ByteOrder.LITTLE_ENDIAN)
        val sampleCount = byteBuffer.remaining() / 2
        val samples = FloatArray(sampleCount)
        for (i in 0 until sampleCount) {
            samples[i] = byteBuffer.short.toFloat() / 32768f
        }

        val target = 16000.0
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

// ── Mel-spectrogram (direct port of iOS Swift version) ─────────────────

/**
 * Computes a mel-spectrogram from raw audio, matching the iOS Swift version
 * so ECAPA embeddings are cross-platform compatible.
 *
 * Pipeline: Hann-windowed STFT → power spectrum → mel filterbank →
 * log(1 + 10*mel). Params match SpeechBrain Fbank defaults: n_fft=400,
 * hop=160, n_mels=80, sample_rate=16000.
 *
 * FFT via JTransforms' FloatFFT_1D — a mature, well-tested Java FFT that
 * produces numerically correct results identical to numpy/scipy/vDSP.
 * Using a real library instead of hand-rolling Cooley-Tukey eliminates
 * a whole class of sign/scaling bugs.
 */
private class MelSpectrogram(
    private val sampleRate: Int,
    private val nFft: Int,
    private val hopLength: Int,
    private val nMels: Int
) {
    private val paddedN: Int = nFft.nextPowerOfTwo()
    private val halfN: Int = paddedN / 2
    private val window: FloatArray
    private val melFilterbank: Array<FloatArray>
    private val fft: FloatFFT_1D

    init {
        // Hann window matching vDSP_hann_window with vDSP_HANN_NORM flag:
        // 0.5 * (1 - cos(2π * n / (N-1)))
        // The N-1 denominator is Apple's normalization. Denominator matters —
        // using N vs N-1 changes edge tapering enough to shift the FFT bins.
        window = FloatArray(nFft) { i ->
            0.5f * (1f - cos(2f * PI.toFloat() * i / (nFft - 1)))
        }

        melFilterbank = buildMelFilterbank(nMels, halfN + 1, sampleRate)
        fft = FloatFFT_1D(paddedN.toLong())
    }

/**
     * pcm: mono 16 kHz Float32 in [-1, 1].
     * Returns flat FloatArray of length [timeFrames * nMels], row-major
     * (time-major, mel-minor). Matches SpeechBrain's [batch, time, mel] shape.
     *
     * Feature pipeline (must match speechbrain/spkrec-ecapa-voxceleb exactly):
     *   1. Hann-windowed STFT (n_fft=400, hop=160)
     *   2. Power spectrum via JTransforms real FFT
     *   3. Triangular mel filterbank (80 bins, HTK)
     *   4. Log-mel in dB: 10*log10(max(mel, 1e-10))
     *   5. Top-db clip: floor at (max - 80 dB)
     *   6. Sentence mean-var norm: subtract per-mel-bin mean over time
     *
     * Steps 4-6 match SpeechBrain's Filterbank(log_mel=True, amin=1e-10,
     * ref_value=1.0, top_db=80.0) followed by InputNormalization(
     * norm_type='sentence', std_norm=False). ECAPA-TDNN was trained
     * expecting this exact input; skip step 6 and cosine collapses to ~0.
     */
    fun compute(pcm: FloatArray): FloatArray {
        if (pcm.size < nFft) return FloatArray(0)

        // center=False, no end-padding — same as iOS Swift.
        val timeFrames = 1 + (pcm.size - nFft) / hopLength
        val result = FloatArray(timeFrames * nMels)

        // Buffer for JTransforms realForward: size N, contains real signal,
        // gets overwritten with packed complex output.
        val fftBuffer = FloatArray(paddedN)
        val power = FloatArray(halfN + 1)

        for (frameIdx in 0 until timeFrames) {
            val start = frameIdx * hopLength

            // 1. Windowed segment, zero-padded to paddedN.
            for (i in 0 until nFft) {
                fftBuffer[i] = pcm[start + i] * window[i]
            }
            for (i in nFft until paddedN) fftBuffer[i] = 0f

            // 2. Real-to-complex FFT via JTransforms.
            //    Output packing for FloatFFT_1D.realForward:
            //      a[0]    = Re[0]     (DC, real)
            //      a[1]    = Re[N/2]   (Nyquist, real — packed here for space)
            //      a[2k]   = Re[k]     for k = 1..N/2-1
            //      a[2k+1] = Im[k]     for k = 1..N/2-1
            fft.realForward(fftBuffer)

            // 3. Power spectrum |X|^2.
            power[0] = fftBuffer[0] * fftBuffer[0]                // DC
            power[halfN] = fftBuffer[1] * fftBuffer[1]            // Nyquist
            for (k in 1 until halfN) {
                val re = fftBuffer[2 * k]
                val im = fftBuffer[2 * k + 1]
                power[k] = re * re + im * im
            }

            // 4. Mel filterbank + log-mel in dB, matching SpeechBrain's
            //    Filterbank(log_mel=True, amin=1e-10, ref_value=1.0):
            //    result = 10 * log10(max(mel_energy, 1e-10))
            for (m in 0 until nMels) {
                var acc = 0f
                val filter = melFilterbank[m]
                for (k in 0..halfN) {
                    acc += power[k] * filter[k]
                }
                val clamped = maxOf(acc, 1e-10f)
                result[frameIdx * nMels + m] = 10f * log10(clamped)
            }
        }

        // 5. Top-db clip. SpeechBrain floors values at (max - top_db).
        //    top_db=80 is the SpeechBrain default. Keeps the log-mel
        //    dynamic range bounded so very quiet frames don't dominate
        //    the sentence mean.
        var maxVal = Float.NEGATIVE_INFINITY
        for (v in result) if (v > maxVal) maxVal = v
        val floor = maxVal - 80f
        for (i in result.indices) {
            if (result[i] < floor) result[i] = floor
        }

        // 6. Sentence-level mean normalization. SpeechBrain's
        //    InputNormalization(norm_type='sentence', std_norm=False):
        //    subtract per-mel-bin mean across all time frames.
        //    std_norm=False means we do NOT divide by std.
        //    Without this, cosine similarity between on-device and
        //    cloud-enrolled embeddings collapses to ~0 — the ECAPA body
        //    was trained expecting mean-subtracted log-mel input.
        val means = FloatArray(nMels)
        for (t in 0 until timeFrames) {
            for (m in 0 until nMels) {
                means[m] += result[t * nMels + m]
            }
        }
        val invT = 1f / timeFrames
        for (m in 0 until nMels) means[m] *= invT
        for (t in 0 until timeFrames) {
            for (m in 0 until nMels) {
                result[t * nMels + m] -= means[m]
            }
        }

        return result
    }

    private fun buildMelFilterbank(nMels: Int, nFftBins: Int, sampleRate: Int): Array<FloatArray> {
        val fMin = 0f
        val fMax = sampleRate / 2f

        fun hzToMel(hz: Float) = 2595f * log10(1 + hz / 700f)
        fun melToHz(mel: Float) = 700f * (10f.pow(mel / 2595f) - 1)

        val melMin = hzToMel(fMin)
        val melMax = hzToMel(fMax)
        val melPoints = FloatArray(nMels + 2) { i ->
            melMin + i * (melMax - melMin) / (nMels + 1)
        }
        val hzPoints = FloatArray(nMels + 2) { melToHz(melPoints[it]) }
        val binPoints = FloatArray(nMels + 2) { i ->
            hzPoints[i] * (nFftBins - 1) / (sampleRate / 2f)
        }

        val filters = Array(nMels) { FloatArray(nFftBins) }
        for (m in 0 until nMels) {
            val leftBin = binPoints[m]
            val centerBin = binPoints[m + 1]
            val rightBin = binPoints[m + 2]

            for (k in 0 until nFftBins) {
                val bin = k.toFloat()
                if (bin < leftBin || bin > rightBin) continue
                filters[m][k] = if (bin < centerBin) {
                    (bin - leftBin) / (centerBin - leftBin)
                } else {
                    (rightBin - bin) / (rightBin - centerBin)
                }
            }
        }
        return filters
    }
}

private fun Int.nextPowerOfTwo(): Int {
    var n = 1
    while (n < this) n = n shl 1
    return n
}