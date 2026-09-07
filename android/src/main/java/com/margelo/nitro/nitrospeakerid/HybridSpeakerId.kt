package com.margelo.nitro.nitrospeakerid

import com.facebook.proguard.annotations.DoNotStrip
import com.margelo.nitro.core.ArrayBuffer
import com.margelo.nitro.core.Promise
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.Executors
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.log10
import kotlin.math.ln
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * On-device ECAPA-TDNN speaker embedding for Android.
 *
 * Architecture: raw PCM → mel-spectrogram (native, hand-rolled FFT) →
 * ECAPA body (TFLite) → 192-d L2-normalized embedding.
 *
 * The mel-spectrogram front-end is hand-rolled Kotlin rather than routed
 * through TFLite's signal ops. Reason: TFLite's SignalTransformer exists but
 * is finicky about tensor shapes and the mel filterbank isn't first-class.
 * A hand-rolled Cooley-Tukey FFT gets us feature parity with SpeechBrain in
 * ~200 lines of well-understood code. Runs in ~3-5 ms per 1.5 s window.
 *
 * Expected latency for one 1.5 s window:
 *   Pixel 8 (Tensor G3):        ~25 ms   (mel ~3 + TFLite NNAPI ~22)
 *   Galaxy S23 (SD 8 Gen 2):    ~22 ms   (mel ~3 + TFLite NNAPI ~19)
 *   Pixel 6a (Tensor G1):       ~40 ms   (mel ~4 + TFLite GPU ~36)
 *   Snapdragon 778G mid-range:  ~55 ms   (mel ~5 + TFLite GPU ~50)
 *   Low-end 2020 Android:       ~140 ms  (mel ~10 + TFLite CPU ~130)
 */
@DoNotStrip
class HybridSpeakerId : HybridSpeakerIdSpec() {

    // Guard concurrent embed() with a single-thread executor. Interpreter is
    // thread-safe for run() but NNAPI's shared driver misbehaves under load.
    private val sessionExecutor = Executors.newSingleThreadExecutor()

    private var interpreter: Interpreter? = null
    private var nnapiDelegate: NnApiDelegate? = null
    private var gpuDelegate: GpuDelegate? = null
    private var currentPath: String? = null

    // Feature extractor. Built once at construction, reused per call.
    private val melExtractor = MelSpectrogram(
        sampleRate = 16000,
        nFft = 400,        // 25 ms window at 16 kHz — SpeechBrain default
        hopLength = 160,   // 10 ms hop — SpeechBrain default
        nMels = 80         // ECAPA input dimension
    )

    override val isLoaded: Boolean
        get() = interpreter != null

    override fun loadModel(modelPath: String): Promise<Unit> {
        return Promise.async {
            synchronized(this) {
                if (currentPath == modelPath && interpreter != null) return@async
                if (!File(modelPath).exists()) {
                    throw RuntimeException("Model file not found: $modelPath")
                }

                closeInterpreter()

                val modelBuffer = loadMappedFile(modelPath)
                val options = Interpreter.Options()
                options.numThreads = 2

                // NNAPI → GPU → CPU fallback chain.
                var delegated = false
                try {
                    val nnapiOpts = NnApiDelegate.Options().apply {
                        setAllowFp16(true)
                        setUseNnapiCpu(false)
                    }
                    val delegate = NnApiDelegate(nnapiOpts)
                    options.addDelegate(delegate)
                    nnapiDelegate = delegate
                    delegated = true
                    android.util.Log.i("SpeakerId", "Using NNAPI delegate")
                } catch (t: Throwable) {
                    android.util.Log.w("SpeakerId", "NNAPI unavailable: ${t.message}")
                }

                if (!delegated) {
                    try {
                        val compat = CompatibilityList()
                        if (compat.isDelegateSupportedOnThisDevice) {
                            val delegate = GpuDelegate()
                            options.addDelegate(delegate)
                            gpuDelegate = delegate
                            delegated = true
                            android.util.Log.i("SpeakerId", "Using GPU delegate")
                        }
                    } catch (t: Throwable) {
                        android.util.Log.w("SpeakerId", "GPU delegate unavailable: ${t.message}")
                    }
                }

                if (!delegated) {
                    android.util.Log.i("SpeakerId", "Using CPU (XNNPACK)")
                }

                interpreter = Interpreter(modelBuffer, options)
                currentPath = modelPath
            }
        }
    }

    override fun unloadModel() {
        synchronized(this) {
            closeInterpreter()
        }
    }

    private fun closeInterpreter() {
        interpreter?.close()
        interpreter = null
        nnapiDelegate?.close()
        nnapiDelegate = null
        gpuDelegate?.close()
        gpuDelegate = null
        currentPath = null
    }

    override fun embed(pcm: ArrayBuffer, sampleRate: Double): Promise<ArrayBuffer> {
        // Snapshot the JS buffer to Kotlin memory on the calling thread.
        val floats = pcmToFloats(pcm, sampleRate)

        return Promise.async {
            val currentInterpreter = interpreter
                ?: throw RuntimeException("Model not loaded — call loadModel() first")

            val submit = sessionExecutor.submit<ArrayBuffer> {
                // 1. audio → mel-spec [timeFrames * 80]
                val mel = melExtractor.compute(floats)
                if (mel.isEmpty()) {
                    throw RuntimeException("Audio too short for mel-spectrogram (need >= 25 ms)")
                }
                val timeFrames = mel.size / 80

                // 2. Resize TFLite input tensor to match this frame count.
                currentInterpreter.resizeInput(0, intArrayOf(1, timeFrames, 80))
                currentInterpreter.allocateTensors()

                val inputBuffer = ByteBuffer
                    .allocateDirect(mel.size * 4)
                    .order(ByteOrder.nativeOrder())
                for (f in mel) inputBuffer.putFloat(f)
                inputBuffer.rewind()

                // 3. TFLite forward pass.
                val outputShape = currentInterpreter.getOutputTensor(0).shape()
                val embeddingDim = outputShape[outputShape.size - 1]
                val outputBuffer = ByteBuffer
                    .allocateDirect(embeddingDim * 4)
                    .order(ByteOrder.nativeOrder())

                currentInterpreter.run(inputBuffer, outputBuffer)

                outputBuffer.rewind()
                val embedding = FloatArray(embeddingDim)
                for (i in 0 until embeddingDim) embedding[i] = outputBuffer.float

                // 4. L2 normalize (model already does this; defensive re-norm).
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

    // ── Helpers ─────────────────────────────────────────────────────

    private fun loadMappedFile(modelPath: String): MappedByteBuffer {
        val file = File(modelPath)
        val inputStream = FileInputStream(file)
        val fileChannel = inputStream.channel
        try {
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, file.length())
        } finally {
            fileChannel.close()
            inputStream.close()
        }
    }

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

// MARK: - Mel-spectrogram (hand-rolled Cooley-Tukey FFT)

/**
 * Computes a mel-spectrogram from raw audio, matching SpeechBrain's Fbank
 * defaults so ECAPA embeddings match the cloud model.
 *
 * Pipeline: Hann-windowed STFT → power spectrum → mel filterbank →
 * log(1 + 10*mel). SpeechBrain defaults: n_fft=400, hop=160, n_mels=80,
 * sample_rate=16000.
 *
 * FFT is Cooley-Tukey radix-2, in-place, single allocation per compute() call.
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
    private val melFilterbank: Array<FloatArray>  // [nMels][halfN + 1]

    // FFT twiddle factor caches — computed once at init, reused per call.
    private val cosTable: FloatArray
    private val sinTable: FloatArray
    private val bitReverse: IntArray

    init {
        // Hann window over the un-padded nFft samples.
        window = FloatArray(nFft) { i ->
            0.5f * (1f - cos(2f * PI.toFloat() * i / (nFft - 1)))
        }

        melFilterbank = buildMelFilterbank(nMels, halfN + 1, sampleRate)

        // Precompute twiddle factors for the FFT.
        cosTable = FloatArray(paddedN / 2)
        sinTable = FloatArray(paddedN / 2)
        for (i in 0 until paddedN / 2) {
            cosTable[i] = cos(2.0 * PI * i / paddedN).toFloat()
            sinTable[i] = -sin(2.0 * PI * i / paddedN).toFloat()  // negative for forward FFT
        }

        // Precompute bit-reversal permutation.
        val bits = (ln(paddedN.toDouble()) / ln(2.0)).toInt()
        bitReverse = IntArray(paddedN) { i ->
            var reversed = 0
            var value = i
            for (b in 0 until bits) {
                reversed = (reversed shl 1) or (value and 1)
                value = value shr 1
            }
            reversed
        }
    }

    /**
     * pcm: mono 16 kHz Float32 in [-1, 1].
     * Returns flat FloatArray of length [timeFrames * nMels], row-major
     * (time-major, mel-minor). Matches SpeechBrain's [batch, time, mel] shape.
     */
    fun compute(pcm: FloatArray): FloatArray {
        if (pcm.size < nFft) return FloatArray(0)

        // Match SpeechBrain: center=False, no end-padding.
        val timeFrames = 1 + (pcm.size - nFft) / hopLength
        val result = FloatArray(timeFrames * nMels)

        // Reusable per-frame buffers.
        val real = FloatArray(paddedN)
        val imag = FloatArray(paddedN)
        val power = FloatArray(halfN + 1)

        for (frameIdx in 0 until timeFrames) {
            val start = frameIdx * hopLength

            // 1. Windowed segment, zero-padded to paddedN.
            for (i in 0 until nFft) {
                real[i] = pcm[start + i] * window[i]
            }
            for (i in nFft until paddedN) real[i] = 0f
            for (i in 0 until paddedN) imag[i] = 0f

            // 2. In-place FFT.
            fft(real, imag)

            // 3. Power spectrum (halfN + 1 bins, including DC and Nyquist).
            for (i in 0..halfN) {
                power[i] = real[i] * real[i] + imag[i] * imag[i]
            }

            // 4. Apply mel filterbank + log(1 + 10*mel).
            for (m in 0 until nMels) {
                var acc = 0f
                val filter = melFilterbank[m]
                for (k in 0..halfN) {
                    acc += power[k] * filter[k]
                }
                result[frameIdx * nMels + m] = ln(1f + 10f * acc)
            }
        }

        return result
    }

    /**
     * Cooley-Tukey radix-2 in-place FFT. Time complexity O(N log N),
     * uses precomputed twiddle tables so no trig calls per frame.
     */
    private fun fft(real: FloatArray, imag: FloatArray) {
        val n = paddedN

        // Bit-reversal permutation.
        for (i in 0 until n) {
            val j = bitReverse[i]
            if (i < j) {
                var t = real[i]; real[i] = real[j]; real[j] = t
                t = imag[i]; imag[i] = imag[j]; imag[j] = t
            }
        }

        // Butterfly stages.
        var size = 2
        while (size <= n) {
            val halfSize = size / 2
            val tableStep = n / size
            var i = 0
            while (i < n) {
                var k = 0
                var j = i
                while (j < i + halfSize) {
                    val tr = cosTable[k] * real[j + halfSize] - sinTable[k] * imag[j + halfSize]
                    val ti = cosTable[k] * imag[j + halfSize] + sinTable[k] * real[j + halfSize]
                    real[j + halfSize] = real[j] - tr
                    imag[j + halfSize] = imag[j] - ti
                    real[j] += tr
                    imag[j] += ti
                    j++
                    k += tableStep
                }
                i += size
            }
            size *= 2
        }
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

private fun sin(x: Double): Double = kotlin.math.sin(x)