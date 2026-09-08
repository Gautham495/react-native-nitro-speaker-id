import Foundation
import NitroModules
import ExecuTorch

/**
 * On-device ECAPA-TDNN speaker embedding for iOS via ExecuTorch.
 *
 * Uses mlboydaisuke/ECAPA-TDNN-Speaker-ExecuTorch — a repackaging of
 * SpeechBrain's spkrec-ecapa-voxceleb where the ENTIRE feature pipeline
 * (STFT → mel filterbank → per-utterance mean subtraction → ECAPA-TDNN)
 * is baked into the model graph. Verified cosine similarity of 0.9985
 * against cloud SpeechBrain's EncoderClassifier.encode_batch on the same
 * audio. Same-voice pairs land 0.65+ above different-voice pairs.
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
 * Backend: Core ML fp16 (via ExecuTorch's Core ML backend). ~2.4 ms per
 * inference on Apple Silicon Mac, expect ~10-30 ms on iPhone.
 *
 * The 3-second window is fixed — shorter audio is zero-padded, longer is
 * center-cropped. This matches how the model was exported and what its
 * per-utterance mean subtraction expects to see.
 */
class HybridSpeakerId: HybridSpeakerIdSpec {

    // Fixed input contract from the exported .pte model.
    private static let TARGET_SAMPLE_RATE: Double = 16000
    private static let TARGET_SAMPLE_COUNT: Int = 48000  // 3 seconds at 16 kHz
    private static let EMBEDDING_DIM: Int = 192

    private var module: Module?
    private var currentPath: String?
    private let sessionQueue = DispatchQueue(
        label: "com.margelo.nitro.speakerid.session",
        qos: .userInitiated
    )

    var isLoaded: Bool {
        return sessionQueue.sync { module != nil }
    }

    func loadModel(modelPath: String) throws -> Promise<Void> {
        return Promise.async { [weak self] in
            guard let self = self else { return }

            try self.sessionQueue.sync {
                if self.currentPath == modelPath && self.module != nil {
                    return
                }

                let path = modelPath.replacingOccurrences(of: "file://", with: "")
                if !FileManager.default.fileExists(atPath: path) {
                    throw RuntimeError.error(withMessage: "Model file not found: \(path)")
                }

                // ModuleLoadMode.mmap keeps the ~42 MB model out of resident
                // memory until inference touches it. Cold start is faster and
                // memory pressure is lower when the app is backgrounded.
                let module = Module(filePath: path, loadMode: .mmap)

                // Force-load the forward method so the first inference doesn't
                // pay compilation cost. Blocks until Core ML has compiled the
                // graph — ~200 ms one-time hit.
                try module.load("forward")

                self.module = module
                self.currentPath = modelPath
            }
        }
    }

    func unloadModel() throws {
        sessionQueue.sync {
            module = nil
            currentPath = nil
        }
    }

    func embed(pcm: ArrayBuffer, sampleRate: Double) throws -> Promise<ArrayBuffer> {
        // Convert PCM Int16 → Float32 [-1, 1] on the caller thread — the
        // ArrayBuffer lifetime is scoped to this turn, don't race the runtime.
        let floats = pcmToFloats(pcm, sampleRate: sampleRate)

        return Promise.async { [weak self] in
            guard let self = self else {
                throw RuntimeError.error(withMessage: "SpeakerId released before embed completed")
            }

            return try self.sessionQueue.sync {
                guard let module = self.module else {
                    throw RuntimeError.error(withMessage: "Model not loaded — call loadModel() first")
                }

                // The model expects EXACTLY 48000 samples. Anvil emits 1.5s
                // windows (24000 samples), so zero-pad to 3 seconds. Zero
                // padding is safe here because the model's per-utterance
                // mean subtraction naturally suppresses silence regions.
                let fixedPcm = padOrCropToFixedLength(floats, target: Self.TARGET_SAMPLE_COUNT)

                // Wrap as [1, 48000] float32 tensor.
                let tensor = Tensor(fixedPcm, shape: [1, Self.TARGET_SAMPLE_COUNT])

                // Forward pass — Core ML handles STFT, mel, mean subtraction,
                // ECAPA body, all inside the graph.
                let outputs = try module.forward([tensor])

                guard let output = outputs.first,
                      let outputTensor = output.tensor(),
                      let outputFloats: [Float] = try? outputTensor.scalars() else {
                    throw RuntimeError.error(
                        withMessage: "Model produced no output tensor"
                    )
                }

                if outputFloats.count != Self.EMBEDDING_DIM {
                    throw RuntimeError.error(
                        withMessage: "Expected \(Self.EMBEDDING_DIM)-d embedding, got \(outputFloats.count)"
                    )
                }

                // The model already L2-normalizes internally, but re-norm as
                // defense against fp16 drift.
                let normalized = l2Normalize(outputFloats)
                return arrayBufferFromFloats(normalized)
            }
        }
    }

    func cosine(a: ArrayBuffer, b: ArrayBuffer) throws -> Double {
        let aFloats = float32ArrayFromArrayBuffer(a)
        let bFloats = float32ArrayFromArrayBuffer(b)
        let n = min(aFloats.count, bFloats.count)
        var dot: Double = 0
        for i in 0..<n {
            dot += Double(aFloats[i]) * Double(bFloats[i])
        }
        return dot
    }
}

// MARK: - PCM helpers

private func pcmToFloats(_ buffer: ArrayBuffer, sampleRate: Double) -> [Float] {
    let byteCount = buffer.size
    let sampleCount = byteCount / MemoryLayout<Int16>.size

    var floats = [Float](repeating: 0, count: sampleCount)
    buffer.data.withMemoryRebound(to: Int16.self, capacity: sampleCount) { int16Ptr in
        for i in 0..<sampleCount {
            floats[i] = Float(int16Ptr[i]) / 32768.0
        }
    }

    let target = 16000.0
    if abs(sampleRate - target) < 1.0 || sampleCount == 0 {
        return floats
    }

    // Linear resampling only when input isn't 16 kHz. In practice Anvil
    // already emits at 16 kHz, so this is a fallback for third-party
    // recorders that might not.
    let ratio = sampleRate / target
    let outCount = Int(Double(sampleCount) / ratio)
    var out = [Float](repeating: 0, count: outCount)
    for i in 0..<outCount {
        let srcPos = Double(i) * ratio
        let lo = Int(srcPos)
        let hi = min(lo + 1, sampleCount - 1)
        let frac = Float(srcPos - Double(lo))
        out[i] = floats[lo] * (1 - frac) + floats[hi] * frac
    }
    return out
}

/**
 * Pad-or-crop to exactly `target` samples.
 *
 * Shorter → zero-pad at the end. The model's per-utterance mean subtraction
 * treats silence as approximately mean, so trailing zeros contribute little
 * to the embedding.
 *
 * Longer → center crop. Speaker identity is stable across a voice sample,
 * and the middle is usually the most articulated / least noisy portion.
 */
private func padOrCropToFixedLength(_ input: [Float], target: Int) -> [Float] {
    if input.count == target { return input }

    if input.count > target {
        let start = (input.count - target) / 2
        return Array(input[start..<(start + target)])
    }

    var out = [Float](repeating: 0, count: target)
    for i in 0..<input.count { out[i] = input[i] }
    return out
}

private func l2Normalize(_ v: [Float]) -> [Float] {
    var sum: Float = 0
    for x in v { sum += x * x }
    let norm = sqrtf(sum) + 1e-9
    return v.map { $0 / norm }
}

private func arrayBufferFromFloats(_ floats: [Float]) -> ArrayBuffer {
    let byteCount = floats.count * MemoryLayout<Float>.size
    return floats.withUnsafeBufferPointer { buf -> ArrayBuffer in
        let bytePtr = UnsafeRawPointer(buf.baseAddress!).assumingMemoryBound(to: UInt8.self)
        return ArrayBuffer.copy(of: bytePtr, size: byteCount)
    }
}

private func float32ArrayFromArrayBuffer(_ buffer: ArrayBuffer) -> [Float] {
    let count = buffer.size / MemoryLayout<Float>.size
    var floats = [Float](repeating: 0, count: count)
    buffer.data.withMemoryRebound(to: Float.self, capacity: count) { ptr in
        for i in 0..<count {
            floats[i] = ptr[i]
        }
    }
    return floats
}