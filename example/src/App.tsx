import { useEffect, useState } from 'react';

import {
  ActivityIndicator,
  Platform,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
  useColorScheme,
} from 'react-native';

import ReactNativeBlobUtil from 'react-native-blob-util';

import {
  Anvil,
  createFileMarkerStore,
  createRecordingService,
} from 'react-native-nitro-audio-anvil';

import { SpeakerId } from 'react-native-nitro-speaker-id';

/**
 * Speaker-id example.
 *
 * Two flows:
 *   1. Enroll — record ~5 s, embed via Core ML (iOS) or TFLite (Android),
 *      cache the voiceprint in memory.
 *   2. Match  — record another ~5 s, embed, cosine against enrolled prints.
 *
 * Everything runs on-device. Setup is copying the platform-appropriate model
 * from bundled assets to a readable path on first launch:
 *   iOS      → ecapa-tdnn-192.mlpackage  (Core ML)
 *   Android  → ecapa-tdnn-192.tflite     (TFLite)
 *
 * Pairs with react-native-nitro-audio-anvil for the recording side. This
 * example uses Anvil because it's the same stack SHINE uses, but you can
 * feed SpeakerId.embed() any 16 kHz mono PCM ArrayBuffer.
 */

// One filename per platform. Core ML wants a `.mlpackage` directory,
// TFLite wants a `.tflite` file. Same underlying ECAPA model, different
// output of the conversion pipeline (see the library README for the
// exact Colab notebooks).
const MODEL_FILENAME =
  Platform.OS === 'ios' ? 'ecapa-tdnn-192.mlpackage' : 'ecapa-tdnn-192.tflite';

const RECORDINGS_DIR = `${ReactNativeBlobUtil.fs.dirs.DocumentDir}/speaker-id-example`;

const MODEL_PATH = `${ReactNativeBlobUtil.fs.dirs.DocumentDir}/${MODEL_FILENAME}`;

const MATCH_THRESHOLD = 0.4;

const markerBridge = {
  async readText(path: string) {
    try {
      return await ReactNativeBlobUtil.fs.readFile(path, 'utf8');
    } catch {
      return null;
    }
  },
  async writeText(path: string, contents: string) {
    await ReactNativeBlobUtil.fs.writeFile(path, contents, 'utf8');
  },
  async delete(path: string) {
    if (await ReactNativeBlobUtil.fs.exists(path)) {
      await ReactNativeBlobUtil.fs.unlink(path);
    }
  },
  async list(directory: string) {
    try {
      return (await ReactNativeBlobUtil.fs.ls(directory)).map(
        (name) => `${directory}/${name}`
      );
    } catch {
      return [];
    }
  },
};

const anvilService = createRecordingService({
  outputDirectory: RECORDINGS_DIR,
  markerStore: createFileMarkerStore(markerBridge, RECORDINGS_DIR),
});

const RECORDER_CONFIG = {
  outputDirectory: RECORDINGS_DIR,
  segmentDurationMs: 30_000,
  fsyncIntervalMs: 500,
  sampleRate: 16000,
  streamChunkMs: 100,
  speakerWindowMs: 5000,
  speakerWindowHopMs: 5000,
  onInterruption: 'resume' as const,
  keepAwakeInBackground: false,
  storageWarningBytes: 50 * 1024 * 1024,
  notification: { title: 'Speaker ID', text: 'Recording sample' },
};

type Voiceprint = { name: string; vector: Float32Array };

export default function App() {
  const scheme = useColorScheme();
  const isDark = scheme === 'dark';
  const styles = createStyles(isDark);

  const [modelReady, setModelReady] = useState(false);
  const [enrolled, setEnrolled] = useState<Voiceprint[]>([]);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState(
    `Ready. Tap 'Load Model' — expects ${MODEL_FILENAME} in the documents dir.`
  );
  const [lastMatch, setLastMatch] = useState<{
    name: string;
    score: number;
  } | null>(null);

  useEffect(() => {
    (async () => {
      if (!(await ReactNativeBlobUtil.fs.exists(RECORDINGS_DIR))) {
        await ReactNativeBlobUtil.fs.mkdir(RECORDINGS_DIR);
      }
    })();
  }, []);

  const handleLoadModel = async () => {
    setBusy(true);
    setStatus('Loading model…');
    try {
      // Copy from bundle to documents dir if not already present. On iOS
      // MainBundleDir is the .app bundle, on Android bundle-assets:// is
      // the assets folder — blob-util normalizes both. In production you'd
      // download from a CDN as a fallback if the bundle copy is missing.
      if (!(await ReactNativeBlobUtil.fs.exists(MODEL_PATH))) {
        setStatus(
          `Model not found at ${MODEL_PATH}\n\n` +
            `Bundle ${MODEL_FILENAME} in ${
              Platform.OS === 'ios'
                ? 'your Xcode target'
                : 'android/app/src/main/assets/'
            } and copy it here on first launch. See README for the conversion + bundling steps.`
        );
        setBusy(false);
        return;
      }
      await SpeakerId.loadModel(MODEL_PATH);
      setModelReady(true);
      setStatus(
        `Model loaded (${Platform.OS === 'ios' ? 'Core ML → Neural Engine' : 'TFLite → NNAPI/GPU'}). Enroll voices to test.`
      );
    } catch (err: any) {
      setStatus(`Load failed: ${err?.message}`);
    } finally {
      setBusy(false);
    }
  };

  // Records ~5 seconds, returns the PCM buffer of the first speaker window.
  const captureSample = async (): Promise<ArrayBuffer | null> => {
    if (Anvil.getPermissionStatus() !== 'granted') {
      const p = await Anvil.requestPermission();
      if (p !== 'granted') {
        setStatus('Microphone permission required.');
        return null;
      }
    }

    return new Promise(async (resolve) => {
      let received: ArrayBuffer | null = null;
      try {
        const recorder = await anvilService.begin({
          logicalId: `sample-${Date.now()}`,
          config: RECORDER_CONFIG,
        });

        const sub = recorder.addSpeakerWindowListener((window) => {
          if (received) return;
          received = window.buffer;
          sub.remove();
          setTimeout(async () => {
            try {
              await anvilService.end();
            } catch {}
            resolve(received);
          }, 50);
        });
      } catch (err: any) {
        setStatus(`Recording failed: ${err?.message}`);
        resolve(null);
      }
    });
  };

  const handleEnroll = async () => {
    if (!modelReady) return;
    setBusy(true);
    setStatus('Recording sample… speak for 5 seconds.');
    const pcm = await captureSample();
    if (!pcm) {
      setBusy(false);
      return;
    }

    setStatus('Embedding…');
    try {
      const embedding = await SpeakerId.embed(pcm, 16000);
      const name = `Voice ${enrolled.length + 1}`;
      setEnrolled((prev) => [...prev, { name, vector: embedding }]);
      setStatus(`Enrolled ${name}. Total: ${enrolled.length + 1}.`);
    } catch (err: any) {
      setStatus(`Embed failed: ${err?.message}`);
    } finally {
      setBusy(false);
    }
  };

  const handleMatch = async () => {
    if (!modelReady || enrolled.length === 0) return;
    setBusy(true);
    setStatus('Recording for matching…');
    const pcm = await captureSample();
    if (!pcm) {
      setBusy(false);
      return;
    }

    setStatus('Matching…');
    try {
      const embedding = await SpeakerId.embed(pcm, 16000);
      let best: { name: string; score: number } | null = null;
      for (const v of enrolled) {
        const score = SpeakerId.cosine(embedding, v.vector);
        if (!best || score > best.score) best = { name: v.name, score };
      }
      setLastMatch(best);
      if (best && best.score >= MATCH_THRESHOLD) {
        setStatus(`Match: ${best.name} (${best.score.toFixed(3)})`);
      } else {
        setStatus(
          `No match — best was ${best?.score.toFixed(3)} vs threshold ${MATCH_THRESHOLD}`
        );
      }
    } catch (err: any) {
      setStatus(`Match failed: ${err?.message}`);
    } finally {
      setBusy(false);
    }
  };

  const handleClear = () => {
    setEnrolled([]);
    setLastMatch(null);
    setStatus('Cleared.');
  };

  const backendLabel =
    Platform.OS === 'ios'
      ? 'Core ML · Neural Engine'
      : 'TensorFlow Lite · NNAPI/GPU';

  return (
    <SafeAreaView style={styles.safe}>
      <ScrollView contentContainerStyle={styles.container}>
        <Text style={styles.title}>Speaker ID · Example</Text>
        <Text style={styles.subtitle}>
          On-device ECAPA-TDNN. Enroll one or more voices, then record again to
          see which one the model matches you to.
        </Text>
        <Text style={styles.backend}>Backend: {backendLabel}</Text>

        <View style={styles.card}>
          <Text style={styles.status}>{status}</Text>
          {busy ? (
            <ActivityIndicator
              size="small"
              color={styles.spinnerColor.color}
              style={{ marginTop: 12 }}
            />
          ) : null}
        </View>

        <View style={styles.actions}>
          <PrimaryButton
            label="Load Model"
            onPress={handleLoadModel}
            disabled={busy}
            styles={styles}
          />
          <PrimaryButton
            label="Enroll Sample"
            onPress={handleEnroll}
            disabled={!modelReady || busy}
            styles={styles}
          />
          <PrimaryButton
            label="Match Sample"
            onPress={handleMatch}
            disabled={!modelReady || busy || enrolled.length === 0}
            styles={styles}
          />
          <SecondaryButton
            label="Clear Enrolled"
            onPress={handleClear}
            disabled={enrolled.length === 0}
            styles={styles}
          />
        </View>

        {enrolled.length > 0 ? (
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Enrolled voiceprints</Text>
            {enrolled.map((v, i) => (
              <Text key={i} style={styles.listItem}>
                • {v.name} — {v.vector.length}-d
              </Text>
            ))}
          </View>
        ) : null}

        {lastMatch ? (
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Last match</Text>
            <Text style={styles.listItem}>
              Best: <Text style={{ fontWeight: '700' }}>{lastMatch.name}</Text>
            </Text>
            <Text style={styles.listItem}>
              Score: {lastMatch.score.toFixed(4)} (
              {lastMatch.score >= MATCH_THRESHOLD ? 'match' : 'no match'})
            </Text>
            <Text style={styles.hint}>
              Cosine ∈ [-1, 1]. Same speaker clean: 0.55-0.85. Same on phone:
              0.35-0.55.
            </Text>
          </View>
        ) : null}
      </ScrollView>
    </SafeAreaView>
  );
}

function PrimaryButton({ label, onPress, disabled, styles }: any) {
  return (
    <TouchableOpacity
      onPress={onPress}
      disabled={disabled}
      style={[styles.btn, styles.btnPrimary, disabled && styles.btnDisabled]}
    >
      <Text style={styles.btnPrimaryLabel}>{label}</Text>
    </TouchableOpacity>
  );
}

function SecondaryButton({ label, onPress, disabled, styles }: any) {
  return (
    <TouchableOpacity
      onPress={onPress}
      disabled={disabled}
      style={[styles.btn, styles.btnSecondary, disabled && styles.btnDisabled]}
    >
      <Text style={styles.btnSecondaryLabel}>{label}</Text>
    </TouchableOpacity>
  );
}

const createStyles = (isDark: boolean) => ({
  ...StyleSheet.create({
    safe: { flex: 1, backgroundColor: isDark ? '#0b1220' : '#ffffff' },
    container: { padding: 20, gap: 16 },
    title: {
      color: isDark ? '#f1f5f9' : '#0f172a',
      fontSize: 24,
      fontWeight: '700' as const,
    },
    subtitle: {
      color: isDark ? '#94a3b8' : '#64748b',
      fontSize: 14,
      lineHeight: 20,
    },
    backend: {
      color: isDark ? '#38bdf8' : '#0284c7',
      fontSize: 12,
      fontWeight: '600' as const,
      letterSpacing: 0.4,
    },
    card: {
      backgroundColor: isDark ? '#111827' : '#f8fafc',
      borderRadius: 16,
      padding: 16,
      gap: 8,
      borderWidth: 1,
      borderColor: isDark ? '#1e293b' : '#e2e8f0',
    },
    cardTitle: {
      color: isDark ? '#f1f5f9' : '#0f172a',
      fontSize: 14,
      fontWeight: '700' as const,
      marginBottom: 4,
      textTransform: 'uppercase' as const,
      letterSpacing: 0.6,
    },
    status: {
      color: isDark ? '#e2e8f0' : '#1e293b',
      fontSize: 14,
      lineHeight: 20,
    },
    listItem: { color: isDark ? '#cbd5e1' : '#475569', fontSize: 14 },
    hint: {
      color: isDark ? '#64748b' : '#94a3b8',
      fontSize: 12,
      marginTop: 8,
      fontStyle: 'italic' as const,
    },
    actions: { gap: 10 },
    btn: {
      paddingVertical: 14,
      paddingHorizontal: 16,
      borderRadius: 12,
      alignItems: 'center' as const,
      justifyContent: 'center' as const,
    },
    btnPrimary: { backgroundColor: '#3b82f6' },
    btnPrimaryLabel: {
      color: '#ffffff',
      fontSize: 15,
      fontWeight: '700' as const,
    },
    btnSecondary: {
      backgroundColor: 'transparent',
      borderWidth: 1,
      borderColor: isDark ? '#334155' : '#cbd5e1',
    },
    btnSecondaryLabel: {
      color: isDark ? '#f1f5f9' : '#0f172a',
      fontSize: 15,
      fontWeight: '600' as const,
    },
    btnDisabled: { opacity: 0.4 },
  }),
  spinnerColor: { color: isDark ? '#94a3b8' : '#64748b' },
});
