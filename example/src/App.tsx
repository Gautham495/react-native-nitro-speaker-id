import { useEffect, useState } from 'react';
import {
  ActivityIndicator,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
  // useColorScheme,
} from 'react-native';

import ReactNativeBlobUtil from 'react-native-blob-util';

import { unzip } from 'react-native-zip-archive';

import {
  Anvil,
  createFileMarkerStore,
  createRecordingService,
} from 'react-native-nitro-audio-anvil';

import { SpeakerId } from 'react-native-nitro-speaker-id';

/**
 * Speaker-id example.
 *
 * Two loading paths demonstrated side by side:
 *   1. From bundled assets — model shipped inside the app binary.
 *      Instant, offline on first launch, bigger install size.
 *   2. From CDN — model downloaded on first launch. Smaller binary,
 *      needs internet once, cached forever.
 *
 * iOS wrinkle: .mlpackage is a directory, not a file. Core ML can't parse
 * it if you serve the raw directory over HTTP — the download becomes a
 * random binary blob. Fix: zip it, upload the .zip, download the .zip,
 * unzip on device. Handled in ensureModelFile('cdn') below.
 *
 * Both paths converge to the same MODEL_PATH in the documents dir before
 * calling SpeakerId.loadModel().
 */

const MODEL_FILENAME =
  Platform.OS === 'ios' ? 'ecapa-body-192.mlpackage' : 'ecapa-body-192.tflite';

// Public CDN hosting the pre-converted models.
// ⚠️  This is my (Gautham's) personal Cloudflare R2 bucket — I might delete
//     or reshuffle it whenever. Fine for kicking the tires; don't build
//     production against it. Convert your own model (see ECAPA-CONVERSION.md) and
//     host on your own storage before shipping.
const CDN_BASE = 'https://ml-models-bucket.gauthamvijay.com';
const CDN_URL =
  Platform.OS === 'ios'
    ? `${CDN_BASE}/ecapa-body-192.mlpackage.zip`
    : `${CDN_BASE}/ecapa-body-192.tflite`;

const RECORDINGS_DIR = `${ReactNativeBlobUtil.fs.dirs.DocumentDir}/speaker-id-example`;

const MODEL_PATH = `${ReactNativeBlobUtil.fs.dirs.DocumentDir}/${MODEL_FILENAME}`;

const DOWNLOAD_TMP_PATH =
  Platform.OS === 'ios'
    ? `${ReactNativeBlobUtil.fs.dirs.DocumentDir}/ecapa-body-192.mlpackage.zip`
    : MODEL_PATH;

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
type LoadSource = 'bundle' | 'cdn';

export default function App() {
  // const scheme = useColorScheme();
  // const isDark = scheme === 'dark';
  const isDark = false;

  const s = createStyles(isDark);

  const [modelReady, setModelReady] = useState(false);
  const [enrolled, setEnrolled] = useState<Voiceprint[]>([]);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState('Choose a source to load the model.');
  const [downloadProgress, setDownloadProgress] = useState(0);
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

  const ensureModelFile = async (source: LoadSource): Promise<boolean> => {
    // Always start clean. User picked a source; honor it. The old file might
    // be from a different source or a corrupted download.
    if (await ReactNativeBlobUtil.fs.exists(MODEL_PATH)) {
      // isDir check because on iOS MODEL_PATH is a directory (.mlpackage)
      const isDir = await ReactNativeBlobUtil.fs.isDir(MODEL_PATH);
      if (isDir) {
        // Delete directory recursively — blob-util's unlink handles this.
        await ReactNativeBlobUtil.fs.unlink(MODEL_PATH);
      } else {
        await ReactNativeBlobUtil.fs.unlink(MODEL_PATH);
      }
    }
    // Also clean up any leftover zip from a previous CDN attempt.
    if (
      Platform.OS === 'ios' &&
      (await ReactNativeBlobUtil.fs.exists(DOWNLOAD_TMP_PATH))
    ) {
      await ReactNativeBlobUtil.fs.unlink(DOWNLOAD_TMP_PATH);
    }

    if (source === 'bundle') {
      try {
        const bundledSource =
          Platform.OS === 'ios'
            ? `${ReactNativeBlobUtil.fs.dirs.MainBundleDir}/${MODEL_FILENAME}`
            : `bundle-assets://${MODEL_FILENAME}`;
        await ReactNativeBlobUtil.fs.cp(bundledSource, MODEL_PATH);
        return true;
      } catch (err: any) {
        setStatus(
          `Bundle copy failed: ${err?.message}\n\nBundle ${MODEL_FILENAME} in ${
            Platform.OS === 'ios'
              ? 'your Xcode target'
              : 'android/app/src/main/assets/'
          } to enable this path.`
        );
        return false;
      }
    }

    // CDN path
    setStatus('Downloading model from CDN...');
    setDownloadProgress(0);

    try {
      if (Platform.OS === 'android') {
        // Android DownloadManager needs shared storage. Download there, then
        // copy into app-private storage so the model loader can find it.
        const externalDownloadPath = `${ReactNativeBlobUtil.fs.dirs.LegacyDownloadDir}/${MODEL_FILENAME}`;

        await ReactNativeBlobUtil.config({
          timeout: 300_000,
          addAndroidDownloads: {
            useDownloadManager: true,
            notification: true,
            title: 'Speaker ID model',
            description: 'One-time model download',
            path: externalDownloadPath,
            mime: 'application/octet-stream',
            mediaScannable: false,
          },
        }).fetch('GET', CDN_URL);

        // Copy into app storage (mv fails across the shared/private boundary
        // under scoped storage on Android 10+).
        if (await ReactNativeBlobUtil.fs.exists(DOWNLOAD_TMP_PATH)) {
          await ReactNativeBlobUtil.fs.unlink(DOWNLOAD_TMP_PATH);
        }
        await ReactNativeBlobUtil.fs.cp(
          externalDownloadPath,
          DOWNLOAD_TMP_PATH
        );
        // Best-effort cleanup of the shared-storage copy — user can delete
        // manually from Downloads if this fails.
        try {
          await ReactNativeBlobUtil.fs.unlink(externalDownloadPath);
        } catch {}
      } else {
        // iOS — download directly to app private with progress reporting.
        await ReactNativeBlobUtil.config({
          path: DOWNLOAD_TMP_PATH,
          timeout: 300_000,
          followRedirect: true,
        })
          .fetch('GET', CDN_URL)
          .progress({ interval: 200 }, (received, total) => {
            setDownloadProgress(Number(received) / Number(total));
          });

        // iOS unpacks the .mlpackage.zip
        setStatus('Unpacking model...');
        await unzip(DOWNLOAD_TMP_PATH, ReactNativeBlobUtil.fs.dirs.DocumentDir);
        await ReactNativeBlobUtil.fs.unlink(DOWNLOAD_TMP_PATH);
      }

      setDownloadProgress(0);
      return true;
    } catch (err: any) {
      setStatus(`Download failed: ${err?.message}`);
      setDownloadProgress(0);
      return false;
    }
  };

  const handleLoad = async (source: LoadSource) => {
    setBusy(true);

    setStatus(
      source === 'bundle'
        ? 'Copying from bundle...'
        : 'Preparing CDN download...'
    );

    try {
      const ok = await ensureModelFile(source);

      if (!ok) {
        setBusy(false);
        return;
      }

      setStatus('Loading model into runtime...');

      const exists = await ReactNativeBlobUtil.fs.exists(MODEL_PATH);

      const isDir = exists && (await ReactNativeBlobUtil.fs.isDir(MODEL_PATH));

      console.log('[speaker-id] MODEL_PATH:', MODEL_PATH);

      console.log('[speaker-id] exists:', exists, 'isDir:', isDir);

      if (isDir) {
        const contents = await ReactNativeBlobUtil.fs.ls(MODEL_PATH);
        console.log('[speaker-id] mlpackage contents:', contents);
      }

      // If a model was previously loaded, unload before loading a new one.
      // SpeakerId.loadModel is idempotent for the same path but doesn't handle
      // swaps cleanly across all runtimes.
      if (SpeakerId.isLoaded) {
        SpeakerId.unloadModel();
        setModelReady(false);
      }

      await SpeakerId.loadModel(MODEL_PATH);

      setModelReady(true);
      const sizeMb =
        Platform.OS === 'ios'
          ? 14 // .mlpackage is a directory; approx size for display
          : (await ReactNativeBlobUtil.fs.stat(MODEL_PATH)).size / 1e6;
      setStatus(
        `Model loaded (~${sizeMb.toFixed(0)} MB, ${
          Platform.OS === 'ios' ? 'Core ML → Neural Engine' : 'LiteRT → GPU/CPU'
        }).`
      );
    } catch (err: any) {
      setStatus(`Load failed: ${err?.message}`);
    } finally {
      setBusy(false);
    }
  };

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
    setStatus('Recording sample... speak for 5 seconds.');
    const pcm = await captureSample();
    if (!pcm) {
      setBusy(false);
      return;
    }

    setStatus('Embedding...');
    try {
      const embedding = await SpeakerId.embed(pcm, 16000);
      const name = `Voice ${enrolled.length + 1}`;
      setEnrolled((prev) => [...prev, { name, vector: embedding }]);
      setStatus(`Enrolled ${name}. Total enrolled: ${enrolled.length + 1}.`);
    } catch (err: any) {
      setStatus(`Embed failed: ${err?.message}`);
    } finally {
      setBusy(false);
    }
  };

  const handleMatch = async () => {
    if (!modelReady || enrolled.length === 0) return;
    setBusy(true);
    setStatus('Recording for matching...');
    const pcm = await captureSample();
    if (!pcm) {
      setBusy(false);
      return;
    }

    setStatus('Matching against enrolled voiceprints...');
    try {
      const embedding = await SpeakerId.embed(pcm, 16000);
      let best: { name: string; score: number } | null = null;
      for (const v of enrolled) {
        const score = SpeakerId.cosine(embedding, v.vector);
        if (!best || score > best.score) best = { name: v.name, score };
      }
      setLastMatch(best);
      if (best && best.score >= MATCH_THRESHOLD) {
        setStatus(`Match: ${best.name} — score ${best.score.toFixed(3)}`);
      } else {
        setStatus(
          `No match. Best score ${best?.score.toFixed(3) ?? '—'} vs threshold ${MATCH_THRESHOLD}.`
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
    setStatus('Cleared enrolled voiceprints.');
  };

  // Add near the top of the component
  const handleClearModel = async () => {
    setBusy(true);
    setStatus('Clearing cached model...');
    try {
      // Unload from runtime first
      if (SpeakerId.isLoaded) {
        SpeakerId.unloadModel();
      }
      setModelReady(false);

      // Delete the model file/directory
      if (await ReactNativeBlobUtil.fs.exists(MODEL_PATH)) {
        await ReactNativeBlobUtil.fs.unlink(MODEL_PATH);
      }
      // Clean up iOS zip leftover if any
      if (
        Platform.OS === 'ios' &&
        (await ReactNativeBlobUtil.fs.exists(DOWNLOAD_TMP_PATH))
      ) {
        await ReactNativeBlobUtil.fs.unlink(DOWNLOAD_TMP_PATH);
      }
      setEnrolled([]);
      setLastMatch(null);
      setStatus('Model cleared. Choose a source to load again.');
    } catch (err: any) {
      setStatus(`Clear failed: ${err?.message}`);
    } finally {
      setBusy(false);
    }
  };

  const backendLabel =
    Platform.OS === 'ios' ? 'Core ML · Neural Engine' : 'LiteRT · GPU / CPU';

  return (
    <View style={s.container}>
      <ScrollView
        contentContainerStyle={s.scroll}
        showsVerticalScrollIndicator={false}
      >
        <View style={s.header}>
          <View style={s.pillRow}>
            <View style={s.pill}>
              <View
                style={[
                  s.dot,
                  { backgroundColor: modelReady ? '#10b981' : '#64748b' },
                ]}
              />
              <Text style={s.pillText}>
                {modelReady ? 'Model ready' : 'No model'}
              </Text>
            </View>
            <View style={s.pill}>
              <Text style={s.pillText}>{backendLabel}</Text>
            </View>
          </View>

          <Text style={s.title}>Speaker ID</Text>
          <Text style={s.subtitle}>
            On-device ECAPA-TDNN speaker embeddings. Enroll voices, match new
            recordings against them — no cloud, no bandwidth.
          </Text>
        </View>

        <View style={s.section}>
          <Text style={s.sectionLabel}>Load the model</Text>
          <View style={s.loaderRow}>
            <LoaderCard
              styles={s}
              title="From bundle"
              subtitle="Ships with the app. Instant. Bigger binary."
              disabled={busy}
              onPress={() => handleLoad('bundle')}
            />
            <LoaderCard
              styles={s}
              title="From CDN"
              subtitle="~20 MB download on first launch. Cached forever."
              disabled={busy || modelReady}
              onPress={() => handleLoad('cdn')}
              highlight
            />
          </View>

          <View style={s.warnCard}>
            <Text style={s.warnText}>
              The CDN URL points at Gautham's personal R2 bucket. It might
              disappear or change. For anything you're actually shipping,
              convert your own model (see ECAPA-CONVERSION.md) and host it
              yourself.
            </Text>
          </View>

          {modelReady ? (
            <Pressable
              onPress={handleClearModel}
              disabled={busy}
              style={({ pressed }) => [
                s.clearButton,
                pressed && { opacity: 0.7 },
                busy && { opacity: 0.4 },
              ]}
            >
              <Text style={s.clearButtonText}>
                Reset model (delete cached file)
              </Text>
            </Pressable>
          ) : null}

          {downloadProgress > 0 && downloadProgress < 1 ? (
            <View style={s.progressWrap}>
              <View
                style={[
                  s.progressFill,
                  { width: `${downloadProgress * 100}%` },
                ]}
              />
              <Text style={s.progressText}>
                {Math.round(downloadProgress * 100)}%
              </Text>
            </View>
          ) : Platform.OS === 'android' &&
            busy &&
            status.includes('Download') ? (
            <View style={s.androidDownloadHint}>
              <ActivityIndicator
                size="small"
                color={isDark ? '#f1f5f9' : '#0f172a'}
              />
              <Text style={s.androidDownloadText}>
                Downloading via Android system. Check the notification shade for
                progress.
              </Text>
            </View>
          ) : null}
        </View>

        <View style={s.statusCard}>
          <View style={s.statusRow}>
            {busy ? (
              <ActivityIndicator
                size="small"
                color={isDark ? '#f1f5f9' : '#0f172a'}
              />
            ) : (
              <View style={s.statusDot} />
            )}
            <Text style={s.statusText}>{status}</Text>
          </View>
        </View>

        <View style={s.section}>
          <Text style={s.sectionLabel}>Test it</Text>
          <ActionButton
            styles={s}
            label="Enroll a voice"
            hint="Record 5 s, embed, cache as a voiceprint"
            onPress={handleEnroll}
            disabled={!modelReady || busy}
            primary
          />
          <ActionButton
            styles={s}
            label="Match against enrolled"
            hint="Record 5 s, embed, cosine against enrolled voiceprints"
            onPress={handleMatch}
            disabled={!modelReady || busy || enrolled.length === 0}
            primary
          />
          <ActionButton
            styles={s}
            label="Clear enrolled"
            onPress={handleClear}
            disabled={enrolled.length === 0}
          />
        </View>

        {enrolled.length > 0 ? (
          <View style={s.dataCard}>
            <Text style={s.dataCardLabel}>Enrolled voiceprints</Text>
            {enrolled.map((v, i) => (
              <View key={i} style={s.dataRow}>
                <View style={s.dataDot} />
                <Text style={s.dataText}>{v.name}</Text>
                <Text style={s.dataDim}>{v.vector.length}-d</Text>
              </View>
            ))}
          </View>
        ) : null}

        {lastMatch ? (
          <View
            style={[
              s.dataCard,
              {
                borderColor:
                  lastMatch.score >= MATCH_THRESHOLD
                    ? '#10b981'
                    : isDark
                      ? '#334155'
                      : '#e2e8f0',
              },
            ]}
          >
            <Text style={s.dataCardLabel}>Last match</Text>
            <View style={s.matchRow}>
              <Text style={s.matchName}>{lastMatch.name}</Text>
              <Text
                style={[
                  s.matchScore,
                  {
                    color:
                      lastMatch.score >= MATCH_THRESHOLD
                        ? '#10b981'
                        : '#f59e0b',
                  },
                ]}
              >
                {lastMatch.score.toFixed(3)}
              </Text>
            </View>
            <Text style={s.hint}>
              Cosine ∈ [-1, 1]. Same speaker (clean audio) lands 0.55-0.85. Same
              on phone audio 0.35-0.55. Different speakers &lt; 0.35.
            </Text>
          </View>
        ) : null}

        <View style={s.footer}>
          <Text style={s.footerText}>
            Backed by SpeechBrain's ECAPA-TDNN. Runs on{' '}
            {Platform.OS === 'ios'
              ? "Apple's Neural Engine"
              : "Google's LiteRT"}
            .
          </Text>
        </View>
      </ScrollView>
    </View>
  );
}

function LoaderCard({
  styles,
  title,
  subtitle,
  disabled,
  onPress,
  highlight,
}: any) {
  return (
    <Pressable
      onPress={onPress}
      disabled={disabled}
      style={({ pressed }) => [
        styles.loaderCard,
        highlight && styles.loaderCardHighlight,
        pressed && !disabled && { opacity: 0.7 },
        disabled && { opacity: 0.4 },
      ]}
    >
      <Text
        style={[styles.loaderTitle, highlight && styles.loaderTitleHighlight]}
      >
        {title}
      </Text>
      <Text
        style={[
          styles.loaderSubtitle,
          highlight && styles.loaderSubtitleHighlight,
        ]}
      >
        {subtitle}
      </Text>
    </Pressable>
  );
}

function ActionButton({
  styles,
  label,
  hint,
  onPress,
  disabled,
  primary,
}: any) {
  return (
    <Pressable
      onPress={onPress}
      disabled={disabled}
      style={({ pressed }) => [
        styles.actionButton,
        primary && styles.actionButtonPrimary,
        pressed && !disabled && { opacity: 0.75 },
        disabled && { opacity: 0.35 },
      ]}
    >
      <View style={styles.actionButtonInner}>
        <Text style={primary ? styles.actionLabelPrimary : styles.actionLabel}>
          {label}
        </Text>
        {hint ? (
          <Text style={primary ? styles.actionHintPrimary : styles.actionHint}>
            {hint}
          </Text>
        ) : null}
      </View>
    </Pressable>
  );
}

const createStyles = (isDark: boolean) => {
  const bg = isDark ? '#0a0f1a' : '#f8fafc';
  const surface = isDark ? '#0f172a' : '#ffffff';
  const border = isDark ? '#1e293b' : '#e2e8f0';
  const text = isDark ? '#f1f5f9' : '#0f172a';
  const textDim = isDark ? '#94a3b8' : '#64748b';
  const accent = '#3b82f6';
  const warnBg = isDark ? '#422006' : '#fef3c7';
  const warnBorder = isDark ? '#78350f' : '#fde68a';
  const warnText = isDark ? '#fbbf24' : '#78350f';

  return StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: bg,
      paddingTop: 60,
    },
    scroll: {
      paddingHorizontal: 20,
      paddingBottom: 40,
      gap: 20,
    },

    // Header
    header: { gap: 12 },
    pillRow: { flexDirection: 'row', gap: 8, flexWrap: 'wrap' },
    pill: {
      flexDirection: 'row',
      alignItems: 'center',
      gap: 6,
      backgroundColor: surface,
      borderWidth: 1,
      borderColor: border,
      paddingHorizontal: 10,
      paddingVertical: 5,
      borderRadius: 999,
    },
    dot: { width: 8, height: 8, borderRadius: 4 },
    pillText: { color: textDim, fontSize: 12, fontWeight: '600' as const },
    title: {
      color: text,
      fontSize: 32,
      fontWeight: '800' as const,
      letterSpacing: -0.5,
    },
    subtitle: { color: textDim, fontSize: 14, lineHeight: 20, marginTop: -2 },

    // Sections
    section: { gap: 10 },
    sectionLabel: {
      color: textDim,
      fontSize: 11,
      fontWeight: '700' as const,
      letterSpacing: 1,
      textTransform: 'uppercase' as const,
      marginBottom: 2,
    },

    // Loader cards
    loaderRow: { flexDirection: 'row', gap: 10 },
    loaderCard: {
      flex: 1,
      backgroundColor: surface,
      borderWidth: 1,
      borderColor: border,
      borderRadius: 14,
      padding: 14,
      gap: 6,
      minHeight: 96,
    },
    loaderCardHighlight: {
      backgroundColor: accent,
      borderColor: accent,
    },
    loaderTitle: { color: text, fontSize: 15, fontWeight: '700' as const },
    loaderTitleHighlight: { color: '#ffffff' },
    loaderSubtitle: { color: textDim, fontSize: 12, lineHeight: 17 },
    loaderSubtitleHighlight: { color: 'rgba(255,255,255,0.85)' },

    // Warning card
    warnCard: {
      backgroundColor: warnBg,
      borderWidth: 1,
      borderColor: warnBorder,
      borderRadius: 12,
      padding: 12,
    },
    warnText: {
      color: warnText,
      fontSize: 12,
      lineHeight: 17,
      fontWeight: '500' as const,
    },

    // Status card
    statusCard: {
      backgroundColor: surface,
      borderWidth: 1,
      borderColor: border,
      borderRadius: 14,
      padding: 14,
    },
    statusRow: { flexDirection: 'row', alignItems: 'flex-start', gap: 10 },
    statusDot: {
      width: 8,
      height: 8,
      borderRadius: 4,
      backgroundColor: textDim,
      marginTop: 6,
    },
    statusText: { flex: 1, color: text, fontSize: 13, lineHeight: 19 },

    // Action buttons
    actionButton: {
      backgroundColor: surface,
      borderWidth: 1,
      borderColor: border,
      borderRadius: 14,
      padding: 14,
    },
    actionButtonPrimary: {
      backgroundColor: accent,
      borderColor: accent,
    },
    actionButtonInner: { gap: 3 },
    actionLabel: { color: text, fontSize: 15, fontWeight: '600' as const },
    actionLabelPrimary: {
      color: '#ffffff',
      fontSize: 15,
      fontWeight: '700' as const,
    },
    actionHint: { color: textDim, fontSize: 12 },
    actionHintPrimary: { color: 'rgba(255,255,255,0.8)', fontSize: 12 },

    // Data cards
    dataCard: {
      backgroundColor: surface,
      borderWidth: 1,
      borderColor: border,
      borderRadius: 14,
      padding: 14,
      gap: 10,
    },
    dataCardLabel: {
      color: textDim,
      fontSize: 11,
      fontWeight: '700' as const,
      letterSpacing: 1,
      textTransform: 'uppercase' as const,
    },
    dataRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
    dataDot: { width: 6, height: 6, borderRadius: 3, backgroundColor: accent },
    dataText: {
      color: text,
      fontSize: 14,
      fontWeight: '500' as const,
      flex: 1,
    },
    dataDim: { color: textDim, fontSize: 12 },

    matchRow: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'baseline',
    },
    matchName: { color: text, fontSize: 20, fontWeight: '700' as const },
    matchScore: {
      fontSize: 22,
      fontWeight: '800' as const,
      fontVariant: ['tabular-nums'],
    },

    hint: {
      color: textDim,
      fontSize: 12,
      fontStyle: 'italic' as const,
      lineHeight: 17,
      marginTop: 4,
    },

    // Footer
    footer: { paddingVertical: 8 },
    footerText: {
      color: textDim,
      fontSize: 11,
      textAlign: 'center' as const,
      opacity: 0.7,
    },

    // Reset button
    clearButton: {
      backgroundColor: 'transparent',
      borderWidth: 1,
      borderColor: isDark ? '#7f1d1d' : '#fecaca',
      borderRadius: 12,
      paddingVertical: 10,
      paddingHorizontal: 14,
      alignItems: 'center',
    },
    clearButtonText: {
      color: isDark ? '#fca5a5' : '#b91c1c',
      fontSize: 13,
      fontWeight: '600' as const,
    },

    // Android download hint
    androidDownloadHint: {
      flexDirection: 'row',
      alignItems: 'center',
      gap: 10,
      backgroundColor: surface,
      borderWidth: 1,
      borderColor: border,
      borderRadius: 12,
      padding: 12,
    },
    androidDownloadText: {
      color: textDim,
      fontSize: 12,
      lineHeight: 17,
      flex: 1,
    },

    // Better progress bar with visible fill
    progressWrap: {
      height: 32,
      backgroundColor: surface,
      borderWidth: 1,
      borderColor: border,
      borderRadius: 999,
      overflow: 'hidden',
      justifyContent: 'center',
      position: 'relative',
    },
    progressFill: {
      position: 'absolute',
      left: 0,
      top: 0,
      bottom: 0,
      backgroundColor: accent,
      opacity: 0.9,
    },
    progressText: {
      textAlign: 'center' as const,
      color: text,
      fontSize: 12,
      fontWeight: '700' as const,
      zIndex: 1,
    },
  });
};
