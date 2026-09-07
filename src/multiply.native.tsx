import { NitroModules } from 'react-native-nitro-modules';
import type { NitroSpeakerId } from './NitroSpeakerId.nitro';

const NitroSpeakerIdHybridObject =
  NitroModules.createHybridObject<NitroSpeakerId>('NitroSpeakerId');

export function multiply(a: number, b: number): number {
  return NitroSpeakerIdHybridObject.multiply(a, b);
}
