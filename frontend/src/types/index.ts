export type DetectorChoice = 'baseline' | 'neural';

export interface WaveformData {
  learner_mfcc: number[][];
  native_mfcc: number[][];
  dtw_path: number[][];
}

export interface PhoneVerdict {
  phone: string;
  word: string;
  prob_mispronounced: number;
  is_mispronounced: boolean;
  substitution: string | null;
  dtw_distance: number | null;
  is_systematic: boolean;
  text_hint: string | null;
  learner_audio_uri: string | null;
  waveform_data: WaveformData | null;
}

export interface AnalysisResult {
  transcript: string;
  learner_id: string;
  detector: DetectorChoice;
  phones: PhoneVerdict[];
  match_confidence: number;
  match_warning: boolean;
  articulation_rate: number | null;
  content_ratio: number | null;
  content_mismatch: boolean;
  unknown_words: string[];
  no_speech: boolean;
}

export interface DiscoveredRule {
  phone: string;
  type: 'substitution' | 'distortion';
  substituted_with: string | null;
  support: number;
  confidence: number;
  mean_dtw: number;
}

export interface CalibrationResult {
  learner_id: string;
  rules: DiscoveredRule[];
}

export interface PhoneStat {
  count: number;
  mean_dtw: number;
  is_systematic: boolean;
}

export interface LearnerProfileData {
  learner_id: string;
  phone_stats: Record<string, PhoneStat>;
  discovered_rules: Record<string, DiscoveredRule>;
}
