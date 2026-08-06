export type DetectorChoice = 'baseline' | 'neural';

export interface Articulation {
  phone: string;
  lip_rounding: number;
  jaw_openness: number;
  tongue_front: number;
  tongue_height: number;
  tongue_tip: number;
  manner: string;
  voiced: boolean;
  tense: boolean;
  place: string;
}

export interface ArticulationBlock {
  target: Articulation;
  /** Null when the detector cannot name what was produced (classical, or an omission). */
  heard: Articulation | null;
  /** How to move from the produced sound to the target; empty when `heard` is null. */
  differences: string[];
  /** How to produce the target sound. Derived from the text, so always available. */
  instructions: string[];
}

export interface PhoneVerdict {
  phone: string;
  word: string;
  prob_mispronounced: number;
  is_mispronounced: boolean;
  substitution: string | null;
  dtw_distance: number | null;
  is_systematic: boolean;
  articulation: ArticulationBlock | null;
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
