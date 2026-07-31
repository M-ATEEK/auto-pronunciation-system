export type DetectorChoice = 'baseline' | 'neural';

export interface PhoneVerdict {
  phone: string;
  word: string;
  prob_mispronounced: number;
  is_mispronounced: boolean;
  substitution: string | null;
  dtw_distance: number | null;
}

export interface AnalysisResult {
  transcript: string;
  learner_id: string;
  detector: DetectorChoice;
  phones: PhoneVerdict[];
}
