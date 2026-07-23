export interface PhoneVerdict {
  phone: string;
  word: string;
  prob_mispronounced: number;
  is_mispronounced: boolean;
  dtw_distance: number | null;
}

export interface AnalysisResult {
  transcript: string;
  learner_id: string;
  phones: PhoneVerdict[];
}
