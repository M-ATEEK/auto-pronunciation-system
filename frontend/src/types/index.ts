export interface PhoneVerdict {
  phone: string;
  word: string;
  prob_mispronounced: number;
  is_mispronounced: boolean;
}

export interface AnalysisResult {
  transcript: string;
  phones: PhoneVerdict[];
}
