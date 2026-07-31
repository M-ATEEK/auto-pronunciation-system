import axios from 'axios';
import { AnalysisResult, DetectorChoice } from '../types';

const BASE_URL = import.meta.env.VITE_API_URL || '';

export const analyzeAudio = async (
  audio: Blob,
  transcript: string,
  detector: DetectorChoice = 'baseline'
): Promise<AnalysisResult> => {
  const formData = new FormData();
  formData.append('audio', audio, 'recording.webm');
  formData.append('transcript', transcript);
  const response = await axios.post(`${BASE_URL}/api/analyze?detector=${detector}`, formData);
  return response.data;
};

export const checkHealth = async (): Promise<{ status: string; version: string }> => {
  const response = await axios.get(`${BASE_URL}/api/health`);
  return response.data;
};
