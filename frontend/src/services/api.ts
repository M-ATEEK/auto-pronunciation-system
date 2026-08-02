import axios from 'axios';
import { AnalysisResult, CalibrationResult, DetectorChoice, LearnerProfileData } from '../types';

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

export const calibrate = async (
  sessions: { blob: Blob; transcript: string }[],
  learnerId: string
): Promise<CalibrationResult> => {
  const formData = new FormData();
  sessions.forEach((s, i) => {
    formData.append('audio', s.blob, `calibration-${i}.webm`);
    formData.append('transcripts', s.transcript);
  });
  formData.append('learner_id', learnerId);
  const response = await axios.post(`${BASE_URL}/api/calibrate`, formData);
  return response.data;
};

export const getProfile = async (learnerId: string): Promise<LearnerProfileData> => {
  const response = await axios.get(`${BASE_URL}/api/profile/${learnerId}`);
  return response.data;
};
