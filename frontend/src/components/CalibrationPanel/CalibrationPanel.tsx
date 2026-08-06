import React, { useState } from 'react';
import AudioRecorder from '../AudioRecorder/AudioRecorder';
import ProfileDashboard from '../ProfileDashboard/ProfileDashboard';
import { calibrate, getProfile } from '../../services/api';
import { LearnerProfileData } from '../../types';

const CALIBRATION_SENTENCES = [
  'the quick brown fox jumps over the lazy dog',
  'she sells seashells by the seashore',
];

interface CalibrationPanelProps {
  learnerId: string;
}

const CalibrationPanel: React.FC<CalibrationPanelProps> = ({ learnerId }) => {
  const [recordings, setRecordings] = useState<(Blob | null)[]>(
    CALIBRATION_SENTENCES.map(() => null)
  );
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [rules, setRules] = useState<LearnerProfileData['discovered_rules'] | null>(null);
  const [profile, setProfile] = useState<LearnerProfileData | null>(null);
  const [isLoadingProfile, setIsLoadingProfile] = useState(false);

  const allRecorded = recordings.every((r) => r !== null);

  const handleRunCalibration = async () => {
    if (!allRecorded) {
      setError('Please record all calibration sentences first.');
      return;
    }
    setIsRunning(true);
    setError(null);
    try {
      const sessions = CALIBRATION_SENTENCES.map((transcript, i) => ({
        blob: recordings[i] as Blob,
        transcript,
      }));
      const result = await calibrate(sessions, learnerId);
      const byPhone: Record<string, (typeof result.rules)[number]> = {};
      result.rules.forEach((r) => { byPhone[r.phone] = r; });
      setRules(byPhone);
    } catch (err) {
      setError('Calibration failed. Please check your connection and try again.');
      console.error(err);
    } finally {
      setIsRunning(false);
    }
  };

  const handleViewProfile = async () => {
    setIsLoadingProfile(true);
    setError(null);
    try {
      const data = await getProfile(learnerId);
      setProfile(data);
    } catch (err) {
      setError('Could not load profile.');
      console.error(err);
    } finally {
      setIsLoadingProfile(false);
    }
  };

  return (
    <section className="calibration-section">
      <h2>Calibrate Your Profile</h2>
      <p className="calibration-help">
        Record yourself reading each sentence below. This helps the system discover any sounds
        you consistently struggle with, so future feedback can be more targeted.
      </p>

      {CALIBRATION_SENTENCES.map((sentence, i) => (
        <div className="calibration-sentence" key={i}>
          <p className="calibration-sentence-text">
            {i + 1}. {sentence}
            {recordings[i] && <span className="badge badge-ok" style={{ marginLeft: 8 }}>Recorded</span>}
          </p>
          <AudioRecorder
            onAudioReady={(blob) =>
              setRecordings((prev) => prev.map((r, idx) => (idx === i ? blob : r)))
            }
          />
        </div>
      ))}

      {error && (
        <div className="error-message" role="alert">
          {error}
        </div>
      )}

      <button
        type="button"
        className="submit-button"
        disabled={isRunning || !allRecorded}
        onClick={handleRunCalibration}
      >
        {isRunning ? 'Calibrating...' : 'Run Calibration'}
      </button>

      {rules && (
        <div className="calibration-results">
          <h3>Discovered Patterns</h3>
          {Object.keys(rules).length === 0 ? (
            <p className="feedback-placeholder">No systematic error patterns discovered.</p>
          ) : (
            <table className="phone-results">
              <thead>
                <tr>
                  <th>Phone</th>
                  <th>Type</th>
                  <th>Substituted With</th>
                  <th>Support</th>
                  <th>Confidence</th>
                </tr>
              </thead>
              <tbody>
                {Object.values(rules).map((r) => (
                  <tr key={r.phone} className="row-flagged">
                    <td>{r.phone}</td>
                    <td>{r.type}</td>
                    <td>{r.substituted_with ?? '—'}</td>
                    <td>{r.support}</td>
                    <td>{r.confidence.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      <button
        type="button"
        className="submit-button"
        style={{ marginTop: 12, background: '#3182ce' }}
        onClick={handleViewProfile}
        disabled={isLoadingProfile}
      >
        {isLoadingProfile ? 'Loading...' : 'View My Profile'}
      </button>

      {profile && (
        <div className="calibration-results">
          <h3>Learner Profile — {profile.learner_id}</h3>
          <ProfileDashboard profile={profile} />
        </div>
      )}
    </section>
  );
};

export default CalibrationPanel;
