import React, { useState } from 'react';
import AudioRecorder from '../components/AudioRecorder/AudioRecorder';
import TranscriptInput from '../components/TranscriptInput/TranscriptInput';
import LoadingSpinner from '../components/LoadingSpinner/LoadingSpinner';
import { analyzeAudio } from '../services/api';
import { AnalysisResult, DetectorChoice } from '../types';

const AnalysisPage: React.FC = () => {
  const [transcript, setTranscript] = useState('');
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [detector, setDetector] = useState<DetectorChoice>('baseline');
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!audioBlob) {
      setError('Please record your pronunciation first.');
      return;
    }
    if (!transcript.trim()) {
      setError('Please enter the sentence you are pronouncing.');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const analysisResult = await analyzeAudio(audioBlob, transcript, detector);
      setResult(analysisResult);
    } catch (err) {
      setError('Failed to analyze pronunciation. Please check your connection and try again.');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="analysis-page">
      <header className="page-header">
        <h1>Pronunciation Feedback</h1>
      </header>

      <main className="page-main">
        <section className="analysis-section">
          <h2>Record Your Pronunciation</h2>

          <form onSubmit={handleSubmit} noValidate>
            <TranscriptInput
              value={transcript}
              onChange={setTranscript}
              disabled={isLoading}
            />

            <AudioRecorder onAudioReady={setAudioBlob} />

            <fieldset className="detector-toggle" disabled={isLoading}>
              <legend>Detection model</legend>
              <label>
                <input
                  type="radio"
                  name="detector"
                  value="baseline"
                  checked={detector === 'baseline'}
                  onChange={() => setDetector('baseline')}
                />
                Classical (Random Forest)
              </label>
              <label>
                <input
                  type="radio"
                  name="detector"
                  value="neural"
                  checked={detector === 'neural'}
                  onChange={() => setDetector('neural')}
                />
                Neural (wav2vec2)
              </label>
            </fieldset>

            {error && (
              <div className="error-message" role="alert">
                {error}
              </div>
            )}

            <button
              type="submit"
              className="submit-button"
              disabled={isLoading || !audioBlob}
            >
              {isLoading ? 'Analyzing...' : 'Analyze Pronunciation'}
            </button>
          </form>
        </section>

        {isLoading && <LoadingSpinner message="Analyzing your pronunciation..." />}

        <section className="feedback-section">
          <h2>Feedback</h2>
          {!result && !isLoading && (
            <p className="feedback-placeholder">Record and analyze to see per-sound results here.</p>
          )}
          {result && (
            <>
              <p className="feedback-summary">
                Analysed with <strong>{result.detector === 'neural' ? 'Neural (wav2vec2)' : 'Classical (Random Forest)'}</strong> —{' '}
                {result.phones.filter((p) => !p.is_mispronounced).length} / {result.phones.length} sounds correct
              </p>
              <table className="phone-results">
                <thead>
                  <tr>
                    <th>Word</th>
                    <th>Phone</th>
                    <th>P(mispronounced)</th>
                    <th>Substitution</th>
                    <th>DTW distance</th>
                    <th>Verdict</th>
                  </tr>
                </thead>
                <tbody>
                  {result.phones.map((p, i) => (
                    <tr key={i} className={p.is_mispronounced ? 'row-flagged' : 'row-ok'}>
                      <td>{p.word}</td>
                      <td>{p.phone}</td>
                      <td>{p.prob_mispronounced.toFixed(2)}</td>
                      <td>{p.substitution ?? '—'}</td>
                      <td>{p.dtw_distance !== null ? p.dtw_distance.toFixed(2) : '—'}</td>
                      <td>
                        <span className={p.is_mispronounced ? 'badge badge-flagged' : 'badge badge-ok'}>
                          {p.is_mispronounced ? 'Mispronounced' : 'Correct'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
        </section>
      </main>
    </div>
  );
};

export default AnalysisPage;
