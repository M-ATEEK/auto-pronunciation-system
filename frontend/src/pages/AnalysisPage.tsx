import React, { useState } from 'react';
import AudioRecorder from '../components/AudioRecorder/AudioRecorder';
import TranscriptInput from '../components/TranscriptInput/TranscriptInput';
import LoadingSpinner from '../components/LoadingSpinner/LoadingSpinner';
import CalibrationPanel from '../components/CalibrationPanel/CalibrationPanel';
import MouthDiagram from '../components/MouthDiagram/MouthDiagram';
import { analyzeAudio } from '../services/api';
import { AnalysisResult, DetectorChoice } from '../types';

/**
 * Learner identity for this page load.
 *
 * `?learner=P07` pins the identity, so a participant's profile survives a
 * refresh and accumulates across a study session. With no parameter a fresh
 * random id is minted per page load, so a casual visitor always starts from an
 * empty profile and two visitors never share one.
 *
 * The id reaches a filename on the server, so it is restricted to characters
 * that cannot escape a directory. The server sanitises it again regardless.
 */
const resolveLearnerId = (): string => {
  const fromUrl = new URLSearchParams(window.location.search).get('learner');
  const cleaned = (fromUrl ?? '').replace(/[^A-Za-z0-9_-]/g, '').slice(0, 64);
  if (cleaned) return cleaned;
  const rand =
    globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(36).slice(2)}`;
  return `learner-${rand}`;
};

const LEARNER_ID = resolveLearnerId();

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
      const analysisResult = await analyzeAudio(audioBlob, transcript, detector, LEARNER_ID);
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
          {result && result.no_speech && (
            <div className="mismatch-banner" role="alert">
              <strong>⚠ No speech detected in that recording.</strong>{' '}
              The microphone picked up only silence or background noise. Check your
              microphone permissions and input level, then record again.
            </div>
          )}
          {result && result.unknown_words.length > 0 && (
            <div className="mismatch-banner" role="alert">
              <strong>
                ⚠ No known pronunciation for:{' '}
                {result.unknown_words.map((w) => `"${w}"`).join(', ')}.
              </strong>{' '}
              {result.unknown_words.length === 1 ? 'That word is' : 'Those words are'} not in
              the pronunciation dictionary, so {result.unknown_words.length === 1 ? 'it was' : 'they were'}{' '}
              skipped rather than scored against guessed sounds. Try a different spelling.
            </div>
          )}
          {/* A recording that may not be the target sentence is scored against the
              wrong expectation, so the analysis is withheld rather than shown with
              a caveat the learner might read past. */}
          {result && !result.no_speech && result.match_warning && (
            <div className="mismatch-banner" role="alert">
              <strong>⚠ This recording may not match the sentence.</strong>{' '}
              {result.content_mismatch
                ? 'The audio does not sound like this sentence being read. '
                : result.articulation_rate !== null
                ? `The transcript has too many sounds to fit naturally in this recording (${result.articulation_rate} phones/sec). `
                : 'Only a small fraction of expected sounds were detected. '}
              Pronunciation feedback is not shown for this recording. Check that the
              sentence matches what you said, then record again.
            </div>
          )}
          {result && !result.no_speech && !result.match_warning && (
            <>
              <p className="feedback-summary">
                Analysed with <strong>{result.detector === 'neural' ? 'Neural (wav2vec2)' : 'Classical (Random Forest)'}</strong>:{' '}
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
                        {p.is_systematic && (
                          <span className="badge badge-flagged" style={{ marginLeft: 6 }}>
                            Systematic
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>

              {result.phones.some((p) => p.articulation) && (
                <div className="mouth-feedback">
                  <h3>How to fix each sound</h3>
                  {result.phones
                    .filter((p) => p.articulation)
                    .map((p, i) => (
                      <div className="mouth-row" key={i}>
                        <div className="mouth-row-head">
                          In &ldquo;{p.word}&rdquo; &mdash; /{p.phone}/
                        </div>
                        <div className="mouth-panels">
                          <MouthDiagram
                            shape={p.articulation!.target}
                            from={p.articulation!.heard}
                            label="Target"
                            highlight
                          />
                          {p.articulation!.heard && (
                            <MouthDiagram shape={p.articulation!.heard} label="What we heard" />
                          )}
                          {/* With a produced sound named we can say how to close the gap;
                              without one we still know the target, so teach that instead. */}
                          <div className="mouth-advice">
                            <h4>
                              {p.articulation!.heard
                                ? 'How to correct it'
                                : `How to make the /${p.phone}/ sound`}
                            </h4>
                            <ul className="mouth-diffs">
                              {(p.articulation!.heard && p.articulation!.differences.length > 0
                                ? p.articulation!.differences
                                : p.articulation!.instructions
                              ).map((d, j) => (
                                <li key={j}>{d}</li>
                              ))}
                            </ul>
                          </div>
                        </div>
                      </div>
                    ))}
                </div>
              )}
            </>
          )}
        </section>

        <CalibrationPanel learnerId={LEARNER_ID} />
      </main>
    </div>
  );
};

export default AnalysisPage;
