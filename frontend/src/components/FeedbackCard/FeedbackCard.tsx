import React from 'react';
import { PhoneVerdict } from '../../types';

interface FeedbackCardProps {
  verdict: PhoneVerdict;
}

const confidenceLabel = (score: number): string => {
  if (score >= 0.75) return 'Needs a lot of work';
  if (score >= 0.55) return 'Needs some work';
  return 'Slightly off';
};

const FeedbackCard: React.FC<FeedbackCardProps> = ({ verdict }) => {
  const scorePercent = Math.round(verdict.prob_mispronounced * 100);

  const playLearnerAudio = () => {
    if (verdict.learner_audio_uri) {
      new Audio(verdict.learner_audio_uri).play().catch(() => {});
    }
  };

  const playReferenceAudio = () => {
    const say = verdict.word || verdict.phone;
    new Audio(`/api/reference?text=${encodeURIComponent(say)}`).play().catch(() => {});
  };

  return (
    <div className={`feedback-card ${verdict.is_systematic ? 'feedback-card--systematic' : ''}`}>
      <div className="feedback-card-header">
        {verdict.word && <span className="feedback-card-word">In &ldquo;{verdict.word}&rdquo;</span>}
        <div className="feedback-card-phones">
          <span className="feedback-card-phone">/{verdict.phone}/</span>
          {verdict.substitution ? (
            <>
              <span className="feedback-card-arrow">→</span>
              <span className="feedback-card-phone feedback-card-phone--sub">/{verdict.substitution}/</span>
              <span className="feedback-card-note">
                (you said /{verdict.substitution}/ instead of /{verdict.phone}/)
              </span>
            </>
          ) : (
            <span className="feedback-card-note">— mispronounced</span>
          )}
        </div>
      </div>

      <div className="feedback-card-score">
        <div className="feedback-card-score-labels">
          <span>{confidenceLabel(verdict.prob_mispronounced)}</span>
          <span>{scorePercent}% different from native</span>
        </div>
        <div className="feedback-card-bar">
          <div className="feedback-card-bar-fill" style={{ width: `${scorePercent}%` }} />
        </div>
      </div>

      {verdict.text_hint && (
        <div className="feedback-card-hint">
          <strong>How to fix it: </strong>
          {verdict.text_hint}
        </div>
      )}

      <div className="feedback-card-audio">
        {verdict.learner_audio_uri && (
          <button type="button" className="audio-button audio-button--learner" onClick={playLearnerAudio}>
            <span aria-hidden="true"></span> Hear your version
          </button>
        )}
        <button type="button" className="audio-button audio-button--reference" onClick={playReferenceAudio}>
          <span aria-hidden="true"></span> Hear correct pronunciation
        </button>
      </div>

      {verdict.is_systematic && (
        <div className="feedback-card-systematic-note">
          ⚠ This is a recurring error across your sessions — focus practice here.
        </div>
      )}
    </div>
  );
};

export default FeedbackCard;
