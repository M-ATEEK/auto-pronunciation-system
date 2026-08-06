import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import FeedbackCard from './FeedbackCard';
import { PhoneVerdict } from '../../types';

const baseVerdict: PhoneVerdict = {
  phone: 'TH',
  word: 'think',
  prob_mispronounced: 0.85,
  is_mispronounced: true,
  substitution: null,
  dtw_distance: 42.5,
  is_systematic: false,
  text_hint: 'Voiceless dental fricative /th/: tongue between teeth.',
  learner_audio_uri: 'data:audio/wav;base64,AAAA',
  waveform_data: null,
};

describe('FeedbackCard', () => {
  it('renders the expected phone', () => {
    render(<FeedbackCard verdict={baseVerdict} />);
    expect(screen.getByText('/TH/')).toBeInTheDocument();
  });

  it('renders the source word', () => {
    render(<FeedbackCard verdict={baseVerdict} />);
    expect(screen.getByText(/think/i)).toBeInTheDocument();
  });

  it('renders the articulatory hint', () => {
    render(<FeedbackCard verdict={baseVerdict} />);
    expect(screen.getByText(/tongue between teeth/i)).toBeInTheDocument();
  });

  it('shows the substitution when present', () => {
    render(<FeedbackCard verdict={{ ...baseVerdict, substitution: 'S' }} />);
    expect(screen.getByText('/S/')).toBeInTheDocument();
    expect(screen.getByText(/instead of \/TH\//i)).toBeInTheDocument();
  });

  it('shows a plain "mispronounced" note when there is no substitution', () => {
    render(<FeedbackCard verdict={baseVerdict} />);
    expect(screen.getByText(/mispronounced/i)).toBeInTheDocument();
  });

  it('shows the "hear your version" button only when learner audio exists', () => {
    render(<FeedbackCard verdict={{ ...baseVerdict, learner_audio_uri: null }} />);
    expect(screen.queryByText(/hear your version/i)).not.toBeInTheDocument();
  });

  it('always shows the "hear correct pronunciation" button', () => {
    render(<FeedbackCard verdict={baseVerdict} />);
    expect(screen.getByText(/hear correct pronunciation/i)).toBeInTheDocument();
  });

  it('shows a systematic-error note when is_systematic is true', () => {
    render(<FeedbackCard verdict={{ ...baseVerdict, is_systematic: true }} />);
    expect(screen.getByText(/recurring error/i)).toBeInTheDocument();
  });

  it('does not render a waveform when waveform_data is null', () => {
    render(<FeedbackCard verdict={baseVerdict} />);
    expect(document.querySelector('canvas')).not.toBeInTheDocument();
  });

  it('renders a waveform when waveform_data is present', () => {
    const withWaveform: PhoneVerdict = {
      ...baseVerdict,
      waveform_data: {
        learner_mfcc: [[1, 2, 3]],
        native_mfcc: [[1, 2, 3]],
        dtw_path: [[0, 0]],
      },
    };
    render(<FeedbackCard verdict={withWaveform} />);
    expect(document.querySelector('canvas')).toBeInTheDocument();
  });
});
