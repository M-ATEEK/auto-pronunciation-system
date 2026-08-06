import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import WaveformView from './WaveformView';

const sampleMfcc = [
  [1.0, 0.5, -0.3, 0.8, 0.1, -0.5, 0.4, 0.2, -0.1, 0.7, -0.2, 0.3, 0.6],
  [0.9, 0.4, -0.2, 0.7, 0.2, -0.4, 0.3, 0.1, 0.0, 0.6, -0.3, 0.2, 0.5],
];

describe('WaveformView', () => {
  it('renders a canvas element', () => {
    render(
      <WaveformView
        learnerMfcc={sampleMfcc}
        nativeMfcc={sampleMfcc}
      />
    );
    const canvas = document.querySelector('canvas');
    expect(canvas).toBeInTheDocument();
  });

  it('renders without errors when given MFCC data', () => {
    expect(() =>
      render(
        <WaveformView
          learnerMfcc={sampleMfcc}
          nativeMfcc={sampleMfcc}
          dtwPath={[[0, 0], [0, 1], [1, 1]]}
        />
      )
    ).not.toThrow();
  });

  it('renders canvas with accessible label', () => {
    render(
      <WaveformView
        learnerMfcc={sampleMfcc}
        nativeMfcc={sampleMfcc}
      />
    );
    expect(screen.getByRole('img')).toBeInTheDocument();
  });

  it('renders without crashing when MFCC arrays are empty', () => {
    expect(() =>
      render(<WaveformView learnerMfcc={[]} nativeMfcc={[]} />)
    ).not.toThrow();
  });
});
