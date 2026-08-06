import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import MouthDiagram from './MouthDiagram';
import { Articulation } from '../../types';

const shape = (over: Partial<Articulation> = {}): Articulation => ({
  phone: 'TH',
  lip_rounding: 0.15,
  jaw_openness: 0.18,
  tongue_front: 1.0,
  tongue_height: 0.7,
  tongue_tip: 1.0,
  manner: 'fricative',
  voiced: false,
  tense: false,
  ...over,
});

describe('MouthDiagram', () => {
  it('renders an accessible diagram naming the phone', () => {
    render(<MouthDiagram shape={shape()} />);
    expect(screen.getByRole('img', { name: /TH/ })).toBeInTheDocument();
  });

  it('shows the phone label', () => {
    render(<MouthDiagram shape={shape({ phone: 'UW' })} />);
    expect(screen.getByText('/UW/')).toBeInTheDocument();
  });

  it('shows an optional caption label', () => {
    render(<MouthDiagram shape={shape()} label="Target" />);
    expect(screen.getByText('Target')).toBeInTheDocument();
  });

  it('draws a voicing indicator only for voiced phones', () => {
    const { container: voiced } = render(<MouthDiagram shape={shape({ voiced: true })} />);
    expect(voiced.querySelector('.mouth-voicing')).toBeInTheDocument();

    const { container: voiceless } = render(<MouthDiagram shape={shape({ voiced: false })} />);
    expect(voiceless.querySelector('.mouth-voicing')).not.toBeInTheDocument();
  });

  it('renders tongue and lips for every phone shape', () => {
    const { container } = render(<MouthDiagram shape={shape()} />);
    expect(container.querySelector('.mouth-tongue')).toBeInTheDocument();
    expect(container.querySelectorAll('.mouth-lip')).toHaveLength(2);
  });

  it('does not throw when animating from another shape', () => {
    expect(() =>
      render(<MouthDiagram shape={shape({ phone: 'TH' })} from={shape({ phone: 'S', tongue_tip: 0.65 })} />)
    ).not.toThrow();
  });

  it('produces different geometry for different phones', () => {
    const { container: a } = render(<MouthDiagram shape={shape({ phone: 'UW', lip_rounding: 0.95, tongue_front: 0.15 })} />);
    const { container: b } = render(<MouthDiagram shape={shape({ phone: 'IY', lip_rounding: 0.10, tongue_front: 1.0 })} />);
    const pathA = a.querySelector('.mouth-tongue')!.getAttribute('d');
    const pathB = b.querySelector('.mouth-tongue')!.getAttribute('d');
    expect(pathA).not.toEqual(pathB);
  });
});
