import React, { useEffect, useRef, useState } from 'react';
import { Articulation } from '../../types';

interface MouthDiagramProps {
  /** Articulation to draw. */
  shape: Articulation;
  /** If given, the diagram animates from this shape towards `shape`. */
  from?: Articulation | null;
  label?: string;
  highlight?: boolean;
}

/**
 * Side-profile cross-section of the mouth, facing right.
 *
 * Everything is driven by the articulatory parameters, so one component draws
 * every phone. When `from` is supplied the diagram loops an animation from
 * that shape to `shape`, which is what actually communicates the correction --
 * two static pictures leave the learner to spot the difference themselves.
 */

// Cavity extents in viewBox units.
const BACK_X = 42;
const LIP_X = 150;
const ROOF_Y = 66;
const FLOOR_Y = 140;

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

function blend(a: Articulation, b: Articulation, t: number) {
  return {
    lip_rounding: lerp(a.lip_rounding, b.lip_rounding, t),
    jaw_openness: lerp(a.jaw_openness, b.jaw_openness, t),
    tongue_front: lerp(a.tongue_front, b.tongue_front, t),
    tongue_height: lerp(a.tongue_height, b.tongue_height, t),
    tongue_tip: lerp(a.tongue_tip, b.tongue_tip, t),
  };
}

function tonguePath(f: ReturnType<typeof blend>): string {
  // Body sits back-to-front and low-to-high with the two body parameters.
  const bodyX = lerp(62, 122, f.tongue_front);
  const bodyY = lerp(126, 82, f.tongue_height);
  // The tip is a separate articulator: it reaches towards the teeth/ridge.
  const tipX = lerp(118, 150, f.tongue_tip);
  const tipY = lerp(126, 84, f.tongue_tip);
  return [
    `M ${BACK_X} ${FLOOR_Y - 6}`,
    `Q ${bodyX} ${bodyY} ${tipX} ${tipY}`,
    `L ${tipX - 4} ${tipY + 15}`,
    `Q ${bodyX} ${bodyY + 26} ${BACK_X} ${FLOOR_Y + 4}`,
    'Z',
  ].join(' ');
}

// The lips meet at this height, so a closed mouth (/b/, /p/, /m/) actually
// closes. Anchoring upper and lower at fixed heights left a permanent gap.
const LIP_CENTER_Y = 97;

function lipPaths(f: ReturnType<typeof blend>) {
  // Rounding pushes the lips forward; openness separates them about the centre.
  const protrude = f.lip_rounding * 13;
  const gap = f.jaw_openness * 30;
  const x = LIP_X + protrude;
  const upperY = LIP_CENTER_Y - gap / 2;
  const lowerY = LIP_CENTER_Y + gap / 2;
  // Rounded lips curl inwards at the opening; spread lips stay flatter.
  const curl = f.lip_rounding * 7;
  return {
    upper: `M ${x - 30} 84 Q ${x + 2} ${upperY - curl} ${x + 9} ${upperY}`,
    lower: `M ${x - 30} 112 Q ${x + 2} ${lowerY + curl} ${x + 9} ${lowerY}`,
  };
}

const MouthDiagram: React.FC<MouthDiagramProps> = ({ shape, from, label, highlight }) => {
  const [t, setT] = useState(from ? 0 : 1);
  const raf = useRef<number | null>(null);

  useEffect(() => {
    if (!from) {
      setT(1);
      return;
    }
    let start: number | null = null;
    const DURATION = 1400;
    const HOLD = 600;
    const step = (now: number) => {
      if (start === null) start = now;
      const elapsed = (now - start) % (DURATION + HOLD);
      // Ease in/out, then hold at the target before looping.
      const raw = Math.min(elapsed / DURATION, 1);
      setT(raw < 1 ? 0.5 - Math.cos(Math.PI * raw) / 2 : 1);
      raf.current = requestAnimationFrame(step);
    };
    raf.current = requestAnimationFrame(step);
    return () => {
      if (raf.current !== null) cancelAnimationFrame(raf.current);
    };
  }, [from, shape]);

  const f = from ? blend(from, shape, t) : blend(shape, shape, 1);
  const lips = lipPaths(f);

  return (
    <figure className={`mouth-diagram ${highlight ? 'mouth-diagram--target' : ''}`}>
      <svg viewBox="0 0 200 170" role="img"
           aria-label={`Mouth position for ${shape.phone}`}>
        {/* head/cavity outline */}
        <path d={`M ${BACK_X - 20} 40 Q 110 26 ${LIP_X + 26} 62
                  L ${LIP_X + 26} 128 Q 110 164 ${BACK_X - 20} 150 Z`}
              className="mouth-face" />
        {/* hard palate (roof) */}
        <path d={`M ${BACK_X} ${ROOF_Y + 10} Q 100 ${ROOF_Y - 6} ${LIP_X - 14} ${ROOF_Y + 14}`}
              className="mouth-palate" />
        {/* teeth */}
        <rect x={LIP_X - 20} y={ROOF_Y + 14} width="9" height="12" rx="2" className="mouth-teeth" />
        <rect x={LIP_X - 20} y="108" width="9" height="12" rx="2" className="mouth-teeth" />
        {/* tongue */}
        <path d={tonguePath(f)} className="mouth-tongue" />
        {/* lips */}
        <path d={lips.upper} className="mouth-lip" />
        <path d={lips.lower} className="mouth-lip" />
        {/* voicing indicator at the larynx */}
        {shape.voiced && <circle cx={BACK_X - 12} cy="120" r="6" className="mouth-voicing" />}
      </svg>
      <figcaption>
        <span className="mouth-phone">/{shape.phone}/</span>
        {label && <span className="mouth-label">{label}</span>}
      </figcaption>
    </figure>
  );
};

export default MouthDiagram;
