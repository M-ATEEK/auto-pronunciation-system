import React, { useEffect, useRef } from 'react';

interface WaveformViewProps {
  learnerMfcc: number[][];
  nativeMfcc: number[][];
  dtwPath?: number[][];
}

const SPECTRO_H = 120;
const MATRIX_H = 120;
const MARGIN_L = 42;
const MARGIN_B = 24;
const TITLE_H = 18;
const GAP = 10;
const CANVAS_W = 600;
const CANVAS_H_FULL = SPECTRO_H + GAP + TITLE_H + MATRIX_H + MARGIN_B;
const CANVAS_H_SPECTRO = SPECTRO_H;

function drawSpectrogram(
  ctx: CanvasRenderingContext2D,
  data: number[][],
  xOffset: number,
  yOffset: number,
  panelW: number,
  panelH: number,
) {
  if (!data || data.length === 0) return;
  const frames = data.length;
  const coeffs = data[0].length;
  let minVal = Infinity, maxVal = -Infinity;
  for (const frame of data) for (const v of frame) {
    if (v < minVal) minVal = v;
    if (v > maxVal) maxVal = v;
  }
  const range = maxVal - minVal || 1;
  const cw = panelW / frames;
  const ch = panelH / coeffs;
  for (let f = 0; f < frames; f++) {
    for (let c = 0; c < coeffs; c++) {
      const n = (data[f][c] - minVal) / range;
      const v = Math.round(n * 255);
      ctx.fillStyle = `rgb(${v},${Math.round(v * 0.6)},${255 - v})`;
      ctx.fillRect(xOffset + f * cw, yOffset + c * ch, Math.max(1, cw), Math.max(1, ch));
    }
  }
}

const WaveformView: React.FC<WaveformViewProps> = ({ learnerMfcc, nativeMfcc, dtwPath }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const hasDtw = dtwPath != null && dtwPath.length > 0;
  const canvasH = hasDtw ? CANVAS_H_FULL : CANVAS_H_SPECTRO;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.height = canvasH;
    ctx.clearRect(0, 0, CANVAS_W, canvasH);

    // -- 1. Spectrograms (side-by-side) --
    const halfW = Math.floor(CANVAS_W / 2);
    const sGap = 6;
    const leftW = halfW - Math.floor(sGap / 2);
    const rightX = halfW + Math.ceil(sGap / 2);
    const rightW = CANVAS_W - rightX;

    ctx.fillStyle = '#eef2ff';
    ctx.fillRect(0, 0, leftW, SPECTRO_H);
    ctx.fillStyle = '#fff0f4';
    ctx.fillRect(rightX, 0, rightW, SPECTRO_H);

    drawSpectrogram(ctx, learnerMfcc, 0, 0, leftW, SPECTRO_H);
    drawSpectrogram(ctx, nativeMfcc, rightX, 0, rightW, SPECTRO_H);

    ctx.fillStyle = 'rgba(0,0,0,0.55)';
    ctx.font = 'bold 10px sans-serif';
    ctx.fillText('Your pronunciation', 4, 13);
    ctx.fillText('Native reference', rightX + 4, 13);

    ctx.fillStyle = 'rgba(0,0,0,0.35)';
    ctx.font = '9px sans-serif';
    if (learnerMfcc?.length) ctx.fillText(`${learnerMfcc.length} fr`, leftW - 28, SPECTRO_H - 3);
    if (nativeMfcc?.length) ctx.fillText(`${nativeMfcc.length} fr`, CANVAS_W - 26, SPECTRO_H - 3);

    // -- 2. DTW Alignment Matrix --
    if (!dtwPath || dtwPath.length === 0 || !learnerMfcc?.length || !nativeMfcc?.length) return;

    const T1 = learnerMfcc.length;   // learner frames  (y-axis)
    const T2 = nativeMfcc.length;    // native frames   (x-axis)

    const matrixY = SPECTRO_H + GAP + TITLE_H;
    const matrixX = MARGIN_L;
    const matrixW = CANVAS_W - MARGIN_L - 8;

    ctx.fillStyle = '#4a5568';
    ctx.font = 'bold 10px sans-serif';
    ctx.fillText('DTW Time Alignment  (how your speech frames map to native frames)', matrixX, SPECTRO_H + GAP + 13);

    ctx.fillStyle = '#f7fafc';
    ctx.fillRect(matrixX, matrixY, matrixW, MATRIX_H);
    ctx.strokeStyle = '#cbd5e0';
    ctx.lineWidth = 1;
    ctx.strokeRect(matrixX, matrixY, matrixW, MATRIX_H);

    ctx.strokeStyle = '#e2e8f0';
    ctx.lineWidth = 0.5;
    const gx = Math.max(1, Math.floor(T2 / 6));
    const gy = Math.max(1, Math.floor(T1 / 6));
    for (let j = 0; j <= T2; j += gx) {
      const x = matrixX + (j / T2) * matrixW;
      ctx.beginPath(); ctx.moveTo(x, matrixY); ctx.lineTo(x, matrixY + MATRIX_H); ctx.stroke();
    }
    for (let i = 0; i <= T1; i += gy) {
      const y = matrixY + (i / T1) * MATRIX_H;
      ctx.beginPath(); ctx.moveTo(matrixX, y); ctx.lineTo(matrixX + matrixW, y); ctx.stroke();
    }

    // Diagonal reference -- perfect 1:1 timing
    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = '#a0aec0';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(matrixX, matrixY);
    ctx.lineTo(matrixX + matrixW, matrixY + MATRIX_H);
    ctx.stroke();
    ctx.setLineDash([]);

    // DTW warp path -- colored by deviation from diagonal
    const safe = (n: number, d: number) => (d <= 1 ? 0 : n / (d - 1));
    ctx.lineWidth = 2.5;
    ctx.lineJoin = 'round';
    for (let k = 1; k < dtwPath.length; k++) {
      const [i0, j0] = dtwPath[k - 1];
      const [i1, j1] = dtwPath[k];
      const x0 = matrixX + safe(j0, T2) * matrixW;
      const y0 = matrixY + safe(i0, T1) * MATRIX_H;
      const x1 = matrixX + safe(j1, T2) * matrixW;
      const y1 = matrixY + safe(i1, T1) * MATRIX_H;

      const deviation = safe(i1, T1) - safe(j1, T2);
      let r: number, g: number, b: number;
      if (Math.abs(deviation) < 0.07) {
        r = 56; g = 161; b = 105;  // green -- well matched
      } else if (deviation > 0) {
        const t = Math.min(deviation / 0.35, 1);
        r = Math.round(56 + t * (229 - 56));
        g = Math.round(161 + t * (62 - 161));
        b = Math.round(105 + t * (62 - 105));   // -> red -- you were faster
      } else {
        const t = Math.min(-deviation / 0.35, 1);
        r = Math.round(56 + t * (49 - 56));
        g = Math.round(161 + t * (130 - 161));
        b = Math.round(105 + t * (206 - 105));  // -> blue -- you were slower
      }

      ctx.strokeStyle = `rgb(${r},${g},${b})`;
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
    }

    ctx.fillStyle = '#718096';
    ctx.font = '9px sans-serif';
    ctx.fillText(`Native frames (0 - ${T2})`, matrixX + matrixW / 2 - 42, matrixY + MATRIX_H + 16);

    ctx.save();
    ctx.translate(matrixX - 30, matrixY + MATRIX_H / 2 + 30);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(`Your frames (0 - ${T1})`, 0, 0);
    ctx.restore();

    ctx.fillStyle = '#a0aec0';
    ctx.font = '8px sans-serif';
    ctx.fillText('0', matrixX + 2, matrixY + MATRIX_H - 3);
    ctx.fillText(`${T2}`, matrixX + matrixW - 10, matrixY + MATRIX_H - 3);
    ctx.fillText(`${T1}`, matrixX + 2, matrixY + 9);

    const lx = matrixX + matrixW - 168;
    const ly = matrixY + 6;
    ctx.font = '9px sans-serif';
    [
      { color: '#38a169', label: 'matched' },
      { color: '#e53e3e', label: 'you faster' },
      { color: '#3182ce', label: 'you slower' },
    ].forEach(({ color, label }, idx) => {
      const x = lx + idx * 58;
      ctx.fillStyle = color;
      ctx.fillRect(x, ly, 8, 8);
      ctx.fillStyle = '#4a5568';
      ctx.fillText(label, x + 10, ly + 8);
    });

    ctx.setLineDash([4, 3]);
    ctx.strokeStyle = '#a0aec0';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(lx - 36, ly + 4); ctx.lineTo(lx - 20, ly + 4); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#718096';
    ctx.font = '9px sans-serif';
    ctx.fillText('perfect', lx - 18, ly + 8);

  }, [learnerMfcc, nativeMfcc, dtwPath, canvasH]);

  return (
    <div style={{ overflowX: 'auto' }}>
      <canvas
        ref={canvasRef}
        width={CANVAS_W}
        height={canvasH}
        style={{ width: '100%', height: 'auto', display: 'block' }}
        aria-label="MFCC spectrogram comparison and DTW time-alignment matrix"
        role="img"
      />
    </div>
  );
};

export default WaveformView;
