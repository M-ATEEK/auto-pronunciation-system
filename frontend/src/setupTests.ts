import '@testing-library/jest-dom';
import { vi } from 'vitest';

// Mock URL.createObjectURL / revokeObjectURL (not available in jsdom)
(globalThis as typeof globalThis & { URL: typeof URL }).URL.createObjectURL = vi.fn(() => 'blob:mock-url');
(globalThis as typeof globalThis & { URL: typeof URL }).URL.revokeObjectURL = vi.fn();

// Mock MediaRecorder for AudioRecorder tests
(globalThis as Record<string, unknown>).MediaRecorder = class MockMediaRecorder {
  state = 'inactive';
  ondataavailable: ((e: { data: Blob }) => void) | null = null;
  onstop: (() => void) | null = null;
  start() {
    this.state = 'recording';
  }
  stop() {
    this.state = 'inactive';
    if (this.ondataavailable) {
      this.ondataavailable({ data: new Blob(['audio'], { type: 'audio/webm' }) });
    }
    if (this.onstop) {
      this.onstop();
    }
  }
  static isTypeSupported() {
    return true;
  }
} as unknown as typeof MediaRecorder;

// Mock getUserMedia
Object.defineProperty(globalThis.navigator, 'mediaDevices', {
  value: {
    getUserMedia: vi.fn().mockResolvedValue({
      getTracks: () => [{ stop: vi.fn() }],
    }),
  },
  writable: true,
});

// Mock canvas for WaveformView tests
HTMLCanvasElement.prototype.getContext = vi.fn().mockReturnValue({
  fillRect: vi.fn(),
  clearRect: vi.fn(),
  fillText: vi.fn(),
  strokeRect: vi.fn(),
  drawImage: vi.fn(),
  putImageData: vi.fn(),
  getImageData: vi.fn(() => ({ data: new Array(4) })),
  beginPath: vi.fn(),
  closePath: vi.fn(),
  stroke: vi.fn(),
  fill: vi.fn(),
  moveTo: vi.fn(),
  lineTo: vi.fn(),
  arc: vi.fn(),
  scale: vi.fn(),
  rotate: vi.fn(),
  translate: vi.fn(),
  save: vi.fn(),
  restore: vi.fn(),
  setLineDash: vi.fn(),
  measureText: vi.fn(() => ({ width: 50 })),
  fillStyle: '',
  strokeStyle: '',
  lineWidth: 0,
  lineJoin: '',
  font: '',
  canvas: { width: 600, height: 150 },
}) as unknown as typeof HTMLCanvasElement.prototype.getContext;

// jsdom does not implement HTMLMediaElement playback -- mock play() so
// TranscriptInput/FeedbackCard's `new Audio().play().catch(...)` calls resolve.
Object.defineProperty(window.HTMLMediaElement.prototype, 'play', {
  configurable: true,
  value: vi.fn().mockResolvedValue(undefined),
});
Object.defineProperty(window.HTMLMediaElement.prototype, 'pause', {
  configurable: true,
  value: vi.fn(),
});
