import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import AudioRecorder from './AudioRecorder';

describe('AudioRecorder', () => {
  it('renders record button', () => {
    render(<AudioRecorder onAudioReady={vi.fn()} />);
    expect(screen.getByRole('button', { name: /record/i })).toBeInTheDocument();
  });

  it('record button click calls getUserMedia', async () => {
    const onAudioReady = vi.fn();
    render(<AudioRecorder onAudioReady={onAudioReady} />);

    const recordButton = screen.getByRole('button', { name: /record/i });
    fireEvent.click(recordButton);

    await waitFor(() => {
      expect(navigator.mediaDevices.getUserMedia).toHaveBeenCalledWith({ audio: true });
    });
  });

  it('shows recording state when active', async () => {
    render(<AudioRecorder onAudioReady={vi.fn()} />);

    const recordButton = screen.getByRole('button', { name: /record/i });
    fireEvent.click(recordButton);

    await waitFor(() => {
      expect(screen.getByText(/recording/i)).toBeInTheDocument();
    });
  });

  it('shows stop button while recording', async () => {
    render(<AudioRecorder onAudioReady={vi.fn()} />);

    fireEvent.click(screen.getByRole('button', { name: /record/i }));

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /stop/i })).toBeInTheDocument();
    });
  });

  it('calls onAudioReady with a blob after stopping', async () => {
    const onAudioReady = vi.fn();
    render(<AudioRecorder onAudioReady={onAudioReady} />);

    fireEvent.click(screen.getByRole('button', { name: /record/i }));

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /stop/i })).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole('button', { name: /stop/i }));

    await waitFor(() => {
      expect(onAudioReady).toHaveBeenCalledWith(expect.any(Blob));
    });
  });
});
