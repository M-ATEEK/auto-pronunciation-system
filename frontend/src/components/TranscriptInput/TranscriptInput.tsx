import React, { useRef, useState } from 'react';

interface TranscriptInputProps {
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
}

const TranscriptInput: React.FC<TranscriptInputProps> = ({
  value,
  onChange,
  disabled = false,
}) => {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [playing, setPlaying] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canPlay = value.trim().length > 0;

  const handleListen = () => {
    if (!canPlay) return;
    setError(null);
    const url = `/api/reference?text=${encodeURIComponent(value.trim())}`;
    if (!audioRef.current) {
      audioRef.current = new Audio();
      audioRef.current.onended = () => setPlaying(false);
      audioRef.current.onerror = () => {
        setPlaying(false);
        setError('Could not load reference audio.');
      };
    }
    audioRef.current.src = url;
    audioRef.current.currentTime = 0;
    setPlaying(true);
    audioRef.current.play().catch(() => {
      setPlaying(false);
      setError('Could not play reference audio.');
    });
  };

  return (
    <div className="transcript-input">
      <label htmlFor="transcript-field" className="transcript-label">
        Sentence to pronounce
      </label>
      <textarea
        id="transcript-field"
        className="transcript-textarea"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        placeholder="Type the sentence you will pronounce..."
        rows={3}
      />
      <div className="listen-row">
        <button
          type="button"
          className="listen-button"
          onClick={handleListen}
          disabled={!canPlay || playing}
          title="Hear the correct native pronunciation"
        >
          <span aria-hidden="true">{playing ? '' : ''}</span>{' '}
          {playing ? 'Playing…' : 'Listen to correct pronunciation'}
        </button>
        {error && <span className="error-message">{error}</span>}
      </div>
    </div>
  );
};

export default TranscriptInput;
