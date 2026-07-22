import React, { useState, useRef, useEffect } from 'react';

interface AudioRecorderProps {
  onAudioReady: (blob: Blob) => void;
}

const AudioRecorder: React.FC<AudioRecorderProps> = ({ onAudioReady }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [duration, setDuration] = useState(0);
  const [hasRecording, setHasRecording] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
    };
  }, []);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      chunksRef.current = [];
      setDuration(0);

      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (e: BlobEvent) => {
        if (e.data && e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
        const url = URL.createObjectURL(blob);
        setAudioUrl(url);
        setHasRecording(true);
        onAudioReady(blob);
        stream.getTracks().forEach((t) => t.stop());
      };

      mediaRecorder.start(100);
      setIsRecording(true);

      intervalRef.current = setInterval(() => {
        setDuration((prev) => prev + 1);
      }, 1000);
    } catch (err) {
      console.error('Error accessing microphone:', err);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    }
  };

  const formatDuration = (secs: number) => {
    const m = Math.floor(secs / 60).toString().padStart(2, '0');
    const s = (secs % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
  };

  return (
    <div className="audio-recorder">
      <div className="recorder-controls">
        {!isRecording ? (
          <button
            onClick={startRecording}
            className="record-button"
            aria-label="Start recording"
          >
            Record
          </button>
        ) : (
          <button
            onClick={stopRecording}
            className="stop-button"
            aria-label="Stop recording"
          >
            Stop
          </button>
        )}
      </div>

      {isRecording && (
        <div className="recording-indicator" aria-live="polite">
          <span className="pulse-dot" aria-hidden="true" />
          <span>Recording... {formatDuration(duration)}</span>
        </div>
      )}

      {hasRecording && !isRecording && (
        <div className="playback-section">
          <p>Recording ready ({formatDuration(duration)})</p>
          {audioUrl && (
            <audio controls src={audioUrl} aria-label="Recorded audio playback" />
          )}
        </div>
      )}
    </div>
  );
};

export default AudioRecorder;
