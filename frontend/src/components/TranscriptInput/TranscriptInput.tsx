import React from 'react';

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
        aria-describedby="transcript-help"
      />
      <p id="transcript-help" className="transcript-help">
        Enter the exact sentence you are recording so the system can compare your pronunciation.
      </p>
    </div>
  );
};

export default TranscriptInput;
