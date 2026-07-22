import React from 'react';

interface LoadingSpinnerProps {
  message?: string;
}

const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  message = 'Analyzing...',
}) => {
  return (
    <div className="loading-spinner" role="status" aria-live="polite">
      <div className="spinner-circle" aria-hidden="true" />
      <p className="spinner-message">{message}</p>
    </div>
  );
};

export default LoadingSpinner;
