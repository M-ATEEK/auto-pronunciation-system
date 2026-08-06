import React from 'react';
import { LearnerProfileData } from '../../types';

interface ProfileDashboardProps {
  profile: LearnerProfileData | null;
}

const ProfileDashboard: React.FC<ProfileDashboardProps> = ({ profile }) => {
  const entries = profile ? Object.entries(profile.phone_stats) : [];

  if (entries.length === 0) {
    return (
      <div className="profile-dashboard profile-dashboard--empty">
        <p className="feedback-placeholder">
          No analysis history yet. Analyze or calibrate a few recordings to build a profile.
        </p>
      </div>
    );
  }

  const maxCount = Math.max(...entries.map(([, stat]) => stat.count), 1);
  const sorted = [...entries].sort(([, a], [, b]) => b.count - a.count);

  return (
    <div className="profile-dashboard">
      <p className="profile-dashboard-summary">
        {entries.length} sounds tracked — {entries.filter(([, s]) => s.is_systematic).length} flagged as systematic
      </p>
      <div className="profile-dashboard-chart">
        {sorted.map(([phone, stat]) => {
          const barWidth = (stat.count / maxCount) * 100;
          return (
            <div
              key={phone}
              className={`profile-entry ${stat.is_systematic ? 'profile-entry--systematic' : ''}`}
              data-testid="profile-entry"
            >
              <span className="profile-entry-phone">/{phone}/</span>
              <div className="profile-entry-bar-track">
                <div
                  className="profile-entry-bar"
                  style={{ width: `${barWidth}%` }}
                  role="meter"
                  aria-valuenow={stat.count}
                  aria-valuemin={0}
                  aria-valuemax={maxCount}
                  aria-label={`/${phone}/: ${stat.count} occurrences${stat.is_systematic ? ' (systematic)' : ''}`}
                />
              </div>
              <span className="profile-entry-count">{stat.count}</span>
              <span className="profile-entry-dtw">avg DTW {stat.mean_dtw.toFixed(1)}</span>
              {stat.is_systematic && (
                <span className="badge badge-flagged" aria-label="systematic error">Systematic</span>
              )}
            </div>
          );
        })}
      </div>
      <div className="profile-dashboard-legend">
        <span className="legend-swatch legend-swatch--regular" /> Regular
        <span className="legend-swatch legend-swatch--systematic" /> Systematic
      </div>
    </div>
  );
};

export default ProfileDashboard;
