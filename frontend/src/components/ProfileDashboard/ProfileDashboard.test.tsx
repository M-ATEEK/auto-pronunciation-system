import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import ProfileDashboard from './ProfileDashboard';
import { LearnerProfileData } from '../../types';

const sampleProfile: LearnerProfileData = {
  learner_id: 'test-learner',
  phone_stats: {
    TH: { count: 5, mean_dtw: 42.3, is_systematic: true },
    S: { count: 2, mean_dtw: 12.1, is_systematic: false },
  },
  discovered_rules: {},
};

describe('ProfileDashboard', () => {
  it('shows an empty-state message when there is no profile', () => {
    render(<ProfileDashboard profile={null} />);
    expect(screen.getByText(/no analysis history yet/i)).toBeInTheDocument();
  });

  it('shows an empty-state message when phone_stats is empty', () => {
    render(<ProfileDashboard profile={{ learner_id: 'x', phone_stats: {}, discovered_rules: {} }} />);
    expect(screen.getByText(/no analysis history yet/i)).toBeInTheDocument();
  });

  it('renders one entry per tracked phone', () => {
    render(<ProfileDashboard profile={sampleProfile} />);
    expect(screen.getAllByTestId('profile-entry')).toHaveLength(2);
    expect(screen.getByText('/TH/')).toBeInTheDocument();
    expect(screen.getByText('/S/')).toBeInTheDocument();
  });

  it('marks systematic phones with a badge', () => {
    render(<ProfileDashboard profile={sampleProfile} />);
    expect(screen.getByText('Systematic')).toBeInTheDocument();
  });

  it('sorts entries by count descending (highest count first)', () => {
    render(<ProfileDashboard profile={sampleProfile} />);
    const phones = screen.getAllByText(/^\/[A-Z]+\/$/).map((el) => el.textContent);
    expect(phones[0]).toBe('/TH/'); // count 5 > count 2
  });

  it('shows the summary count of tracked sounds', () => {
    render(<ProfileDashboard profile={sampleProfile} />);
    expect(screen.getByText(/2 sounds tracked/i)).toBeInTheDocument();
  });
});
