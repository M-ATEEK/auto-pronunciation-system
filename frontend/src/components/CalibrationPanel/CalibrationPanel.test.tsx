import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import CalibrationPanel from './CalibrationPanel';

vi.mock('../../services/api', () => ({
  calibrate: vi.fn(),
  getProfile: vi.fn().mockResolvedValue({ learner_id: 'x', phone_stats: {}, discovered_rules: {} }),
}));

import { calibrate } from '../../services/api';

async function runWith(rules: unknown[]) {
  (calibrate as ReturnType<typeof vi.fn>).mockResolvedValue({ rules });
  render(<CalibrationPanel learnerId="test" />);
  const recorders = screen.getAllByRole('button', { name: /start recording/i });
  for (const r of recorders) {
    await userEvent.click(r);
    const stop = screen.getAllByRole('button', { name: /stop recording/i })[0];
    await userEvent.click(stop);
  }
  await userEvent.click(screen.getByRole('button', { name: /run calibration/i }));
}

describe('CalibrationPanel results', () => {
  beforeEach(() => vi.clearAllMocks());

  it('describes a substitution in learner-facing words', async () => {
    await runWith([{ phone: 'DH', type: 'substitution', substituted_with: 'D',
                     support: 9, confidence: 0.75, mean_dtw: 0 }]);
    await waitFor(() => expect(screen.getByText('/DH/')).toBeInTheDocument());
    expect(screen.getByText('said as /D/')).toBeInTheDocument();
    expect(screen.getByText('75% of the time')).toBeInTheDocument();
  });

  it('grades evidence strength by how many examples support the rule', async () => {
    await runWith([
      { phone: 'DH', type: 'substitution', substituted_with: 'D', support: 9, confidence: 0.8, mean_dtw: 0 },
      { phone: 'R', type: 'substitution', substituted_with: 'L', support: 5, confidence: 0.7, mean_dtw: 0 },
      { phone: 'K', type: 'substitution', substituted_with: 'G', support: 2, confidence: 0.67, mean_dtw: 0 },
    ]);
    await waitFor(() => expect(screen.getByText('strong')).toBeInTheDocument());
    expect(screen.getByText('moderate')).toBeInTheDocument();
    expect(screen.getByText('weak')).toBeInTheDocument();
  });

  it('warns when a rule rests on only a few examples', async () => {
    await runWith([{ phone: 'K', type: 'substitution', substituted_with: 'G',
                     support: 2, confidence: 0.67, mean_dtw: 0 }]);
    await waitFor(() =>
      expect(screen.getByText(/rest on only a few examples/i)).toBeInTheDocument());
  });

  it('does not warn when every rule is well supported', async () => {
    await runWith([{ phone: 'DH', type: 'substitution', substituted_with: 'D',
                     support: 12, confidence: 0.9, mean_dtw: 0 }]);
    await waitFor(() => expect(screen.getByText('strong')).toBeInTheDocument());
    expect(screen.queryByText(/rest on only a few examples/i)).not.toBeInTheDocument();
  });

  it('says plainly when no pattern was found', async () => {
    await runWith([]);
    await waitFor(() =>
      expect(screen.getByText(/nothing was wrong often enough/i)).toBeInTheDocument());
  });

  it('does not reuse the profile history wording', async () => {
    await runWith([{ phone: 'DH', type: 'substitution', substituted_with: 'D',
                     support: 9, confidence: 0.8, mean_dtw: 0 }]);
    await waitFor(() => expect(screen.getByText('/DH/')).toBeInTheDocument());
    expect(screen.queryByText('Recurring')).not.toBeInTheDocument();
    expect(screen.queryByText('Support')).not.toBeInTheDocument();
  });
});
