import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import TranscriptInput from './TranscriptInput';

describe('TranscriptInput', () => {
  it('renders the textarea with the given value', () => {
    render(<TranscriptInput value="hello world" onChange={vi.fn()} />);
    expect(screen.getByLabelText(/sentence to pronounce/i)).toHaveValue('hello world');
  });

  it('calls onChange when the user types', () => {
    const onChange = vi.fn();
    render(<TranscriptInput value="" onChange={onChange} />);
    fireEvent.change(screen.getByLabelText(/sentence to pronounce/i), {
      target: { value: 'a new sentence' },
    });
    expect(onChange).toHaveBeenCalledWith('a new sentence');
  });

  it('disables the Listen button when there is no text', () => {
    render(<TranscriptInput value="" onChange={vi.fn()} />);
    expect(screen.getByRole('button', { name: /listen to correct pronunciation/i })).toBeDisabled();
  });

  it('enables the Listen button once text is entered', () => {
    render(<TranscriptInput value="hello" onChange={vi.fn()} />);
    expect(screen.getByRole('button', { name: /listen to correct pronunciation/i })).toBeEnabled();
  });

  it('does not throw when the Listen button is clicked', () => {
    render(<TranscriptInput value="hello" onChange={vi.fn()} />);
    expect(() =>
      fireEvent.click(screen.getByRole('button', { name: /listen to correct pronunciation/i }))
    ).not.toThrow();
  });

  it('disables the textarea when disabled prop is true', () => {
    render(<TranscriptInput value="hello" onChange={vi.fn()} disabled />);
    expect(screen.getByLabelText(/sentence to pronounce/i)).toBeDisabled();
  });
});
