import { useMemo } from 'react';

/**
 * Hook to calculate live IELTS essay word count and target progress.
 * IELTS Writing Task 2 standard minimum requirement is 250 words.
 */
export function useWordCount(text = '', targetWords = 250) {
  return useMemo(() => {
    const trimmed = text.trim();
    const words = trimmed ? trimmed.split(/\s+/).filter(Boolean) : [];
    const wordCount = words.length;

    const paragraphs = text
      .split(/\n\s*\n/)
      .map(p => p.trim())
      .filter(Boolean);

    const progressPercent = Math.min(100, Math.round((wordCount / targetWords) * 100));
    const isTargetReached = wordCount >= targetWords;
    const wordsRemaining = Math.max(0, targetWords - wordCount);

    return {
      wordCount,
      charCount: text.length,
      paragraphCount: paragraphs.length,
      progressPercent,
      isTargetReached,
      wordsRemaining,
      targetWords,
    };
  }, [text, targetWords]);
}
