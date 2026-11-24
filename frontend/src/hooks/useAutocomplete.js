import { useState, useEffect, useRef, useCallback } from 'react';
import { streamAutocomplete } from '../services/api';

/**
 * Custom hook managing debounced real-time autocomplete suggestions.
 * Controls generation streaming, Tab to accept, and Esc to dismiss.
 */
export function useAutocomplete({
  question,
  draft,
  enabled = true,
  debounceMs = 500,
  maxTokens = 30,
}) {
  const [suggestion, setSuggestion] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const abortControllerRef = useRef(null);
  const debounceTimerRef = useRef(null);

  // Clear current suggestion and cancel any active stream
  const clearSuggestion = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setSuggestion('');
    setIsStreaming(false);
  }, []);

  // Request new autocomplete stream from backend
  const fetchSuggestion = useCallback(async () => {
    // Only trigger if enabled, question has text, and draft is at least 3 words
    if (!enabled || !question.trim() || draft.trim().split(/\s+/).length < 2) {
      clearSuggestion();
      return;
    }

    // Cancel existing stream before starting a new one
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    const controller = new AbortController();
    abortControllerRef.current = controller;

    setSuggestion('');
    setIsStreaming(true);

    await streamAutocomplete({
      question,
      draft,
      maxTokens,
      signal: controller.signal,
      onToken: (token) => {
        setSuggestion((prev) => prev + token);
      },
      onDone: () => {
        setIsStreaming(false);
      },
      onError: () => {
        setIsStreaming(false);
      },
    });
  }, [enabled, question, draft, maxTokens, clearSuggestion]);

  // Debounce typing: trigger suggestion fetch after user stops typing
  useEffect(() => {
    clearTimeout(debounceTimerRef.current);

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }

    setSuggestion('');
    setIsStreaming(false);

    debounceTimerRef.current = setTimeout(() => {
      fetchSuggestion();
    }, debounceMs);

    return () => {
      clearTimeout(debounceTimerRef.current);
    };
  }, [draft, question, debounceMs, fetchSuggestion]);

  // Accept current suggestion
  const acceptSuggestion = useCallback(() => {
    if (!suggestion) return draft;
    const acceptedText = draft + suggestion;
    clearSuggestion();
    return acceptedText;
  }, [draft, suggestion, clearSuggestion]);

  return {
    suggestion,
    isStreaming,
    clearSuggestion,
    acceptSuggestion,
    triggerManual: fetchSuggestion,
  };
}
