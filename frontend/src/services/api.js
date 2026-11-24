/**
 * API service for IELTS Copilot backend.
 * Handles Server-Sent Events (SSE) streaming and health checks.
 */

export async function streamAutocomplete({
  question,
  draft,
  maxTokens = 30,
  temperature = 0.7,
  onToken,
  onDone,
  onError,
  signal,
}) {
  try {
    const response = await fetch('/api/autocomplete/stream', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        question,
        draft,
        max_tokens: maxTokens,
        temperature,
      }),
      signal,
    });

    if (!response.ok) {
      throw new Error(`Server returned HTTP ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || ''; // Keep incomplete trailing fragment

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || !trimmed.startsWith('data:')) continue;

        const jsonStr = trimmed.slice(5).trim();
        try {
          const payload = JSON.parse(jsonStr);
          if (payload.token) {
            onToken(payload.token);
          }
          if (payload.done) {
            onDone();
            return;
          }
        } catch (e) {
          console.warn('Failed to parse SSE JSON chunk:', jsonStr);
        }
      }
    }

    onDone();
  } catch (error) {
    if (error.name === 'AbortError') {
      // Stream aborted by user typing - clean exit
      return;
    }
    console.error('Streaming request error:', error);
    if (onError) onError(error);
  }
}

export async function checkBackendHealth() {
  try {
    const res = await fetch('/health');
    return await res.json();
  } catch {
    return { status: 'offline', model_loaded: false };
  }
}
