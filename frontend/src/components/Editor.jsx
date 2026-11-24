import React, { useRef, useEffect } from 'react';

/**
 * Editor component with synchronized Ghost-Text overlay.
 * Replicates the GitHub Copilot / Cursor inline autocomplete experience.
 */
export function Editor({
  draft,
  onDraftChange,
  suggestion,
  onAcceptSuggestion,
  onDismissSuggestion,
  isStreaming,
}) {
  const textareaRef = useRef(null);
  const overlayRef = useRef(null);

  // Sync scroll position between textarea and ghost overlay
  const handleScroll = () => {
    if (textareaRef.current && overlayRef.current) {
      overlayRef.current.scrollTop = textareaRef.current.scrollTop;
      overlayRef.current.scrollLeft = textareaRef.current.scrollLeft;
    }
  };

  // Keyboard shortcut listener: Tab to accept, Esc to dismiss
  const handleKeyDown = (e) => {
    if (e.key === 'Tab' && suggestion) {
      e.preventDefault(); // Prevent focus switching
      const newDraft = onAcceptSuggestion();
      onDraftChange(newDraft);
    } else if (e.key === 'Escape' && suggestion) {
      e.preventDefault();
      onDismissSuggestion();
    }
  };

  // Auto-focus editor on load
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.focus();
    }
  }, []);

  return (
    <div className="relative w-full h-[520px] bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-2xl focus-within:border-indigo-500/60 transition-colors">
      {/* Ghost-Text Overlay Layer (Renders beneath transparent textarea) */}
      <div
        ref={overlayRef}
        aria-hidden="true"
        className="absolute inset-0 p-6 font-sans text-base leading-relaxed tracking-normal whitespace-pre-wrap break-words pointer-events-none overflow-y-auto select-none"
        style={{ fontKerning: 'normal' }}
      >
        {/* User's typed text rendered in invisible ink */}
        <span className="invisible opacity-0">{draft}</span>

        {/* Ghost continuation rendered in faded grey italic right at the cursor */}
        {suggestion && (
          <span className="text-slate-500/70 italic bg-indigo-500/10 px-0.5 rounded transition-opacity duration-200">
            {suggestion}
          </span>
        )}
      </div>

      {/* Interactive Textarea Layer (Transparent background so overlay shows through) */}
      <textarea
        ref={textareaRef}
        value={draft}
        onChange={(e) => onDraftChange(e.target.value)}
        onKeyDown={handleKeyDown}
        onScroll={handleScroll}
        placeholder="Begin writing your IELTS Task 2 essay here... As you type, your AI Copilot will suggest Band 8+ academic completions in ghost text. Press Tab to accept."
        spellCheck="true"
        className="relative z-10 w-full h-full p-6 bg-transparent text-slate-100 placeholder-slate-600 font-sans text-base leading-relaxed tracking-normal resize-none focus:outline-none caret-indigo-400 overflow-y-auto"
        style={{ fontKerning: 'normal' }}
      />
    </div>
  );
}
