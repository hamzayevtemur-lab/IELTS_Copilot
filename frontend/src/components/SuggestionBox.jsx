import React from 'react';
import { Sparkles, CornerDownLeft, XCircle } from 'lucide-react';

export function SuggestionBox({ suggestion, isStreaming, onAccept, onDismiss }) {
  if (!suggestion && !isStreaming) return null;

  return (
    <div className="flex items-center justify-between gap-3 px-4 py-2 bg-indigo-950/40 border border-indigo-500/30 rounded-xl text-xs text-indigo-200 backdrop-blur-sm animate-fade-in">
      <div className="flex items-center gap-2 truncate">
        <Sparkles className={`w-3.5 h-3.5 text-indigo-400 shrink-0 ${isStreaming ? 'animate-pulse text-amber-400' : ''}`} />
        {isStreaming ? (
          <span className="text-slate-300 italic">Thinking and continuing draft...</span>
        ) : (
          <span className="truncate text-slate-200">
            Suggestion: <span className="text-indigo-300 font-medium">"{suggestion}"</span>
          </span>
        )}
      </div>

      {suggestion && (
        <div className="flex items-center gap-2 shrink-0">
          <button
            onClick={onAccept}
            className="flex items-center gap-1 px-2.5 py-1 bg-indigo-600 hover:bg-indigo-500 text-white rounded-md font-mono text-[11px] shadow transition active:scale-95"
          >
            <span>Tab</span>
            <CornerDownLeft className="w-3 h-3" />
          </button>
          <button
            onClick={onDismiss}
            title="Dismiss suggestion (Esc)"
            className="p-1 text-slate-400 hover:text-slate-200 transition"
          >
            <XCircle className="w-3.5 h-3.5" />
          </button>
        </div>
      )}
    </div>
  );
}
