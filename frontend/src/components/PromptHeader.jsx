import React, { useState, useEffect } from 'react';
import { Timer, BookOpen, Sparkles, AlertCircle, RotateCcw } from 'lucide-react';

const PRESET_QUESTIONS = [
  "Some people believe that university education should be free for everyone. To what extent do you agree or disagree?",
  "In many countries, traditional customs and cultural practices are disappearing. Is this a positive or negative development?",
  "Many people believe that unpaid community service should be a compulsory part of high school education. Discuss both views and give your opinion.",
  "The increasing use of social media is replacing face-to-face interaction among young people. What are the causes and solutions?",
  "Some people think that strict government laws are the only way to solve environmental pollution. To what extent do you agree?"
];

export function PromptHeader({
  question,
  onQuestionChange,
  wordStats,
  copilotEnabled,
  onToggleCopilot,
}) {
  // 40-Minute IELTS Exam Timer
  const [secondsLeft, setSecondsLeft] = useState(40 * 60);
  const [isTimerRunning, setIsTimerRunning] = useState(false);

  useEffect(() => {
    let interval = null;
    if (isTimerRunning && secondsLeft > 0) {
      interval = setInterval(() => {
        setSecondsLeft((s) => s - 1);
      }, 1000);
    } else if (secondsLeft === 0) {
      setIsTimerRunning(false);
    }
    return () => clearInterval(interval);
  }, [isTimerRunning, secondsLeft]);

  const formatTime = (secs) => {
    const mins = Math.floor(secs / 60);
    const remainingSecs = secs % 60;
    return `${mins.toString().padStart(2, '0')}:${remainingSecs.toString().padStart(2, '0')}`;
  };

  const resetTimer = () => {
    setSecondsLeft(40 * 60);
    setIsTimerRunning(false);
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl mb-6">
      {/* Top Bar: Timer, Controls, Word Count Badge */}
      <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-800/80 rounded-xl border border-slate-700/60 font-mono text-sm">
            <Timer className={`w-4 h-4 ${secondsLeft < 300 ? 'text-rose-400 animate-pulse' : 'text-indigo-400'}`} />
            <span className={secondsLeft < 300 ? 'text-rose-300 font-bold' : 'text-slate-200'}>
              {formatTime(secondsLeft)}
            </span>
            <button
              onClick={() => setIsTimerRunning(!isTimerRunning)}
              className="text-xs ml-1 px-2 py-0.5 rounded bg-slate-700 hover:bg-slate-600 text-slate-300 transition"
            >
              {isTimerRunning ? 'Pause' : 'Start'}
            </button>
            <button
              onClick={resetTimer}
              title="Reset Timer"
              className="p-1 text-slate-400 hover:text-slate-200 transition"
            >
              <RotateCcw className="w-3.5 h-3.5" />
            </button>
          </div>

          {/* Word Count Indicator */}
          <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-800/80 rounded-xl border border-slate-700/60 text-sm">
            <span className="text-slate-400">Words:</span>
            <span className={`font-semibold ${wordStats.isTargetReached ? 'text-emerald-400' : 'text-amber-400'}`}>
              {wordStats.wordCount}
            </span>
            <span className="text-slate-500">/ 250 min</span>
          </div>
        </div>

        {/* Copilot Toggle */}
        <div className="flex items-center gap-3">
          <button
            onClick={onToggleCopilot}
            className={`flex items-center gap-2 px-4 py-1.5 rounded-xl text-xs font-medium transition-all ${
              copilotEnabled
                ? 'bg-indigo-600/30 text-indigo-300 border border-indigo-500/50 shadow-lg shadow-indigo-500/10'
                : 'bg-slate-800 text-slate-400 border border-slate-700 hover:bg-slate-700'
            }`}
          >
            <Sparkles className={`w-3.5 h-3.5 ${copilotEnabled ? 'text-indigo-400 animate-spin-slow' : 'text-slate-400'}`} />
            Copilot: {copilotEnabled ? 'Active' : 'Disabled'}
          </button>
        </div>
      </div>

      {/* Progress Bar for Word Count */}
      <div className="w-full bg-slate-800/60 h-1.5 rounded-full mt-3 overflow-hidden">
        <div
          className={`h-full transition-all duration-300 ${
            wordStats.isTargetReached
              ? 'bg-emerald-500'
              : wordStats.wordCount > 150
              ? 'bg-amber-500'
              : 'bg-indigo-500'
          }`}
          style={{ width: `${wordStats.progressPercent}%` }}
        />
      </div>

      {/* Question Selector & Input */}
      <div className="mt-4">
        <div className="flex items-center justify-between mb-2">
          <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
            <BookOpen className="w-3.5 h-3.5 text-indigo-400" />
            IELTS Task 2 Essay Topic
          </label>
          <select
            onChange={(e) => {
              if (e.target.value) onQuestionChange(e.target.value);
            }}
            defaultValue=""
            className="text-xs bg-slate-800 border border-slate-700 text-slate-300 rounded-lg px-2.5 py-1 focus:outline-none focus:border-indigo-500 cursor-pointer"
          >
            <option value="" disabled>Load Sample IELTS Topic...</option>
            {PRESET_QUESTIONS.map((q, idx) => (
              <option key={idx} value={q}>
                Topic #{idx + 1}: {q.slice(0, 55)}...
              </option>
            ))}
          </select>
        </div>
        <textarea
          value={question}
          onChange={(e) => onQuestionChange(e.target.value)}
          placeholder="Paste or type your IELTS Writing Task 2 prompt here..."
          rows={2}
          className="w-full bg-slate-950/70 border border-slate-800 rounded-xl p-3 text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500/70 transition resize-none leading-relaxed"
        />
      </div>
    </div>
  );
}
