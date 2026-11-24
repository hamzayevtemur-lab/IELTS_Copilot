import React, { useState, useEffect } from 'react';
import { PromptHeader } from './components/PromptHeader';
import { Editor } from './components/Editor';
import { SuggestionBox } from './components/SuggestionBox';
import { useWordCount } from './hooks/useWordCount';
import { useAutocomplete } from './hooks/useAutocomplete';
import { checkBackendHealth } from './services/api';
import { Sparkles, CheckCircle2, Award, ShieldCheck, HelpCircle } from 'lucide-react';

const DEFAULT_QUESTION = "Some people believe that university education should be free for everyone. To what extent do you agree or disagree?";

const DEFAULT_DRAFT = "Over the past few decades, the accessibility of tertiary education has sparked intense global debate. While opponents argue that funding higher education imposes an immense fiscal burden on the state, I firmly believe that";

export default function App() {
  const [question, setQuestion] = useState(DEFAULT_QUESTION);
  const [draft, setDraft] = useState(DEFAULT_DRAFT);
  const [copilotEnabled, setCopilotEnabled] = useState(true);
  const [backendStatus, setBackendStatus] = useState('checking');

  // Word statistics hook
  const wordStats = useWordCount(draft);

  // Autocomplete streaming hook
  const {
    suggestion,
    isStreaming,
    clearSuggestion,
    acceptSuggestion,
  } = useAutocomplete({
    question,
    draft,
    enabled: copilotEnabled,
    debounceMs: 400,
    maxTokens: 30,
  });

  // Verify backend connection on load
  useEffect(() => {
    checkBackendHealth().then((res) => {
      setBackendStatus(res.status === 'healthy' ? 'online' : 'offline');
    });
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Top Navbar */}
      <header className="border-b border-slate-800 bg-slate-900/60 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-indigo-600 to-violet-500 flex items-center justify-center shadow-lg shadow-indigo-500/20">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="font-bold text-base tracking-tight text-white flex items-center gap-2">
                IELTS Writing Copilot
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-indigo-500/20 text-indigo-300 font-mono border border-indigo-500/30">
                  Task 2
                </span>
              </h1>
              <p className="text-[11px] text-slate-400">Context-Aware Real-Time Autocomplete Assistant</p>
            </div>
          </div>

          {/* Model Status Badge */}
          <div className="flex items-center gap-3 text-xs">
            <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-slate-800/80 border border-slate-700/60">
              <span className={`w-2 h-2 rounded-full ${backendStatus === 'online' ? 'bg-emerald-400 animate-pulse' : 'bg-rose-400'}`} />
              <span className="text-slate-300 font-mono text-[11px]">
                {backendStatus === 'online' ? 'Qwen2.5-1.5B (Band 8+)' : 'Backend Offline (Run main.py)'}
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="max-w-6xl mx-auto px-6 py-8 flex-1 w-full grid grid-cols-1 lg:grid-cols-4 gap-8">
        {/* Editor Area (Left 3 Columns) */}
        <div className="lg:col-span-3 flex flex-col">
          <PromptHeader
            question={question}
            onQuestionChange={setQuestion}
            wordStats={wordStats}
            copilotEnabled={copilotEnabled}
            onToggleCopilot={() => setCopilotEnabled(!copilotEnabled)}
          />

          <div className="relative flex flex-col flex-1">
            <Editor
              draft={draft}
              onDraftChange={setDraft}
              suggestion={suggestion}
              onAcceptSuggestion={acceptSuggestion}
              onDismissSuggestion={clearSuggestion}
              isStreaming={isStreaming}
            />

            <div className="mt-3">
              <SuggestionBox
                suggestion={suggestion}
                isStreaming={isStreaming}
                onAccept={() => setDraft(acceptSuggestion())}
                onDismiss={clearSuggestion}
              />
            </div>
          </div>
        </div>

        {/* Sidebar: IELTS Task 2 Criteria Guide (Right Column) */}
        <div className="space-y-6">
          {/* Target Assessment Card */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl">
            <h3 className="font-semibold text-sm text-slate-200 flex items-center gap-2 mb-3">
              <Award className="w-4 h-4 text-indigo-400" />
              Band Score Criteria
            </h3>
            <div className="space-y-3 text-xs">
              <div className="p-2.5 rounded-xl bg-slate-950/60 border border-slate-800/80">
                <span className="font-semibold text-indigo-300 block mb-0.5">Task Achievement</span>
                <p className="text-slate-400 text-[11px] leading-relaxed">
                  Fully address all prompt parts with a clear, sustained position throughout.
                </p>
              </div>
              <div className="p-2.5 rounded-xl bg-slate-950/60 border border-slate-800/80">
                <span className="font-semibold text-indigo-300 block mb-0.5">Coherence & Cohesion</span>
                <p className="text-slate-400 text-[11px] leading-relaxed">
                  Logically organize ideas with clear paragraphing and varied linking devices.
                </p>
              </div>
              <div className="p-2.5 rounded-xl bg-slate-950/60 border border-slate-800/80">
                <span className="font-semibold text-indigo-300 block mb-0.5">Lexical Resource</span>
                <p className="text-slate-400 text-[11px] leading-relaxed">
                  Use advanced academic collocations and precise vocabulary naturally.
                </p>
              </div>
              <div className="p-2.5 rounded-xl bg-slate-950/60 border border-slate-800/80">
                <span className="font-semibold text-indigo-300 block mb-0.5">Grammatical Range</span>
                <p className="text-slate-400 text-[11px] leading-relaxed">
                  Mix complex sentence structures with high grammatical accuracy.
                </p>
              </div>
            </div>
          </div>

          {/* Keyboard Shortcuts Card */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl">
            <h3 className="font-semibold text-sm text-slate-200 flex items-center gap-2 mb-3">
              <HelpCircle className="w-4 h-4 text-indigo-400" />
              Keyboard Shortcuts
            </h3>
            <div className="space-y-2 text-xs">
              <div className="flex items-center justify-between text-slate-400">
                <span>Accept Suggestion</span>
                <kbd className="px-2 py-0.5 bg-slate-800 text-slate-200 rounded border border-slate-700 font-mono text-[11px]">Tab</kbd>
              </div>
              <div className="flex items-center justify-between text-slate-400">
                <span>Dismiss Suggestion</span>
                <kbd className="px-2 py-0.5 bg-slate-800 text-slate-200 rounded border border-slate-700 font-mono text-[11px]">Esc</kbd>
              </div>
              <div className="flex items-center justify-between text-slate-400">
                <span>Trigger Suggestion</span>
                <span className="text-[11px] text-slate-500">Auto (400ms pause)</span>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
