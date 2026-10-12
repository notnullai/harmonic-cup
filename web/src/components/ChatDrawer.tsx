import { useEffect, useRef, useState } from "react";
import { MessageSquareText, Send, X } from "lucide-react";
import { api, type ChatAnswer } from "../api";

interface Turn { q: string; a?: ChatAnswer; err?: string }

const SUGGESTIONS = [
  "What is wrong with the machine?",
  "How do I replace the drive-end bearing?",
  "What should I check before any work?",
];

export default function ChatDrawer({ open, onClose, epoch }: { open: boolean; onClose: () => void; epoch: number }) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const end = useRef<HTMLDivElement>(null);

  useEffect(() => { setTurns([]); }, [epoch]);
  useEffect(() => { end.current?.scrollIntoView({ behavior: "smooth" }); }, [turns]);
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  const ask = async (q: string) => {
    if (!q.trim() || busy) return;
    setText("");
    setBusy(true);
    setTurns((t) => [...t, { q }]);
    try {
      const a = await api.chat(q);
      setTurns((t) => t.map((x, i) => (i === t.length - 1 ? { ...x, a } : x)));
    } catch (e) {
      setTurns((t) => t.map((x, i) => (i === t.length - 1 ? { ...x, err: (e as Error).message } : x)));
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <div className="scrim" onClick={onClose} />
      <aside className="drawer" role="dialog" aria-modal="true" aria-label="Ask about this machine">
        <div className="card-h" style={{ padding: "16px 18px", borderBottom: "1px solid var(--border)" }}>
          <MessageSquareText size={18} /><h2>Ask about this machine</h2><span className="grow" />
          <button className="btn ghost icon sm" onClick={onClose} aria-label="Close"><X size={16} /></button>
        </div>
        <div className="chat">
          {turns.length === 0 && (
            <div className="empty">
              <div>Answers are grounded in the knowledge graph and the maintenance manual, with sources.</div>
              <div className="row" style={{ justifyContent: "center" }}>
                {SUGGESTIONS.map((s) => <button key={s} className="btn sm" onClick={() => ask(s)}>{s}</button>)}
              </div>
            </div>
          )}
          {turns.map((t, i) => (
            <div key={i} style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              <div className="msg q">{t.q}</div>
              {t.a ? (
                <div className="msg a">
                  {t.a.answer}
                  {t.a.references?.length > 0 && (
                    <div className="refs" style={{ marginTop: 8 }}>{t.a.references.map((r) => <code key={r}>{r}</code>)}</div>
                  )}
                  {!t.a.grounded && <div className="hint" style={{ marginTop: 6 }}>Some citations were removed: they were not in what was retrieved.</div>}
                </div>
              ) : t.err ? <div className="msg a" style={{ color: "var(--critical)" }}>{t.err}</div>
                : <div className="msg a"><span className="skeleton" style={{ display: "inline-block", width: 160, height: 12 }} /></div>}
            </div>
          ))}
          <div ref={end} />
        </div>
        <form className="chat-in" onSubmit={(e) => { e.preventDefault(); void ask(text); }}>
          <input value={text} onChange={(e) => setText(e.target.value)} placeholder="e.g. what's wrong with the pump?"
            aria-label="Question" autoFocus />
          <button className="btn primary icon" disabled={busy || !text.trim()} aria-label="Send"><Send size={15} /></button>
        </form>
      </aside>
    </>
  );
}
