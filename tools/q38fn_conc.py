#!/usr/bin/env python3
"""Concurrency bench: agent-style load against one OpenAI-compatible server.
Agent-like load against one OpenAI-compatible server: every agent shares one long system prompt,
has its own history, and sends TURNS follow-ups that extend its own conversation (exercises the prompt cache).
Per level N: N agents run at once. Reports aggregate output tok/s, per-agent tok/s, TTFT p50/p90, errors.
Stdlib only."""
import argparse, json, random, statistics, threading, time, urllib.request

WORDS = ("agent gateway lane model queue lease shard tensor kernel prompt cache context window router fallback "
         "watchdog cron handoff canon brief checklist recipe quant expert layer token decode prefill").split()

def text(rnd, n_words):
    return " ".join(rnd.choice(WORDS) for _ in range(n_words))

def stream_chat(url, model, messages, max_tokens, timeout=1800):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; out = []; usage = {}
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            j = json.loads(data)
            if j.get("usage"):
                usage = j["usage"]
            for ch in j.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content") or ""
                if piece:
                    if ttft is None:
                        ttft = time.time() - t0
                    out.append(piece)
    return "".join(out), usage, ttft, time.time() - t0

def agent(i, a, system, results, lock):
    rnd = random.Random(1000 + i)
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": f"Agent {i} history: " + text(rnd, a.history_words) +
             "\n\nSummarize the state of the queue in five bullet points."}]
    for turn in range(a.turns):
        try:
            reply, usage, ttft, dt = stream_chat(a.url, a.model, msgs, a.max_tokens)
            row = {"agent": i, "turn": turn, "ok": True, "ttft": ttft, "seconds": dt,
                   "completion_tokens": usage.get("completion_tokens", 0), "prompt_tokens": usage.get("prompt_tokens", 0),
                   "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)}
            msgs += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": "Follow-up: " + text(rnd, 60) + "\nWhat should happen next? Five bullets."}]
        except Exception as e:  # noqa: BLE001
            row = {"agent": i, "turn": turn, "ok": False, "error": str(e)[:300]}
        with lock:
            results.append(row)

def level(a, n, system):
    results, lock = [], threading.Lock()
    t0 = time.time()
    th = [threading.Thread(target=agent, args=(i, a, system, results, lock)) for i in range(n)]
    for t in th: t.start()
    for t in th: t.join()
    wall = time.time() - t0
    ok = [r for r in results if r["ok"]]
    toks = sum(r["completion_tokens"] for r in ok)
    tt = sorted(r["ttft"] for r in ok if r["ttft"] is not None)
    pct = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None
    per = [r["completion_tokens"] / (r["seconds"] - (r["ttft"] or 0)) for r in ok
           if r["seconds"] - (r["ttft"] or 0) > 0 and r["completion_tokens"]]
    return {"agents": n, "requests": len(results), "errors": len(results) - len(ok), "wall_s": round(wall, 1),
            "agg_out_tok_s": round(toks / wall, 1), "per_agent_decode_tok_s_median": round(statistics.median(per), 1) if per else None,
            "ttft_p50_s": round(pct(tt, 0.5), 2) if tt else None, "ttft_p90_s": round(pct(tt, 0.9), 2) if tt else None,
            "prompt_tokens_mean": round(statistics.mean(r["prompt_tokens"] for r in ok)) if ok else None,
            "cached_tokens_mean": round(statistics.mean(r["cached_tokens"] for r in ok)) if ok else None,
            "error_samples": [r["error"] for r in results if not r["ok"]][:3]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080"); ap.add_argument("--model", required=True)
    ap.add_argument("--levels", default="1,4,8,16"); ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--system-words", type=int, default=6000); ap.add_argument("--history-words", type=int, default=1500)
    ap.add_argument("--max-tokens", type=int, default=256); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    system = "You are an operations agent in a home AI lab. Shared operating notes follow.\n" + text(random.Random(7), a.system_words)
    stream_chat(a.url, a.model, [{"role": "user", "content": "Say ok."}], 8)  # warm-up
    rows = []
    for n in [int(x) for x in a.levels.split(",")]:
        r = level(a, n, system); rows.append(r); print(json.dumps(r), flush=True)
    json.dump({"model": a.model, "settings": vars(a), "levels": rows}, open(a.out, "w"), indent=1)

if __name__ == "__main__":
    main()
