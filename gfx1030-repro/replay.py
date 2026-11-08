#!/usr/bin/env python3
"""Issue-27 replay: synthetic ~23k-token tool prompt then an unrelated short
prompt, per the reporter's recipe, against the repro server (:9932).

usage:
  replay.py wait                       block until the server answers /models
  replay.py big [target_tokens]        one long request (default 22970)
  replay.py short <tokens>             one unrelated prompt of ~N tokens
  replay.py replay                     big (22970) -> 5 s idle -> short (535)
  replay.py sweep t1 t2 ...            big -> 5 s idle -> each size, in order,
                                       5 s idle between (unique prompts)

Token counts are calibrated per run from the server's own tokenizer via
usage.prompt_tokens, so sizes land within ~1%. Every prompt draws fresh random
filler: consecutive requests share no prefix, like opencode's title request.
"""
import json
import random
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:9932"
WORDS = (
    "river mountain clock garden tower window silver harbor engine meadow patch "
    "signal timber anchor valley mirror circuit ember pasture ledger compass "
    "kernel orchard shuttle ribbon quarry lantern ferry thicket anvil cobalt "
    "saddle rampart windmill copse girder relay thresher pulley moraine beacon "
    "cable culvert damsel forge gantry hatchery irrigator junction kiln loom "
    "mastic navigator observatory pier quadrant rudder sawmill tramway umbrella"
).split()


def tools():
    out = []
    rng = random.Random(4)
    for i in range(20):
        name = "read_" + rng.choice(WORDS) + "_" + str(i)
        props = {}
        for j in range(rng.randint(4, 8)):
            props[rng.choice(WORDS) + "_" + str(j)] = {
                "type": "string",
                "description": "The " + " ".join(rng.sample(WORDS, 12)) + ".",
            }
        out.append({
            "type": "function",
            "function": {
                "name": name,
                "description": "Calls the " + " ".join(rng.sample(WORDS, 40))
                               + " on the remote host.",
                "parameters": {"type": "object", "properties": props, "required": []},
            },
        })
    return out


def filler(seed, n_chars):
    rng = random.Random(seed)
    parts = []
    total = 0
    while total < n_chars:
        w = rng.choice(WORDS)
        parts.append(w)
        total += len(w) + 1
    return " ".join(parts)[:n_chars]


def send(system, user, tl, max_tokens):
    body = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "tools": tl,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    ttft = None
    usage = {}
    with urllib.request.urlopen(req) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            d = json.loads(chunk)
            if d.get("choices"):
                delta = d["choices"][0].get("delta", {})
                if (delta.get("content") or delta.get("reasoning_content")
                        or delta.get("tool_calls")) and ttft is None:
                    ttft = time.time() - t0
            if d.get("usage"):
                usage = d["usage"]
    return usage, ttft, time.time() - t0


def report(tag, usage, ttft, total):
    pe = usage.get("prompt_eval_count") or usage.get("prompt_tokens", 0)
    ped = usage.get("prompt_eval_duration") or 0
    rate = pe / (ped / 1e6) if ped else float("nan")
    print("%-14s prompt=%6d compl=%4d ttft=%.3fs prefill=%.1f t/s wall=%.2fs" %
          (tag, pe, usage.get("completion_tokens", 0),
           ttft or -1, rate, total))
    return pe


def wait_ready(timeout=1800):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(BASE + "/models", timeout=2).read()
            print("server ready after %.1fs" % (time.time() - t0))
            return
        except Exception:
            time.sleep(2)
    sys.exit("server did not come up")


def calibrate(tools_list):
    """Return (base_tokens, chars_per_run, tokens_per_char) on the big skeleton."""
    base = "You are a coding agent. " + filler(99, 60)
    u, _, _ = send(base, "hi", tools_list, 1)
    t0 = u.get("prompt_tokens", 0)
    probe = filler(100, 6000)
    u, _, _ = send(base + " " + probe, "hi", tools_list, 1)
    t1 = u.get("prompt_tokens", 0)
    ratio = (t1 - t0) / 6000.0
    print("calib: skeleton=%d tokens, %.5f tok/char" % (t0, ratio))
    return t0, ratio


def big_prompt(target, tools_list, base_tokens, ratio, seed):
    pad = max(0, int((target - base_tokens) / ratio))
    system = "You are a coding agent. Tools operate on the remote workspace. " \
             + filler(seed, pad)
    user = "What tools do you have? Answer briefly."
    return system, user


def short_prompt(target, seed):
    system = "You are a naming assistant. Reply with a two or three word title."
    user = "Title this session: " + filler(seed, max(0, int(target / ratio) - 20)) \
           + " What is a short title?"
    return system, user


def main():
    global ratio
    cmd = sys.argv[1] if len(sys.argv) > 1 else "replay"
    if cmd == "wait":
        wait_ready()
        return
    wait_ready()
    tl = tools()
    base_tokens, ratio = calibrate(tl)

    if cmd == "big":
        target = int(sys.argv[2]) if len(sys.argv) > 2 else 22970
        s, u = big_prompt(target, tl, base_tokens, ratio, seed=1)
        report("big", *send(s, u, tl, 16))
    elif cmd == "short":
        target = int(sys.argv[2])
        s, u = short_prompt(target, seed=int(time.time()))
        report("short", *send(s, u, [], 16))
    elif cmd == "replay":
        s, u = big_prompt(22970, tl, base_tokens, ratio, seed=1)
        report("big", *send(s, u, tl, 16))
        print("idle 5s"); time.sleep(5)
        s, u = short_prompt(535, seed=int(time.time()))
        report("short", *send(s, u, [], 64))
    elif cmd == "busy":
        # tail-chunk case: the short hosted batch lands while the GPU is still
        # busy from the big request (no idle gap) - reporter says this is stable
        s, u = big_prompt(22970, tl, base_tokens, ratio, seed=1)
        report("big", *send(s, u, tl, 16))
        s, u = short_prompt(535, seed=int(time.time()))
        report("short@busy", *send(s, u, [], 64))
    elif cmd == "sweep":
        s, u = big_prompt(22970, tl, base_tokens, ratio, seed=1)
        report("big", *send(s, u, tl, 16))
        for tok in sys.argv[2:]:
            print("idle 5s"); time.sleep(5)
            s, u = short_prompt(int(tok), seed=int(time.time()))
            report("short@" + tok, *send(s, u, [], 16))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
