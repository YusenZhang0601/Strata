"""Unit tests for RFC-S2A-2: unmark_think_literals range regex + pure-text differential IPC (STRATA_IPC_DELTA)."""
import io
import os
import queue
import re
import threading
import unittest

import serve.frontend as fe
import serve.server as srv


def _cpp_ipc_digest_i32(ids: list[int]) -> int:
    """Exact bit-for-bit reference mirror of ipc_digest_i32() in src/program/generate.cpp."""
    crc_tab = []
    for i in range(256):
        c = i
        for _ in range(8):
            c = (0xEDB88320 ^ (c >> 1)) if (c & 1) else (c >> 1)
        crc_tab.append(c)
    crc = 0xFFFFFFFF
    a, b = 1, 0
    mod = 65521
    run = 0
    for v in ids:
        v &= 0xFFFFFFFF
        for sh in (0, 8, 16, 24):
            byte = (v >> sh) & 0xFF
            crc = crc_tab[(crc ^ byte) & 0xFF] ^ (crc >> 8)
            a += byte
            b += a
        run += 1
        if run == 1024:
            a %= mod
            b %= mod
            run = 0
    if run > 0:
        a %= mod
        b %= mod
    crc ^= 0xFFFFFFFF
    return ((crc & 0xFFFFFFFF) << 32) | (((b << 16) | a) & 0xFFFFFFFF)


class _FakePipeProc:
    def __init__(self, on_write):
        self._on_write = on_write
        self._buf = ""
        self.stdin = self

    def write(self, data: str):
        self._buf += data
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._on_write(line)
        return len(data)

    def flush(self):
        pass

    def poll(self):
        return None


class UnmarkThinkLiteralsTest(unittest.TestCase):
    def test_exact_parity_all_cases(self):
        def orig_unmark(prompt, tags=fe.LITERAL_TAGS):
            marks = fe.THINK_MARKS if tags is fe.LITERAL_TAGS else {v: k for k, v in tags.items()}
            mark_re = re.compile("|".join(fe.THINK_MARKS)) if tags is fe.LITERAL_TAGS else \
                re.compile("|".join(map(re.escape, marks)))
            out, spans, pos, n = [], [], 0, 0
            for m in mark_re.finditer(prompt):
                out.append(prompt[pos:m.start()])
                n += m.start() - pos
                tag = marks[m.group(0)]
                out.append(tag)
                spans.append((n, n + len(tag)))
                n += len(tag)
                pos = m.end()
            out.append(prompt[pos:])
            return "".join(out), spans

        controls = [f"<|ctrl_{i}|>" for i in range(20)] + ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]
        tags = fe.literal_tags(controls)
        cases = [
            (tags["<think>"] + "hello" + tags["<|im_end|>"] + tags["</tool_call>"] + "world", tags),
            ("plain text with no marks at all", tags),
            (fe.LITERAL_TAGS["<think>"] + "abc" + fe.LITERAL_TAGS["</think>"], fe.LITERAL_TAGS),
            ("a-b]c^d\\e", {"<A>": "-", "<B>": "]", "<C>": "^", "<D>": "\\"}),
            ("foo@@BAR@@baz", {"<X>": "@@"}),
        ]
        for text, tg in cases:
            self.assertEqual(fe.unmark_think_literals(text, tg), orig_unmark(text, tg))
        self.assertEqual(fe.unmark_think_literals("hello", {}), ("hello", []))


class IpcDeltaCodecTest(unittest.TestCase):
    def test_digest_matches_cpp_reference(self):
        codec = srv.IpcDeltaCodec()
        for sz in (0, 1, 256, 1024, 1025, 3000):
            sample = [(i * 2654435761) & 0x7FFFFFFF for i in range(sz)]
            buf = codec._to_le32_bytes(sample)
            self.assertIsNotNone(buf)
            self.assertEqual(codec.digest_bytes(memoryview(buf)), _cpp_ipc_digest_i32(sample))

    def test_multiturn_roundtrip_and_divergence(self):
        codec = srv.IpcDeltaCodec()
        refs: list[tuple[int, list[int]]] = []

        def decode_cmd(cmd: str):
            toks = cmd.split()
            idx = 3 if toks[0] in ("BGEN", "BGENI") else 2
            kv = {}
            while idx < len(toks) and "=" in toks[idx]:
                k, v = toks[idx].split("=", 1)
                kv[k] = v
                idx += 1
            emb = toks[idx] if toks[0] in ("GENI", "BGENI") else None
            if emb is not None:
                idx += 1
            payload = toks[idx]
            if "ref" in kv:
                rid, keep, h = int(kv["ref"]), int(kv["keep"]), int(kv["h"], 16)
                hit = next((r for r in refs if r[0] == rid), None)
                if hit is None or keep > len(hit[1]) or _cpp_ipc_digest_i32(hit[1][:keep]) != h:
                    return "REF_MISS", rid, emb
                delta = [] if payload == "-" else [int(x) for x in payload.split(",")]
                ids = list(hit[1][:keep]) + delta
            else:
                ids = [int(x) for x in payload.split(",")]
            if "ref_save" in kv:
                sid = int(kv["ref_save"])
                refs[:] = [r for r in refs if r[0] != sid]
                if len(refs) >= 16:
                    refs.pop(0)
                refs.append((sid, list(ids)))
            return "OK", ids, emb

        # Short prompt (<256): plain CSV
        dk, pl, used, sid = codec.encode([10, 20, 30])
        self.assertEqual((dk, pl, used, sid), ("", "10,20,30", None, None))

        # 6 incremental turns
        cur = list(range(100, 1100))
        for turn in range(6):
            dk, pl, used, sid = codec.encode(cur)
            if turn == 0:
                self.assertIsNone(used)
                self.assertEqual(sid, 1)
                self.assertIn("ref_save=1", dk)
            else:
                self.assertIsNotNone(used)
                self.assertIsNotNone(sid)
                self.assertIn(f"keep={len(cur) - 47} ", dk)
            st, got, _ = decode_cmd(f"BGEN 0 32{dk} {pl}")
            self.assertEqual((st, got), ("OK", cur))
            cur = cur + [50000 + turn * 100 + j for j in range(47)]

        # Mid-prompt divergence
        div = cur[:600] + [99999] * 25
        dk, pl, _, _ = codec.encode(div)
        self.assertIn("keep=600 ", dk)
        self.assertEqual(decode_cmd(f"GEN 10{dk} {pl}"), ("OK", div, None))

        # Exact repeat with vision embedding path (empty delta sentinel '-')
        dk_rep, pl_rep, _, sid_rep = codec.encode(div)
        self.assertEqual(pl_rep, "-")
        self.assertIsNone(sid_rep)
        self.assertNotIn("ref_save=", dk_rep)
        self.assertEqual(decode_cmd(f"GENI 1{dk_rep} /tmp/emb.bin {pl_rep}"), ("OK", div, "/tmp/emb.bin"))

        # Engine restart / eviction -> REF_MISS -> drop BOTH used_ref and phantom save_id -> resend full
        refs.clear()
        dk_m, pl_m, used_m, sid_m = codec.encode(div + [42])
        self.assertEqual(decode_cmd(f"GEN 10{dk_m} {pl_m}"), ("REF_MISS", used_m, None))
        codec.drop(used_m)
        if sid_m is not None:
            codec.drop(sid_m)
        self.assertNotIn(sid_m, codec._entries)
        dk_f, pl_f, used_f, sid_f = codec.encode(div + [42], allow_delta=False)
        self.assertIsNone(used_f)
        self.assertIn(sid_f, codec._entries)
        self.assertEqual(decode_cmd(f"GEN 10{dk_f} {pl_f}"), ("OK", div + [42], None))

    def test_engine_default_off_and_ref_miss_recovery(self):
        sent: list[str] = []
        lines_q: queue.Queue = queue.Queue()
        miss_once = {"done": False}

        def on_write(line: str):
            sent.append(line)
            if "ref=" in line and not miss_once["done"]:
                miss_once["done"] = True
                rid = next(tok.split("=")[1] for tok in line.split() if tok.startswith("ref="))
                lines_q.put(f"REF_MISS {rid}\n")
            else:
                lines_q.put("T 777\n")
                lines_q.put("DONE 1 400 1.0 1.0 stop\n")

        eng = srv.StrataEngine.__new__(srv.StrataEngine)
        eng.proc = _FakePipeProc(on_write)
        eng.ended = False
        eng.batch = 0
        eng.multi = 0
        eng.gen = 1
        eng.can_stop = True
        eng.silence_s = 0
        eng.info = {"ipc_delta": 1}
        eng.lines = lines_q
        eng.wlock = threading.Lock()
        eng.last = {}
        eng._ipc_delta = srv.IpcDeltaCodec()
        eng._ipc_delta_gen = 1

        prompt1 = list(range(500, 900))
        prompt2 = prompt1 + [901, 902, 903]
        old_env = os.environ.get("STRATA_IPC_DELTA")
        try:
            # Default OFF: exact original format, no ref/ref_save keys
            os.environ.pop("STRATA_IPC_DELTA", None)
            toks = list(eng.generate(prompt1, 4, {}, threading.Event()))
            self.assertEqual(toks, [777])
            self.assertNotIn("ref_save=", sent[-1])
            self.assertNotIn("ref=", sent[-1])

            # Enabled: Turn 1 saves ref, Turn 2 hits REF_MISS once and transparently resends full
            os.environ["STRATA_IPC_DELTA"] = "1"
            sent.clear()
            self.assertEqual(list(eng.generate(prompt1, 4, {}, threading.Event())), [777])
            self.assertIn("ref_save=1", sent[-1])

            sent.clear()
            self.assertEqual(list(eng.generate(prompt2, 4, {}, threading.Event())), [777])
            self.assertEqual(len(sent), 2)
            self.assertIn("ref=1 keep=400 ", sent[0])
            failed_save_id = int(next(tok.split("=")[1] for tok in sent[0].split() if tok.startswith("ref_save=")))
            retry_save_id = int(next(tok.split("=")[1] for tok in sent[1].split() if tok.startswith("ref_save=")))
            self.assertNotIn("ref=", sent[1])
            # Red-team #69 defect A: failed delta's phantom save_id must be evicted from _entries
            self.assertNotIn(1, eng._ipc_delta._entries)
            self.assertNotIn(failed_save_id, eng._ipc_delta._entries)
            self.assertIn(retry_save_id, eng._ipc_delta._entries)
        finally:
            if old_env is None:
                os.environ.pop("STRATA_IPC_DELTA", None)
            else:
                os.environ["STRATA_IPC_DELTA"] = old_env

    def test_batched_ref_miss_drops_phantom_save_and_calls_started_once(self):
        """Red-team #69 defects A & B on BOTH solo and batch-admit paths of generate_batched()."""
        for mode in ("solo", "admit"):
            sent: list[str] = []
            lines_q: queue.Queue = queue.Queue()
            slot_q = [queue.Queue(), queue.Queue()]
            miss_once = {"done": False}

            def on_write(line: str, _sent=sent, _lq=lines_q, _sq=slot_q, _mo=miss_once, _mode=mode):
                _sent.append(line)
                if "ref=" in line and not _mo["done"]:
                    _mo["done"] = True
                    rid = next(tok.split("=")[1] for tok in line.split() if tok.startswith("ref="))
                    _lq.put(f"REF_MISS {rid}\n")
                elif line.startswith("BGEN "):
                    _lq.put("T 888\n")
                    _lq.put("DONE 1 403 1.0 1.0 stop\n")
                    _lq.put("BADM 0 0\n")
                else:
                    _lq.put("T 888\n")
                    _lq.put("DONE 1 403 1.0 1.0 stop\n")

            eng = srv.StrataEngine.__new__(srv.StrataEngine)
            eng.proc = _FakePipeProc(on_write)
            eng.ended = False
            eng.batch = 2
            eng.multi = 0
            eng.gen = 1
            eng.can_stop = True
            eng.silence_s = 0
            eng.info = {"ipc_delta": 1, "slot_cache": 1}
            eng.lines = lines_q
            eng.wlock = threading.Lock()
            eng.ctl = threading.Lock()
            eng.slot_cv = threading.Condition()
            eng.ctl_epoch = 0
            eng.waiting = 0
            eng.wait_lens = []
            eng.arrivals = []
            eng.admitting = 0
            eng.slot_order = [0, 1]
            eng.slot_group = [0, 0]
            # For 'admit' mode mark slot 1 busy so generate_batched skips solo path and enters BGEN into slot 0
            eng.slot_busy = [False, mode == "admit"]
            eng.slot_claims = {1} if mode == "admit" else set()
            eng.slot_releasing = set()
            eng.slot_held = [[], []]
            eng.slot_used = [0.0, 0.0]
            eng.slot_live = [None, None]
            eng.slot_q = slot_q
            eng.last = {}
            eng._yielded = None
            eng._ipc_delta = srv.IpcDeltaCodec()
            eng._ipc_delta_gen = 1

            prompt1 = list(range(500, 900))
            prompt2 = prompt1 + [901, 902, 903]
            old_env = os.environ.get("STRATA_IPC_DELTA")
            try:
                os.environ["STRATA_IPC_DELTA"] = "1"
                # Seed ref_id=1
                eng._ipc_delta.encode(prompt1)
                cancel = threading.Event()
                started_calls = {"n": 0}
                cancel._strata_queue_started = lambda: started_calls.__setitem__("n", started_calls["n"] + 1)

                toks = list(eng.generate_batched(prompt2, 4, {}, cancel))
                self.assertEqual(toks, [888], f"mode={mode}")
                self.assertEqual(len(sent), 2, f"mode={mode} should send delta then full retry")
                self.assertIn("ref=1 keep=400 ", sent[0], f"mode={mode}")
                failed_sid = int(next(tok.split("=")[1] for tok in sent[0].split() if tok.startswith("ref_save=")))
                retry_sid = int(next(tok.split("=")[1] for tok in sent[1].split() if tok.startswith("ref_save=")))
                self.assertNotIn("ref=", sent[1], f"mode={mode}")
                # Defect A: no phantom save_id retained
                self.assertNotIn(1, eng._ipc_delta._entries, f"mode={mode}")
                self.assertNotIn(failed_sid, eng._ipc_delta._entries, f"mode={mode}")
                self.assertIn(retry_sid, eng._ipc_delta._entries, f"mode={mode}")
                # Defect B: _strata_queue_started invoked strictly once across retry
                self.assertEqual(started_calls["n"], 1, f"mode={mode}")
            finally:
                if old_env is None:
                    os.environ.pop("STRATA_IPC_DELTA", None)
                else:
                    os.environ["STRATA_IPC_DELTA"] = old_env


if __name__ == "__main__":
    unittest.main()
