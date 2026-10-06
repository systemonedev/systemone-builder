"""Full general-suite evaluation of a trained Kenning-XL model, per family, vs v0.5 / Clef / Jev,
and the reflex+XL cascade number. In-process (no HTTP server needed)."""
import asyncio
import collections
import os
from pathlib import Path

from systemone_builder.kenning.xl import KenningXL
from systemone_builder.system_one import general_suite as g
from systemone_builder.system_one.contract import SystemOneRequest

# recorded baselines (general suite, 1328 items)
V05 = {"text": 0.756, "conversation": 0.979, "quality": 0.479, "agent": 0.727, "records": 0.587, "table": 0.520, "logs": 0.520}
CLEF = {"text": 0.841, "conversation": 1.0, "quality": 0.447, "agent": 0.793, "records": 0.857, "table": 0.860, "logs": 0.740}
JEV = {"text": 0.834, "conversation": 1.0, "quality": 0.498, "agent": 0.900, "records": 0.921, "table": 0.940, "logs": 0.720}
# families where XL (deliberate reasoner) should take over from the reflex
XL_FAMILIES = {"records", "table", "logs", "agent"}


def ok(qtype, ans, label):
    if qtype == "noul":
        t = int(label) if isinstance(label, (int, float)) else (1 if str(label).lower() in ("1", "yes", "true") else 0)
        return int(ans.noul >= 0.5) == t
    if qtype == "choice":
        return str(ans.choice) == str(label)
    return round(ans.score) == int(label)


async def main():
    model_dir = os.environ.get("XL_MODEL", "/work/models/kenning-xl-v0.6")
    suite = await g.general_suite(50, 42, Path("/cache/bench_cache"))
    qs = suite.questions
    m = KenningXL(model_dir, load_4bit=False, max_length=2048)
    hits = collections.defaultdict(list)
    for k, it in enumerate(suite.items):
        ask = {q: qs[q] for q in it.labels if q in qs}
        if not ask:
            continue
        resp = m.system_one(SystemOneRequest(state=it.state, questions=ask))
        for q, a in resp.answers.items():
            hits[g.FAMILY.get(q)].append(ok(qs[q].type, a, it.labels[q]))
        if k % 100 == 0:
            print(f"  {k}/{len(suite.items)}", flush=True)
    xl = {fam: sum(v) / len(v) for fam, v in hits.items()}
    fams = sorted(xl)
    print("\n=== Kenning-XL v0.6 (trained) per family ===")
    print(f"{'family':<13}{'XL':>8}{'v0.5':>8}{'Clef':>8}{'Jev':>8}")
    for f in fams:
        print(f"{f:<13}{xl[f]:>8.3f}{V05.get(f,0):>8.3f}{CLEF.get(f,0):>8.3f}{JEV.get(f,0):>8.3f}")
    xl_macro = sum(xl.values()) / len(xl)
    # cascade: XL on its families, v0.5 reflex elsewhere
    casc = {f: (xl[f] if f in XL_FAMILIES else V05.get(f, xl[f])) for f in fams}
    print(f"\nXL macro {xl_macro:.3f} | v0.5 0.679 | Clef 0.813 | Jev 0.840")
    print(f"CASCADE (XL on {sorted(XL_FAMILIES)}, v0.5 elsewhere) macro {sum(casc.values())/len(casc):.3f}")


if __name__ == "__main__":
    asyncio.run(main())
