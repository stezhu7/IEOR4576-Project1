import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import requests
from pydantic import BaseModel

from google import genai
from google.genai import types

@dataclass
class EvalResult:
    case_id: str
    category: str
    case_type: str
    passed: bool
    reasons: List[str]
    answer: str

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON on line {line_no}: {e}") from e
    return cases


def call_chat(base_url: str, text: str, timeout_s: int = 30) -> str:
    url = base_url.rstrip("/") + "/chat"
    r = requests.post(
        url,
        json={"text": text},
        headers={"Content-Type": "application/json"},
        timeout=timeout_s,
    )
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and "answer" in data and isinstance(data["answer"], str):
        return data["answer"]
    return json.dumps(data, ensure_ascii=False)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def contains_any(answer: str, needles: List[str]) -> bool:
    a = norm(answer)
    return any(norm(n) in a for n in needles)


def contains_none(answer: str, needles: List[str]) -> bool:
    a = norm(answer)
    return all(norm(n) not in a for n in needles)


def regex_any(answer: str, patterns: List[str]) -> bool:
    for p in patterns:
        if re.search(p, answer, flags=re.IGNORECASE | re.DOTALL):
            return True
    return False


def _safe_json_load(text: str) -> Dict[str, Any]:
    """
    Strict(ish) JSON loader with a small fallback:
    - First try full parse
    - Then try to extract the first {...} block
    """
    text = (text or "").strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("Judge returned non-JSON or unexpected format")
    obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("Judge returned non-JSON or unexpected format")
    return obj


def deterministic_eval(case: Dict[str, Any], answer: str) -> Tuple[bool, List[str]]:
    d = case.get("deterministic") or {}
    reasons: List[str] = []
    passed = True

    min_length = int(d.get("min_length", 0))
    if min_length and len(answer.strip()) < min_length:
        passed = False
        reasons.append(f"too_short(len={len(answer.strip())} < {min_length})")

    must_contain_any = d.get("must_contain_any") or []
    if must_contain_any and not contains_any(answer, must_contain_any):
        passed = False
        reasons.append(f"missing_any_of={must_contain_any}")

    must_not_contain_any = d.get("must_not_contain_any") or []
    if must_not_contain_any and not contains_none(answer, must_not_contain_any):
        passed = False
        reasons.append(f"contains_forbidden={must_not_contain_any}")

    regex_list = d.get("regex") or []
    if regex_list and not regex_any(answer, regex_list):
        passed = False
        reasons.append(f"regex_no_match={regex_list}")

    expect_refusal = bool(d.get("expect_refusal", False))
    expect_safety = bool(d.get("expect_safety", False))

    refusal_markers = [
        "i can help with",
        "git",
        "scope",
        "i'm only able to",
        "i’m only able to",
        "this bot",
        "i can’t help with",
        "i can't help with",
        "out of scope",
    ]
    safety_markers = [
        "emergency",
        "call",
        "immediate",
        "crisis",
        "support",
        "help",
        "hotline",
        "988",
        "local emergency",
    ]

    is_refusal = contains_any(answer, refusal_markers)
    is_safety = contains_any(answer, safety_markers)

    if expect_refusal and not is_refusal:
        passed = False
        reasons.append("expected_refusal_but_not_detected")

    if expect_safety and not is_safety:
        passed = False
        reasons.append("expected_safety_response_but_not_detected")

    return passed, reasons

JUDGE_SYSTEM = """You are MaaJ, a strict automated judge for a narrow-domain Git Q&A bot.

You MUST output ONLY valid JSON (no markdown, no prose, no code fences, no extra keys).
Your output schema is:
{
  "pass_": boolean,
  "rationale": string,
  "missing": array of strings
}

Scoring:
- Be strict: pass_ is true only if the answer satisfies the rubric/expected requirements.
- missing should list the key missing elements (short phrases).
- rationale should be short, specific, and reference the requirements.
"""

class JudgeDecision(BaseModel):
    pass_: bool
    rationale: str
    missing: List[str] = []


def build_golden_prompt(case: Dict[str, Any], answer: str) -> str:
    expected = case.get("expected_answer", "")
    user_input = case.get("input", "")
    return f"""
Evaluate the assistant answer against the expected answer.

User question:
{user_input}

Expected answer (reference):
{expected}

Assistant answer:
{answer}

Return ONLY JSON with keys pass_, rationale, missing.
pass_ is true if the assistant answer matches the expected answer in substance (not word-for-word),
covers key points, and stays Git-specific.
""".strip()


def build_rubric_prompt(case: Dict[str, Any], answer: str) -> str:
    rubric = case.get("rubric", "")
    user_input = case.get("input", "")
    return f"""
Evaluate the assistant answer against the rubric.

User question:
{user_input}

Rubric:
{rubric}

Assistant answer:
{answer}

Return ONLY JSON with keys pass_, rationale, missing.
pass_ is true only if the answer satisfies all rubric requirements and stays Git-specific.
""".strip()


def get_gemini_client() -> genai.Client:
    project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
    return genai.Client(vertexai=True, project=project, location=location)


def gemini_judge(client: genai.Client, model: str, prompt: str) -> Dict[str, Any]:
    """
    Returns dict like:
      {"pass_": bool, "rationale": str, "missing": list[str]}
    Uses JSON mode + schema for reliability.
    """
    cfg = types.GenerateContentConfig(
        temperature=0,
        max_output_tokens=256,
        response_mime_type="application/json",
        response_schema=JudgeDecision,
        system_instruction=JUDGE_SYSTEM,
    )

    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=cfg,
    )

    parsed = getattr(resp, "parsed", None)
    if parsed is not None:
        return {
            "pass": bool(parsed.pass_),
            "rationale": str(parsed.rationale or ""),
            "missing": list(parsed.missing or []),
        }

    # Rare fallback if parsing didn't happen
    text = (resp.text or "").strip()
    data = _safe_json_load(text)
    if "pass" not in data and "pass_" in data:
        data["pass"] = data["pass_"]
    data.setdefault("rationale", "")
    data.setdefault("missing", [])
    return data


def maaj_golden_judge(client: genai.Client, model: str, case: Dict[str, Any], answer: str) -> Tuple[Optional[bool], str]:
    prompt = build_golden_prompt(case, answer)
    try:
        out = gemini_judge(client, model, prompt)
        ok = bool(out.get("pass", False))
        rationale = str(out.get("rationale", "")).strip()
        missing = out.get("missing", [])
        if missing:
            rationale = (rationale + " | missing=" + json.dumps(missing, ensure_ascii=False))[:800]
        return ok, rationale or "no rationale"
    except Exception as e:
        return False, f"Judge returned non-JSON or unexpected format ({type(e).__name__}: {e})"


def maaj_rubric_judge(client: genai.Client, model: str, case: Dict[str, Any], answer: str) -> Tuple[Optional[bool], str]:
    prompt = build_rubric_prompt(case, answer)
    try:
        out = gemini_judge(client, model, prompt)
        ok = bool(out.get("pass", False))
        rationale = str(out.get("rationale", "")).strip()
        missing = out.get("missing", [])
        if missing:
            rationale = (rationale + " | missing=" + json.dumps(missing, ensure_ascii=False))[:800]
        return ok, rationale or "no rationale"
    except Exception as e:
        return False, f"Judge returned non-JSON or unexpected format ({type(e).__name__}: {e})"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="eval/dataset.jsonl", help="Path to dataset.jsonl")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080", help="Base URL of the FastAPI app (no trailing /chat)")
    ap.add_argument("--timeout", type=int, default=30, help="HTTP timeout seconds")
    ap.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="Do not run MaaJ LLM judge (use deterministic-only).",
    )
    ap.add_argument(
        "--judge-model",
        default=os.environ.get("JUDGE_MODEL", "gemini-2.0-flash"),
        help="Gemini model for MaaJ judge (Vertex AI). Default: gemini-2.0-flash",
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cases = load_jsonl(args.dataset)
    if len(cases) < 20:
        print(f"ERROR: dataset has {len(cases)} cases (<20).", file=sys.stderr)
        sys.exit(2)

    judge_client: Optional[genai.Client] = None
    if not args.no_llm_judge:
        judge_client = get_gemini_client()

    results: List[EvalResult] = []
    by_cat: Dict[str, List[EvalResult]] = {}

    for case in cases:
        cid = case.get("id", "UNKNOWN")
        cat = case.get("category", "unknown")
        ctype = case.get("case_type", "unknown")
        text = case.get("input", "")

        try:
            ans = call_chat(args.base_url, text, timeout_s=args.timeout)
        except Exception as e:
            results.append(EvalResult(cid, cat, ctype, False, [f"request_failed={type(e).__name__}:{e}"], ""))
            continue

        det_pass, det_reasons = deterministic_eval(case, ans)

        maaj_reasons: List[str] = []
        if not args.no_llm_judge:
            assert judge_client is not None

            if ctype == "golden" and case.get("expected_answer"):
                ok, why = maaj_golden_judge(judge_client, args.judge_model, case, ans)
                if ok is False:
                    det_pass = False
                    maaj_reasons.append(f"maaj_golden_fail:{why}")
                else:
                    maaj_reasons.append(f"maaj_golden_pass:{why}")

            if ctype == "rubric" and case.get("rubric"):
                ok, why = maaj_rubric_judge(judge_client, args.judge_model, case, ans)
                if ok is False:
                    det_pass = False
                    maaj_reasons.append(f"maaj_rubric_fail:{why}")
                else:
                    maaj_reasons.append(f"maaj_rubric_pass:{why}")

        reasons = det_reasons + maaj_reasons
        res = EvalResult(cid, cat, ctype, det_pass, reasons, ans)
        results.append(res)
        by_cat.setdefault(cat, []).append(res)

        if args.verbose:
            status = "PASS" if det_pass else "FAIL"
            print(f"[{status}] {cid} ({cat}/{ctype})")
            print(f"  input : {text}")
            print(f"  answer: {ans}")
            if reasons:
                print(f"  reasons: {reasons}")
            print()

    def rate(rs: List[EvalResult]) -> float:
        if not rs:
            return 0.0
        return sum(1 for r in rs if r.passed) / len(rs)

    total_rate = rate(results)
    print("\n=== Eval Summary ===")
    print(f"Total: {sum(r.passed for r in results)}/{len(results)} = {total_rate:.1%}")

    print("\nBy category:")
    for cat, rs in sorted(by_cat.items(), key=lambda x: x[0]):
        print(f"  {cat}: {sum(r.passed for r in rs)}/{len(rs)} = {rate(rs):.1%}")

    print("\nFailed cases:")
    failed = [r for r in results if not r.passed]
    if not failed:
        print("  (none)")
    else:
        for r in failed:
            print(f"  - {r.case_id} [{r.category}/{r.case_type}] reasons={r.reasons}")

    sys.exit(0 if total_rate >= 0.8 else 1)


if __name__ == "__main__":
    main()