from typing import Dict, Any, List, Tuple
from openai import OpenAI
from settings import (
    OPENAI_MODEL,
    MAX_OPENAI_BUDGET_USD, OPENAI_PRICE_IN_PER_1K, OPENAI_PRICE_OUT_PER_1K
)

client = OpenAI(api_key=OPENAI_API_KEY)

class BudgetGuard:
    def __init__(self, max_usd: float, price_in: float, price_out: float):
        self.max_usd = max_usd
        self.price_in = price_in
        self.price_out = price_out
        self.spent = 0.0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0

    def estimate_cost(self, in_tokens: int, out_tokens: int) -> float:
        return (in_tokens/1000.0)*self.price_in + (out_tokens/1000.0)*self.price_out

    def can(self, in_tokens: int, out_tokens: int) -> bool:
        return (self.spent + self.estimate_cost(in_tokens, out_tokens)) <= self.max_usd

    def add(self, in_tokens: int, out_tokens: int):
        cost = self.estimate_cost(in_tokens, out_tokens)
        self.spent += cost
        self.prompt_tokens += in_tokens
        self.completion_tokens += out_tokens
        self.calls += 1

def summarize_many(job_desc: str, items: List[Dict[str, Any]],
                   max_in_chars: int = 1600, max_out_tokens: int = 120
                   ) -> Tuple[List[str], Dict[str, Any]]:

    guard = BudgetGuard(MAX_OPENAI_BUDGET_USD, OPENAI_PRICE_IN_PER_1K, OPENAI_PRICE_OUT_PER_1K)
    outs: List[str] = []

    for c in items:
        text = (c.get("text") or "")[:max_in_chars]
        prompt = (
            "You are a tech recruiter. Given a JOB DESCRIPTION and a CANDIDATE RESUME, "
            "write a concise 2–3 sentence summary of why this candidate fits. Be specific—"
            "mention matching skills, tools, years, or projects.\n\n"
            f"JOB DESCRIPTION:\n{job_desc}\n\n"
            f"CANDIDATE ({c.get('name','Unknown')}):\n{text}\n\n"
            "Summary:"
        )
        # rough estimate if API doesn't return usage
        in_est = max(1, len(prompt)//4)
        out_est = max_out_tokens
        if not guard.can(in_est, out_est):
            outs.append("(Skipped to stay under $1 budget.)")
            continue

        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_out_tokens,
                temperature=0.2
            )
            msg = (resp.choices[0].message.content or "").strip()
            outs.append(msg if msg else "(No summary.)")

            # Real usage if present; else estimated
            usage = getattr(resp, "usage", None)
            if usage and hasattr(usage, "prompt_tokens"):
                pt = int(getattr(usage, "prompt_tokens", in_est) or in_est)
                ct = int(getattr(usage, "completion_tokens", out_est) or out_est)
                guard.add(pt, ct)
            else:
                guard.add(in_est, out_est)
        except Exception:
            outs.append("(Summary failed.)")

    return outs, {
        "spent_usd": round(guard.spent, 6),
        "prompt_tokens": int(guard.prompt_tokens),
        "completion_tokens": int(guard.completion_tokens),
        "calls": int(guard.calls),
    }
