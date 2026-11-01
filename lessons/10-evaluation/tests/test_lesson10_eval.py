"""Lesson 10 — evaluation. Run with: python -m grader run 10"""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType

import pytest

from grader.paths import REPO_ROOT
from parsrag import FakeLLM, Message
from parsrag.eval import QAItem
from parsrag.types import Chunk

hint = pytest.mark.hint


# -------------------------------------------------------------- correctness


@hint(
    fa="ارجاع‌های [n] را حذف، normalize و lower کنید؛ جداکنندهٔ هزارگان را بردارید و نیم‌فاصله را فاصله کنید؛ بعد ایست‌واژه‌ها را کنار بگذارید.",
    en="Drop [n] markers, normalize + lower, remove thousands separators, ZWNJ → space, then drop stop words.",
)
def test_answer_tokens(gm: ModuleType) -> None:
    """answer_tokens makes Persian answers comparable"""
    assert gm.answer_tokens("۲۰۰ هزار تومان است [1].") == ["200", "هزار", "تومان"]
    assert gm.answer_tokens("۱۰٬۴۲۰ عضو") == ["10420", "عضو"]
    assert (
        gm.answer_tokens("نمي‌شود") == gm.answer_tokens("نمی شود") == ["نمی"]
    )  # «شود» is a stop word
    assert gm.answer_tokens("") == []


@hint(
    fa="Counter(pred) & Counter(ref) برای هم‌پوشانی؛ precision = common/len(pred)، recall = common/len(ref)؛ F1 = میانگین هارمونیک.",
    en="Overlap = Counter(pred) & Counter(ref); P = common/len(pred), R = common/len(ref); F1 = harmonic mean.",
)
@pytest.mark.parametrize(
    ("pred", "ref", "f1"),
    [
        ("۲۰۰ هزار تومان.", "200 هزار تومان", 1.0),
        ("هزینهٔ پیک 200 هزار تومان است [1].", "۲۰۰ هزار تومان.", 2 * 0.6 * 1.0 / 1.6),
        ("اصفهان", "تهران", 0.0),
        ("", "تهران", 0.0),
    ],
)
def test_token_f1(gm: ModuleType, pred: str, ref: str, f1: float) -> None:
    """token_f1 is SQuAD-style overlap after Persian normalisation"""
    assert gm.token_f1(pred, ref) == pytest.approx(f1)


# ---------------------------------------------------------------- citations


def _item() -> QAItem:
    return QAItem(
        id="q", question="؟", answer="200", doc_id="ship", evidence="پیک در تهران 200 هزار"
    )


@hint(
    fa="سهم قطعه‌های ارجاع‌شده‌ای که contains_evidence دارند؛ اگر چیزی ارجاع نشده یا evidence ندارد → None.",
    en="Share of cited chunks with contains_evidence; None if nothing is cited or there is no evidence.",
)
def test_citation_precision(gm: ModuleType) -> None:
    """citation precision = cited chunks that really contain the evidence"""
    good = Chunk(id="a", doc_id="ship", text="هزینهٔ پیک در تهران ۲۰۰ هزار تومان است.")
    bad = Chunk(id="b", doc_id="faq", text="ساعت کاری")
    assert gm.citation_precision([good, bad], _item()) == 0.5
    assert gm.citation_precision([], _item()) is None
    assert gm.citation_precision([good], QAItem(id="u", question="?", answer="-")) is None
    assert gm.citation_hit([bad, good], _item()) and not gm.citation_hit([bad], _item())


# ------------------------------------------------------------- faithfulness


@hint(
    fa="ارجاع‌ها را حذف و متن را در . ! ? ؟ و خط جدید بشکنید؛ تکه‌های بدون توکن را دور بریزید.",
    en="Remove citations, split at . ! ? ؟ and newlines, drop pieces without tokens.",
)
def test_split_claims(gm: ModuleType) -> None:
    """split_claims cuts an answer into sentences"""
    assert gm.split_claims("هزینه ۲۰۰ هزار است [1]. ارسال رایگان است؟ [2]\nبله.") == [
        "هزینه ۲۰۰ هزار است .",
        "ارسال رایگان است؟",
        "بله.",
    ]
    assert gm.split_claims("[1]") == []


@hint(
    fa="برای هر متن: سهم واژه‌های ادعا که در متن هست، و آیا همهٔ عددها هست. بهترین متن را نگه دارید؛ پشتیبانی = سهم ≥ آستانه و عددها درست.",
    en="Per context: share of claim words present, and whether all numbers are present; keep the best; supported = share ≥ threshold and numbers ok.",
)
def test_check_claim_and_faithfulness(gm: ModuleType) -> None:
    """faithfulness counts claims supported by one retrieved chunk, numbers included"""
    ctx = ["هزینهٔ پیک در تهران 200 هزار تومان است.", "ارسال به شهرستان‌ها با پست پیشتاز است."]
    ok = gm.check_claim("هزینهٔ پیک در تهران ۲۰۰ هزار تومان است.", ctx)
    assert ok.supported and ok.numbers_ok and ok.support == 1.0
    wrong_number = gm.check_claim("هزینهٔ پیک در تهران 250 هزار تومان است.", ctx)
    assert not wrong_number.supported and not wrong_number.numbers_ok
    result = gm.faithfulness(
        "هزینهٔ پیک 200 هزار تومان است [1]. ارسال با هواپیما در یک ساعت است.", ctx
    )
    assert result.score == 0.5 and [c.supported for c in result.claims] == [True, False]
    assert gm.faithfulness("", ctx).score == 1.0


@hint(
    fa="پیام system + پیام user با قالب؛ امتیاز را با «امتیاز: N» بخوانید و بر ۱۰ تقسیم کنید؛ «دلیل: ...» را هم بردارید.",
    en="System + user message from the template; parse 'امتیاز: N' and divide by 10; also read 'دلیل: ...'.",
)
def test_llm_judge(gm: ModuleType) -> None:
    """LLMJudge sends a Persian judge prompt and parses its verdict"""
    seen: list[list[Message]] = []

    def handler(messages: list[Message]) -> str:
        seen.append(messages)
        return "امتیاز: ۸\nدلیل: یک ادعا در متن نیست."

    judge = gm.LLMJudge(FakeLLM(handler=handler))
    r = judge.faithfulness("هزینهٔ پیک؟", "۲۰۰ هزار تومان", ["پیک ۲۰۰ هزار تومان"])
    assert r.score == pytest.approx(0.8) and r.reason == "یک ادعا در متن نیست."
    assert seen[0][0].role == "system" and "[1] پیک ۲۰۰ هزار تومان" in seen[0][1].content
    c = judge.correctness("هزینهٔ پیک؟", "۲۰۰ هزار", "200 هزار تومان")
    assert c.score == pytest.approx(0.8) and "پاسخ مرجع: 200 هزار تومان" in seen[1][1].content
    assert gm.LLMJudge(FakeLLM(["خوب بود"])).faithfulness("?", "x", []).score is None


# ---------------------------------------------------------------- statistics


@hint(
    fa="rng = np.random.default_rng(seed)؛ اندیس‌ها با integers(0, n, size=(n_resamples, n))؛ میانگین هر سطر؛ quantile در alpha/2 و 1-alpha/2.",
    en="rng = default_rng(seed); indices integers(0, n, (n_resamples, n)); row means; quantiles at alpha/2, 1-alpha/2.",
)
def test_bootstrap_ci(st: ModuleType) -> None:
    """bootstrap_ci gives a deterministic interval around the mean"""
    values = [1.0] * 45 + [0.0] * 14
    ci = st.bootstrap_ci(values)
    assert ci.mean == pytest.approx(45 / 59)
    assert 0.6 < ci.low < ci.mean < ci.high < 0.9
    assert st.bootstrap_ci(values) == ci  # same seed, same interval
    assert st.bootstrap_ci([0.5] * 10).low == pytest.approx(0.5)
    with pytest.raises(ValueError):
        st.bootstrap_ci([])


@hint(
    fa="تفاضل هر پرسش (b - a) را بسازید و روی همان bootstrap_ci بگیرید — جفت‌ها با هم نمونه‌گیری می‌شوند.",
    en="Bootstrap the per-question differences b - a, so pairs are resampled together.",
)
def test_paired_bootstrap(st: ModuleType) -> None:
    """paired_bootstrap: a consistent small gain is real, a coin flip is not"""
    a = [0.5] * 30
    assert st.paired_bootstrap(a, [0.6] * 30).low > 0
    noisy = st.paired_bootstrap([0.0, 1.0] * 15, [1.0, 0.0] * 15)
    assert noisy.low < 0 < noisy.high
    with pytest.raises(ValueError):
        st.paired_bootstrap([1.0], [1.0, 2.0])


# ----------------------------------------------------------------------- gate


@hint(
    fa="برای هر معیار baseline: اگر در current نیست یا کمتر از baseline - tolerance است، یک Change برگردانید.",
    en="For every baseline metric: missing, or below baseline - tolerance → a Change.",
)
def test_compare_metrics(gate: ModuleType) -> None:
    """the gate fails on drops beyond the tolerance and on missing metrics"""
    base = {"original": {"mrr": 0.90, "f1": 0.30}, "paraphrase": {"mrr": 0.50}}
    same = {"original": {"mrr": 0.90, "f1": 0.30}, "paraphrase": {"mrr": 0.50}}
    assert gate.compare_metrics(base, same) == []
    small = {"original": {"mrr": 0.885, "f1": 0.35}, "paraphrase": {"mrr": 0.60, "new": 1.0}}
    assert gate.compare_metrics(base, small, tolerance=0.02) == []
    worse = {"original": {"mrr": 0.85, "f1": 0.30}, "paraphrase": {}}
    failures = gate.compare_metrics(base, worse, tolerance=0.02)
    assert [(c.group, c.metric, c.current) for c in failures] == [
        ("original", "mrr", 0.85),
        ("paraphrase", "mrr", None),
    ]
    assert failures[0].delta == pytest.approx(-0.05)
    with pytest.raises(ValueError):
        gate.compare_metrics(base, same, tolerance=-1)


# ------------------------------------------------------------------ the payoff


def test_committed_baseline_is_well_formed(gate: ModuleType) -> None:
    """the course's committed baseline covers both question sets"""
    path = Path(REPO_ROOT) / "docs" / "eval" / "baseline.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert set(data["metrics"]) == {"original", "paraphrase"}
    assert (
        data["metrics"]["original"]["retrieval_r@1"]
        > data["metrics"]["paraphrase"]["retrieval_r@1"]
    )
    assert gate.compare_metrics(data["metrics"], data["metrics"], data["tolerance"]) == []


def test_your_metrics_reproduce_the_baseline(gm: ModuleType) -> None:
    """your token_f1 and faithfulness give the committed baseline numbers on the course pipeline"""
    from parsrag.eval.benchmark import load_fixture_benchmark
    from parsrag.eval.pipeline import build_course_pipeline

    baseline = json.loads((Path(REPO_ROOT) / "docs/eval/baseline.json").read_text(encoding="utf-8"))
    generator = build_course_pipeline()
    _, qa = load_fixture_benchmark()
    f1s, faith = [], []
    for item in qa:
        ans = generator.answer(item.question)
        if item.answerable:
            f1s.append(0.0 if ans.abstained else gm.token_f1(ans.text, item.answer))
        if not ans.abstained:
            contexts = [r.chunk.text for r in ans.sources]
            faith.append(gm.faithfulness(ans.text, contexts).score)
    assert sum(f1s) / len(f1s) == pytest.approx(
        baseline["metrics"]["original"]["answer_f1"], abs=1e-3
    )
    assert sum(faith) / len(faith) == pytest.approx(
        baseline["metrics"]["original"]["faithfulness"], abs=1e-3
    )
