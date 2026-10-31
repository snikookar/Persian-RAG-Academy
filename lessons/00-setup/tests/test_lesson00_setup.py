"""Lesson 00 — setup. Run with: python -m grader run 00"""

from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest

from parsrag import FakeLLM, HashingEmbedding

hint = pytest.mark.hint


@hint(
    fa="فقط دو عدد اول نسخه (major, minor) را به صورت tuple مقایسه کنید.",
    en="Compare only (major, minor) as tuples.",
)
@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ((3, 10, 0), True),
        ((3, 12, 4), True),
        ((4, 0, 0), True),
        ((3, 9, 18), False),
        ((2, 7, 18), False),
    ],
)
def test_python_version_ok(doctor: ModuleType, version: tuple[int, ...], expected: bool) -> None:
    """python_version_ok compares against the minimum version"""
    assert doctor.python_version_ok((3, 10), version) is expected


@hint(
    fa="بدون آرگومان، نسخهٔ پایتون فعلی را بررسی کنید.",
    en="With no argument, check sys.version_info.",
)
def test_python_version_ok_uses_running_python(doctor: ModuleType) -> None:
    """python_version_ok() checks the running interpreter by default"""
    assert doctor.python_version_ok() is True


@hint(
    fa="برای هر extra در EXTRA_MODULES بررسی کنید که همهٔ ماژول‌هایش با module_available در دسترس باشند.",
    en="For each extra in EXTRA_MODULES, all of its modules must pass module_available().",
)
def test_extras_status(doctor: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """extras_status reports which optional extras are installed"""
    monkeypatch.setattr(doctor, "module_available", lambda name: name in {"httpx", "fastapi"})
    status = doctor.extras_status()
    assert set(status) == set(doctor.EXTRA_MODULES)
    assert status["openai"] is True  # needs only httpx
    assert status["api"] is False  # needs fastapi AND uvicorn
    assert status["st"] is False


@hint(
    fa="InMemoryVectorStore + HashingEmbedding برای بازیابی، و FakeLLM(mode='extractive') برای پاسخ.",
    en="Use InMemoryVectorStore + HashingEmbedding to retrieve, FakeLLM(mode='extractive') to answer.",
)
def test_smoke_test_answers_offline(doctor: ModuleType) -> None:
    """smoke_test() retrieves the right document and answers offline"""
    answer = doctor.smoke_test()
    assert isinstance(answer, str)
    assert "ساعت شش" in answer


def test_fake_llm_is_deterministic() -> None:
    """FakeLLM gives the same answer every time (no network, no key)"""
    llm = FakeLLM(responses=["سلام!"])
    messages = [{"role": "user", "content": "یک سؤال"}]
    assert llm.generate(messages) == llm.generate(messages) == "سلام!"


def test_hashing_embedding_is_deterministic() -> None:
    """HashingEmbedding is deterministic and returns unit-length vectors"""
    a = HashingEmbedding().embed(["کتابخانه", "فروشگاه"])
    b = HashingEmbedding().embed(["کتابخانه", "فروشگاه"])
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, rtol=1e-5)
