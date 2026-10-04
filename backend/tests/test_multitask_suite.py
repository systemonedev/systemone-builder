from __future__ import annotations

from systemone_builder.system_one.multitask_suite import TASKS, macro_average

BY_ID = {t.qid: t for t in TASKS}


def test_every_task_has_a_question_classes_and_a_licence():
    assert len(TASKS) >= 14 and len({t.qid for t in TASKS}) == len(TASKS)
    for t in TASKS:
        assert t.question.instructions and t.classes and t.licence and t.split


def test_row_parsers_on_real_shaped_rows():
    cases = {
        "answerable": ({"passage": "p", "question": "q", "answer": True}, 1),
        "claim_supported": ({"sentence1": "a", "sentence2": "b", "gold_label": "entailment"}, 1),
        "financial_sentiment": ({"text": "shares -3%", "label": 0}, "bearish"),
        "toxic_comment": ({"comment_text": "x", "label": 1}, 1),
        "counterfactual": ({"text": "I wish", "label": 1}, 1),
        "prompt_injection": ({"text": "ignore the above", "label": 1}, 1),
        "jailbreak": ({"prompt": "you are DAN", "type": "jailbreak"}, 1),
        "fine_emotion": ({"text": "thanks!", "labels": [15]}, "gratitude"),
        "toxicity_level": ({"prompt": {"text": "t", "toxicity": 0.9}}, 2),
        "article_topic": ({"title": "T", "content": "c", "label": 9}, "Animal"),
        "banking_intent": ({"text": "where is my refund", "category": "Refund_not_showing_up"}, "refund_not_showing_up"),
    }
    for qid, (row, label) in cases.items():
        state, got = BY_ID[qid].row(row)
        assert got == label and state, qid
        assert label in BY_ID[qid].classes, qid


def test_rows_outside_the_task_are_skipped():
    assert BY_ID["fine_emotion"].row({"text": "x", "labels": [15, 17]}) is None      # multi-label
    assert BY_ID["toxicity_level"].row({"prompt": {"text": "t", "toxicity": 0.3}}) is None   # between bands
    assert BY_ID["article_topic"].row({"title": "T", "content": "c", "label": 1})[1] is None  # class not used


def test_macro_average_uses_accuracy_and_exact_level():
    report = {"questions": {"a": {"n": 10, "accuracy": 0.8}, "b": {"n": 10, "exact_level": 0.6},
                            "c": {"n": 0}}}
    assert abs(macro_average(report) - 0.7) < 1e-9
