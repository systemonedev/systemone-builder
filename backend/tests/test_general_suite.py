"""The general suite's offline logic: question coverage, the dialogue filter, per-family averages."""

from systemone_builder.system_one.general_generators import GENERATORS
from systemone_builder.system_one.general_suite import (
    FAMILY, GENERATED, QUESTIONS, SGD_SERVICES, _sgd_clean, family_averages,
)


def test_every_question_has_a_family_and_every_generator_is_asked():
    for kind, (primary, asked) in GENERATED.items():
        assert kind in GENERATORS
        assert primary in asked
        for qid in asked:
            assert qid in QUESTIONS and qid in FAMILY


def _dialogue(*turns: str, latest: str) -> dict:
    return {"conversation": [{"user" if i % 2 == 0 else "assistant": t} for i, t in enumerate(turns)],
            "latest_user_message": latest}


def test_dialogue_kept_only_when_visibly_about_its_service():
    hotel = _dialogue("I need a hotel in Sydney.", "I found 10 hotels.", latest="Can I smoke in the hotel?")
    assert _sgd_clean(hotel, "Hotels")
    assert not _sgd_clean(hotel, "Events")  # label and conversation disagree
    mixed = _dialogue("Book 4 bus tickets to Long Beach.", "Your tickets are booked.",
                      "Find me a two star hotel there.", latest="sounds right")
    assert not _sgd_clean(mixed, "Hotels")  # mentions another service
    vague = _dialogue("Yes, please do.", "How many?", latest="Four.")
    assert not any(_sgd_clean(vague, s) for s in SGD_SERVICES)  # no service visible at all


def test_family_averages_macro_per_family_and_score_exact_level():
    report = {"questions": {
        "refund_eligible": {"n": 10, "accuracy": 1.0},
        "access_allowed": {"n": 10, "accuracy": 0.5},
        "ticket_priority": {"n": 10, "exact_level": 0.4},
        "goal_completed": {"n": 0, "accuracy": None},  # not answered: left out
        "not_a_task": {"n": 10, "accuracy": 0.0},
    }}
    assert family_averages(report) == {"records": (1.0 + 0.5 + 0.4) / 3}
