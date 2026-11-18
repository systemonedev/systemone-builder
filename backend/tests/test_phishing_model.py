"""Unit tests for Stage-0 email pre-parser, counterfactual pair splitting, frontier engine factory, and benchmark metrics."""

from __future__ import annotations

from systemone_builder.config import Settings
from systemone_builder.kenning.phishing_data import (
    extract_email_signals,
    load_workspace_counterfactuals,
    phishing_counterfactual_suite,
    phishing_counterfactual_synthetic_suite,
    phishing_layouts_suite,
    split_counterfactual_dataset,
    to_training_rows,
)
from systemone_builder.system_one.bench import (
    pairwise_flip_accuracy,
    pr_auc,
    question_metrics,
    recall_at_fpr,
    roc_auc,
)
from systemone_builder.system_one.factory import build_engine


def test_workspace_counterfactuals_have_no_placeholders_and_strict_pair_split():
    ws = load_workspace_counterfactuals()
    assert len(ws) == 160, f"expected 160 workspace counterfactual items, got {len(ws)}"
    for item in ws:
        assert "[suspicious link removed]" not in str(item), f"placeholder left in {item['id']}"

    train_items, held_items = split_counterfactual_dataset(seed=42)
    train_pairs = {x["pair_id"] for x in train_items}
    held_pairs = {x["pair_id"] for x in held_items}
    assert not (train_pairs & held_pairs), f"counterfactual twin leakage across splits: {train_pairs & held_pairs}"
    assert len(held_items) == 80  # 40 counterfactual pairs


def test_stage0_signal_extractor_separates_counterfactual_twins():
    ws = {x["id"]: x for x in load_workspace_counterfactuals()}
    # p001a vs p001b: link_domain_mismatch
    s_1a = extract_email_signals(ws["p001a"])
    s_1b = extract_email_signals(ws["p001b"])
    assert s_1a["link_vs_sender_domain_mismatch"] is True
    assert s_1b["link_vs_sender_domain_mismatch"] is False

    # q001a vs q001b: sharer_outside_org
    s_q1a = extract_email_signals(ws["q001a"])
    s_q1b = extract_email_signals(ws["q001b"])
    assert s_q1a["sharer_same_org"] is False
    assert s_q1b["sharer_same_org"] is True

    # q009a vs q009b: recipient_org_spread
    s_q9a = extract_email_signals(ws["q009a"])
    s_q9b = extract_email_signals(ws["q009b"])
    assert s_q9a["multi_org_recipient_spread"] is True
    assert s_q9b["multi_org_recipient_spread"] is False

    # Fixed false positives on benign counterfactuals: p012b (ESP bounce), p029b (brand tenant),
    # p037b (-portal), p039b (-authority), q031b/q032b (multi CC within <=2 orgs)
    for benign_id in ("p012b", "p029b", "p037b", "p039b", "q031b", "q032b"):
        sb = extract_email_signals(ws[benign_id])
        assert sb["lookalike_or_homoglyph_domain"] is False, f"{benign_id} falsely flagged lookalike"
        assert sb["reply_to_mismatch"] is False, f"{benign_id} falsely flagged reply_to_mismatch"
        assert sb["multi_org_recipient_spread"] is False, f"{benign_id} falsely flagged spread"

    # Fixed true positives on subtle counterfactuals: p029a (rnicrosoft homoglyph),
    # q026a (role vs address mismatch), q033a (SaaS platform claim mismatch), q015a (role vs doc mismatch)
    assert extract_email_signals(ws["p029a"])["lookalike_or_homoglyph_domain"] is True
    assert extract_email_signals(ws["q026a"])["role_vs_address_mismatch"] is True
    assert extract_email_signals(ws["q033a"])["saas_platform_claim_mismatch"] is True
    assert extract_email_signals(ws["q015a"])["recipient_role_vs_document_mismatch"] is True


def test_stage0_flat_email_enrichment_and_trusted_links():
    from systemone_builder.kenning.phishing_data import maybe_enrich_email_state
    from systemone_builder.kenning.cascade import _fast_path_answer
    from systemone_builder.system_one.contract import Question

    legit_flat = {
        "sender": "Patrick Wright",
        "from": "patrick.wright@cedarlogic.com",
        "subject": "quarterly plan now in Microsoft Teams workspace",
        "body": "Join here: https://teams.microsoft.com/",
        "link_display_text": "Join Teams workspace",
        "link_url": "https://teams.microsoft.com/",
    }
    enriched_legit = maybe_enrich_email_state(legit_flat)
    assert "signals" in enriched_legit
    assert "signals" not in legit_flat  # non-mutating
    assert enriched_legit["signals"]["trusted_cross_domain_link"] is True
    assert enriched_legit["signals"]["link_vs_sender_domain_mismatch"] is False
    assert enriched_legit["signals"]["ephemeral_or_shortener_link"] is False

    phish_flat = {
        "sender": "Mackenzie Baker",
        "from": "mackenzie22@gmail.com",
        "subject": "Mutual Non-Disclosure Agreement: qrco.de Partnership Inquiry",
        "body": "Review via https://qrco.de/bfMWom",
        "link_display_text": "View",
        "link_url": "https://qrco.de/bfMWom",
    }
    enriched_phish = maybe_enrich_email_state(phish_flat)
    assert enriched_phish["signals"]["ephemeral_or_shortener_link"] is True
    assert enriched_phish["signals"]["freemail_sender_with_link_or_business_lure"] is True

    # Non-email webpage state must remain untouched
    web = {"webpage_triage": {"url": "https://example.com"}}
    assert maybe_enrich_email_state(web) == web

    # Cascade fast-path choice consistency
    q_verdict = Question(type="choice", instructions="Verdict?", criteria={"phishing": "bad", "legitimate": "good"})
    assert _fast_path_answer(q_verdict, 0.95).choice == "phishing"
    assert _fast_path_answer(q_verdict, 0.03).choice == "legitimate"


def test_phishing_training_rows_and_suites_build_cleanly():
    train_items, _ = split_counterfactual_dataset(seed=42)
    rows = to_training_rows(train_items[:20], seed=7, signal_dropout=0.5)
    assert len(rows) >= 60
    assert all("pair_id" in r and "questions" in r and "targets" in r for r in rows)
    # Verify signal dropout actually produces structured_json rows both with and without signals
    sj_rows = [r for r in rows if isinstance(r["state"], dict) and "headers" in r["state"] and "body" in r["state"]]
    with_sig = sum(1 for r in sj_rows if "signals" in r["state"])
    without_sig = sum(1 for r in sj_rows if "signals" not in r["state"])
    assert with_sig > 0 and without_sig > 0, f"expected signal dropout mix, got {with_sig} with / {without_sig} without"

    cf_suite = phishing_counterfactual_suite(seed=42)
    assert cf_suite.name == "phishing_counterfactual"
    assert len(cf_suite.items) == 48
    assert all(not it.id.startswith("cf") for it in cf_suite.items)
    assert cf_suite.gate == "is_phish"

    cf_syn = phishing_counterfactual_synthetic_suite(seed=42)
    assert cf_syn.name == "phishing_counterfactual_synthetic"
    assert len(cf_syn.items) == 32
    assert all(it.id.startswith("cf") for it in cf_syn.items)

    ly_suite = phishing_layouts_suite(seed=42, n_pairs=5)
    assert ly_suite.name == "phishing_layouts"
    assert len(ly_suite.items) == 40  # 10 emails x 4 layouts


def test_ranking_and_counterfactual_flip_metrics():
    ps = [0.95, 0.85, 0.70, 0.20, 0.10, 0.02]
    ys = [1, 1, 1, 0, 0, 0]
    assert roc_auc(ps, ys) == 1.0
    assert pr_auc(ps, ys) == 1.0
    assert recall_at_fpr(ps, ys, max_fpr=0.01) == 1.0

    suite = phishing_counterfactual_suite(seed=42)
    # Perfect predictions -> 1.0 pairwise flip accuracy
    perfect_preds = {it.id: (0.95 if it.labels["is_phish"] == 1 else 0.05, it.labels["is_phish"]) for it in suite.items}
    assert pairwise_flip_accuracy(suite, perfect_preds) == 1.0

    # Always predicting phishing (p=0.9) gets 50% accuracy but 0% pairwise flip accuracy!
    always_yes = {it.id: (0.90, it.labels["is_phish"]) for it in suite.items}
    assert pairwise_flip_accuracy(suite, always_yes) == 0.0

    fake_results = [
        {"id": it.id, "answers": {"is_phish": {"type": "noul", "noul": 0.95 if it.labels["is_phish"] == 1 else 0.05}}, "error": None}
        for it in suite.items
    ]
    qm = question_metrics("is_phish", suite.questions["is_phish"], suite, fake_results, hi=0.9, lo=0.1)
    assert qm["accuracy"] == 1.0
    assert qm["roc_auc"] == 1.0
    assert qm["pairwise_flip_accuracy"] == 1.0


def test_frontier_engine_factory_supports_multi_provider_specs():
    s = Settings(
        openai_api_key="sk-test",
        anthropic_api_key="sk-ant-test",
        gemini_api_key="AIza-test",
        openrouter_api_key="sk-or-test",
    )
    e_oai = build_engine("openai:gpt-4o", s)
    assert e_oai.name == "openai (gpt-4o)"

    e_ant = build_engine("anthropic:claude-sonnet-4-5", s)
    assert e_ant.name == "anthropic (claude-sonnet-4-5)"

    e_gem = build_engine("gemini:gemini-2.5-flash", s)
    assert e_gem.name == "gemini (gemini-2.5-flash)"

    e_or = build_engine("openrouter:openai/gpt-4o-mini", s)
    assert e_or.name == "openrouter (openai/gpt-4o-mini)"

    e_llm = build_engine("llm:anthropic/claude-haiku-4-5", s)
    assert e_llm.name == "anthropic (claude-haiku-4-5)"


async def test_decision_index_suite_requires_external_rows():
    import pytest
    from systemone_builder.system_one.decision_index_suite import decision_index_suite

    # decision_index_suite must fail loudly if external rows are not present
    with pytest.raises(RuntimeError, match="Real Jev Decision Index rows not found"):
        await decision_index_suite(n=10, seed=42, cache_dir=None)


def test_decision_index_chance_corrected_scoring():
    from systemone_builder.system_one.decision_index_suite import (
        AREA_WEIGHTS,
        _build_builtin_items,
        _canonical_questions,
        chance_corrected_question_score,
        decision_index_breakdown,
        num_options,
    )

    qs = _canonical_questions()
    items = _build_builtin_items(seed=42)
    assert len(qs) == 15
    assert len(items) == 100

    # Random guessing (1/K) must yield 0.0 chance-corrected score; 100% must yield 100.0
    assert chance_corrected_question_score({"n": 20, "accuracy": 0.50}, k_options=2, total_asked=20) == 0.0
    assert chance_corrected_question_score({"n": 20, "accuracy": 1.00}, k_options=2, total_asked=20) == 100.0
    assert chance_corrected_question_score({"n": 20, "accuracy": 0.25}, k_options=4, total_asked=20) == 0.0
    assert chance_corrected_question_score({"n": 20, "accuracy": 1.00}, k_options=4, total_asked=20) == 100.0

    # Coverage adjustment: if only 10 of 20 items were answered (at 100% accuracy), score is halved (50.0)
    assert chance_corrected_question_score({"n": 10, "accuracy": 1.00}, k_options=4, total_asked=20) == 50.0

    # Construct raw accuracies that correspond to exactly 70.0 chance-corrected points for each K options
    fake_report = {
        "questions": {
            qid: {
                "n": 7,
                "type": q.type,
                "k_options": num_options(q),
                "accuracy": (1.0 / num_options(q)) + 0.70 * (1.0 - 1.0 / num_options(q)),
            }
            for qid, q in qs.items()
        }
    }
    di = decision_index_breakdown(fake_report)
    assert di["index_score"] == 70.0
    assert set(di["area_scores"].keys()) == set(AREA_WEIGHTS.keys())


def test_phreshphish_webpage_signal_extractor_and_samples():
    from systemone_builder.system_one.phreshphish_suite import (
        _builtin_phreshphish_samples,
        extract_webpage_signals,
    )

    samples = _builtin_phreshphish_samples(seed=42)
    assert len(samples) == 80
    phish_row = next(s for s in samples if s["label"] == "phish" and s["target"] == "Microsoft")
    benign_row = next(s for s in samples if s["label"] == "benign" and s["target"] == "Microsoft" and "<form" in s["html"])

    p_sig = extract_webpage_signals(phish_row["url"], phish_row["html"], phish_row["target"])["webpage_triage"]["signals"]
    b_sig = extract_webpage_signals(benign_row["url"], benign_row["html"], benign_row["target"])["webpage_triage"]["signals"]

    assert p_sig["brand_vs_domain_mismatch"] is True
    assert p_sig["has_password_input"] is True
    assert p_sig["external_form_action"] is True
    assert "brand_vs_domain_mismatch" in p_sig["risk_flags"]

    assert b_sig["domain_matches_brand"] is True
    assert b_sig["brand_vs_domain_mismatch"] is False
    assert b_sig["external_form_action"] is False


def test_unwrap_safelinks_and_proofpoint_urldefense():
    from systemone_builder.kenning.phishing_data import _url_domain, unwrap_gateway_url

    inner = "https://docs.google.com/document/d/123/edit"
    pp = f"https://urldefense.com/v3/__{inner}__;!!O0uQkS0!abc123$"
    wrapped = (
        "https://nam11.safelinks.protection.outlook.com/?url="
        + pp.replace(":", "%3A").replace("/", "%2F")
        + "&data=05%7C02%7Cuser%40example.edu&reserved=0"
    )
    assert unwrap_gateway_url(wrapped) == inner
    assert _url_domain(wrapped) == "google.com"

    sig = extract_email_signals({
        "headers": {
            "from_display": "Colleague (via Google Docs)",
            "from_addr": "drive-shares-dm-noreply@google.com",
            "reply_to": "colleague@example.edu",
            "to_addr": "team@example.edu",
            "subject": "Shared notes",
        },
        "auth": {"spf": "pass", "dkim": "pass", "dmarc": "pass"},
        "links": [{"href": wrapped, "anchor_text": "Open"}],
        "attachments": [],
        "body": "See notes.",
    })
    assert sig["link_domains"] == ["google.com"]
    assert sig["link_vs_sender_domain_mismatch"] is False

