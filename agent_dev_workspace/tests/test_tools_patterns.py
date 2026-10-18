import os
import json
import tempfile
import types
import pytest
import numpy as np
import pandas as pd
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools as make_patterns_tools


def test_mine_patterns_tool_known_answer_17_tfidf_fpgrowth():
    """Tests Known Answer #17: tfidf filtering with fpgrowth algorithm on catA and catB."""
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    patterns_tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in patterns_tools if t.name == "mine_patterns_tool")

    expected_keys = [
        "result_id", "category_name", "filtering_method", "algorithm",
        "surviving_terms_count", "pattern_count", "patterns", "note"
    ]

    # catA tfidf fpgrowth min_sup=1
    res_catA = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })

    assert list(res_catA.keys()) == expected_keys
    assert res_catA["category_name"] == "catA"
    assert res_catA["filtering_method"] == "tfidf"
    assert res_catA["algorithm"] == "fpgrowth"
    assert res_catA["surviving_terms_count"] == 1
    assert res_catA["pattern_count"] == 1
    assert res_catA["patterns"] == [{"pattern": ["alpha"], "support": 3}]

    artifact_key_A = "patterns_catA_tfidf_fpgrowth"
    assert session.artifacts[artifact_key_A] == [{"pattern": ["alpha"], "support": 3}]

    full_report_A = session.results_store[res_catA["result_id"]]["full_report"]
    assert list(full_report_A.columns) == ["pattern", "support"]
    assert len(full_report_A) == 1
    assert full_report_A.iloc[0]["pattern"] == "alpha"
    assert full_report_A.iloc[0]["support"] == 3

    assert isinstance(json.dumps(res_catA), str)

    # catB tfidf fpgrowth min_sup=1
    res_catB = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })

    assert list(res_catB.keys()) == expected_keys
    assert res_catB["category_name"] == "catB"
    assert res_catB["surviving_terms_count"] == 1
    assert res_catB["pattern_count"] == 1
    assert res_catB["patterns"] == [{"pattern": ["gamma"], "support": 4}]

    artifact_key_B = "patterns_catB_tfidf_fpgrowth"
    assert session.artifacts[artifact_key_B] == [{"pattern": ["gamma"], "support": 4}]

    full_report_B = session.results_store[res_catB["result_id"]]["full_report"]
    assert list(full_report_B.columns) == ["pattern", "support"]
    assert len(full_report_B) == 1
    assert full_report_B.iloc[0]["pattern"] == "gamma"
    assert full_report_B.iloc[0]["support"] == 4


def test_mine_patterns_tool_known_answer_17_zero_terms_survived():
    """Tests Known Answer #17: variance and term_frequency keep zero terms for both categories."""
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    patterns_tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in patterns_tools if t.name == "mine_patterns_tool")

    for cat in ["catA", "catB"]:
        for filt in ["variance", "term_frequency"]:
            res = mine_patterns_tool.invoke({
                "category_name": cat,
                "filtering_method": filt,
                "algorithm": "fpgrowth",
                "min_sup": 1
            })

            assert res["surviving_terms_count"] == 0
            assert res["pattern_count"] == 0
            assert res["patterns"] == []
            assert res["note"] == "no terms survived filtering"

            artifact_key = f"patterns_{cat}_{filt}_fpgrowth"
            assert session.artifacts[artifact_key] == []

            full_report = session.results_store[res["result_id"]]["full_report"]
            assert list(full_report.columns) == ["pattern", "support"]
            assert len(full_report) == 0
            assert isinstance(json.dumps(res), str)


def test_mine_patterns_tool_known_answer_20_topk_and_maxfpgrowth():
    """Tests Known Answer #20: topk and maxfpgrowth behavior on tfidf-filtered fixture data."""
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    patterns_tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in patterns_tools if t.name == "mine_patterns_tool")

    expected_keys = [
        "result_id", "category_name", "filtering_method", "algorithm",
        "surviving_terms_count", "pattern_count", "patterns", "note"
    ]

    # Top-K (k=1) catA
    res_topk_A = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    assert list(res_topk_A.keys()) == expected_keys
    assert res_topk_A["patterns"] == [{"pattern": ["alpha"], "support": 3}]
    assert session.artifacts["patterns_catA_tfidf_topk"] == [{"pattern": ["alpha"], "support": 3}]
    full_report_topk_A = session.results_store[res_topk_A["result_id"]]["full_report"]
    assert list(full_report_topk_A.columns) == ["pattern", "support"]
    assert len(full_report_topk_A) == 1
    assert full_report_topk_A.iloc[0]["pattern"] == "alpha"
    assert full_report_topk_A.iloc[0]["support"] == 3
    assert isinstance(json.dumps(res_topk_A), str)

    # Top-K (k=1) catB
    res_topk_B = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    assert list(res_topk_B.keys()) == expected_keys
    assert res_topk_B["patterns"] == [{"pattern": ["gamma"], "support": 4}]
    assert session.artifacts["patterns_catB_tfidf_topk"] == [{"pattern": ["gamma"], "support": 4}]
    full_report_topk_B = session.results_store[res_topk_B["result_id"]]["full_report"]
    assert list(full_report_topk_B.columns) == ["pattern", "support"]
    assert len(full_report_topk_B) == 1
    assert full_report_topk_B.iloc[0]["pattern"] == "gamma"
    assert full_report_topk_B.iloc[0]["support"] == 4
    assert isinstance(json.dumps(res_topk_B), str)

    # MaxFPGrowth (min_sup=1) -> Expected 0 patterns
    for cat in ["catA", "catB"]:
        res_max = mine_patterns_tool.invoke({
            "category_name": cat,
            "filtering_method": "tfidf",
            "algorithm": "maxfpgrowth",
            "min_sup": 1
        })
        assert list(res_max.keys()) == expected_keys
        assert res_max["pattern_count"] == 0
        assert res_max["patterns"] == []
        assert res_max["note"] == "no patterns found"
        artifact_key_max = f"patterns_{cat}_tfidf_maxfpgrowth"
        assert session.artifacts[artifact_key_max] == []
        full_report_max = session.results_store[res_max["result_id"]]["full_report"]
        assert list(full_report_max.columns) == ["pattern", "support"]
        assert len(full_report_max) == 0
        assert isinstance(json.dumps(res_max), str)


def test_mine_patterns_tool_temp_file_cleanup(monkeypatch):
    """Tests that temporary CSV database files created during PAMI execution are cleaned up on success and error."""
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    patterns_tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in patterns_tools if t.name == "mine_patterns_tool")

    created_temp_files = []
    real_mkstemp = tempfile.mkstemp

    def mock_mkstemp(*args, **kwargs):
        fd, path = real_mkstemp(*args, **kwargs)
        created_temp_files.append(path)
        return fd, path

    monkeypatch.setattr(tempfile, "mkstemp", mock_mkstemp)

    # 1. Normal success path
    mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert len(created_temp_files) == 1
    assert not os.path.exists(created_temp_files[0])

    # 2. Forced PAMI error path
    def mock_fpgrowth_raise(*args, **kwargs):
        raise RuntimeError("Forced PAMI error")

    monkeypatch.setattr("tools_patterns.FPGrowth.FPGrowth", mock_fpgrowth_raise)

    with pytest.raises(RuntimeError, match="Forced PAMI error"):
        mine_patterns_tool.invoke({
            "category_name": "catA",
            "filtering_method": "tfidf",
            "algorithm": "fpgrowth",
            "min_sup": 1
        })

    assert len(created_temp_files) == 2
    for tmp_file in created_temp_files:
        assert not os.path.exists(tmp_file)


def test_mine_patterns_tool_validation():
    """Tests parameter validation and unloaded session handling."""
    session = SessionState()
    patterns_tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in patterns_tools if t.name == "mine_patterns_tool")

    # Unloaded
    with pytest.raises(ValueError, match="No dataset loaded"):
        mine_patterns_tool.invoke({"category_name": "catA", "filtering_method": "tfidf", "min_sup": 1})

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Empty category_name
    with pytest.raises(ValueError, match="non-empty string"):
        mine_patterns_tool.invoke({"category_name": "   ", "filtering_method": "tfidf", "min_sup": 1})

    # Invalid category
    with pytest.raises(ValueError, match="not found in dataset categories"):
        mine_patterns_tool.invoke({"category_name": "catXYZ", "filtering_method": "tfidf", "min_sup": 1})

    # Invalid filter
    with pytest.raises(ValueError, match="Invalid filtering_method"):
        mine_patterns_tool.invoke({"category_name": "catA", "filtering_method": "invalid", "min_sup": 1})

    # Invalid algorithm
    with pytest.raises(ValueError, match="Invalid algorithm"):
        mine_patterns_tool.invoke({"category_name": "catA", "filtering_method": "tfidf", "algorithm": "invalid", "min_sup": 1})

    # min_sup validation (missing, 0, negative, non-int float, str, bool)
    with pytest.raises(ValueError, match="min_sup must be a positive integer"):
        mine_patterns_tool.invoke({"category_name": "catA", "filtering_method": "tfidf", "algorithm": "fpgrowth"})

    with pytest.raises(ValueError, match="min_sup must be a positive integer"):
        mine_patterns_tool.invoke({"category_name": "catA", "filtering_method": "tfidf", "algorithm": "fpgrowth", "min_sup": 0})

    with pytest.raises(ValueError, match="min_sup must be a positive integer"):
        mine_patterns_tool.invoke({"category_name": "catA", "filtering_method": "tfidf", "algorithm": "fpgrowth", "min_sup": -1})

    with pytest.raises(ValueError, match="min_sup must be a positive integer"):
        mine_patterns_tool.func(category_name="catA", filtering_method="tfidf", algorithm="fpgrowth", min_sup=1.5)

    with pytest.raises(ValueError, match="min_sup must be a positive integer"):
        mine_patterns_tool.func(category_name="catA", filtering_method="tfidf", algorithm="fpgrowth", min_sup="1")

    with pytest.raises(ValueError, match="min_sup must be a positive integer"):
        mine_patterns_tool.func(category_name="catA", filtering_method="tfidf", algorithm="fpgrowth", min_sup=True)

    # k validation (missing, 0, negative, non-int float, str, bool)
    with pytest.raises(ValueError, match="k must be a positive integer"):
        mine_patterns_tool.invoke({"category_name": "catA", "filtering_method": "tfidf", "algorithm": "topk"})

    with pytest.raises(ValueError, match="k must be a positive integer"):
        mine_patterns_tool.invoke({"category_name": "catA", "filtering_method": "tfidf", "algorithm": "topk", "k": 0})

    with pytest.raises(ValueError, match="k must be a positive integer"):
        mine_patterns_tool.func(category_name="catA", filtering_method="tfidf", algorithm="topk", k=1.5)

    with pytest.raises(ValueError, match="k must be a positive integer"):
        mine_patterns_tool.func(category_name="catA", filtering_method="tfidf", algorithm="topk", k="1")

    with pytest.raises(ValueError, match="k must be a positive integer"):
        mine_patterns_tool.func(category_name="catA", filtering_method="tfidf", algorithm="topk", k=True)


def test_mine_patterns_tool_empty_vocabulary_and_no_transactions(monkeypatch):
    """Tests edge cases: empty vocabulary and no non-empty transactions without executing PAMI."""
    session = SessionState()
    session.dataframe = pd.DataFrame({"text": ["the the", "the"], "category_name": ["catA", "catA"]})

    pami_called_vocab = {"called": False}

    def mock_df2db_vocab(*args, **kwargs):
        pami_called_vocab["called"] = True
        raise AssertionError("PAMI DF2DB should not be called for empty vocabulary")

    monkeypatch.setattr("tools_patterns.DF2DB.DF2DB", mock_df2db_vocab)

    patterns_tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in patterns_tools if t.name == "mine_patterns_tool")

    # 1. Empty vocabulary path ("the" is stopword)
    res_empty_vocab = mine_patterns_tool.invoke({"category_name": "catA", "filtering_method": "tfidf", "min_sup": 1})
    assert pami_called_vocab["called"] is False
    assert res_empty_vocab["surviving_terms_count"] == 0
    assert res_empty_vocab["pattern_count"] == 0
    assert res_empty_vocab["patterns"] == []
    assert res_empty_vocab["note"] == "no terms survived filtering"

    artifact_key = "patterns_catA_tfidf_fpgrowth"
    assert session.artifacts[artifact_key] == []

    full_report_empty = session.results_store[res_empty_vocab["result_id"]]["full_report"]
    assert list(full_report_empty.columns) == ["pattern", "support"]
    assert len(full_report_empty) == 0
    assert isinstance(json.dumps(res_empty_vocab), str)

    # 2. No non-empty transactions path
    pami_called_trans = {"called": False}

    def mock_df2db_trans(*args, **kwargs):
        pami_called_trans["called"] = True
        raise AssertionError("PAMI DF2DB should not be called when transaction DataFrame is empty")

    monkeypatch.setattr("tools_patterns.DF2DB.DF2DB", mock_df2db_trans)

    session2 = SessionState()
    session2.dataframe = pd.DataFrame({"text": ["alpha"], "category_name": ["catA"]})
    patterns_tools2 = make_patterns_tools(session2)
    mine_patterns_tool2 = next(t for t in patterns_tools2 if t.name == "mine_patterns_tool")

    import tools_patterns
    real_pd_dataframe = pd.DataFrame

    class ZeroTransDataFrame(real_pd_dataframe):
        @property
        def T(self):
            return real_pd_dataframe(
                np.zeros((len(self.columns), len(self.index)), dtype=int),
                index=self.columns,
                columns=self.index
            )

    def mock_dataframe_factory(*args, **kwargs):
        cols = kwargs.get("columns", args[1] if len(args) > 1 else None)
        normalized_cols = list(cols) if cols is not None else None
        if normalized_cols == ["alpha"]:
            return ZeroTransDataFrame(*args, **kwargs)
        return real_pd_dataframe(*args, **kwargs)

    # Proxy pd module specifically for tools_patterns without mutating global pandas
    tools_patterns_pd_proxy = types.SimpleNamespace(
        DataFrame=mock_dataframe_factory
    )
    for attr in dir(pd):
        if not attr.startswith("_") and attr != "DataFrame":
            setattr(tools_patterns_pd_proxy, attr, getattr(pd, attr))

    monkeypatch.setattr(tools_patterns, "pd", tools_patterns_pd_proxy)

    res_no_trans = mine_patterns_tool2.invoke({"category_name": "catA", "filtering_method": "tfidf", "min_sup": 1})
    assert pami_called_trans["called"] is False
    assert res_no_trans["surviving_terms_count"] == 1
    assert res_no_trans["pattern_count"] == 0
    assert res_no_trans["patterns"] == []
    assert res_no_trans["note"] == "no non-empty transactions"

    full_report_no_trans = session2.results_store[res_no_trans["result_id"]]["full_report"]
    assert list(full_report_no_trans.columns) == ["pattern", "support"]
    assert len(full_report_no_trans) == 0
    assert session2.artifacts["patterns_catA_tfidf_fpgrowth"] == []
    assert isinstance(json.dumps(res_no_trans), str)


def test_mine_patterns_tool_non_ascii_terms():
    """Tests that non-ASCII terms are safely encoded for PAMI and mapped back to original strings."""
    session = SessionState()
    session.dataframe = pd.DataFrame({
        "text": ["café résumé", "café résumé"],
        "category_name": ["catA", "catA"]
    })

    patterns_tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in patterns_tools if t.name == "mine_patterns_tool")

    res = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })

    assert res["surviving_terms_count"] == 2
    assert res["pattern_count"] > 0

    all_pattern_items = [item for p in res["patterns"] for item in p["pattern"]]

    # Verify that BOTH original non-ASCII terms appear in the returned patterns
    assert "café" in all_pattern_items
    assert "résumé" in all_pattern_items

    # Verify that NO ASCII placeholder identifiers (e.g. item_0, item_1) leak through
    assert not any(item.startswith("item_") for item in all_pattern_items)
