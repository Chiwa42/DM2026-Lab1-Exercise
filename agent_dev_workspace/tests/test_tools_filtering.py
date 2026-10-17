import json
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools


def test_variance_filter_tool_known_answer_5():
    # 1. Initialize single shared SessionState
    session = SessionState()

    # 2. Load premade load_dataset_tool and load sample_fixture.csv
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 3. Build DTM using build_dtm_tool from tools_dtm.py
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Record matrix and feature_names state before filtering to check immutability.
    # A sparse difference check (a - b).nnz == 0 is used to avoid dense conversions.
    matrix_before = session.feature_matrix.copy()
    names_before = list(session.feature_names)

    # 4. Instantiate filtering tools and run variance_filter_tool with threshold=0.15
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")
    res = variance_filter_tool.invoke({"threshold": 0.15})

    # Check that session.feature_matrix and session.feature_names were NOT mutated
    assert (session.feature_matrix - matrix_before).nnz == 0
    assert session.feature_names == names_before

    # Verify return dictionary expected keys ("kept_terms" and "removed_terms")
    kept_terms = res["kept_terms"]
    removed_terms = res["removed_terms"]

    assert sorted(kept_terms) == ["alpha", "beta", "delta", "gamma"]
    assert removed_terms == ["always"]

    # Verify full_report stored in results_store
    result_id = res["result_id"]
    stored_entry = session.results_store[result_id]
    full_report = stored_entry["full_report"]

    # Verify full_report columns: [feature, variance, keep]
    assert list(full_report.columns) == ["feature", "variance", "keep"]

    # Verify full_report sorting (variance descending)
    variances = full_report["variance"].tolist()
    assert variances == sorted(variances, reverse=True)

    # Verify full_report["keep"] boolean values for each term via dictionary assertion
    keep_dict = dict(zip(full_report["feature"], full_report["keep"]))
    assert keep_dict == {
        "alpha": True,
        "always": False,
        "beta": True,
        "delta": True,
        "gamma": True,
    }

    # Verify per-term variances against Known Answer #5 from TEST_FIXTURE.md
    expected_variances = {
        "alpha": 1.1875,
        "always": 0.109375,
        "beta": 0.234375,
        "delta": 0.25,
        "gamma": 1.109375,
    }

    report_dict = dict(zip(full_report["feature"], full_report["variance"]))
    for term, exp_var in expected_variances.items():
        assert report_dict[term] == pytest.approx(exp_var, abs=1e-4)


def test_pearson_filter_tool_known_answer_6():
    """Tests pearson_filter_tool against Known Answer #6 in TEST_FIXTURE.md using sample_fixture.csv."""
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    matrix_before = session.feature_matrix.copy()
    df_before = session.dataframe.copy()

    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    # Invoke pearson_filter_tool for target_class="catB" and threshold=0.5
    res = pearson_filter_tool.invoke({"target_class": "catB", "threshold": 0.5})

    # Verify session.feature_matrix and session.dataframe were unchanged
    assert (session.feature_matrix - matrix_before).nnz == 0
    assert session.dataframe.equals(df_before)

    # Known Answer #6 correlations for target_class="catB"
    expected_correlations = {
        "alpha": -0.6882,
        "always": 0.3780,
        "beta": -0.7746,
        "delta": 1.0000,
        "gamma": 0.8307,
    }

    # Verify full_report in results_store
    result_id = res["result_id"]
    stored_entry = session.results_store[result_id]
    full_report = stored_entry["full_report"]

    assert list(full_report.columns) == ["feature", "pearson_r", "keep"]

    # Verify exact sorted feature order: delta, gamma, always, alpha, beta
    expected_feature_order = ["delta", "gamma", "always", "alpha", "beta"]
    assert list(full_report["feature"]) == expected_feature_order

    report_dict = dict(zip(full_report["feature"], full_report["pearson_r"]))
    for term, exp_r in expected_correlations.items():
        assert report_dict[term] == pytest.approx(exp_r, abs=1e-3)

    # Verify keep values for threshold=0.5 (|r| > 0.5)
    keep_dict = dict(zip(full_report["feature"], full_report["keep"]))
    assert keep_dict == {
        "delta": True,
        "gamma": True,
        "always": False,
        "alpha": True,
        "beta": True,
    }

    # Verify kept_terms and removed_terms preserve pearson_r-descending order
    assert res["kept_terms"] == ["delta", "gamma", "alpha", "beta"]
    assert res["removed_terms"] == ["always"]

    # Assert json.dumps succeeds
    assert isinstance(json.dumps(res), str)


def test_pearson_filter_tool_validation_and_unbuilt():
    """Tests pearson_filter_tool validation for unbuilt DTM, misaligned inputs, and invalid parameters."""
    session = SessionState()
    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    # 1. Unbuilt DTM raises ValueError
    with pytest.raises(ValueError, match="No feature matrix"):
        pearson_filter_tool.invoke({"target_class": "catB"})

    # Load fixture & build DTM for parameter validation
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # 2. Unknown target_class raises ValueError
    with pytest.raises(ValueError, match="not found in dataset labels"):
        pearson_filter_tool.invoke({"target_class": "catXYZ"})

    # 3. Invalid threshold (< 0.0 or > 1.0) raises ValueError
    with pytest.raises(ValueError, match="numeric value in \\[0.0, 1.0\\]"):
        pearson_filter_tool.invoke({"target_class": "catB", "threshold": -0.1})

    # 4. Strict bool rejection via wrapped function
    with pytest.raises(ValueError, match="numeric value in \\[0.0, 1.0\\]"):
        pearson_filter_tool.func(target_class="catB", threshold=True)

    # 5. Misaligned feature names raises ValueError
    original_feature_names = list(session.feature_names)
    session.feature_names = original_feature_names[:2]
    with pytest.raises(ValueError, match="does not match DTM column count"):
        pearson_filter_tool.invoke({"target_class": "catB"})

    # Restore feature_names before testing label misalignment independently
    session.feature_names = original_feature_names

    # 6. Misaligned labels raises ValueError
    session.labels = session.labels[:3]
    with pytest.raises(ValueError, match="does not match DTM row count"):
        pearson_filter_tool.invoke({"target_class": "catB"})


def test_spearman_filter_tool_known_answer_6():
    """Tests spearman_filter_tool against Known Answer #6 in TEST_FIXTURE.md using sample_fixture.csv."""
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    matrix_before = session.feature_matrix.copy()
    df_before = session.dataframe.copy()

    filtering_tools = make_filtering_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    # Invoke spearman_filter_tool for target_class="catB" and threshold=0.5
    res = spearman_filter_tool.invoke({"target_class": "catB", "threshold": 0.5})

    # Verify session.feature_matrix and session.dataframe were unchanged
    assert (session.feature_matrix - matrix_before).nnz == 0
    assert session.dataframe.equals(df_before)

    # Known Answer #6 Spearman rank correlations for target_class="catB"
    expected_correlations = {
        "alpha": -0.7500,
        "always": 0.3780,
        "beta": -0.7746,
        "delta": 1.0000,
        "gamma": 0.9363,
    }

    # Verify full_report in results_store
    result_id = res["result_id"]
    stored_entry = session.results_store[result_id]
    full_report = stored_entry["full_report"]

    assert list(full_report.columns) == ["feature", "spearman_r", "keep"]

    # Verify exact sorted feature order: delta, gamma, always, alpha, beta
    expected_feature_order = ["delta", "gamma", "always", "alpha", "beta"]
    assert list(full_report["feature"]) == expected_feature_order

    report_dict = dict(zip(full_report["feature"], full_report["spearman_r"]))
    for term, exp_r in expected_correlations.items():
        assert report_dict[term] == pytest.approx(exp_r, abs=1e-3)

    # Verify keep values for threshold=0.5 (|r| > 0.5)
    keep_dict = dict(zip(full_report["feature"], full_report["keep"]))
    assert keep_dict == {
        "delta": True,
        "gamma": True,
        "always": False,
        "alpha": True,
        "beta": True,
    }

    # Verify kept_terms and removed_terms preserve spearman_r-descending order
    assert res["kept_terms"] == ["delta", "gamma", "alpha", "beta"]
    assert res["removed_terms"] == ["always"]

    # Assert json.dumps succeeds
    assert isinstance(json.dumps(res), str)


def test_spearman_filter_tool_validation_and_unbuilt():
    """Tests spearman_filter_tool validation for unbuilt DTM, misaligned inputs, and invalid parameters."""
    session = SessionState()
    filtering_tools = make_filtering_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    # 1. Unbuilt DTM raises ValueError
    with pytest.raises(ValueError, match="No feature matrix"):
        spearman_filter_tool.invoke({"target_class": "catB"})

    # Load fixture & build DTM for parameter validation
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # 2. Unknown target_class raises ValueError
    with pytest.raises(ValueError, match="not found in dataset labels"):
        spearman_filter_tool.invoke({"target_class": "catBXYZ"})

    # 3. Invalid threshold (< 0.0 or > 1.0) raises ValueError
    with pytest.raises(ValueError, match="numeric value in \\[0.0, 1.0\\]"):
        spearman_filter_tool.invoke({"target_class": "catB", "threshold": -0.1})

    # 4. Strict bool rejection via wrapped function
    with pytest.raises(ValueError, match="numeric value in \\[0.0, 1.0\\]"):
        spearman_filter_tool.func(target_class="catB", threshold=True)

    # 5. Misaligned feature names raises ValueError
    original_feature_names = list(session.feature_names)
    session.feature_names = original_feature_names[:2]
    with pytest.raises(ValueError, match="does not match DTM column count"):
        spearman_filter_tool.invoke({"target_class": "catB"})

    # Restore feature_names before testing label misalignment independently
    session.feature_names = original_feature_names

    # 6. Misaligned labels raises ValueError
    session.labels = session.labels[:3]
    with pytest.raises(ValueError, match="does not match DTM row count"):
        spearman_filter_tool.invoke({"target_class": "catB"})
