import json
import matplotlib.figure
import pandas as pd
import pytest
from pydantic import ValidationError
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


def test_build_dtm_tool_known_answer_4():
    """Tests build_dtm_tool against Known Answer #4 in TEST_FIXTURE.md using sample_fixture.csv."""
    # Create single shared session
    session = SessionState()

    # Load premade load_dataset_tool bound to this session
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load sample fixture dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Instantiate build_dtm_tool using the same session
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    # Invoke build_dtm_tool with default settings
    res = build_dtm_tool.invoke({})

    # Known Answer #4 expectations:
    # 1. Vocabulary (5 terms): alpha, always, beta, delta, gamma
    expected_vocab = ["alpha", "always", "beta", "delta", "gamma"]
    assert session.feature_names == expected_vocab

    # 2. Matrix shape: 8 documents x 5 terms
    assert session.feature_matrix is not None
    assert session.feature_matrix.shape == (8, 5)

    # 3. Sparsity: 21 non-zero cells out of 40 total elements -> sparsity_pct = 47.5
    assert res["n_documents"] == 8
    assert res["n_terms"] == 5
    assert res["non_zero"] == 21
    assert res["total_elements"] == 40
    assert res["sparsity_pct"] == pytest.approx(47.5, abs=1e-3)

    # 4. Result ID format
    assert res["result_id"].startswith("dtm_")

    # 5. Check vectorizer artifact stored in session
    assert "count_vectorizer" in session.artifacts


def test_term_frequency_tool_known_answer_14():
    """Tests term_frequency_tool against Known Answer #14 in TEST_FIXTURE.md using sample_fixture.csv."""
    # 1. Single shared SessionState
    session = SessionState()

    # 2. Load premade load_dataset_tool and load sample_fixture.csv
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 3. Build DTM using build_dtm_tool
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Record state before running term_frequency_tool to verify immutability
    matrix_before = session.feature_matrix.copy()
    names_before = list(session.feature_names)

    # 4. Run term_frequency_tool
    term_frequency_tool = next(t for t in dtm_tools if t.name == "term_frequency_tool")
    res = term_frequency_tool.invoke({})

    # Verify session.feature_matrix and session.feature_names were unchanged
    assert (session.feature_matrix - matrix_before).nnz == 0
    assert session.feature_names == names_before

    # Fetch stored full_report
    result_id = res["result_id"]
    stored_entry = session.results_store[result_id]
    full_report = stored_entry["full_report"]

    # Verify columns: exactly ["term", "frequency"]
    assert list(full_report.columns) == ["term", "frequency"]

    # Verify expected term totals from Known Answer #14
    report_dict = dict(zip(full_report["term"], full_report["frequency"]))
    assert report_dict["alpha"] == 6
    assert report_dict["always"] == 7
    assert report_dict["beta"] == 3
    assert report_dict["delta"] == 4
    assert report_dict["gamma"] == 7

    # Verify row order sorted by frequency descending, term ascending for ties:
    # always (7), gamma (7), alpha (6), delta (4), beta (3)
    expected_order = ["always", "gamma", "alpha", "delta", "beta"]
    assert list(full_report["term"]) == expected_order


def test_dtm_heatmap_tool_known_answer_21():
    """Tests dtm_heatmap_tool against Known Answer #21 in TEST_FIXTURE.md using sample_fixture.csv."""
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    dtm_heatmap_tool = next(t for t in dtm_tools if t.name == "dtm_heatmap_tool")

    # 1. Test default slice (n_terms=20, n_documents=20) against Known Answer #21
    res = dtm_heatmap_tool.invoke({})

    expected_terms = ["alpha", "always", "beta", "delta", "gamma"]
    expected_doc_labels = [f"doc_{i}" for i in range(8)]
    expected_matrix = [
        [3, 1, 1, 0, 0],
        [2, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 3],
        [0, 1, 0, 1, 2],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0],
    ]

    assert res["n_documents"] == 8
    assert res["n_terms"] == 5
    assert res["terms"] == expected_terms
    assert res["document_labels"] == expected_doc_labels
    assert res["matrix"] == expected_matrix

    # Verify full_report in results_store
    result_id = res["result_id"]
    stored_entry = session.results_store[result_id]
    full_report = stored_entry["full_report"]
    assert list(full_report.columns) == expected_terms
    assert list(full_report.index) == list(range(8))
    assert full_report.values.tolist() == expected_matrix

    # Verify pending_figure is a matplotlib Figure
    assert isinstance(session.pending_figure, matplotlib.figure.Figure)

    # Verify json.dumps succeeds
    serialized_json = json.dumps(res)
    assert isinstance(serialized_json, str)

    # 2. Test corner slice (n_terms=2, n_documents=3)
    res_corner = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})
    assert res_corner["n_terms"] == 2
    assert res_corner["n_documents"] == 3
    assert res_corner["terms"] == ["alpha", "always"]
    assert res_corner["document_labels"] == ["doc_0", "doc_1", "doc_2"]
    assert res_corner["matrix"] == [[3, 1], [2, 1], [1, 1]]


def test_dtm_heatmap_tool_validation_and_unbuilt():
    """Tests dtm_heatmap_tool error handling for unbuilt DTM and invalid slice parameters."""
    session = SessionState()
    dtm_tools = make_tools(session)
    dtm_heatmap_tool = next(t for t in dtm_tools if t.name == "dtm_heatmap_tool")

    # Test unbuilt DTM raises ValueError
    with pytest.raises(ValueError, match="No Document-Term Matrix found"):
        dtm_heatmap_tool.invoke({})

    # Load fixture & build DTM for parameter validation
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Test n_terms <= 0
    with pytest.raises(ValueError, match="positive integer"):
        dtm_heatmap_tool.invoke({"n_terms": 0})

    # Test n_documents <= 0
    with pytest.raises(ValueError, match="positive integer"):
        dtm_heatmap_tool.invoke({"n_documents": -1})

    # Test boolean n_terms / n_documents raises ValueError via wrapped function
    with pytest.raises(ValueError, match="positive integer"):
        dtm_heatmap_tool.func(n_terms=True)

    with pytest.raises(ValueError, match="positive integer"):
        dtm_heatmap_tool.func(n_documents=True)


def test_build_dtm_tool_max_df_explicit_mode_and_semantics():
    """Tests build_dtm_tool with max_df_mode='proportion' vs 'count' and asserts exact vocabulary filtering and schema validation."""
    session = SessionState()
    session.dataframe = pd.DataFrame({
        "text": [
            "apple banana",   # "apple" in doc 0, 1; "banana" in doc 0, 2
            "apple cherry",   # "cherry" in doc 1 only
            "banana dragon",  # "dragon" in doc 2, 3
            "dragon elder",   # "elder" in doc 3 only
        ],
        "category_name": ["catA", "catA", "catB", "catB"]
    })

    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    # 1. Default mode (max_df_mode omitted with max_df=1.0): proportion 100% (all 5 terms survive)
    res_default_float = build_dtm_tool.invoke({"max_df": 1.0})
    vectorizer_default_float = session.artifacts["count_vectorizer"]
    assert type(vectorizer_default_float.max_df) is float
    assert vectorizer_default_float.max_df == 1.0
    assert session.feature_names == ["apple", "banana", "cherry", "dragon", "elder"]
    assert res_default_float["n_terms"] == 5

    # 2. Default mode (max_df_mode omitted with max_df=1): interprets as float proportion 1.0 (all 5 terms survive)
    res_default_int = build_dtm_tool.invoke({"max_df": 1})
    vectorizer_default_int = session.artifacts["count_vectorizer"]
    assert type(vectorizer_default_int.max_df) is float
    assert vectorizer_default_int.max_df == 1.0
    assert session.feature_names == ["apple", "banana", "cherry", "dragon", "elder"]
    assert res_default_int["n_terms"] == 5

    # 3. Explicit max_df_mode="proportion": max_df=1 -> proportion 1.0 (all 5 terms survive)
    res_prop = build_dtm_tool.invoke({"max_df": 1, "max_df_mode": "proportion"})
    vectorizer_prop = session.artifacts["count_vectorizer"]
    assert type(vectorizer_prop.max_df) is float
    assert vectorizer_prop.max_df == 1.0
    assert session.feature_names == ["apple", "banana", "cherry", "dragon", "elder"]
    assert res_prop["n_terms"] == 5

    # 4. Explicit max_df_mode="count": max_df=1 -> int count <= 1 doc (only 'cherry' and 'elder' survive)
    res_count = build_dtm_tool.invoke({"max_df": 1, "max_df_mode": "count"})
    vectorizer_count = session.artifacts["count_vectorizer"]
    assert type(vectorizer_count.max_df) is int
    assert vectorizer_count.max_df == 1
    assert session.feature_names == ["cherry", "elder"]
    assert res_count["n_terms"] == 2

    # 5. Non-integer in count mode raises ValueError
    with pytest.raises(ValueError, match="must be an integer count"):
        build_dtm_tool.invoke({"max_df": 1.5, "max_df_mode": "count"})

    # 6. Invalid max_df_mode string rejected at schema boundary with pydantic ValidationError
    with pytest.raises(ValidationError, match="proportion|count"):
        build_dtm_tool.invoke({"max_df": 1, "max_df_mode": "invalid_mode"})
