import matplotlib.figure
import numpy as np
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_exploration import make_tools as make_exploration_tools


@pytest.fixture
def session_and_tools():
    session = SessionState()
    
    # 1. Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # 2. Instantiate build_dtm_tool and exploration tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    
    exploration_tools = make_exploration_tools(session)
    cosine_similarity_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")
    feature_correlation_matrix_tool = next(t for t in exploration_tools if t.name == "feature_correlation_matrix_tool")
    
    # 3. Setup dataset and DTM
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    build_dtm_tool.invoke({})
    
    return session, cosine_similarity_tool, feature_correlation_matrix_tool


def test_cosine_similarity_tool_known_answer_16(session_and_tools):
    session, cosine_similarity_tool, _ = session_and_tools
    
    # Pair 1: 0 vs 1 (both catA, near-identical text) -> 0.9847
    res_0_1 = cosine_similarity_tool.invoke({"doc_idx1": 0, "doc_idx2": 1})
    assert res_0_1["doc_idx1"] == 0
    assert res_0_1["doc_idx2"] == 1
    assert res_0_1["similarity"] == pytest.approx(0.9847, abs=1e-3)
    assert session.artifacts["cosine_similarity_0_1"] == pytest.approx(0.9847, abs=1e-3)
    assert res_0_1["result_id"] in session.results_store
    assert session.results_store[res_0_1["result_id"]]["record"].summary["similarity"] == pytest.approx(0.9847, abs=1e-3)
    
    # Pair 2: 0 vs 3 (catA vs catB, share only "always") -> 0.0909
    res_0_3 = cosine_similarity_tool.invoke({"doc_idx1": 0, "doc_idx2": 3})
    assert res_0_3["similarity"] == pytest.approx(0.0909, abs=1e-3)
    assert session.artifacts["cosine_similarity_0_3"] == pytest.approx(0.0909, abs=1e-3)
    assert res_0_3["result_id"] in session.results_store
    assert session.results_store[res_0_3["result_id"]]["record"].summary["similarity"] == pytest.approx(0.0909, abs=1e-3)
    
    # Pair 3: 3 vs 4 (both catB, near-identical text) -> 0.9847
    res_3_4 = cosine_similarity_tool.invoke({"doc_idx1": 3, "doc_idx2": 4})
    assert res_3_4["similarity"] == pytest.approx(0.9847, abs=1e-3)
    assert session.artifacts["cosine_similarity_3_4"] == pytest.approx(0.9847, abs=1e-3)
    assert res_3_4["result_id"] in session.results_store
    assert session.results_store[res_3_4["result_id"]]["record"].summary["similarity"] == pytest.approx(0.9847, abs=1e-3)
    
    # Pair 4: 0 vs 7 (catA vs empty-text row) -> 0.0
    res_0_7 = cosine_similarity_tool.invoke({"doc_idx1": 0, "doc_idx2": 7})
    assert res_0_7["similarity"] == 0.0
    assert session.artifacts["cosine_similarity_0_7"] == 0.0
    assert res_0_7["result_id"] in session.results_store
    assert session.results_store[res_0_7["result_id"]]["record"].summary["similarity"] == 0.0


def test_cosine_similarity_tool_validation_and_unbuilt():
    session = SessionState()
    exploration_tools = make_exploration_tools(session)
    cosine_similarity_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")
    
    # Unloaded dataset / unbuilt vectorizer raises ValueError
    with pytest.raises(ValueError, match="No dataset loaded"):
        cosine_similarity_tool.invoke({"doc_idx1": 0, "doc_idx2": 1})
    
    # Load dataset without DTM raises vectorizer ValueError
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    with pytest.raises(ValueError, match="Fitted vectorizer missing"):
        cosine_similarity_tool.invoke({"doc_idx1": 0, "doc_idx2": 1})
    
    # Build DTM for index validation testing
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})
    
    # Out of bounds indices raise ValueError
    with pytest.raises(ValueError, match="out of bounds"):
        cosine_similarity_tool.invoke({"doc_idx1": -1, "doc_idx2": 0})
        
    with pytest.raises(ValueError, match="out of bounds"):
        cosine_similarity_tool.invoke({"doc_idx1": 0, "doc_idx2": 99})


def test_feature_correlation_matrix_tool_known_answer_19(session_and_tools):
    session, _, feature_correlation_matrix_tool = session_and_tools
    
    session.pending_figure = None
    
    # Known Answer #19: Top 20 terms by variance (5 terms total in fixture)
    res = feature_correlation_matrix_tool.invoke({})
    
    expected_terms = ["alpha", "gamma", "delta", "beta", "always"]
    expected_corr = [
        [1.0000, -0.5718, -0.6882, 0.8885, 0.2601],
        [-0.5718, 1.0000, 0.8307, -0.6435, 0.3140],
        [-0.6882, 0.8307, 1.0000, -0.7746, 0.3780],
        [0.8885, -0.6435, -0.7746, 1.0000, 0.2928],
        [0.2601, 0.3140, 0.3780, 0.2928, 1.0000],
    ]
    
    assert res["terms"] == expected_terms
    assert np.allclose(res["correlation_matrix"], expected_corr, atol=1e-3)
    
    # Check session artifacts and pending figure
    assert "feature_correlation" in session.artifacts
    assert np.allclose(session.artifacts["feature_correlation"], expected_corr, atol=1e-3)
    assert isinstance(session.pending_figure, matplotlib.figure.Figure)
    
    # Check results_store record
    result_id = res["result_id"]
    assert result_id in session.results_store
    stored_summary = session.results_store[result_id]["record"].summary
    assert stored_summary["terms"] == expected_terms
    assert np.allclose(stored_summary["correlation_matrix"], expected_corr, atol=1e-3)


def test_feature_correlation_matrix_tool_unbuilt():
    session = SessionState()
    exploration_tools = make_exploration_tools(session)
    feature_correlation_matrix_tool = next(t for t in exploration_tools if t.name == "feature_correlation_matrix_tool")
    
    # Unbuilt DTM raises ValueError
    with pytest.raises(ValueError, match="No Document-Term Matrix found"):
        feature_correlation_matrix_tool.invoke({})
