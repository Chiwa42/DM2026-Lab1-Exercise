import numpy as np
import pandas as pd
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_reduction import make_tools as make_reduction_tools


@pytest.fixture
def session_and_tools():
    session = SessionState()
    
    # 1. Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # 2. Load student tools_dtm and tools_reduction bound to session
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    
    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")
    binarize_labels_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")
    
    # 3. Populate session dataframe and feature_matrix using sample_fixture.csv
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    build_dtm_tool.invoke({})
    
    return session, reduce_dimensions_tool, binarize_labels_tool


def test_reduce_dimensions_pca(session_and_tools):
    session, reduce_dimensions_tool, _ = session_and_tools
    
    session.pending_figure = None
    
    # Known answer #7: PCA
    res = reduce_dimensions_tool.invoke({"method": "pca"})
    
    assert res["method"] == "pca"
    assert res["n_components"] == 2
    assert session.pending_figure is not None
    assert "reduced_pca" in session.artifacts
    
    expected_explained_variance = [0.7532, 0.1965]
    assert np.allclose(res["explained_variance_ratio"], expected_explained_variance, atol=1e-3)
    
    expected_coords = [
        [2.3669, 0.8871],
        [1.7039, 0.2642],
        [1.0408, -0.3587],
        [-2.0759, 1.0129],
        [-1.4554, 0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [0.0894, -1.4822]
    ]
    assert np.allclose(res["coordinates"], expected_coords, atol=1e-3)
    assert np.allclose(session.artifacts["reduced_pca"], expected_coords, atol=1e-3)


def test_reduce_dimensions_tsne(session_and_tools):
    session, reduce_dimensions_tool, _ = session_and_tools
    
    session.pending_figure = None
    
    # Known answer #8: t-SNE structural check (perplexity=2, random_state=42)
    res = reduce_dimensions_tool.invoke({
        "method": "tsne",
        "perplexity": 2.0,
        "random_state": 42
    })
    
    assert res["method"] == "tsne"
    assert res["n_components"] == 2
    assert session.pending_figure is not None
    assert "reduced_tsne" in session.artifacts
    
    coords = np.array(res["coordinates"])
    artifact_coords = np.array(session.artifacts["reduced_tsne"])
    
    # Validate 8x2 finite numerical coordinate structure
    assert coords.shape == (8, 2)
    assert np.all(np.isfinite(coords))
    assert np.allclose(coords, artifact_coords)


def test_reduce_dimensions_umap(session_and_tools):
    session, reduce_dimensions_tool, _ = session_and_tools
    
    session.pending_figure = None
    
    # Known answer #9: UMAP (n_neighbors=3, random_state=42)
    res = reduce_dimensions_tool.invoke({
        "method": "umap",
        "n_neighbors": 3,
        "random_state": 42
    })
    
    assert res["method"] == "umap"
    assert res["n_components"] == 2
    assert session.pending_figure is not None
    assert "reduced_umap" in session.artifacts
    
    expected_coords = [
        [6.6811, -6.8218],
        [6.9907, -7.2245],
        [7.7022, -7.3712],
        [10.2228, -6.2310],
        [9.8759, -5.6537],
        [9.3831, -6.5748],
        [9.2153, -5.8297],
        [8.5860, -7.2622]
    ]
    assert np.allclose(res["coordinates"], expected_coords, atol=1e-3)
    assert np.allclose(session.artifacts["reduced_umap"], expected_coords, atol=1e-3)


def test_binarize_labels_tool_fixture(session_and_tools):
    session, _, binarize_labels_tool = session_and_tools
    
    res = binarize_labels_tool.invoke({"label_column": "category_name"})
    
    expected_cols = ["category_name_catA", "category_name_catB"]
    assert res["total_documents"] == 8
    assert res["label_column"] == "category_name"
    assert res["binary_columns"] == expected_cols
    
    # Verify artifact and stored result
    artifact_df = session.artifacts["binarized_labels"]
    assert isinstance(artifact_df, pd.DataFrame)
    assert list(artifact_df.columns) == expected_cols
    assert len(artifact_df) == 8
    
    result_id = res["result_id"]
    assert result_id in session.results_store
    stored_summary = session.results_store[result_id]["record"].summary
    assert stored_summary["total_documents"] == 8
    assert stored_summary["binary_columns"] == expected_cols
    
    # Verify exact 8x2 row-order values (rows 0,1,2,7 are catA=1; 3,4,5,6 are catB=1)
    expected_matrix = [
        [1, 0],  # row 0
        [1, 0],  # row 1
        [1, 0],  # row 2
        [0, 1],  # row 3
        [0, 1],  # row 4
        [0, 1],  # row 5
        [0, 1],  # row 6
        [1, 0],  # row 7
    ]
    assert artifact_df.values.tolist() == expected_matrix


def test_binarize_labels_tool_validation():
    session = SessionState()
    reduction_tools = make_reduction_tools(session)
    binarize_labels_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")
    
    # Unloaded dataset raises ValueError
    with pytest.raises(ValueError, match="No dataset loaded"):
        binarize_labels_tool.invoke({"label_column": "category_name"})
        
    # Setup dataset
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Missing column raises ValueError
    with pytest.raises(ValueError, match="Column 'non_existent_column' not found"):
        binarize_labels_tool.invoke({"label_column": "non_existent_column"})
        
    # Column with missing/NaN values raises ValueError
    session.dataframe["missing_col"] = ["catA", None, "catB", "catA", "catB", "catA", "catB", "catA"]
    with pytest.raises(ValueError, match="contains missing/NaN values"):
        binarize_labels_tool.invoke({"label_column": "missing_col"})
