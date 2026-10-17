import inspect
import sklearn
from sklearn.manifold import TSNE
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools


def test_tsne_default_vs_explicit_diagnostic():
    print(f"\n[DEFAULT_VS_EXPLICIT] sklearn version: {sklearn.__version__}")
    print(f"[DEFAULT_VS_EXPLICIT] TSNE signature: {inspect.signature(TSNE)}")
    
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    build_dtm_tool.invoke({})
    
    X = session.feature_matrix.toarray()
    
    expected_coords = [
        [1330.9991, 32.1039],
        [951.2459, 393.5588],
        [530.8209, 682.9791],
        [-1558.6448, -314.1625],
        [-915.5153, -346.3907],
        [-280.0313, -1261.2115],
        [-280.0313, -1261.2115],
        [-701.3674, 1465.3973]
    ]
    
    # 1. Omitting method (using constructor default)
    tsne_omitted = TSNE(n_components=2, perplexity=2.0, random_state=42, init="random", learning_rate=200.0)
    coords_omitted = tsne_omitted.fit_transform(X).tolist()
    match_omitted = np.allclose(coords_omitted, expected_coords, atol=1e-3)
    
    # 2. Explicit method="barnes_hut"
    tsne_explicit = TSNE(n_components=2, perplexity=2.0, random_state=42, init="random", learning_rate=200.0, method="barnes_hut")
    coords_explicit = tsne_explicit.fit_transform(X).tolist()
    match_explicit = np.allclose(coords_explicit, expected_coords, atol=1e-3)
    
    # Compare omitted vs explicit
    omitted_vs_explicit = np.allclose(coords_omitted, coords_explicit, atol=1e-6)
    
    print(f"\n1. Omitted method (default):")
    print(f"Match with Known Answer #8: {match_omitted}")
    print(f"Full coordinates:\n{coords_omitted}")
    
    print(f"\n2. Explicit method='barnes_hut':")
    print(f"Match with Known Answer #8: {match_explicit}")
    print(f"Full coordinates:\n{coords_explicit}")
    
    print(f"\n3. Do omitted and explicit outputs match each other?")
    print(f"Omitted vs Explicit match: {omitted_vs_explicit}")
