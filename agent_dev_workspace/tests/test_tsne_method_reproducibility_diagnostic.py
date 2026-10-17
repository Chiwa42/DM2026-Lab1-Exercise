import sklearn
from sklearn.manifold import TSNE
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools


def test_tsne_method_reproducibility_diagnostic():
    print(f"\n[METHOD_DIAGNOSTIC] sklearn version: {sklearn.__version__}")
    
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
    
    # 1. method="barnes_hut" (Run 1 & Run 2)
    tsne_bh1 = TSNE(n_components=2, perplexity=2.0, random_state=42, init="random", learning_rate=200.0, method="barnes_hut")
    coords_bh1 = tsne_bh1.fit_transform(X).tolist()
    tsne_bh2 = TSNE(n_components=2, perplexity=2.0, random_state=42, init="random", learning_rate=200.0, method="barnes_hut")
    coords_bh2 = tsne_bh2.fit_transform(X).tolist()
    
    match_bh_expected = np.allclose(coords_bh1, expected_coords, atol=1e-3)
    reproducible_bh = np.allclose(coords_bh1, coords_bh2, atol=1e-6)
    
    print(f"\n[METHOD_DIAGNOSTIC] method='barnes_hut':")
    print(f"Match with Known Answer #8: {match_bh_expected}")
    print(f"Reproducible across runs: {reproducible_bh}")
    print(f"Full coordinates (Run 1):\n{coords_bh1}")
    
    # 2. method="exact" (Run 1 & Run 2)
    tsne_ex1 = TSNE(n_components=2, perplexity=2.0, random_state=42, init="random", learning_rate=200.0, method="exact")
    coords_ex1 = tsne_ex1.fit_transform(X).tolist()
    tsne_ex2 = TSNE(n_components=2, perplexity=2.0, random_state=42, init="random", learning_rate=200.0, method="exact")
    coords_ex2 = tsne_ex2.fit_transform(X).tolist()
    
    match_ex_expected = np.allclose(coords_ex1, expected_coords, atol=1e-3)
    reproducible_ex = np.allclose(coords_ex1, coords_ex2, atol=1e-6)
    
    print(f"\n[METHOD_DIAGNOSTIC] method='exact':")
    print(f"Match with Known Answer #8: {match_ex_expected}")
    print(f"Reproducible across runs: {reproducible_ex}")
    print(f"Full coordinates (Run 1):\n{coords_ex1}")
