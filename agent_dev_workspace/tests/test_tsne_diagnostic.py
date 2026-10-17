import inspect
import sklearn
from sklearn.manifold import TSNE
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools


def test_tsne_diagnostic():
    print(f"\n[DIAGNOSTIC] Installed scikit-learn version: {sklearn.__version__}")
    print(f"[DIAGNOSTIC] TSNE signature defaults: {inspect.signature(TSNE)}")
    
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
    
    # 1. Test init="pca"
    tsne_pca = TSNE(n_components=2, perplexity=2.0, random_state=42, init="pca")
    coords_pca = tsne_pca.fit_transform(X)
    match_pca = np.allclose(coords_pca, expected_coords, atol=1e-3)
    print(f"\n[DIAGNOSTIC] init='pca' match with Known Answer #8: {match_pca}")
    print(f"[DIAGNOSTIC] Full coordinates (pca):\n{coords_pca.tolist()}")
    
    # 2. Test init="random"
    tsne_random = TSNE(n_components=2, perplexity=2.0, random_state=42, init="random")
    coords_random = tsne_random.fit_transform(X)
    match_random = np.allclose(coords_random, expected_coords, atol=1e-3)
    print(f"\n[DIAGNOSTIC] init='random' match with Known Answer #8: {match_random}")
    print(f"[DIAGNOSTIC] Full coordinates (random):\n{coords_random.tolist()}")
