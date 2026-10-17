import inspect
import sklearn
from sklearn.manifold import TSNE
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools


def test_tsne_learning_rate_diagnostic():
    print(f"\n[LEARNING_RATE_DIAGNOSTIC] sklearn version: {sklearn.__version__}")
    
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
    
    # 1. Test learning_rate="auto", init="random"
    tsne_auto = TSNE(n_components=2, perplexity=2.0, random_state=42, init="random", learning_rate="auto")
    coords_auto = tsne_auto.fit_transform(X).tolist()
    match_auto = np.allclose(coords_auto, expected_coords, atol=1e-3)
    print(f"\n[LEARNING_RATE_DIAGNOSTIC] learning_rate='auto', init='random':")
    print(f"Match with Known Answer #8: {match_auto}")
    print(f"Full coordinates:\n{coords_auto}")
    
    # 2. Test learning_rate=200.0, init="random"
    tsne_200 = TSNE(n_components=2, perplexity=2.0, random_state=42, init="random", learning_rate=200.0)
    coords_200 = tsne_200.fit_transform(X).tolist()
    match_200 = np.allclose(coords_200, expected_coords, atol=1e-3)
    print(f"\n[LEARNING_RATE_DIAGNOSTIC] learning_rate=200.0, init='random':")
    print(f"Match with Known Answer #8: {match_200}")
    print(f"Full coordinates:\n{coords_200}")
    
    # 3. Test learning_rate=200.0, init="pca"
    tsne_200_pca = TSNE(n_components=2, perplexity=2.0, random_state=42, init="pca", learning_rate=200.0)
    coords_200_pca = tsne_200_pca.fit_transform(X).tolist()
    match_200_pca = np.allclose(coords_200_pca, expected_coords, atol=1e-3)
    print(f"\n[LEARNING_RATE_DIAGNOSTIC] learning_rate=200.0, init='pca':")
    print(f"Match with Known Answer #8: {match_200_pca}")
    print(f"Full coordinates:\n{coords_200_pca}")
