import inspect
import sklearn
from sklearn.manifold import TSNE
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
import tools_reduction


def test_tsne_investigation_diagnostic():
    print(f"\n[INVESTIGATION] sklearn version: {sklearn.__version__}")
    print(f"[INVESTIGATION] TSNE signature: {inspect.signature(TSNE)}")
    
    # Print source code of reduce_dimensions_tool from tools_reduction.py
    source_lines = inspect.getsource(tools_reduction)
    tsne_lines = [line.strip() for line in source_lines.split("\n") if "TSNE(" in line]
    print(f"[INVESTIGATION] tools_reduction.py TSNE constructor call line: {tsne_lines}")
    
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    
    reduction_tools = tools_reduction.make_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    build_dtm_tool.invoke({})
    
    # Verify dataframe row order and DTM properties
    print(f"[INVESTIGATION] DataFrame row order:\n{session.dataframe[['text', 'category_name']]}")
    X = session.feature_matrix.toarray()
    print(f"[INVESTIGATION] DTM shape: {X.shape}")
    print(f"[INVESTIGATION] Feature names: {session.feature_names}")
    print(f"[INVESTIGATION] DTM matrix:\n{X}")
    
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
    
    # 1. Run reduce_dimensions_tool
    res_tool = reduce_dimensions_tool.invoke({
        "method": "tsne",
        "perplexity": 2.0,
        "random_state": 42
    })
    coords_tool = res_tool["coordinates"]
    match_tool = np.allclose(coords_tool, expected_coords, atol=1e-3)
    print(f"\n[INVESTIGATION] reduce_dimensions_tool coords:\n{coords_tool}")
    print(f"[INVESTIGATION] reduce_dimensions_tool match with Known Answer #8: {match_tool}")
    
    # 2. Direct TSNE call
    tsne_direct = TSNE(n_components=2, perplexity=2.0, random_state=42, init="random")
    coords_direct = tsne_direct.fit_transform(X).tolist()
    match_direct = np.allclose(coords_direct, expected_coords, atol=1e-3)
    print(f"\n[INVESTIGATION] Direct TSNE(init='random') coords:\n{coords_direct}")
    print(f"[INVESTIGATION] Direct TSNE match with Known Answer #8: {match_direct}")
