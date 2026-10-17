import json
import matplotlib.figure
from pathlib import Path
import pytest
import pandas as pd
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools


def test_check_missing_tool_known_answer_2():
    """Tests check_missing_tool against Known Answer #2 in TEST_FIXTURE.md using sample_fixture.csv."""
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

    # Record dataframe state before tool invocation to verify immutability
    df_before = session.dataframe.copy()

    # 3. Instantiate tools_data and run check_missing_tool
    data_tools = make_tools(session)
    check_missing_tool = next(t for t in data_tools if t.name == "check_missing_tool")
    res = check_missing_tool.invoke({})

    # Verify session.dataframe was unchanged
    assert session.dataframe.equals(df_before)

    # Verify return dictionary values matching Known Answer #2
    assert res["total_documents"] == 8
    assert res["missing_count"] == 1
    assert res["missing_indices"] == [7]
    assert res["missing_percentage"] == pytest.approx(12.5, abs=1e-3)

    # Verify stored full_report in results_store
    result_id = res["result_id"]
    stored_entry = session.results_store[result_id]
    full_report = stored_entry["full_report"]

    # Verify full_report contains exactly row 7
    assert len(full_report) == 1
    assert list(full_report.index) == [7]


def test_check_duplicates_tool_known_answer_3_check_mode():
    """Tests check_duplicates_tool(drop=False) against Known Answer #3 in TEST_FIXTURE.md."""
    # Each test function creates its own fresh session & tools
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    df_before = session.dataframe.copy()

    data_tools = make_tools(session)
    check_duplicates_tool = next(t for t in data_tools if t.name == "check_duplicates_tool")
    res = check_duplicates_tool.invoke({"drop": False})

    # Verify session.dataframe was preserved with 8 rows
    assert session.dataframe.equals(df_before)
    assert len(session.dataframe) == 8

    # Verify summary dict values matching Known Answer #3
    assert res["total_documents_before"] == 8
    assert res["duplicate_count"] == 1
    assert res["dropped"] is False
    assert res["total_documents_after"] == 8

    # Verify stored full_report contains exactly rows 5 and 6
    result_id = res["result_id"]
    stored_entry = session.results_store[result_id]
    full_report = stored_entry["full_report"]

    assert list(full_report.index) == [5, 6]


def test_check_duplicates_tool_known_answer_3_drop_mode():
    """Tests check_duplicates_tool(drop=True) against Known Answer #3 in TEST_FIXTURE.md."""
    # Each test function creates its own fresh session & tools
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_tools(session)
    check_duplicates_tool = next(t for t in data_tools if t.name == "check_duplicates_tool")
    res = check_duplicates_tool.invoke({"drop": True})

    # Verify summary dict values
    assert res["total_documents_before"] == 8
    assert res["duplicate_count"] == 1
    assert res["dropped"] is True
    assert res["total_documents_after"] == 6

    # Verify session.dataframe now has 6 rows and preserved remaining indices [0, 1, 2, 3, 4, 7]
    assert len(session.dataframe) == 6
    assert list(session.dataframe.index) == [0, 1, 2, 3, 4, 7]


def test_tokenize_tool_known_answer_10():
    """Tests tokenize_tool against Known Answer #10 in TEST_FIXTURE.md using sample_fixture.csv."""
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

    # 3. Instantiate tools_data and run tokenize_tool
    data_tools = make_tools(session)
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")
    res = tokenize_tool.invoke({})

    # Expected token lists for all 8 rows according to Known Answer #10
    expected_tokens = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        [],  # Row 7 is empty text
    ]

    # Verify session.dataframe["unigrams"] matches expected lists exactly
    assert list(session.dataframe["unigrams"]) == expected_tokens
    assert session.artifacts["unigrams"] == expected_tokens

    # Verify summary dict values
    assert res["total_documents"] == 8
    assert res["total_tokens"] == 27
    assert res["vocabulary_size"] == 5

    # Verify stored full_report in results_store
    result_id = res["result_id"]
    stored_entry = session.results_store[result_id]
    full_report = stored_entry["full_report"]

    assert list(full_report.columns) == ["text", "unigrams"]
    assert list(full_report["unigrams"]) == expected_tokens


def test_list_files_tool_known_answer_11():
    """Tests list_files_tool against Known Answer #11 in TEST_FIXTURE.md using real workspace state."""
    session = SessionState()
    # Set session.workspace_dir to absolute working directory path where newdataset/ resides
    session.workspace_dir = str(Path.cwd())

    data_tools = make_tools(session)
    list_files_tool = next(t for t in data_tools if t.name == "list_files_tool")

    res = list_files_tool.invoke({"subdirectory": "newdataset"})

    # Known Answer #11 expectations
    assert res["subdirectory"] == "newdataset"
    assert res["directories"] == []
    assert res["files"] == ["Reddit-stock-sentiment.csv"]

    # Verify full_report stored in results_store
    result_id = res["result_id"]
    stored_entry = session.results_store[result_id]
    full_report = stored_entry["full_report"]

    assert list(full_report.columns) == ["type", "name"]
    assert len(full_report) == 1
    assert full_report.iloc[0]["type"] == "file"
    assert full_report.iloc[0]["name"] == "Reddit-stock-sentiment.csv"


def test_list_files_tool_invalid_paths():
    """Tests list_files_tool error handling for escaping paths and missing directories."""
    session = SessionState()
    session.workspace_dir = str(Path.cwd())

    data_tools = make_tools(session)
    list_files_tool = next(t for t in data_tools if t.name == "list_files_tool")

    # Test non-existent directory raises ValueError
    with pytest.raises(ValueError, match="Directory not found"):
        list_files_tool.invoke({"subdirectory": "non_existent_folder_xyz"})

    # Test escaping path raises ValueError
    with pytest.raises(ValueError, match="Access denied"):
        list_files_tool.invoke({"subdirectory": "../outside_workspace"})


def test_inspect_data_tool_loaded():
    """Tests inspect_data_tool on a loaded dataset with n=3, n=0, and with tokenize_tool list cells."""
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_tools(session)
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")
    inspect_data_tool = next(t for t in data_tools if t.name == "inspect_data_tool")

    # Tokenize first to populate unigrams column (testing list-valued cells)
    tokenize_tool.invoke({})

    # 1. Test inspect_data_tool(n=3)
    res = inspect_data_tool.invoke({"n": 3})

    # Verify actual session columns (4 columns after tokenize: text, category_name, category, unigrams)
    assert res["n_documents"] == 8
    assert res["n_columns"] == 4
    assert res["columns"] == ["text", "category_name", "category", "unigrams"]
    assert len(res["preview"]) == 3

    # Assert json.dumps succeeds cleanly for JSON safety validation
    serialized_json = json.dumps(res)
    assert isinstance(serialized_json, str)

    # Verify full_report equals session.dataframe.head(3)
    result_id = res["result_id"]
    stored_entry = session.results_store[result_id]
    full_report = stored_entry["full_report"]
    assert full_report.equals(session.dataframe.head(3))

    # 2. Test n=0
    res_zero = inspect_data_tool.invoke({"n": 0})
    assert res_zero["n_documents"] == 8
    assert res_zero["preview"] == []


def test_inspect_data_tool_invalid_n_and_unloaded():
    """Tests inspect_data_tool validation for invalid n values and unloaded session."""
    session = SessionState()
    data_tools = make_tools(session)
    inspect_data_tool = next(t for t in data_tools if t.name == "inspect_data_tool")

    # Test unloaded session raises ValueError
    with pytest.raises(ValueError, match="No dataset loaded"):
        inspect_data_tool.invoke({"n": 5})

    # Load dataset to test parameter validation
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Test negative n raises ValueError
    with pytest.raises(ValueError, match="non-negative integer"):
        inspect_data_tool.invoke({"n": -1})

    # Test boolean n raises ValueError when invoking wrapped function directly (bypassing LangChain coercion)
    with pytest.raises(ValueError, match="non-negative integer"):
        inspect_data_tool.func(n=True)


def test_sample_data_tool_known_answer_13():
    """Tests sample_data_tool against Known Answer #13 in TEST_FIXTURE.md using sample_fixture.csv."""
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    df_before = session.dataframe.copy()

    data_tools = make_tools(session)
    sample_data_tool = next(t for t in data_tools if t.name == "sample_data_tool")

    # Invoke sample_data_tool with n=4, random_state=42
    res = sample_data_tool.invoke({"n": 4, "random_state": 42})

    # Known Answer #13 assertions
    assert res["total_documents"] == 8
    assert res["n_sampled"] == 4
    assert res["sampled_indices"] == [1, 5, 0, 7]

    # Verify session.dataframe was preserved with 8 rows
    assert session.dataframe.equals(df_before)

    # Verify stored full_report in results_store preserves sampled index order
    result_id = res["result_id"]
    stored_entry = session.results_store[result_id]
    full_report = stored_entry["full_report"]
    assert list(full_report.index) == [1, 5, 0, 7]

    # Verify reproducibility across repeated calls with same random_state
    res2 = sample_data_tool.invoke({"n": 4, "random_state": 42})
    assert res2["sampled_indices"] == [1, 5, 0, 7]


def test_sample_data_tool_validation_and_unloaded():
    """Tests sample_data_tool validation for invalid n, random_state, dataset bounds, and unloaded session."""
    session = SessionState()
    data_tools = make_tools(session)
    sample_data_tool = next(t for t in data_tools if t.name == "sample_data_tool")

    # Test unloaded session raises ValueError
    with pytest.raises(ValueError, match="No dataset loaded"):
        sample_data_tool.invoke({"n": 4})

    # Load dataset for parameter validation
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Test n <= 0
    with pytest.raises(ValueError, match="positive integer"):
        sample_data_tool.invoke({"n": 0})

    # Test n > total_documents (8)
    with pytest.raises(ValueError, match="exceeds total document count"):
        sample_data_tool.invoke({"n": 10})

    # Test strict bool rejection for n and random_state via wrapped function directly
    with pytest.raises(ValueError, match="positive integer"):
        sample_data_tool.func(n=True)

    with pytest.raises(ValueError, match="must be an integer"):
        sample_data_tool.func(n=4, random_state=True)


def test_describe_data_tool_known_answer_18():
    """Tests describe_data_tool against Known Answer #18 in TEST_FIXTURE.md using sample_fixture.csv."""
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_tools(session)
    describe_data_tool = next(t for t in data_tools if t.name == "describe_data_tool")

    res = describe_data_tool.invoke({})

    # Verify text_length column assigned to session.dataframe
    expected_lengths = [29, 23, 17, 30, 24, 18, 18, 0]
    assert list(session.dataframe["text_length"]) == expected_lengths

    # Verify overall_stats from Known Answer #18
    overall = res["overall_stats"]
    assert overall["count"] == 8
    assert overall["mean"] == pytest.approx(19.875, abs=1e-3)
    assert overall["std"] == pytest.approx(9.4330, abs=1e-3)
    assert overall["min"] == 0
    assert overall["25%"] == pytest.approx(17.75, abs=1e-3)
    assert overall["50%"] == pytest.approx(20.5, abs=1e-3)
    assert overall["75%"] == pytest.approx(25.25, abs=1e-3)
    assert overall["max"] == 30

    # Verify by_category_stats from Known Answer #18
    catA = res["by_category_stats"]["catA"]
    assert catA["count"] == 4
    assert catA["mean"] == pytest.approx(17.25, abs=1e-3)
    assert catA["std"] == pytest.approx(12.5, abs=1e-3)

    catB = res["by_category_stats"]["catB"]
    assert catB["count"] == 4
    assert catB["mean"] == pytest.approx(22.5, abs=1e-3)
    assert catB["std"] == pytest.approx(5.7446, abs=1e-3)

    # Verify full_report in results_store
    result_id = res["result_id"]
    stored_entry = session.results_store[result_id]
    full_report = stored_entry["full_report"]
    assert list(full_report.columns) == ["text", "category_name", "text_length"]
    assert list(full_report["text_length"]) == expected_lengths

    # Verify pending_figure is a matplotlib Figure
    assert isinstance(session.pending_figure, matplotlib.figure.Figure)

    # Assert json.dumps succeeds
    serialized_json = json.dumps(res)
    assert isinstance(serialized_json, str)


def test_describe_data_tool_unloaded_missing_cols_and_empty():
    """Tests describe_data_tool error handling for unloaded data, missing columns, and empty DataFrames."""
    session = SessionState()
    data_tools = make_tools(session)
    describe_data_tool = next(t for t in data_tools if t.name == "describe_data_tool")

    # 1. Unloaded session raises ValueError
    with pytest.raises(ValueError, match="No dataset loaded"):
        describe_data_tool.invoke({})

    # 2. Missing required category_name column raises ValueError
    session.dataframe = pd.DataFrame({"text": ["hello", "world"]})
    with pytest.raises(ValueError, match="Required column"):
        describe_data_tool.invoke({})

    # 3. Empty DataFrame yields deterministic empty stats and valid Figure
    session.dataframe = pd.DataFrame(columns=["text", "category_name"])
    res_empty = describe_data_tool.invoke({})

    expected_keys = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    assert list(res_empty["overall_stats"].keys()) == expected_keys
    assert res_empty["overall_stats"]["count"] == 0.0
    assert res_empty["overall_stats"]["mean"] is None
    assert res_empty["by_category_stats"] == {}
    assert isinstance(session.pending_figure, matplotlib.figure.Figure)
    assert isinstance(json.dumps(res_empty), str)
