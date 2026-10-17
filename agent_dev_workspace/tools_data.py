import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import nltk
import pandas as pd
from typing import Dict, Any, Optional
from langchain_core.tools import tool


def make_tools(session):
    """Factory that creates and returns workspace tools bound to the given session."""

    @tool
    def check_missing_tool() -> Dict[str, Any]:
        """Checks for missing or empty text values in session.dataframe.

        Scans the 'text' column of the active session DataFrame for NaN values,
        empty strings, or whitespace-only entries, computes summary counts,
        stores the missing rows in full_report, and returns a summary dict.

        Returns:
            Dict containing result_id, total_documents, missing_count,
            missing_indices, and missing_percentage.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            raise ValueError("No dataset loaded in session or 'text' column missing.")

        df = session.dataframe
        total_documents = len(df)

        # Identify missing: NaN or empty/whitespace string
        is_missing = df["text"].isna() | (df["text"].astype(str).str.strip() == "")
        missing_count = int(is_missing.sum())
        missing_indices = [int(idx) for idx in df.index[is_missing]]

        missing_percentage = (
            0.0 if total_documents == 0 else 100.0 * (missing_count / total_documents)
        )

        missing_rows_df = df[is_missing].copy()

        result_id = session.next_result_id("check_missing")
        summary = {
            "result_id": result_id,
            "total_documents": total_documents,
            "missing_count": missing_count,
            "missing_indices": missing_indices,
            "missing_percentage": missing_percentage,
        }

        session.store_result("check_missing_tool", {}, summary, full_report=missing_rows_df)
        return summary

    @tool
    def check_duplicates_tool(drop: bool = False) -> Dict[str, Any]:
        """Checks for (and optionally drops) duplicate documents in session.dataframe.

        Identifies exact duplicate documents based on the 'text' column. When drop=True,
        removes all copies of duplicated rows (keep=False) to match pipeline requirements.

        Args:
            drop: Whether to drop duplicate documents from session.dataframe. Defaults to False.

        Returns:
            Dict containing result_id, total_documents_before, duplicate_count,
            dropped, and total_documents_after.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            raise ValueError("No dataset loaded in session or 'text' column missing.")

        df = session.dataframe
        total_documents_before = len(df)

        # Count extra duplicate documents (Known Answer #3 semantics)
        duplicate_count = int(df.duplicated(subset=["text"], keep="first").sum())

        # Build full_report containing all duplicate copies before mutation
        is_dup_all = df.duplicated(subset=["text"], keep=False)
        full_report = df[is_dup_all].copy()

        if drop and duplicate_count > 0:
            session.dataframe = df.drop_duplicates(subset=["text"], keep=False)
            total_documents_after = len(session.dataframe)
        else:
            total_documents_after = total_documents_before

        result_id = session.next_result_id("check_duplicates")
        summary = {
            "result_id": result_id,
            "total_documents_before": total_documents_before,
            "duplicate_count": duplicate_count,
            "dropped": drop,
            "total_documents_after": total_documents_after,
        }

        session.store_result("check_duplicates_tool", {"drop": drop}, summary, full_report=full_report)
        return summary

    @tool
    def tokenize_tool() -> Dict[str, Any]:
        """Tokenizes each document's text into lowercased unigram tokens using NLTK.

        Applies nltk.word_tokenize on lowercased document text, assigns the token list
        to session.dataframe['unigrams'], mirrors them in session.artifacts['unigrams'],
        and returns a summary with document count, token count, and vocabulary size.

        Returns:
            Dict containing result_id, total_documents, total_tokens, and vocabulary_size.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            raise ValueError("No dataset loaded in session or 'text' column missing.")

        unigrams_list = []
        for val in session.dataframe["text"]:
            if pd.isna(val) or str(val).strip() == "":
                unigrams_list.append([])
            else:
                tokens = nltk.word_tokenize(str(val).lower(), language="english")
                unigrams_list.append(tokens)

        session.dataframe["unigrams"] = unigrams_list
        session.artifacts["unigrams"] = unigrams_list

        total_documents = len(session.dataframe)
        total_tokens = int(sum(len(toks) for toks in unigrams_list))
        vocabulary_size = int(len(set(tok for toks in unigrams_list for tok in toks)))

        full_report = session.dataframe[["text", "unigrams"]].copy()

        result_id = session.next_result_id("tokenize")
        summary = {
            "result_id": result_id,
            "total_documents": total_documents,
            "total_tokens": total_tokens,
            "vocabulary_size": vocabulary_size,
        }

        session.store_result("tokenize_tool", {}, summary, full_report=full_report)
        return summary

    @tool
    def list_files_tool(subdirectory: str = ".") -> Dict[str, Any]:
        """Lists direct child files and directories inside a workspace subdirectory.

        Inspects the specified subdirectory relative to session.workspace_dir, ensuring
        the target path remains within the workspace root, and returns lists of direct
        file and directory names along with a full_report DataFrame.

        Args:
            subdirectory: Subdirectory path relative to workspace root. Defaults to '.'.

        Returns:
            Dict containing result_id, subdirectory, directories, and files.
        """
        workspace_dir = getattr(session, "workspace_dir", None)
        if not workspace_dir or not Path(workspace_dir).exists() or not Path(workspace_dir).is_dir():
            raise ValueError("session.workspace_dir is missing, does not exist, or is not a directory.")

        base_dir = Path(workspace_dir).resolve()
        target_dir = (base_dir / subdirectory).resolve()

        if not target_dir.is_relative_to(base_dir):
            raise ValueError(f"Access denied: Subdirectory '{subdirectory}' escapes workspace directory.")

        if not target_dir.exists() or not target_dir.is_dir():
            raise ValueError(f"Directory not found: '{subdirectory}'")

        directories = []
        files = []
        report_rows = []

        for entry in target_dir.iterdir():
            if entry.is_dir():
                directories.append(entry.name)
                report_rows.append({"type": "directory", "name": entry.name})
            elif entry.is_file():
                files.append(entry.name)
                report_rows.append({"type": "file", "name": entry.name})

        directories.sort()
        files.sort()

        full_report = pd.DataFrame(report_rows, columns=["type", "name"]).sort_values(by=["type", "name"]).reset_index(drop=True)

        result_id = session.next_result_id("list_files")
        summary = {
            "result_id": result_id,
            "subdirectory": subdirectory,
            "directories": directories,
            "files": files,
        }

        session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary, full_report=full_report)
        return summary

    def _to_json_safe(val: Any) -> Any:
        """Recursively converts values (including lists/dicts/NumPy scalars) to JSON-safe Python types."""
        if val is None:
            return None
        if isinstance(val, (list, tuple)):
            return [_to_json_safe(x) for x in val]
        if isinstance(val, dict):
            return {k: _to_json_safe(v) for k, v in val.items()}
        if hasattr(val, "item"):
            val = val.item()
        if isinstance(val, bool):
            return val
        if pd.api.types.is_scalar(val) and pd.isna(val):
            return None
        if isinstance(val, (int, float, str)):
            return val
        return str(val)

    @tool
    def inspect_data_tool(n: int = 5) -> Dict[str, Any]:
        """Inspects the top n rows of session.dataframe without mutating state.

        Validates parameter n, extracts a preview slice from session.dataframe.head(n),
        converts record values recursively to native Python types for JSON safety,
        stores the slice in full_report, and returns summary metadata with the record preview.

        Args:
            n: Number of top rows to inspect. Defaults to 5.

        Returns:
            Dict containing result_id, n_documents, n_columns, columns, and preview.
        """
        if session.dataframe is None:
            raise ValueError("No dataset loaded in session. Load a dataset first.")

        if type(n) is not int or n < 0:
            raise ValueError("Parameter 'n' must be a non-negative integer (booleans excluded).")

        df = session.dataframe
        n_documents = len(df)
        columns = df.columns.tolist()
        n_columns = len(columns)

        preview_df = df.head(n).copy()

        # Convert preview records explicitly to native Python types via recursive helper
        raw_records = preview_df.to_dict(orient="records")
        preview = _to_json_safe(raw_records)

        result_id = session.next_result_id("inspect_data")
        summary = {
            "result_id": result_id,
            "n_documents": n_documents,
            "n_columns": n_columns,
            "columns": columns,
            "preview": preview,
        }

        session.store_result("inspect_data_tool", {"n": n}, summary, full_report=preview_df)
        return summary

    @tool
    def sample_data_tool(n: int = 5, random_state: Optional[int] = None) -> Dict[str, Any]:
        """Subsamples the working DataFrame without mutating session.dataframe.

        Extracts n random rows using pandas DataFrame.sample(n=n, random_state=random_state),
        validates strict positive int constraints for n and integer constraints for random_state,
        stores the sampled subset in full_report, and returns a summary dict.

        Args:
            n: Number of rows to sample. Defaults to 5.
            random_state: Optional random seed for reproducible sampling. Defaults to None.

        Returns:
            Dict containing result_id, total_documents, n_sampled, and sampled_indices.
        """
        if session.dataframe is None:
            raise ValueError("No dataset loaded in session. Load a dataset first.")

        if type(n) is not int or n <= 0:
            raise ValueError("Parameter 'n' must be a positive integer (booleans excluded).")

        total_documents = len(session.dataframe)
        if n > total_documents:
            raise ValueError(f"Sample size n={n} exceeds total document count ({total_documents}).")

        if random_state is not None and type(random_state) is not int:
            raise ValueError("Parameter 'random_state' must be an integer (booleans excluded).")

        sampled_df = session.dataframe.sample(n=n, random_state=random_state)
        sampled_indices = [int(idx) for idx in sampled_df.index]

        result_id = session.next_result_id("sample_data")
        summary = {
            "result_id": result_id,
            "total_documents": total_documents,
            "n_sampled": n,
            "sampled_indices": sampled_indices,
        }

        session.store_result(
            "sample_data_tool",
            {"n": n, "random_state": random_state},
            summary,
            full_report=sampled_df.copy()
        )
        return summary

    @tool
    def describe_data_tool() -> Dict[str, Any]:
        """Computes text_length statistics on session.dataframe and generates a box plot.

        Calculates character length per document using df['text'].apply(len), assigns
        text_length to session.dataframe, calculates overall and per-category .describe()
        statistics, generates a seaborn boxplot set on session.pending_figure, and stores
        the document-level full_report DataFrame.

        Returns:
            Dict containing result_id, overall_stats, and by_category_stats.
        """
        if session.dataframe is None:
            raise ValueError("No dataset loaded in session. Load a dataset first.")

        if "text" not in session.dataframe.columns or "category_name" not in session.dataframe.columns:
            raise ValueError("Required column 'text' or 'category_name' missing from session.dataframe.")

        df = session.dataframe

        # Compute and assign text_length column
        df["text_length"] = df["text"].apply(len)

        full_report = df[["text", "category_name", "text_length"]].copy()

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df, x="category_name", y="text_length", ax=ax)
        ax.set_title("Text Length Distribution by Category")
        ax.set_xlabel("Category Name")
        ax.set_ylabel("Text Length")

        if len(df) == 0:
            overall_stats = {
                "count": 0.0,
                "mean": None,
                "std": None,
                "min": None,
                "25%": None,
                "50%": None,
                "75%": None,
                "max": None,
            }
            by_category_stats = {}
        else:
            overall_stats = _to_json_safe(df["text_length"].describe().to_dict())
            by_cat_df = df.groupby("category_name")["text_length"].describe()
            by_category_stats = _to_json_safe(by_cat_df.T.to_dict())

        session.pending_figure = fig

        result_id = session.next_result_id("describe_data")
        summary = {
            "result_id": result_id,
            "overall_stats": overall_stats,
            "by_category_stats": by_category_stats,
        }

        session.store_result("describe_data_tool", {}, summary, full_report=full_report)
        return summary

    return [
        check_missing_tool,
        check_duplicates_tool,
        tokenize_tool,
        list_files_tool,
        inspect_data_tool,
        sample_data_tool,
        describe_data_tool,
    ]
