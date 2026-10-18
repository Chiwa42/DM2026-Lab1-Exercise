import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Optional, Union, Dict, Any, Literal
from sklearn.feature_extraction.text import CountVectorizer
from langchain_core.tools import tool


def make_tools(session):
    """Factory that creates and returns workspace tools bound to the given session."""

    @tool
    def build_dtm_tool(
        ngram_range: Optional[List[int]] = None,
        min_df: Union[int, float] = 1,
        max_df: Union[int, float] = 1.0,
        max_df_mode: Literal["proportion", "count"] = "proportion",
        lowercase: bool = True,
    ) -> Dict[str, Any]:
        """Builds a Document-Term Matrix (DTM) from document text in the session.

        Converts the text column of session.dataframe into a sparse document-term matrix
        using scikit-learn's CountVectorizer, stores the matrix, vocabulary, and vectorizer
        in the session, computes matrix sparsity, and returns a summary.

        Args:
            ngram_range: List of two integers [min_n, max_n] specifying the range of n-gram
                lengths to extract. Defaults to [1, 1] (unigrams).
            min_df: Minimum document frequency for vocabulary inclusion (int for absolute count,
                float for document fraction). Defaults to 1.
            max_df: Maximum document frequency for vocabulary inclusion. Interpreted as float
                proportion in (0.0, 1.0] when max_df_mode='proportion', or as int count >= 1
                when max_df_mode='count'. Defaults to 1.0.
            max_df_mode: Mode for interpreting max_df ('proportion' or 'count'). Defaults to 'proportion'.
            lowercase: Whether to convert text to lowercase before tokenizing. Defaults to True.

        Returns:
            Dict containing result_id, n_documents, n_terms, vocabulary_sample,
            non_zero, total_elements, and sparsity_pct.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            raise ValueError("No dataset loaded in session. Load a dataset with text column first.")

        # Default and validate ngram_range
        if ngram_range is None:
            ngram_range = [1, 1]

        if (
            not isinstance(ngram_range, (list, tuple))
            or len(ngram_range) != 2
            or not all(isinstance(x, int) for x in ngram_range)
        ):
            raise ValueError("ngram_range must be a list of exactly two integers [min_n, max_n].")

        min_n, max_n = ngram_range[0], ngram_range[1]
        if min_n < 1 or min_n > max_n:
            raise ValueError(f"Invalid ngram_range [{min_n}, {max_n}]: lower bound must be >= 1 and <= upper bound.")

        ngram_tuple = (min_n, max_n)

        # Unambiguous parsing of max_df via max_df_mode
        if max_df_mode == "proportion":
            try:
                max_df = float(max_df)
            except (ValueError, TypeError):
                raise ValueError(f"max_df='{max_df}' cannot be converted to float for proportion mode.")
            if max_df <= 0.0 or max_df > 1.0:
                raise ValueError(f"max_df={max_df} out of bounds for proportion mode (must be in (0.0, 1.0]).")
        elif max_df_mode == "count":
            if isinstance(max_df, bool) or (isinstance(max_df, float) and not max_df.is_integer()):
                raise ValueError(f"In 'count' mode, max_df must be an integer count (got {max_df}).")
            max_df = int(max_df)
            if max_df < 1:
                raise ValueError(f"max_df={max_df} out of bounds for count mode (must be >= 1).")

        # Handle NaNs / missing text values safely by filling with empty string
        text_data = session.dataframe["text"].fillna("").astype(str)

        vectorizer = CountVectorizer(
            ngram_range=ngram_tuple,
            min_df=min_df,
            max_df=max_df,
            lowercase=lowercase,
        )

        feature_matrix = vectorizer.fit_transform(text_data)
        feature_names = vectorizer.get_feature_names_out().tolist()

        # Update session state
        session.feature_matrix = feature_matrix
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].to_numpy())

        # Compute sparsity metrics
        n_rows, n_cols = feature_matrix.shape
        non_zero = int(feature_matrix.nnz)
        total_elements = int(n_rows * n_cols)

        if total_elements == 0:
            sparsity_pct = 0.0
        else:
            sparsity_pct = 100.0 * (1.0 - (non_zero / total_elements))

        result_id = session.next_result_id("dtm")
        summary = {
            "result_id": result_id,
            "n_documents": n_rows,
            "n_terms": n_cols,
            "vocabulary_sample": feature_names[:10],
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": sparsity_pct,
        }

        session.store_result("build_dtm_tool", {"ngram_range": ngram_range, "min_df": min_df, "max_df": max_df, "max_df_mode": max_df_mode, "lowercase": lowercase}, summary)
        return summary

    @tool
    def term_frequency_tool() -> Dict[str, Any]:
        """Aggregates term frequencies across all documents in session.feature_matrix.

        Sums counts for each term across all documents in the Document-Term Matrix,
        stores the full report sorted by frequency descending (and term ascending),
        and returns a summary with total terms and top terms preview.

        Returns:
            Dict containing result_id, total_terms, and top_terms preview.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("No Document-Term Matrix found in session. Run build_dtm_tool first.")

        counts = np.asarray(session.feature_matrix.sum(axis=0)).ravel()

        full_report = pd.DataFrame({
            "term": session.feature_names,
            "frequency": counts.astype(int)
        })
        full_report = full_report.sort_values(by=["frequency", "term"], ascending=[False, True]).reset_index(drop=True)

        top_preview = full_report.head(10).to_dict(orient="records")

        result_id = session.next_result_id("term_freq")
        summary = {
            "result_id": result_id,
            "total_terms": len(session.feature_names),
            "top_terms": top_preview,
        }

        session.store_result("term_frequency_tool", {}, summary, full_report=full_report)
        return summary

    @tool
    def dtm_heatmap_tool(n_terms: int = 20, n_documents: int = 20) -> Dict[str, Any]:
        """Generates a heatmap visualization of a positional slice of the Document-Term Matrix.

        Slices the top n_documents rows and top n_terms columns from session.feature_matrix without
        reordering, renders a seaborn heatmap assigned to session.pending_figure, and stores
        a DataFrame representation of the slice in full_report.

        Args:
            n_terms: Number of terms (columns) to include from the start of the vocabulary. Defaults to 20.
            n_documents: Number of documents (rows) to include from the start of the dataset. Defaults to 20.

        Returns:
            Dict containing result_id, n_terms, n_documents, terms, document_labels, and matrix.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("No Document-Term Matrix found in session. Run build_dtm_tool first.")

        if type(n_terms) is not int or n_terms <= 0:
            raise ValueError("Parameter 'n_terms' must be a positive integer (booleans excluded).")

        if type(n_documents) is not int or n_documents <= 0:
            raise ValueError("Parameter 'n_documents' must be a positive integer (booleans excluded).")

        # Slice matrix positionally without re-sorting
        sliced_matrix_sparse = session.feature_matrix[0:n_documents, 0:n_terms]
        dense_matrix = sliced_matrix_sparse.toarray()

        sliced_terms = session.feature_names[0:n_terms]
        doc_indices = list(range(dense_matrix.shape[0]))
        document_labels = [f"doc_{i}" for i in doc_indices]

        full_report = pd.DataFrame(dense_matrix, columns=sliced_terms, index=doc_indices)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            dense_matrix,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            xticklabels=sliced_terms,
            yticklabels=document_labels,
            ax=ax,
        )
        ax.set_title(f"DTM Heatmap ({dense_matrix.shape[0]} Docs x {dense_matrix.shape[1]} Terms)")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Document Index")

        session.pending_figure = fig

        result_id = session.next_result_id("dtm_heatmap")
        summary = {
            "result_id": result_id,
            "n_terms": int(dense_matrix.shape[1]),
            "n_documents": int(dense_matrix.shape[0]),
            "terms": sliced_terms,
            "document_labels": document_labels,
            "matrix": dense_matrix.astype(int).tolist(),
        }

        session.store_result(
            "dtm_heatmap_tool",
            {"n_terms": n_terms, "n_documents": n_documents},
            summary,
            full_report=full_report,
        )
        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
