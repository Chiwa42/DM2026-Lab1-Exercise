import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Any
from langchain_core.tools import tool


def make_tools(session):
    """Factory that creates and returns workspace tools bound to the given session."""

    @tool
    def cosine_similarity_tool(doc_idx1: int, doc_idx2: int) -> Dict[str, Any]:
        """Calculates cosine similarity between two documents using the fitted DTM vectorizer.

        Retrieves texts for doc_idx1 and doc_idx2 from session.dataframe, vectorizes them using
        session.artifacts['count_vectorizer'], computes the cosine similarity score (returning 0.0
        if either vector is zero), stores the score in session.artifacts, and returns summary metadata.

        Args:
            doc_idx1: Zero-based row index of the first document in session.dataframe.
            doc_idx2: Zero-based row index of the second document in session.dataframe.

        Returns:
            Dict containing result_id, doc_idx1, doc_idx2, similarity, doc1_text, and doc2_text.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            raise ValueError("No dataset loaded in session or 'text' column missing. Run load_dataset_tool first.")

        if "count_vectorizer" not in session.artifacts or session.artifacts["count_vectorizer"] is None:
            raise ValueError("Fitted vectorizer missing from session.artifacts['count_vectorizer']. Run build_dtm_tool first.")

        if type(doc_idx1) is not int or type(doc_idx2) is not int:
            raise ValueError("Parameters 'doc_idx1' and 'doc_idx2' must be integers (booleans excluded).")

        total_docs = len(session.dataframe)
        if doc_idx1 < 0 or doc_idx1 >= total_docs:
            raise ValueError(f"doc_idx1={doc_idx1} is out of bounds for dataset with {total_docs} documents.")

        if doc_idx2 < 0 or doc_idx2 >= total_docs:
            raise ValueError(f"doc_idx2={doc_idx2} is out of bounds for dataset with {total_docs} documents.")

        vectorizer = session.artifacts["count_vectorizer"]

        # Extract document texts handling missing/NaN text safely
        val1 = session.dataframe["text"].iloc[doc_idx1]
        val2 = session.dataframe["text"].iloc[doc_idx2]

        text1 = "" if pd.isna(val1) else str(val1)
        text2 = "" if pd.isna(val2) else str(val2)

        # Vectorize using fitted count_vectorizer
        vec_matrix = vectorizer.transform([text1, text2])
        v1 = vec_matrix[0].toarray().ravel()
        v2 = vec_matrix[1].toarray().ravel()

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0.0 or norm2 == 0.0:
            similarity = 0.0
        else:
            similarity = float(np.dot(v1, v2) / (norm1 * norm2))

        result_id = session.next_result_id("cosine_similarity")
        artifact_key = f"cosine_similarity_{doc_idx1}_{doc_idx2}"
        session.artifacts[artifact_key] = similarity

        summary = {
            "result_id": result_id,
            "doc_idx1": doc_idx1,
            "doc_idx2": doc_idx2,
            "similarity": similarity,
            "doc1_text": text1,
            "doc2_text": text2,
        }

        full_report = pd.DataFrame([{
            "doc_idx1": doc_idx1,
            "doc_idx2": doc_idx2,
            "doc1_text": text1,
            "doc2_text": text2,
            "similarity": similarity,
        }])

        session.store_result(
            "cosine_similarity_tool",
            {"doc_idx1": doc_idx1, "doc_idx2": doc_idx2},
            summary,
            full_report=full_report,
        )

        return summary

    @tool
    def feature_correlation_matrix_tool() -> Dict[str, Any]:
        """Computes feature-vs-feature Pearson correlation heatmap for top 20 terms by variance.

        Calculates term variances across session.feature_matrix, selects top 20 terms by variance,
        computes their Pearson correlation matrix, plots an annotated seaborn heatmap assigned to
        session.pending_figure, stores the matrix in session.artifacts['feature_correlation'],
        and returns summary metadata.

        Returns:
            Dict containing result_id, terms, and correlation_matrix.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("No Document-Term Matrix found in session. Run build_dtm_tool first.")

        X = session.feature_matrix.toarray() if hasattr(session.feature_matrix, "toarray") else np.asarray(session.feature_matrix)

        # Calculate variance per term column
        variances = np.var(X, axis=0, ddof=0)

        # Sort indices by variance descending; select top 20
        sorted_indices = np.argsort(-variances)
        top_indices = sorted_indices[:20].tolist()

        top_terms = [session.feature_names[i] for i in top_indices]
        X_top = X[:, top_indices]

        # Calculate Pearson correlation matrix and normalize to 2D numpy array
        raw_corr = np.corrcoef(X_top, rowvar=False)
        corr_matrix = np.atleast_2d(raw_corr)
        corr_list = corr_matrix.tolist()

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".4f",
            cmap="coolwarm",
            xticklabels=top_terms,
            yticklabels=top_terms,
            ax=ax,
        )
        ax.set_title("Top 20 Features Correlation Matrix")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("feature_correlation")
        session.artifacts["feature_correlation"] = corr_list

        summary = {
            "result_id": result_id,
            "terms": top_terms,
            "correlation_matrix": corr_list,
        }

        full_report = pd.DataFrame(corr_list, index=top_terms, columns=top_terms)

        session.store_result(
            "feature_correlation_matrix_tool",
            {},
            summary,
            full_report=full_report,
        )

        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
