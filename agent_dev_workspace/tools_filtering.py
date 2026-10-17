from typing import Dict, Any
import scipy.stats as stats
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from langchain_core.tools import tool


def make_tools(session):
    """Factory that creates and returns workspace tools bound to the given session."""

    @tool
    def variance_filter_tool(threshold: float = 0.0) -> Dict[str, Any]:
        """Flags near-constant terms in the document-term matrix by population variance.

        Computes the population variance (ddof=0) for each term across all documents
        in session.feature_matrix efficiently without converting sparse matrices to dense.
        Terms with variance > threshold are kept, while terms with variance <= threshold
        are flagged for removal.

        Args:
            threshold: Minimum variance threshold. Terms with variance strictly greater
                than threshold are kept. Defaults to 0.0.

        Returns:
            Dict containing result_id, threshold, total_terms, terms_kept_count,
            terms_removed_count, kept_terms, and removed_terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("No feature matrix in session. Build a DTM first using build_dtm_tool.")

        X = session.feature_matrix

        # Compute sparse-friendly population variance (ddof=0): E[X^2] - (E[X])^2
        if issparse(X):
            # Mean along axis 0
            mean = np.asarray(X.mean(axis=0)).ravel()
            # Mean of elementwise squares along axis 0
            mean_sq = np.asarray(X.power(2).mean(axis=0)).ravel()
            variances = mean_sq - (mean ** 2)
            # Clip small negative numerical inaccuracies to 0.0
            variances = np.maximum(variances, 0.0)
        else:
            variances = np.var(np.asarray(X), axis=0, ddof=0)

        feature_names = list(session.feature_names)

        # Strict VarianceThreshold condition: variance > threshold
        keep_mask = variances > threshold

        kept_terms = [name for name, keep in zip(feature_names, keep_mask) if keep]
        removed_terms = [name for name, keep in zip(feature_names, keep_mask) if not keep]

        # Construct full_report DataFrame: [feature, variance, keep]
        df_report = pd.DataFrame({
            "feature": feature_names,
            "variance": variances,
            "keep": keep_mask,
        })

        # Sort descending by variance
        df_report = df_report.sort_values(by="variance", ascending=False).reset_index(drop=True)

        result_id = session.next_result_id("variance")
        summary = {
            "result_id": result_id,
            "threshold": threshold,
            "total_terms": len(feature_names),
            "terms_kept_count": len(kept_terms),
            "terms_removed_count": len(removed_terms),
            "kept_terms": kept_terms,
            "removed_terms": removed_terms,
        }

        session.store_result(
            "variance_filter_tool",
            {"threshold": threshold},
            summary,
            full_report=df_report,
        )

        return summary

    @tool
    def pearson_filter_tool(target_class: str, threshold: float = 0.0) -> Dict[str, Any]:
        """Pearson correlation filter between term counts and a target class indicator.

        Computes Pearson correlation r between each term count column in session.feature_matrix
        and a one-vs-rest binary indicator vector for target_class, filters terms using
        abs(pearson_r) > threshold, stores a sorted full_report DataFrame, and returns summary metadata.

        Args:
            target_class: Target category name for one-vs-rest binary correlation.
            threshold: Minimum absolute correlation threshold in [0.0, 1.0]. Defaults to 0.0.

        Returns:
            Dict containing result_id, target_class, threshold, total_terms, terms_kept_count,
            terms_removed_count, kept_terms, and removed_terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("No feature matrix in session. Build a DTM first using build_dtm_tool.")

        X = session.feature_matrix
        feature_names = list(session.feature_names)
        if len(feature_names) != X.shape[1]:
            raise ValueError(f"Feature name count ({len(feature_names)}) does not match DTM column count ({X.shape[1]}).")

        if type(target_class) is not str or not target_class.strip():
            raise ValueError("target_class must be a non-empty string.")

        if type(threshold) not in (int, float) or threshold < 0.0 or threshold > 1.0:
            raise ValueError("threshold must be a numeric value in [0.0, 1.0] (booleans excluded).")

        # Resolve category labels
        labels = session.labels
        if labels is None and session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].to_numpy()

        if labels is None:
            raise ValueError("No category labels found in session.")

        if len(labels) != X.shape[0]:
            raise ValueError(f"Label count ({len(labels)}) does not match DTM row count ({X.shape[0]}).")

        unique_labels = [str(lbl) for lbl in labels]
        if target_class not in unique_labels:
            raise ValueError(f"Target class '{target_class}' not found in dataset labels.")

        # Build one-vs-rest binary target
        y = (np.asarray(unique_labels) == target_class).astype(float)
        y_std = np.std(y, ddof=0)

        pearson_r_list = []

        # Compute Pearson r for each column
        for j in range(X.shape[1]):
            col = X[:, j].toarray().ravel() if issparse(X) else np.asarray(X[:, j]).ravel()
            col_std = np.std(col, ddof=0)
            if col_std == 0.0 or y_std == 0.0:
                pearson_r_list.append(0.0)
            else:
                r, _ = stats.pearsonr(col, y)
                pearson_r_list.append(0.0 if np.isnan(r) else float(r))

        df_report = pd.DataFrame({
            "feature": feature_names,
            "pearson_r": pearson_r_list,
            "keep": [bool(abs(r) > threshold) for r in pearson_r_list],
        })
        df_report = df_report.sort_values(by=["pearson_r", "feature"], ascending=[False, True]).reset_index(drop=True)

        kept_terms = df_report[df_report["keep"]]["feature"].tolist()
        removed_terms = df_report[~df_report["keep"]]["feature"].tolist()

        result_id = session.next_result_id("pearson")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "total_terms": len(feature_names),
            "terms_kept_count": len(kept_terms),
            "terms_removed_count": len(removed_terms),
            "kept_terms": kept_terms,
            "removed_terms": removed_terms,
        }

        session.store_result(
            "pearson_filter_tool",
            {"target_class": target_class, "threshold": threshold},
            summary,
            full_report=df_report,
        )

        return summary

    @tool
    def spearman_filter_tool(target_class: str, threshold: float = 0.0) -> Dict[str, Any]:
        """Spearman rank correlation filter between term counts and a target class indicator.

        Computes Spearman rank correlation rho between each term count column in session.feature_matrix
        and a one-vs-rest binary indicator vector for target_class, filters terms using
        abs(spearman_r) > threshold, stores a sorted full_report DataFrame, and returns summary metadata.

        Args:
            target_class: Target category name for one-vs-rest binary correlation.
            threshold: Minimum absolute correlation threshold in [0.0, 1.0]. Defaults to 0.0.

        Returns:
            Dict containing result_id, target_class, threshold, total_terms, terms_kept_count,
            terms_removed_count, kept_terms, and removed_terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("No feature matrix in session. Build a DTM first using build_dtm_tool.")

        X = session.feature_matrix
        feature_names = list(session.feature_names)
        if len(feature_names) != X.shape[1]:
            raise ValueError(f"Feature name count ({len(feature_names)}) does not match DTM column count ({X.shape[1]}).")

        if type(target_class) is not str or not target_class.strip():
            raise ValueError("target_class must be a non-empty string.")

        if type(threshold) not in (int, float) or threshold < 0.0 or threshold > 1.0:
            raise ValueError("threshold must be a numeric value in [0.0, 1.0] (booleans excluded).")

        # Resolve category labels
        labels = session.labels
        if labels is None and session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].to_numpy()

        if labels is None:
            raise ValueError("No category labels found in session.")

        if len(labels) != X.shape[0]:
            raise ValueError(f"Label count ({len(labels)}) does not match DTM row count ({X.shape[0]}).")

        unique_labels = [str(lbl) for lbl in labels]
        if target_class not in unique_labels:
            raise ValueError(f"Target class '{target_class}' not found in dataset labels.")

        # Build one-vs-rest binary target
        y = (np.asarray(unique_labels) == target_class).astype(float)
        y_std = np.std(y, ddof=0)

        spearman_r_list = []

        # Compute Spearman rank correlation for each column
        for j in range(X.shape[1]):
            col = X[:, j].toarray().ravel() if issparse(X) else np.asarray(X[:, j]).ravel()
            col_std = np.std(col, ddof=0)
            if col_std == 0.0 or y_std == 0.0:
                spearman_r_list.append(0.0)
            else:
                r, _ = stats.spearmanr(col, y)
                spearman_r_list.append(0.0 if np.isnan(r) else float(r))

        df_report = pd.DataFrame({
            "feature": feature_names,
            "spearman_r": spearman_r_list,
            "keep": [bool(abs(r) > threshold) for r in spearman_r_list],
        })
        df_report = df_report.sort_values(by=["spearman_r", "feature"], ascending=[False, True]).reset_index(drop=True)

        kept_terms = df_report[df_report["keep"]]["feature"].tolist()
        removed_terms = df_report[~df_report["keep"]]["feature"].tolist()

        result_id = session.next_result_id("spearman")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "total_terms": len(feature_names),
            "terms_kept_count": len(kept_terms),
            "terms_removed_count": len(removed_terms),
            "kept_terms": kept_terms,
            "removed_terms": removed_terms,
        }

        session.store_result(
            "spearman_filter_tool",
            {"target_class": target_class, "threshold": threshold},
            summary,
            full_report=df_report,
        )

        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
