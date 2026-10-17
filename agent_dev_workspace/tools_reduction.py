from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from langchain_core.tools import tool


def make_tools(session):
    """Factory that creates and returns workspace tools bound to the given session."""

    @tool
    def reduce_dimensions_tool(
        method: str = "pca",
        perplexity: float = 30.0,
        n_neighbors: int = 15,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Reduces document-term matrix dimensions to 2D using PCA, t-SNE, or UMAP, and plots the scatter graph.

        Args:
            method: Reduction algorithm to use ('pca', 'tsne', or 'umap'). Defaults to 'pca'.
            perplexity: Perplexity parameter for t-SNE. Defaults to 30.0.
            n_neighbors: Number of neighbors parameter for UMAP. Defaults to 15.
            random_state: Optional random seed for reproducible stochastic reductions.

        Returns:
            Dict containing result_id, method, n_components (2), coordinates, and explained_variance_ratio (PCA only).
        """
        if session.feature_matrix is None:
            raise ValueError("No feature_matrix found in session. Please run build_dtm_tool first.")

        valid_methods = ["pca", "tsne", "umap"]
        if method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Must be one of {valid_methods}.")

        # Convert sparse or dense feature matrix to 2D numpy array
        X = session.feature_matrix.toarray() if hasattr(session.feature_matrix, "toarray") else np.asarray(session.feature_matrix)

        explained_variance_ratio = None

        if method == "pca":
            reducer = PCA(n_components=2, random_state=random_state)
            coords = reducer.fit_transform(X)
            explained_variance_ratio = reducer.explained_variance_ratio_.tolist()
        elif method == "tsne":
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, init="random", learning_rate=200.0)
            coords = reducer.fit_transform(X)
        elif method == "umap":
            reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=random_state)
            coords = reducer.fit_transform(X)

        coords_list = coords.tolist()

        # Build 2D scatter plot colored by category_name or category
        fig, ax = plt.subplots(figsize=(8, 6))
        
        hue = None
        if session.dataframe is not None:
            if "category_name" in session.dataframe.columns:
                hue = session.dataframe["category_name"]
            elif "category" in session.dataframe.columns:
                hue = session.dataframe["category"]

        df_plot = pd.DataFrame({
            "Component 1": coords[:, 0],
            "Component 2": coords[:, 1],
        })
        if hue is not None:
            df_plot["Category"] = hue.values if hasattr(hue, "values") else hue
            sns.scatterplot(data=df_plot, x="Component 1", y="Component 2", hue="Category", ax=ax, s=70)
        else:
            sns.scatterplot(data=df_plot, x="Component 1", y="Component 2", ax=ax, s=70)

        ax.set_title(f"2D Dimension Reduction ({method.upper()})")
        plt.tight_layout()

        # Set pending figure for notebook display
        session.pending_figure = fig

        result_id = session.next_result_id("reduce_dimensions")
        artifact_key = f"reduced_{method}"
        session.artifacts[artifact_key] = coords_list

        summary = {
            "result_id": result_id,
            "method": method,
            "n_components": 2,
            "coordinates": coords_list,
            "explained_variance_ratio": explained_variance_ratio,
        }

        full_report = pd.DataFrame({
            "Component_1": coords[:, 0],
            "Component_2": coords[:, 1],
        })
        if hue is not None:
            full_report["Category"] = hue.values if hasattr(hue, "values") else hue

        session.store_result(
            "reduce_dimensions_tool",
            {"method": method, "perplexity": perplexity, "n_neighbors": n_neighbors, "random_state": random_state},
            summary,
            full_report=full_report,
        )

        return summary

    @tool
    def binarize_labels_tool(
        label_column: str = "category_name"
    ) -> Dict[str, Any]:
        """One-hot encodes category labels into binary indicator columns.

        Validates label_column existence in session.dataframe, checks for missing/NaN values,
        applies pd.get_dummies with prefix=label_column and dtype=int, preserves row order,
        stores the encoded DataFrame in session.artifacts['binarized_labels'], and returns summary metadata.

        Args:
            label_column: Name of the label column in session.dataframe to encode. Defaults to 'category_name'.

        Returns:
            Dict containing result_id, label_column, total_documents, and binary_columns.
        """
        if session.dataframe is None:
            raise ValueError("No dataset loaded in session. Load a dataset first.")

        if type(label_column) is not str or not label_column.strip():
            raise ValueError("Parameter 'label_column' must be a non-empty string.")

        if label_column not in session.dataframe.columns:
            raise ValueError(f"Column '{label_column}' not found in session.dataframe.")

        labels = session.dataframe[label_column]
        if labels.isna().any():
            raise ValueError(f"Column '{label_column}' contains missing/NaN values.")

        dummies = pd.get_dummies(labels, prefix=label_column, dtype=int)
        binary_cols = dummies.columns.tolist()
        total_docs = len(dummies)

        # Store in session artifacts
        session.artifacts["binarized_labels"] = dummies

        result_id = session.next_result_id("binarize_labels")
        summary = {
            "result_id": result_id,
            "label_column": label_column,
            "total_documents": total_docs,
            "binary_columns": binary_cols,
        }

        session.store_result(
            "binarize_labels_tool",
            {"label_column": label_column},
            summary,
            full_report=dummies,
        )

        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
