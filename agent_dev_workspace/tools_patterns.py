import os
import tempfile
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool
from PAMI.extras.convert import DF2DB
from PAMI.frequentPattern.basic import FPGrowth
from PAMI.frequentPattern.topk import FAE
from PAMI.frequentPattern.maximal import MaxFPGrowth


def make_tools(session):
    """Factory that creates and returns workspace tools bound to the given session."""

    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: str,
        algorithm: str = "fpgrowth",
        min_sup: Optional[int] = None,
        k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Mines category-local frequent patterns using PAMI algorithms.

        Builds a category-local DTM with CountVectorizer(stop_words='english'), applies
        the requested vocabulary filter ('variance', 'tfidf', or 'term_frequency'), converts
        surviving terms into a PAMI transactional database, runs the chosen miner ('fpgrowth',
        'topk', or 'maxfpgrowth'), stores the pattern list in session.artifacts, and returns summary metadata.

        Args:
            category_name: Category label to filter session.dataframe by.
            filtering_method: Vocabulary filtering method ('variance', 'tfidf', or 'term_frequency').
            algorithm: PAMI mining algorithm ('fpgrowth', 'topk', or 'maxfpgrowth'). Defaults to 'fpgrowth'.
            min_sup: Minimum support threshold (integer count, required for 'fpgrowth' and 'maxfpgrowth').
            k: Number of top patterns to mine (integer, required for 'topk').

        Returns:
            Dict containing result_id, category_name, filtering_method, algorithm,
            surviving_terms_count, pattern_count, patterns, and note.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            raise ValueError("No dataset loaded in session or 'text' column missing.")

        if "category_name" not in session.dataframe.columns:
            raise ValueError("Required column 'category_name' missing from session.dataframe.")

        if type(category_name) is not str or not category_name.strip():
            raise ValueError("category_name must be a non-empty string.")

        valid_categories = session.dataframe["category_name"].unique().tolist()
        if category_name not in valid_categories:
            raise ValueError(f"Category '{category_name}' not found in dataset categories.")

        valid_filters = ["variance", "tfidf", "term_frequency"]
        if filtering_method not in valid_filters:
            raise ValueError(f"Invalid filtering_method '{filtering_method}'. Must be one of {valid_filters}.")

        valid_algos = ["fpgrowth", "topk", "maxfpgrowth"]
        if algorithm not in valid_algos:
            raise ValueError(f"Invalid algorithm '{algorithm}'. Must be one of {valid_algos}.")

        if algorithm in ["fpgrowth", "maxfpgrowth"]:
            if type(min_sup) is not int or min_sup <= 0:
                raise ValueError("min_sup must be a positive integer for 'fpgrowth' and 'maxfpgrowth' (booleans excluded).")
        elif algorithm == "topk":
            if type(k) is not int or k <= 0:
                raise ValueError("k must be a positive integer for 'topk' (booleans excluded).")

        result_id = session.next_result_id("mine_patterns")
        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"

        # Helper to return no-terms-survived summary consistently
        def _empty_terms_summary(note_text: str = "no terms survived filtering"):
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "surviving_terms_count": 0,
                "pattern_count": 0,
                "patterns": [],
                "note": note_text,
            }
            full_report = pd.DataFrame(columns=["pattern", "support"])
            session.artifacts[artifact_key] = []
            session.store_result("mine_patterns_tool", {"category_name": category_name, "filtering_method": filtering_method, "algorithm": algorithm, "min_sup": min_sup, "k": k}, summary, full_report=full_report)
            return summary

        # 1. Filter session.dataframe to category_name & build category-local DTM
        df_cat = session.dataframe[session.dataframe["category_name"] == category_name]
        text_cat = df_cat["text"].fillna("").astype(str)

        try:
            vectorizer = CountVectorizer(stop_words="english")
            X_cat = vectorizer.fit_transform(text_cat)
            local_feature_names = vectorizer.get_feature_names_out()
        except ValueError as err:
            if "empty vocabulary" in str(err).lower():
                return _empty_terms_summary()
            raise

        # 2. Vocabulary filtering using strict percentile bounds for middle 90%
        surviving_indices = []
        if len(local_feature_names) > 0:
            if filtering_method == "variance":
                variances = np.var(X_cat.toarray(), axis=0, ddof=0)
                p5, p95 = np.percentile(variances, 5), np.percentile(variances, 95)
                surviving_indices = np.where((variances > p5) & (variances < p95))[0].tolist()
            elif filtering_method == "term_frequency":
                sums = np.asarray(X_cat.sum(axis=0)).ravel()
                p5, p95 = np.percentile(sums, 5), np.percentile(sums, 95)
                surviving_indices = np.where((sums > p5) & (sums < p95))[0].tolist()
            elif filtering_method == "tfidf":
                tfidf_mat = TfidfTransformer().fit_transform(X_cat)
                mean_tfidf = np.asarray(tfidf_mat.mean(axis=0)).ravel()
                p20 = np.percentile(mean_tfidf, 20)
                surviving_indices = np.where(mean_tfidf >= p20)[0].tolist()

        surviving_terms = [local_feature_names[i] for i in surviving_indices]
        surviving_terms_count = len(surviving_terms)

        # Case 1: No terms survived filtering
        if surviving_terms_count == 0:
            return _empty_terms_summary("no terms survived filtering")

        # 3. Build transactional dataframe
        X_surviving = X_cat[:, surviving_indices].toarray()
        df_trans = pd.DataFrame(X_surviving, columns=surviving_terms)
        df_trans = df_trans[(df_trans.T != 0).any()]  # Drop empty transactions

        # Case 2: No non-empty transactions exist
        if len(df_trans) == 0:
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "surviving_terms_count": surviving_terms_count,
                "pattern_count": 0,
                "patterns": [],
                "note": "no non-empty transactions",
            }
            full_report = pd.DataFrame(columns=["pattern", "support"])
            session.artifacts[artifact_key] = []
            session.store_result("mine_patterns_tool", {"category_name": category_name, "filtering_method": filtering_method, "algorithm": algorithm, "min_sup": min_sup, "k": k}, summary, full_report=full_report)
            return summary

        # Map surviving terms to stable ASCII identifiers for PAMI DF2DB compatibility
        term_to_id = {term: f"item_{i}" for i, term in enumerate(surviving_terms)}
        id_to_term = {v: k for k, v in term_to_id.items()}
        df_trans_ascii = df_trans.rename(columns=term_to_id)

        # 4. PAMI mining via temp file
        patterns_list = []
        note = ""

        temp_fd, temp_path = tempfile.mkstemp(suffix=".csv")
        os.close(temp_fd)

        try:
            conv = DF2DB.DF2DB(df_trans_ascii)
            conv.convert2TransactionalDatabase(temp_path, ">=", 1)

            if algorithm == "fpgrowth":
                miner = FPGrowth.FPGrowth(iFile=temp_path, minSup=min_sup, sep="\t")
            elif algorithm == "topk":
                miner = FAE.FAE(iFile=temp_path, k=k, sep="\t")
            elif algorithm == "maxfpgrowth":
                miner = MaxFPGrowth.MaxFPGrowth(iFile=temp_path, minSup=min_sup, sep="\t")

            miner.startMine()
            patterns_df = miner.getPatternsAsDataFrame()

            if patterns_df is not None and len(patterns_df) > 0:
                for _, row in patterns_df.iterrows():
                    pat_str = str(row["Patterns"])
                    sup_val = int(row["Support"])
                    raw_items = pat_str.strip().split("\t") if "\t" in pat_str else pat_str.strip().split()
                    mapped_items = [id_to_term.get(item, item) for item in raw_items]
                    pat_items = sorted(mapped_items)
                    patterns_list.append({"pattern": pat_items, "support": sup_val})
            else:
                note = "no patterns found"
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

        # 5. Format full_report DataFrame
        report_rows = []
        for item in patterns_list:
            report_rows.append({
                "pattern": " ".join(item["pattern"]),
                "support": item["support"]
            })

        full_report = pd.DataFrame(report_rows, columns=["pattern", "support"])
        if len(full_report) > 0:
            full_report = full_report.sort_values(by=["support", "pattern"], ascending=[False, True]).reset_index(drop=True)

        session.artifacts[artifact_key] = patterns_list

        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "surviving_terms_count": surviving_terms_count,
            "pattern_count": len(patterns_list),
            "patterns": patterns_list,
            "note": note if note else ("" if len(patterns_list) > 0 else "no patterns found"),
        }

        session.store_result("mine_patterns_tool", {"category_name": category_name, "filtering_method": filtering_method, "algorithm": algorithm, "min_sup": min_sup, "k": k}, summary, full_report=full_report)
        return summary

    return [mine_patterns_tool]
