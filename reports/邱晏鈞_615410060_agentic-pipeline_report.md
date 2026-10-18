# Agentic Pipeline Report

I loaded the four requested 20 Newsgroups categories in `load_dataset_1`: `alt.atheism` (480 documents), `comp.graphics` (584), `sci.med` (594), and `soc.religion.christian` (599), for 2,257 documents total. In `dtm_2`, I built a unigram DTM with `min_df=1`, `max_df=1.0`, and `max_df_mode="proportion"`. It contains 35,788 terms and 365,886 nonzero entries across 80,773,516 cells, giving 99.5470% sparsity.

I ran variance-filtered FPGrowth with `min_sup=50` for each category. The returned pattern counts were 8 for `comp.graphics`, 3 for `soc.religion.christian`, 7 for `sci.med`, and 39 for `alt.atheism`. Examples include `[newsreader]` (support 58) in `comp.graphics`, `[subject]` (support 594) in `sci.med`, and `[lines]` (support 480) in `alt.atheism`. The chat outputs did not show result IDs for these four mining calls.

In `reduce_dimensions_4`, PCA reduced the documents to two dimensions. The explained variance ratios were 0.65937 for component 1 and 0.04330 for component 2. The resulting category-colored scatter plot was saved to `plots/邱晏鈞_agentic-pipeline_plot_reduce_dimensions_4.png`.

In `binarize_labels_5`, I encoded `category_name` into four binary columns: `category_name_alt.atheism`, `category_name_comp.graphics`, `category_name_sci.med`, and `category_name_soc.religion.christian`, covering all 2,257 documents. In `cosine_similarity_3`, the similarity between documents 0 and 1 was 0.3532413464362985.
