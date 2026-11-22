"""Explain the study replay with the benchmark's fold-held-out Random Forest, using the project's shared model catalog.

Each participant's Random Forest score comes from a forest fitted WITHOUT that participant (the fold's best parameters from
results/model_benchmark). The fitted forest's probability is decomposed along its decision paths. The decomposition is an explanation of the
Random Forest only: it is NOT the headline performance estimate. The headline is the fully nested selection estimate
(results/nested_selection), whose per-participant held-out calls are attached to every case as `nested`, with the per-fold top-3
majority vote (results/nested_top3_per_fold) as `vote`.
"""
import ast
import sys

import numpy as np
import pandas as pd

def explain_study(root):
    sys.path.insert(0, str(root / "src"))
    import model_catalog as mc
    res = root / "results"
    frame = pd.read_csv(root / "data/processed/experiment_dataset.csv", dtype={"subject": str}).set_index("subject")
    X = frame.drop(columns=["group", "side", "target"])
    bench = pd.read_csv(res / "model_benchmark/model_benchmark_predictions.csv").query("model == 'Random_Forest'")
    bench["subject"] = bench.subject.astype(int).astype(str).str.zfill(2)
    config = pd.read_csv(res / "model_benchmark/model_benchmark_fold_results.csv").query("model == 'Random_Forest'").set_index("fold")
    nested = pd.read_csv(res / "nested_selection/nested_selection_predictions.csv")
    nested["subject"] = nested.subject.astype(int).astype(str).str.zfill(2)
    nested = nested.set_index("subject")
    vote = pd.read_csv(res / "nested_top3_per_fold/per_fold_top3_predictions.csv")
    vote["subject"] = vote.subject.astype(int).astype(str).str.zfill(2)
    vote = vote.set_index("subject")
    out = {}
    for fold, held in bench.groupby("fold"):
        ids = held.subject.to_list()
        train = [s for s in X.index if s not in set(ids)]
        params = ast.literal_eval(config.loc[fold, "best_parameters"])
        model = mc.make_pipeline("Random_Forest")
        model.set_params(selector__k=params["k"], **{"model__" + k: v for k, v in params.items() if k != "k"})
        model.fit(X.loc[train], frame.loc[train, "target"])
        values = model[:-1].transform(X.loc[ids])
        names = X.columns[model.named_steps["selector"].get_support()]
        rf = model.named_steps["model"]
        probabilities = rf.predict_proba(values)[:, 1]
        np.testing.assert_allclose(probabilities, held.predicted_score.to_numpy(), atol=1e-8)   # reproduces the saved benchmark exactly
        for i, subject in enumerate(ids):
            terms, base = np.zeros(len(names)), 0.0
            for tree in rf.estimators_:
                t = tree.tree_
                probs = t.value[:, 0, 1] / t.value[:, 0, :].sum(axis=1)
                node = 0
                base += probs[0]
                while t.children_left[node] != t.children_right[node]:
                    feature = t.feature[node]
                    child = t.children_left[node] if np.float32(values[i, feature]) <= t.threshold[node] else t.children_right[node]
                    terms[feature] += probs[child] - probs[node]
                    node = child
            terms /= len(rf.estimators_)
            base /= len(rf.estimators_)
            assert np.isclose(base + terms.sum(), probabilities[i])
            order = np.argsort(-np.abs(terms))
            out[subject] = dict(score=float(probabilities[i]), fold=int(fold), base=float(base),
                                terms=[[names[j], round(float(terms[j]), 8)] for j in order],
                                call=int(held.iloc[i].predicted_target),                      # Random Forest's own held-out call
                                nested_call=int(nested.loc[subject, "predicted_target"]),     # headline: fully nested selection procedure
                                nested_model=str(nested.loc[subject, "winner"]),
                                vote_call=int(vote.loc[subject, "majority_vote_of_top3"]))
    assert len(out) == 51
    return out
