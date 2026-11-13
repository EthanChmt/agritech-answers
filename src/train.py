"""Pipeline d'entraînement complet : données -> optimisation Optuna -> évaluation -> MLflow -> champion.

Usage :
    python -m src.train                        # protocole complet (40 essais par modèle)
    python -m src.train --n-trials 5 --max-rows 50000   # essai rapide
"""

import argparse
import logging
import time

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.model_selection import KFold

from src import config
from src.data import load_clean, split
from src.evaluate import crop_effect, evaluate, rmse
from src.pipeline import build_pipeline

logger = logging.getLogger("train")


def setup_mlflow(tracking_uri: str, experiment: str) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=config.ARTIFACT_ROOT.as_uri())
    mlflow.set_experiment(experiment)


def suggest_params(name: str, trial: optuna.Trial) -> dict:
    if name == "ridge":
        return {"alpha": trial.suggest_float("alpha", 1e-3, 1e3, log=True)}
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_iter": trial.suggest_int("max_iter", 100, 600),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 8, 128),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 400),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-6, 10.0, log=True),
    }


def make_objective(name, X_tune, y_tune, cv_folds, seed):
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(name, trial)
        scores, pruned = [], False
        with mlflow.start_run(run_name=f"{name}-trial-{trial.number:03d}", nested=True):
            mlflow.set_tags({"stage": "optuna-trial", "model_name": name})
            mlflow.log_params(params)
            for fold, (idx_tr, idx_va) in enumerate(kfold.split(X_tune)):
                pipe = build_pipeline(name, params).fit(X_tune.iloc[idx_tr], y_tune.iloc[idx_tr])
                scores.append(rmse(y_tune.iloc[idx_va], pipe.predict(X_tune.iloc[idx_va])))
                trial.report(float(np.mean(scores)), step=fold)
                if trial.should_prune():
                    pruned = True
                    break
            mlflow.log_metrics({
                "cv_rmse_mean": float(np.mean(scores)),
                "cv_rmse_std": float(np.std(scores)),
                "folds_evaluated": len(scores),
            })
            mlflow.set_tag("state", "pruned" if pruned else "complete")
        if pruned:
            raise optuna.TrialPruned()
        return float(np.mean(scores))

    return objective


def tune(name, X_tune, y_tune, n_trials, cv_folds, seed) -> optuna.Study:
    """Recherche d'hyperparamètres (TPE + MedianPruner), un run parent MLflow contenant un run par essai."""
    with mlflow.start_run(run_name=f"{name}-optuna"):
        mlflow.set_tags({
            "stage": "optuna-search",
            "model_name": name,
            "mlflow.note.content": f"Recherche Optuna (TPE, MedianPruner) pour {name}, validation croisée {cv_folds} plis.",
        })
        mlflow.log_params({
            "seed": seed, "n_trials": n_trials, "cv_folds": cv_folds, "n_tune_sample": len(X_tune),
            "sampler": "TPE", "pruner": "MedianPruner",
        })
        study = optuna.create_study(
            study_name=name, direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=min(8, n_trials), n_warmup_steps=1),
        )
        t0 = time.perf_counter()
        study.optimize(make_objective(name, X_tune, y_tune, cv_folds, seed), n_trials=n_trials)
        states = [t.state for t in study.trials]
        mlflow.log_metrics({
            "best_cv_rmse": study.best_value,
            "n_complete": states.count(optuna.trial.TrialState.COMPLETE),
            "n_pruned": states.count(optuna.trial.TrialState.PRUNED),
            "search_time_s": time.perf_counter() - t0,
        })
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
    return study


def fit_and_log(run_name, name, params, stage, data, extra_metrics=None):
    """Entraîne sur tout le train, évalue train et test, journalise le pipeline complet dans MLflow."""
    X_train, X_test, y_train, y_test = data
    pipe = build_pipeline(name, params)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({"stage": stage, "model_name": name, "target": config.TARGET})
        mlflow.log_params({
            "seed": config.SEED, "n_train": len(X_train), "n_test": len(X_test),
            "features": ",".join(config.FEATURES), **{f"param_{k}": v for k, v in params.items()},
        })
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0
        m_train, m_test = evaluate(pipe, X_train, y_train), evaluate(pipe, X_test, y_test)
        mlflow.log_metrics({
            **{f"train_{k}": v for k, v in m_train.items()},
            **{f"test_{k}": v for k, v in m_test.items()},
            "gap_r2_train_minus_test": m_train["r2"] - m_test["r2"],
            "fit_time_s": fit_time,
            **(extra_metrics or {}),
        })
        example = X_train.head(1000)
        mlflow.sklearn.log_model(
            pipe, name="model",
            signature=infer_signature(example, pipe.predict(example)),
            input_example=example.head(3),
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
    return pipe, {"run_id": run.info.run_id, "model": name, "run_name": run_name, **{f"test_{k}": v for k, v in m_test.items()}}


def select_model(best_cv: dict[str, float]) -> str:
    """Le modèle simple est retenu s'il est à moins de TOLERANCE_SELECTION du meilleur RMSE de validation croisée."""
    lowest = min(best_cv.values())
    if config.SIMPLE_MODEL in best_cv and best_cv[config.SIMPLE_MODEL] <= lowest + config.TOLERANCE_SELECTION:
        return config.SIMPLE_MODEL
    return min(best_cv, key=best_cv.get)


def register_champion(result: dict, selected: str, cv_rmse: float, spread_mean: float) -> int:
    """Enregistre le modèle dans le registry MLflow et lui donne l'alias `champion`."""
    client = MlflowClient()
    version = mlflow.register_model(f"runs:/{result['run_id']}/model", config.REGISTRY_NAME)
    client.set_registered_model_alias(config.REGISTRY_NAME, config.CHAMPION_ALIAS, version.version)
    tags = {
        "selected_model": selected,
        "test_rmse": f"{result['test_rmse']:.6f}",
        "test_r2": f"{result['test_r2']:.6f}",
        "cv_rmse": f"{cv_rmse:.6f}",
        "crop_spread_mean": f"{spread_mean:.6f}",
        "sklearn_version": sklearn.__version__,
    }
    for key, value in tags.items():
        client.set_model_version_tag(config.REGISTRY_NAME, version.version, key, value)
    return int(version.version)


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-trials", type=int, default=config.N_TRIALS)
    p.add_argument("--cv-folds", type=int, default=config.CV_FOLDS)
    p.add_argument("--tune-sample", type=int, default=config.N_TUNE_SAMPLE)
    p.add_argument("--max-rows", type=int, default=None, help="Sous-échantillonne le jeu (essais rapides).")
    p.add_argument("--tracking-uri", default=config.TRACKING_URI)
    p.add_argument("--experiment", default=config.EXPERIMENT_NAME)
    p.add_argument("--no-register", action="store_true", help="Ne pas enregistrer le champion dans le registry.")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    np.random.seed(config.SEED)
    setup_mlflow(args.tracking_uri, args.experiment)

    df = load_clean()
    if args.max_rows:
        df = df.sample(min(args.max_rows, len(df)), random_state=config.SEED)
    X_train, X_test, y_train, y_test = data = split(df)
    X_tune = X_train.sample(min(args.tune_sample, len(X_train)), random_state=config.SEED)
    y_tune = y_train.loc[X_tune.index]
    logger.info("Train %s | test %s | recherche %s lignes", f"{len(X_train):,}", f"{len(X_test):,}", f"{len(X_tune):,}")

    studies, results = {}, {}
    for name in config.MODEL_NAMES:
        studies[name] = tune(name, X_tune, y_tune, args.n_trials, args.cv_folds, config.SEED)
        logger.info("%-24s RMSE CV = %.4f | %s", name, studies[name].best_value, studies[name].best_params)
        fit_and_log(f"reference-{name}-defaut", name, {}, "reference-full", data)
        _, results[name] = fit_and_log(
            f"final-{name}-optuna", name, studies[name].best_params, "final-optuna", data,
            extra_metrics={"cv_rmse_optuna": studies[name].best_value},
        )

    best_cv = {name: study.best_value for name, study in studies.items()}
    selected = select_model(best_cv)
    result = results[selected]
    logger.info("Modèle retenu : %s | test RMSE %.4f | R² %.4f", selected, result["test_rmse"], result["test_r2"])

    pipe = mlflow.sklearn.load_model(f"runs:/{result['run_id']}/model")
    by_crop = crop_effect(pipe, X_test)
    spread_mean = float((by_crop.max(axis=1) - by_crop.min(axis=1)).mean())
    logger.info("Écart moyen entre cultures : %.4f t/ha (RMSE du modèle : %.3f t/ha)", spread_mean, result["test_rmse"])

    if not args.no_register:
        version = register_champion(result, selected, best_cv[selected], spread_mean)
        logger.info("Enregistré : models:/%s@%s (version %s)", config.REGISTRY_NAME, config.CHAMPION_ALIAS, version)
    return {"selected": selected, **result, "crop_spread_mean": spread_mean}


if __name__ == "__main__":
    main()