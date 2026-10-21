import logging

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

from pipeline.retrain.fit_catboost import prepare_features
from pipeline.settings import MODELS_DIR

logger = logging.getLogger(__name__)


def check_model_quality(train_info: dict, preproc_info: dict) -> dict:
    """Читает модели и test, сравнивает ROC-AUC, пишет результат в лог.

    Возвращает решение о публикации.
    Данные и модели не изменяет.
    """
    test = pd.read_parquet(preproc_info["test_proc_path"])
    x_test, y_test = prepare_features(test)

    if y_test.isna().any() or set(y_test.unique()) != {0, 1}:
        raise ValueError("Для ROC-AUC нужны оба класса, и 0 и 1")

    current_version = (MODELS_DIR / "latest.txt").read_text().strip()
    current_path = MODELS_DIR / f"my_catboost_{current_version}.cbm"

    scores = []
    for path in [train_info["staging_model"], current_path]:
        model = CatBoostClassifier()
        model.load_model(str(path))
        features = x_test.loc[:, model.feature_names_].copy()

        for index in model.get_cat_feature_indices():
            column = model.feature_names_[index]
            features[column] = features[column].astype(str)

        probabilities = model.predict_proba(features)[:, 1]
        scores.append(float(roc_auc_score(y_test, probabilities)))
    candidate_auc, current_auc = scores
    logger.info(
        "ROC-AUC на месяце %s: кандидат %s = %.6f, текущая модель %s = %.6f",
        preproc_info["test_month"],
        train_info["version"],
        candidate_auc,
        current_version,
        current_auc,
    )

    return {
        **train_info,
        "approved": candidate_auc >= current_auc,
        "candidate_auc": candidate_auc,
        "current_auc": current_auc,
        "current_version": current_version,
    }
