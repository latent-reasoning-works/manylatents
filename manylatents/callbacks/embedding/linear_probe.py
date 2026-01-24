"""
LinearProbeCallback - Evaluate frozen embeddings with linear/simple probes.

This callback runs lightweight classifiers (logistic regression, MLP) on
frozen embeddings to evaluate their quality for downstream tasks.
"""
import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import make_scorer, roc_auc_score, average_precision_score

from manylatents.callbacks.embedding.base import EmbeddingCallback, EmbeddingOutputs

logger = logging.getLogger(__name__)


def _get_classifier(name: str, **kwargs) -> Any:
    """Factory function for sklearn classifiers."""
    classifiers = {
        "logistic": lambda: LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            multi_class="multinomial",
            **kwargs
        ),
        "mlp": lambda: MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            **kwargs
        ),
    }
    if name not in classifiers:
        raise ValueError(f"Unknown classifier: {name}. Available: {list(classifiers.keys())}")
    return classifiers[name]()


def _safe_auroc(y_true, y_pred, **kwargs):
    """AUROC that handles edge cases (single class, etc.)."""
    try:
        n_classes = len(np.unique(y_true))
        if n_classes < 2:
            return np.nan
        if n_classes == 2:
            # Binary: use probability of positive class
            if y_pred.ndim == 2:
                y_pred = y_pred[:, 1]
            return roc_auc_score(y_true, y_pred)
        else:
            # Multiclass: use OvR
            return roc_auc_score(y_true, y_pred, multi_class="ovr", average="macro")
    except Exception as e:
        logger.warning(f"AUROC computation failed: {e}")
        return np.nan


def _safe_auprc(y_true, y_pred, **kwargs):
    """Average precision that handles edge cases."""
    try:
        n_classes = len(np.unique(y_true))
        if n_classes < 2:
            return np.nan
        if n_classes == 2:
            if y_pred.ndim == 2:
                y_pred = y_pred[:, 1]
            return average_precision_score(y_true, y_pred)
        else:
            # Multiclass: macro average
            from sklearn.preprocessing import label_binarize
            y_bin = label_binarize(y_true, classes=np.unique(y_true))
            return average_precision_score(y_bin, y_pred, average="macro")
    except Exception as e:
        logger.warning(f"AUPRC computation failed: {e}")
        return np.nan


class LinearProbeCallback(EmbeddingCallback):
    """
    Evaluate frozen embeddings using linear probes.

    Supports probing:
    - Embeddings from the embeddings dict (via embedding_key)
    - Module attributes like diffusion matrices (via module_attr)

    Example config:
        callbacks:
          embedding:
            probe:
              _target_: manylatents.callbacks.embedding.linear_probe.LinearProbeCallback
              embedding_key: embeddings
              probes: [logistic, mlp]
              cv_folds: 5
              metrics: [accuracy, auroc, auprc]
    """

    def __init__(
        self,
        embedding_key: str = "embeddings",
        module_attr: Optional[str] = None,
        probes: List[str] = None,
        cv_folds: int = 5,
        metrics: List[str] = None,
        scale_features: bool = True,
        random_state: int = 42,
    ) -> None:
        """
        Initialize LinearProbeCallback.

        Args:
            embedding_key: Key in embeddings dict to probe (e.g., "embeddings", "fused")
            module_attr: Alternative: attribute from module to probe (e.g., "diffusion_matrix")
            probes: List of probe types to run. Default: ["logistic"]
            cv_folds: Number of cross-validation folds
            metrics: Metrics to compute. Default: ["accuracy"]
            scale_features: Whether to standardize features before probing
            random_state: Random seed for reproducibility
        """
        super().__init__()
        self.embedding_key = embedding_key
        self.module_attr = module_attr
        self.probes = probes or ["logistic"]
        self.cv_folds = cv_folds
        self.metrics = metrics or ["accuracy"]
        self.scale_features = scale_features
        self.random_state = random_state

        logger.info(
            f"LinearProbeCallback initialized: "
            f"probes={self.probes}, cv_folds={self.cv_folds}, metrics={self.metrics}"
        )

    def _get_embeddings_to_probe(
        self,
        embeddings: EmbeddingOutputs,
        module: Optional[Any] = None
    ) -> Optional[np.ndarray]:
        """Extract embeddings to probe from embeddings dict or module attribute."""
        X = None

        # Try module_attr first if specified
        if self.module_attr is not None and module is not None:
            if hasattr(module, self.module_attr):
                X = getattr(module, self.module_attr)
                logger.info(f"Probing module attribute: {self.module_attr}")
            else:
                logger.warning(
                    f"Module {type(module).__name__} has no attribute '{self.module_attr}'"
                )

        # Fall back to embedding_key
        if X is None:
            X = embeddings.get(self.embedding_key)
            if X is not None:
                logger.info(f"Probing embeddings['{self.embedding_key}']")
            else:
                logger.warning(f"No embeddings found at key '{self.embedding_key}'")
                return None

        # Convert to numpy
        if hasattr(X, "numpy"):
            X = X.numpy()
        elif hasattr(X, "cpu"):
            X = X.cpu().numpy()

        return np.asarray(X)

    def _get_labels(
        self,
        dataset: Any,
        embeddings: EmbeddingOutputs
    ) -> Optional[np.ndarray]:
        """Extract labels from dataset or embeddings."""
        y = None

        # Try embeddings["label"] first
        if "label" in embeddings and embeddings["label"] is not None:
            y = embeddings["label"]
            logger.debug("Using labels from embeddings['label']")

        # Try dataset.get_labels()
        elif hasattr(dataset, "get_labels"):
            y = dataset.get_labels()
            logger.debug("Using labels from dataset.get_labels()")

        # Try dataset.labels
        elif hasattr(dataset, "labels"):
            y = dataset.labels
            logger.debug("Using labels from dataset.labels")

        if y is None:
            return None

        # Convert to numpy
        if hasattr(y, "numpy"):
            y = y.numpy()
        elif hasattr(y, "cpu"):
            y = y.cpu().numpy()

        return np.asarray(y)

    def _run_cv_probe(
        self,
        X: np.ndarray,
        y: np.ndarray,
        probe_name: str,
    ) -> Dict[str, float]:
        """Run cross-validated probe and return metrics."""
        results = {}

        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        n_classes = len(le.classes_)

        # Scale features
        if self.scale_features:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        # Get classifier
        clf = _get_classifier(probe_name, random_state=self.random_state)

        # Setup CV
        cv = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state
        )

        # Compute metrics
        for metric_name in self.metrics:
            if metric_name == "accuracy":
                scores = cross_val_score(clf, X, y_encoded, cv=cv, scoring="accuracy")
                results[f"{probe_name}_accuracy"] = float(np.mean(scores))
                results[f"{probe_name}_accuracy_std"] = float(np.std(scores))

            elif metric_name == "auroc":
                # AUROC requires predict_proba
                if not hasattr(clf, "predict_proba"):
                    logger.warning(f"Classifier {probe_name} doesn't support predict_proba for AUROC")
                    results[f"{probe_name}_auroc"] = np.nan
                    continue

                # Manual CV for AUROC (need probabilities)
                auroc_scores = []
                for train_idx, test_idx in cv.split(X, y_encoded):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

                    clf_fold = _get_classifier(probe_name, random_state=self.random_state)
                    clf_fold.fit(X_train, y_train)
                    y_proba = clf_fold.predict_proba(X_test)

                    auroc = _safe_auroc(y_test, y_proba)
                    auroc_scores.append(auroc)

                results[f"{probe_name}_auroc"] = float(np.nanmean(auroc_scores))
                results[f"{probe_name}_auroc_std"] = float(np.nanstd(auroc_scores))

            elif metric_name == "auprc":
                # Average precision requires predict_proba
                if not hasattr(clf, "predict_proba"):
                    logger.warning(f"Classifier {probe_name} doesn't support predict_proba for AUPRC")
                    results[f"{probe_name}_auprc"] = np.nan
                    continue

                # Manual CV for AUPRC
                auprc_scores = []
                for train_idx, test_idx in cv.split(X, y_encoded):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

                    clf_fold = _get_classifier(probe_name, random_state=self.random_state)
                    clf_fold.fit(X_train, y_train)
                    y_proba = clf_fold.predict_proba(X_test)

                    auprc = _safe_auprc(y_test, y_proba)
                    auprc_scores.append(auprc)

                results[f"{probe_name}_auprc"] = float(np.nanmean(auprc_scores))
                results[f"{probe_name}_auprc_std"] = float(np.nanstd(auprc_scores))

            else:
                logger.warning(f"Unknown metric: {metric_name}")

        return results

    def on_latent_end(
        self,
        dataset: Any,
        embeddings: EmbeddingOutputs,
        module: Optional[Any] = None,
    ) -> Dict[str, float]:
        """
        Run linear probes on frozen embeddings.

        Args:
            dataset: Dataset object (may have get_labels() method)
            embeddings: EmbeddingOutputs dict containing embeddings and optional labels
            module: Optional LatentModule (for probing module attributes)

        Returns:
            Dict of metric results: {"{probe}_{metric}": value, ...}
        """
        # Extract embeddings
        X = self._get_embeddings_to_probe(embeddings, module)
        if X is None:
            logger.error("No embeddings found to probe")
            return {}

        # Extract labels
        y = self._get_labels(dataset, embeddings)
        if y is None:
            logger.warning("No labels found - cannot run linear probe")
            return {}

        # Validate shapes
        if len(X) != len(y):
            logger.error(f"Shape mismatch: X has {len(X)} samples, y has {len(y)}")
            return {}

        logger.info(
            f"Running linear probes: X.shape={X.shape}, n_classes={len(np.unique(y))}"
        )

        # Run all probes
        all_results = {}
        for probe_name in self.probes:
            logger.info(f"Running {probe_name} probe...")
            try:
                probe_results = self._run_cv_probe(X, y, probe_name)
                all_results.update(probe_results)
                logger.info(f"  {probe_name}: {probe_results}")
            except Exception as e:
                logger.error(f"Probe {probe_name} failed: {e}")
                for metric in self.metrics:
                    all_results[f"{probe_name}_{metric}"] = np.nan

        # Register outputs
        for key, value in all_results.items():
            self.register_output(key, value)

        return all_results
