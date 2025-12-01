"""
Data format for ReservoirPy:
- Single timeseries: (timesteps, features)
- Multiple timeseries: list of (timesteps, features) arrays
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from reservoirpy import ESN, set_seed
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
set_seed(42)
np.random.seed(42)


class ReservoirBCI:
    """Simple Reservoir Computing for BCI using ESN class."""

    def __init__(self, units=1000, lr=0.05, sr=0.99, ridge=1e-5, input_scaling=1.0):
        # Use ESN class - combines Reservoir + Readout
        self.esn = ESN(units=units, lr=lr, sr=sr, ridge=ridge, input_scaling=input_scaling, rc_connectivity=0.1)
        self.scaler = StandardScaler()
        self.label_mapping = None

    def fit(self, X_train, y_train):
        """
        Train the model.
        X_train: (n_samples, n_channels, n_timepoints)
        y_train: (n_samples,) - labels
        """
        print(f"\nTraining with {len(X_train)} samples...")

        # Normalize data
        n_samples, n_channels, n_timepoints = X_train.shape
        X_flat = X_train.reshape(n_samples, -1)
        X_flat = self.scaler.fit_transform(X_flat)
        X_train = X_flat.reshape(n_samples, n_channels, n_timepoints)

        # Convert labels to integers and one-hot encode
        y_encoded, self.label_mapping = self._encode_labels(y_train)

        # One-hot encode targets
        n_classes = len(np.unique(y_encoded))
        y_onehot = np.zeros((len(y_encoded), n_classes))
        for i, label in enumerate(y_encoded):
            y_onehot[i, label] = 1

        # Prepare data for ReservoirPy: list of (timesteps, features) arrays
        X_list = [X_train[i].T for i in range(len(X_train))]
        n_timesteps = X_list[0].shape[0]
        y_list = [np.tile(y_onehot[i], (n_timesteps, 1)) for i in range(len(y_onehot))]

        print(f"Sample shape: {X_list[0].shape} (timesteps, channels)")
        print(f"Target shape: {y_list[0].shape} (timesteps, n_classes)")

        # Train ESN
        print("Training ESN...")
        self.esn = self.esn.fit(X_list, y_list, warmup=10)

        print("✓ Training completed!")

    def predict(self, X_test):
        """
        Predict labels.
        X_test: (n_samples, n_channels, n_timepoints)
        """
        # Normalize data
        n_samples, n_channels, n_timepoints = X_test.shape
        X_flat = X_test.reshape(n_samples, -1)
        X_flat = self.scaler.transform(X_flat)
        X_test = X_flat.reshape(n_samples, n_channels, n_timepoints)

        # Prepare data
        X_list = [X_test[i].T for i in range(len(X_test))]

        # Predict with ESN
        predictions = []
        for x in X_list:
            pred = self.esn.run(x)
            predictions.append(np.mean(pred, axis=0))

        predictions = np.array(predictions)
        predictions = np.argmax(predictions, axis=1)

        # Convert back to original labels
        if self.label_mapping:
            int_to_label = {v: k for k, v in self.label_mapping.items()}
            predictions = np.array([int_to_label[p] for p in predictions])

        return predictions

    def _encode_labels(self, y):
        """Convert string labels to integers."""
        if y.dtype.kind in ["U", "S", "O"]:
            unique = np.unique(y)
            mapping = {label: i for i, label in enumerate(unique)}
            print(f"Label mapping: {mapping}")
            y_int = np.array([mapping[label] for label in y])
            return y_int, mapping
        return y, None


def load_data(subject_id: int | None = 1):
    """Load BNCI dataset."""
    print(f"\n{'=' * 60}")
    print(f"Loading BNCI2014-001 (Subject {subject_id})")
    print(f"{'=' * 60}")

    dataset = BNCI2014_001()
    paradigm = MotorImagery(n_classes=4, fmin=7, fmax=35, tmin=0.0, tmax=4.0, resample=128)

    if subject_id is None:
        X, y, _ = paradigm.get_data(dataset=dataset)
    else:
        X, y, _ = paradigm.get_data(dataset=dataset, subjects=[subject_id])

    print(f"\n✓ Loaded: {X.shape}")
    print(f"  Classes: {np.unique(y)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"  Train: {len(X_train)}")
    print(f"  Test:  {len(X_test)}")

    return X_train, X_test, y_train, y_test


def evaluate(y_true, y_pred):
    """Evaluate and plot results."""
    unique = np.unique(np.concatenate([y_true, y_pred]))

    if isinstance(unique[0], (str, np.str_)):
        names = {"left_hand": "Left Hand", "right_hand": "Right Hand", "feet": "Feet", "tongue": "Tongue"}
        class_names = [names.get(str(label), str(label)) for label in unique]
    else:
        class_names = [f"Class {i}" for i in unique]

    acc = accuracy_score(y_true, y_pred)

    print(f"\n{'=' * 60}")
    print(f"Test Accuracy: {acc * 100:.2f}%")
    print(f"{'=' * 60}\n")

    print(classification_report(y_true, y_pred, target_names=class_names, labels=unique))

    cm = confusion_matrix(y_true, y_pred, labels=unique)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)

    plt.title(f"Confusion Matrix - RC-BCI\nTest Accuracy: {acc * 100:.2f}%", fontsize=14, fontweight="bold")
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    return acc, cm


def compare_sizes(X_train, X_test, y_train, y_test):
    """Compare reservoir sizes."""
    sizes = [500, 1000, 1500, 2000]
    results = {}

    print(f"\n{'=' * 60}")
    print("Comparing Reservoir Sizes")
    print(f"{'=' * 60}")

    for size in sizes:
        print(f"\nSize = {size}")
        model = ReservoirBCI(units=size, lr=0.05, sr=0.99)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[size] = acc
        print(f"  Accuracy: {acc * 100:.2f}%")

    plt.figure(figsize=(10, 6))
    plt.plot(list(results.keys()), [v * 100 for v in results.values()], "o-", linewidth=2, markersize=12, color="#2ecc71")
    plt.xlabel("Reservoir Size", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title("Performance vs Reservoir Size", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    plt.tight_layout()

    return results


def main():
    """Main execution."""
    print("\n" + "=" * 60)
    print("RESERVOIR COMPUTING FOR BCI")
    print("=" * 60)

    # using None for loading data for all subjects
    X_train, X_test, y_train, y_test = load_data(subject_id=1)

    print(f"\n{'=' * 60}")
    print("Training Model")
    print(f"{'=' * 60}")

    # Best config: larger reservoir, lower leak rate, higher spectral radius
    model = ReservoirBCI(units=1000, lr=0.05, sr=0.99, ridge=1e-5, input_scaling=1.0)
    model.fit(X_train, y_train)

    print(f"\n{'=' * 60}")
    print("Testing Model")
    print(f"{'=' * 60}")

    y_pred = model.predict(X_test)
    acc, cm = evaluate(y_test, y_pred)

    results = compare_sizes(X_train, X_test, y_train, y_test)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Main model: {acc * 100:.2f}%")
    print(f"Best size: {max(results, key=results.get)} ({max(results.values()) * 100:.2f}%)")
    print(f"{'=' * 60}\n")

    plt.figure(1)
    plt.savefig("rc_bci_confusion_matrix.png", dpi=300, bbox_inches="tight")
    print("✓ Saved: rc_bci_confusion_matrix.png")

    plt.figure(2)
    plt.savefig("rc_bci_comparison.png", dpi=300, bbox_inches="tight")
    print("✓ Saved: rc_bci_comparison.png")

    plt.show()


if __name__ == "__main__":
    main()
