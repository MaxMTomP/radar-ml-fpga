"""Train a simple MLP classifier for radar range-profile classification."""
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier


def load_datasets(base_path: str = "data/training_vectors"):
    """Load training, validation, and test datasets from .npy files."""
    x_train = np.load(f"{base_path}/X_train.npy")
    y_train = np.load(f"{base_path}/y_train.npy")
    x_val = np.load(f"{base_path}/X_val.npy")
    y_val = np.load(f"{base_path}/y_val.npy")
    x_test = np.load(f"{base_path}/X_test.npy")
    y_test = np.load(f"{base_path}/y_test.npy")
    return x_train, y_train, x_val, y_val, x_test, y_test


def build_classifier():
    """Create the MLP classifier with the specified architecture and parameters."""
    return MLPClassifier(
        hidden_layer_sizes=(6,),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42,
    )


def evaluate_and_report(model: MLPClassifier, x_train, y_train, x_val, y_val, x_test, y_test):
    """Generate accuracy metrics, confusion matrix, and classification report."""
    train_predictions = model.predict(x_train)
    val_predictions = model.predict(x_val)
    test_predictions = model.predict(x_test)

    train_accuracy = accuracy_score(y_train, train_predictions)
    val_accuracy = accuracy_score(y_val, val_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)

    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Validation accuracy: {val_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    conf_matrix = confusion_matrix(y_test, test_predictions)
    print("Confusion matrix:")
    print(conf_matrix)

    class_report = classification_report(y_test, test_predictions)
    print("Classification report:")
    print(class_report)

    return conf_matrix, class_report


def save_model_parameters(model: MLPClassifier):
    """Save trained model weights and biases to .npy files."""
    # model.coefs_ contains weight matrices for each layer; intercepts_ holds biases.
    np.save("ml_weights_hidden.npy", model.coefs_[0])
    np.save("ml_biases_hidden.npy", model.intercepts_[0])
    np.save("ml_weights_output.npy", model.coefs_[1])
    np.save("ml_biases_output.npy", model.intercepts_[1])


def main():
    # Load datasets
    x_train, y_train, x_val, y_val, x_test, y_test = load_datasets()

    # Initialize and train the classifier
    classifier = build_classifier()
    classifier.fit(x_train, y_train)

    # Evaluate the trained model and display metrics
    evaluate_and_report(classifier, x_train, y_train, x_val, y_val, x_test, y_test)

    # Save weights and biases for the hidden and output layers
    save_model_parameters(classifier)


if __name__ == "__main__":
    main()
