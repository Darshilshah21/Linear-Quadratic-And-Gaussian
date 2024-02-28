


from models import GaussianNBModel
from utils import load_and_prepare_data
from sklearn.metrics import accuracy_score


def evaluate_gnb_model(X_train, y_train, X_test, y_test):
    # Reshape input data
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # Initialize and fit Gaussian Naive Bayes model
    gnb = GaussianNBModel()
    gnb.fit(X_train, y_train)

    # Make predictions on test data
    y_pred = gnb.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")


# Load and prepare RGB data
X_train_rgb, y_train_rgb, X_test_rgb, y_test_rgb = load_and_prepare_data(as_grayscale=False)

print("Evaluating Gaussian Naive Bayes on RGB data:")
evaluate_gnb_model(X_train_rgb, y_train_rgb, X_test_rgb, y_test_rgb)

# Load and prepare Grayscale data
X_train_gray, y_train_gray, X_test_gray, y_test_gray = load_and_prepare_data(as_grayscale=True)

print("Evaluating Gaussian Naive Bayes on Grayscale data:")
evaluate_gnb_model(X_train_gray, y_train_gray, X_test_gray, y_test_gray)
