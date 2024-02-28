import numpy as np
# models.py


class LDAModel:
    def __init__(self, nComponents):
        self.nComponents = nComponents
        self.ld = None
    
    def fit(self, X, y):
        nSamples, nFeatures = X.shape[:2]  # Extract number of samples and features
        labels = np.unique(y)

        # Calculate class means
        class_means = self.compute_class_means(X, y)

        # Calculate shared covariance matrix
        shared_covariance = self.compute_shared_covariance(X, y, class_means)

        # Compute SW and SB
        SW = np.zeros((nFeatures, nFeatures))
        SB = np.zeros((nFeatures, nFeatures))
        for label in labels:
            class_samples = X[y == label]
            n_c = class_samples.shape[0]
            mean_diff = (class_means[label] - np.mean(X, axis=0)).reshape(-1, 1)
            SB += n_c * mean_diff.dot(mean_diff.T)
            class_diff = class_samples - class_means[label]
            SW += np.dot(class_diff.T, class_diff)

        # Compute SW^-1 * SB
        A = np.linalg.inv(SW).dot(SB)

        # Get eigenvalues and eigenvectors of SW^-1 * SB
        eValues, eVectors = np.linalg.eig(A)

        # Sort eigenvectors based on eigenvalues
        idxs = np.argsort(abs(eValues))[::-1]
        eVectors = eVectors[:, idxs]

        # Select first nComponents eigenvectors
        self.ld = eVectors[:, :self.nComponents]

    def transform(self, X):
        # Project data
        return np.dot(X, self.ld)

    def predict(self, X):
        # Project data onto linear discriminants
        X_projected = self.transform(X)

        # Perform classification
        y_pred = np.argmax(X_projected, axis=1)

        return y_pred

    def compute_class_means(self, X, y):
        class_means = []
        for label in np.unique(y):
            class_samples = X[y == label]
            class_means.append(np.mean(class_samples, axis=0))
        return np.array(class_means)

    def compute_shared_covariance(self, X, y, class_means):
        total_samples = len(X)
        shared_covariance = np.zeros((X.shape[1], X.shape[1]))
        for label, class_mean in enumerate(class_means):
            class_samples = X[y == label]
            class_diff = class_samples - class_mean
            shared_covariance += np.dot(class_diff.T, class_diff) / total_samples
        return shared_covariance







class QDAModel:
    def __init__(self):
        self.class_means = None
        self.class_covariances = None
        self.class_priors = None
    
    def fit(self, X, y):
        nSamples, nFeatures = X.shape[:2]  # Extract number of samples and features
        labels = np.unique(y)

        # Compute the mean and standard deviation of each feature
        feature_means = np.mean(X, axis=0)
        feature_stds = np.std(X, axis=0)

        # Normalize the features
        X_normalized = (X - feature_means) / feature_stds

        # Initialize class means, covariances, and priors
        self.class_means = np.zeros((len(labels), nFeatures))
        self.class_covariances = np.zeros((len(labels), nFeatures, nFeatures))
        self.class_priors = np.zeros(len(labels))

        # Calculate class means, covariances, and priors
        for i, label in enumerate(labels):
            class_samples = X_normalized[y == label]
            self.class_means[i] = np.mean(class_samples, axis=0)
            self.class_covariances[i] = np.cov(class_samples.T)
            self.class_priors[i] = len(class_samples) / nSamples


    def predict(self, X):
        if self.class_means is None or self.class_covariances is None or self.class_priors is None:
            raise RuntimeError("Model has not been trained yet.")

        # Initialize array to store log probabilities for each class and sample
        log_probs = np.zeros((X.shape[0], len(self.class_means)))

        # Compute log probabilities for each class
        for i, (mean, cov, prior) in enumerate(zip(self.class_means, self.class_covariances, self.class_priors)):
            # Compute the difference between X and class mean
            diff = X - mean

            # Compute the determinant of covariance matrix for class
            cov_det = np.linalg.det(cov)

            # Check if the determinant is close to zero
            if cov_det < 1e-10:
                cov_det = 1e-10  # set a small non-zero value to avoid division by zero

            # Compute the inverse of covariance matrix for class
            cov_inv = np.linalg.inv(cov)

            # Compute the exponent term for class
            exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)

            # Compute the log probability for class
            log_probs[:, i] = exponent - 0.5 * np.log(np.abs(cov_det)) + np.log(prior)

        # Predict the class with the highest log probability for each sample
        predictions = np.argmax(log_probs, axis=1)

        return predictions




    

class GaussianNBModel:
    def __init__(self):
        self.class_means = None
        self.class_variances = None
        self.class_priors = None
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_means = {}
        self.class_variances = {}
        self.class_priors = {}

        for c in self.classes:
            X_c = X[y == c]
            self.class_means[c] = np.mean(X_c, axis=0)
            self.class_variances[c] = np.var(X_c, axis=0)  # Compute variance for each feature separately
            self.class_priors[c] = len(X_c) / len(X)

    def predict(self, X):
        # Initialize array to store log probabilities for each class and sample
        log_probs = np.zeros((X.shape[0], len(self.classes)))

        # Compute log probabilities for each class
        for i, c in enumerate(self.classes):
            class_mean = self.class_means[c]
            class_variance = self.class_variances[c]
            class_prior = self.class_priors[c]

            # Compute log likelihoods for each feature and sum them up
            log_likelihoods = -0.5 * np.sum(np.log(2 * np.pi * class_variance) +
                                            ((X - class_mean) ** 2) / class_variance, axis=1)
            # Compute log posterior probabilities by adding log likelihood and log prior
            log_probs[:, i] = log_likelihoods + np.log(class_prior)

        # Predict the class with the highest log posterior probability for each sample
        predictions = np.argmax(log_probs, axis=1)

        return predictions
