import random

class SimpleLinearRegression:
    """A basic linear regression model implemented from scratch."""
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples = len(X)
        self.weights = 0.0
        self.bias = 0.0

        print(f"--- Starting local model training ({self.n_iterations} iterations) ---")
        # This loop simulates the "observable" aspect of the workstation
        for i in range(self.n_iterations):
            # Predict y based on current weights and bias
            y_predicted = [self.weights * x + self.bias for x in X]

            # Calculate gradients (Mean Squared Error derivative)
            dw = sum([(y_pred - y_true) * x for x, y_true, y_pred in zip(X, y, y_predicted)]) / n_samples
            db = sum([y_pred - y_true for y_true, y_pred in zip(y, y_predicted)]) / n_samples

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Basic observability: print loss every N iterations to monitor progress
            if (i + 1) % (self.n_iterations // 10) == 0:
                loss = sum([(y_pred - y_true)**2 for y_true, y_pred in zip(y, y_predicted)]) / n_samples
                print(f"Iteration {i+1}/{self.n_iterations}: Loss = {loss:.4f}")

        print("--- Local model training complete ---")
        print(f"Final Weights: {self.weights:.4f}, Bias: {self.bias:.4f}")

    def predict(self, X):
        """Makes predictions using the trained model."""
        return [self.weights * x + self.bias for x in X]

# --- Main execution ---
if __name__ == "__main__":
    print("XORAS Studio V2 - Offline AI Workstation Simulation")
    print("---------------------------------------------------\n")

    # 1. Simulate local, air-gapped data generation
    # This data is generated entirely within the script, no external access.
    # This represents the "offline" and "secure" data handling aspect.
    print("Generating synthetic local dataset...")
    X_train = [random.uniform(1, 10) for _ in range(50)]
    y_train = [2 * x + 5 + random.uniform(-1, 1) for x in X_train] # y = 2x + 5 + noise

    X_test = [random.uniform(1, 10) for _ in range(10)]
    y_test = [2 * x + 5 + random.uniform(-1, 1) for x in X_test]

    print(f"Generated {len(X_train)} training samples and {len(X_test)} test samples locally.\n")

    # 2. Initialize and train the model locally
    # The entire training process happens on the local machine,
    # without any internet connection or cloud services, demonstrating "offline" capability.
    model = SimpleLinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X_train, y_train) # Training with observability via print statements

    print("\n--- Making local predictions ---")
    # 3. Make predictions using the locally trained model
    predictions = model.predict(X_test)

    # 4. Basic observability for predictions
    print("Test Set Predictions:")
    for i, (x, y_true, y_pred) in enumerate(zip(X_test, y_test, predictions)):
        print(f"  Sample {i+1}: X={x:.2f}, Actual Y={y_true:.2f}, Predicted Y={y_pred:.2f}")

    print("\nSimulation complete. All operations performed locally and observably.")
    # No network requests were made, demonstrating the "offline" and "air-gapped" nature.
