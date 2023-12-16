# data_science_pipeline.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import yaml

# Load configuration from config.yaml
with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

# Feature Engineering Step
class ImputationStep:
    def process(self, data):
        # Code for imputation
        return data

    def visualize_data(self, data):
        # Code for visualizing data (e.g., print data)
        print("Imputation Step Visualization:", data)


# Data Visualization Step
class CustomVisualizationStep:
    def visualize_data(self, data):
        # Code for custom visualization
        print("Custom Visualization:", data)


# Train-Test Split Step
class TrainTestSplitStep:
    def __init__(self, test_size, random_state):
        self.test_size = test_size
        self.random_state = random_state

    def process(self, data):
        # Train-test split
        X = data.drop("class", axis=1)
        y = data["class"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        return X_train, X_test, y_train, y_test


# Pipeline Manager
class DataSciencePipelineManager:
    def __init__(self):
        self.steps = []
        self.data = None

    def add_step(self, step):
        self.steps.append(step)

    def process_data(self):
        for step in self.steps:
            self.data = step.process(self.data)
            step.visualize_data(self.data)

    def build_model(self):
        # Code for building the machine learning model
        return RandomForestClassifier(random_state=config["model"]["random_state"])

    def train_model(self, model, X_train, y_train, X_test, y_test, epochs=1):
        # Train the model
        for epoch in range(epochs):
            model.fit(X_train, y_train)

            # Additional logic if needed

            # Optional: Visualize data during training
            self.process_data()

        return model

    def evaluate_model(self, model, X_test, y_test):
        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))


# Usage
if __name__ == "__main__":
    # Data loading
    df = pd.read_csv(config["data"]["url"], header=None, names=config["data"]["columns"])

    # Data preprocessing
    df["class"] = df["class"].astype("category").cat.codes

    # Creating an instance of the Pipeline Manager
    pipeline_manager = DataSciencePipelineManager()
    pipeline_manager.data = df  # Set the data for the pipeline manager

    # Adding feature engineering steps and data visualization steps dynamically
    pipeline_manager.add_step(ImputationStep())
    pipeline_manager.add_step(CustomVisualizationStep())
    
    # Adding Train-Test Split as a step
    train_test_split_step = TrainTestSplitStep(
        test_size=config["train_test_split"]["test_size"],
        random_state=config["train_test_split"]["random_state"]
    )
    pipeline_manager.add_step(train_test_split_step)

    # Build model
    model = pipeline_manager.build_model()

    # Train and evaluate model
    trained_model = pipeline_manager.train_model(model, *pipeline_manager.data, epochs=config["epochs"])
    pipeline_manager.evaluate_model(trained_model, *pipeline_manager.data[:-1])
