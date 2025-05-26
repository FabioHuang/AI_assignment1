from dataset import FaceEmbeddingDataset
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class KNNFaceClassifier:
    def __init__(self):
        self.model = None
        self.best_k = None
        self.best_metric = None
        
    def train(self, train_embeddings, train_labels, val_embeddings=None, val_labels=None):
        param_grid = {
            'n_neighbors': list(range(1, 21, 2)),
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'cosine']
        }
        
        print("Finding optimal KNN parameters using grid search...")
        grid_search = GridSearchCV(
            KNeighborsClassifier(),
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(train_embeddings, train_labels)
        
        self.model = grid_search.best_estimator_
        self.best_k = grid_search.best_params_['n_neighbors']
        self.best_metric = grid_search.best_params_['metric']
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Cross-validation accuracy: {grid_search.best_score_:.4f}")
        
        if val_embeddings is not None and val_labels is not None:
            val_accuracy = self.model.score(val_embeddings, val_labels)
            print(f"Validation accuracy: {val_accuracy:.4f}")
            
        return self
    
    def evaluate(self, test_embeddings, test_labels):
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        predictions = self.model.predict(test_embeddings)
        
        accuracy = accuracy_score(test_labels, predictions)
        print(f"Test accuracy: {accuracy:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(test_labels, predictions, zero_division=0))
        
        return accuracy
    
    def predict(self, embeddings):
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        return self.model.predict(embeddings)
    
    def predict_proba(self, embeddings):
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        return self.model.predict_proba(embeddings)
    
    def save(self, filepath):
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        instance = cls()
        with open(filepath, 'rb') as f:
            instance.model = pickle.load(f)
        instance.best_k = instance.model.n_neighbors
        instance.best_metric = instance.model.metric
        return instance