import pickle
import numpy as np
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class KNNFaceClassifier:
    def __init__(self):
        self.model = None
        self.best_k = None
        self.best_metric = None
        self.unknown_threshold = None
        
    def train(self, train_embeddings, train_labels, label_encoder_classes, val_embeddings=None, val_labels=None): # Added label_encoder_classes
        self.label_classes_ = label_encoder_classes
        param_grid = {
            'n_neighbors': list(range(1, 5)),
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'cosine']
        }
        
        print("Finding optimal KNN parameters using grid search...")
        grid_search = GridSearchCV(
            KNeighborsClassifier(),
            param_grid,
            cv=4, #20% do dataset de treino
            scoring='accuracy',
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
        target_names_for_report = None
        if self.label_classes_ is not None:
            target_names_for_report = [str(label) for label in self.label_classes_]

        print(classification_report(test_labels, predictions, target_names=target_names_for_report, zero_division=0))
        
        return accuracy
    
    def predict(self, embeddings):
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if self.unknown_threshold is None:
            int_predictions = self.model.predict(embeddings)
            if self.label_classes_ is not None:
                return self.label_classes_[int_predictions.astype(int)]
            else:
                raise ValueError("Label classes not available for converting predictions to names.")
            
        # Prediction with unknown threshold
        distances, _ = self.model.kneighbors(embeddings)
        kth_distances = distances[:, -1] 
        
        string_predictions = []
        for i, dist in enumerate(kth_distances):
            if dist >= self.unknown_threshold:
                string_predictions.append("desconhecido") # Directly use string for unknown
            else:
                int_prediction = self.model.predict(embeddings[i:i+1])[0]
                if self.label_classes_ is not None:
                    string_predictions.append(self.label_classes_[int(int_prediction)])
                else:
                    raise ValueError("Label classes not available for converting prediction to name.")
                
        return np.array(string_predictions)
    
    def predict_proba(self, embeddings):
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        return self.model.predict_proba(embeddings)
    
    def save(self, filepath):
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        data_to_save = {
            'model': self.model,
            'label_classes': self.label_classes_,
            'best_k': self.best_k,
            'best_metric': self.best_metric,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data_to_save, f)
        print(f"Model saved to {filepath}")

    def set_unknown_threshold(self, threshold):
        self.unknown_threshold = threshold

    @classmethod
    def load(cls, filepath):
        instance = cls()
        with open(filepath, 'rb') as f:
            saved_data = pickle.load(f)
        
        instance.model = saved_data['model']
        instance.label_classes_ = saved_data.get('label_classes')
        instance.best_k = saved_data.get('best_k', instance.model.n_neighbors if instance.model else None)
        instance.best_metric = saved_data.get('best_metric', instance.model.metric if instance.model else None)

        if instance.model is not None and instance.label_classes_ is None:
            print(f"Warning: Label classes not found in the loaded model from {filepath}. "
                  "Predictions for known classes might fail if string names are expected. "
                  "Please retrain and save the model with label classes.")

        return instance
    
if __name__ == "__main__":
    from dataset import FaceEmbeddingDataset
    from matplotlib import pyplot as plt

    dataset = FaceEmbeddingDataset("/home/fabio/Repos/embeddings.pkl", exclude_classes=["desconhecido"])
    
    splits = dataset.split_dataset(test_size=0.2)
    train_embeddings, train_labels = splits['train']
    test_embeddings, test_labels = splits['test']
    
    print(f"Train set: {len(train_labels)} samples")
    print(f"Test set: {len(test_labels)} samples")
    
    classifier = KNNFaceClassifier()
    classifier.train(train_embeddings, train_labels, dataset.label_encoder.classes_)
    
    test_accuracy = classifier.evaluate(test_embeddings, test_labels)
    
    classifier.save("/home/fabio/Repos/AI_assignment1/models/knn_face_classifier.pkl")

    # Find optimal threshold for unknown detection
    # Load the unknown dataset
    all_classes = set(train_labels) | set(test_labels)
    classes_to_exclude = [cls for cls in all_classes if cls != "desconhecido"]
    unknown_dataset = FaceEmbeddingDataset("/home/fabio/Repos/embeddings.pkl", exclude_classes=classes_to_exclude)
    unknown_embeddings = unknown_dataset.embeddings
    
    classifier.set_unknown_threshold(0.3)
    
    classifier.save("/home/fabio/Repos/AI_assignment1/models/knn_face_classifier.pkl")
    
    print("\nTesting prediction with unknown detection:")
    predictions = classifier.predict(unknown_embeddings[:10])
    predictions2 = classifier.predict(test_embeddings[:10])
    print(f"Predictions: {predictions}")
    print(f"Predictions2: {predictions2}")