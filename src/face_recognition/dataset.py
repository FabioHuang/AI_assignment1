from pathlib import Path
from typing import Union, Tuple, Dict, List
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

class FaceEmbeddingDataset:
    def __init__(self, pickle_path: Union[str, Path], exclude_classes: List[str] = None):
        self.pickle_path = Path(pickle_path)
        self.embeddings = []
        self.labels = []
        self.label_encoder = LabelEncoder()
        self.exclude_classes = exclude_classes or ["desconhecido"]
        self.load_data()
        
    def load_data(self):
        if not self.pickle_path.exists():
            raise FileNotFoundError(f"Pickle file not found: {self.pickle_path}")
        
        all_embeddings = []
        all_labels = []
        
        with open(self.pickle_path, "rb") as f:
            try:
                while True:
                    data = pickle.load(f)
                    embeddings = data['embeddings']
                    if 'labels' in data:
                        labels = [str(label) for label in data['labels']]
                        
                        if self.exclude_classes:
                            keep_indices = [i for i, label in enumerate(labels) if label not in self.exclude_classes]
                            if keep_indices:
                                filtered_embeddings = embeddings[keep_indices]
                                filtered_labels = [labels[i] for i in keep_indices]
                                all_embeddings.append(filtered_embeddings)
                                all_labels.extend(filtered_labels)
                        else:
                            all_embeddings.append(embeddings)
                            all_labels.extend(labels)
            except EOFError:
                pass
        
        if all_embeddings:
            self.embeddings = np.vstack(all_embeddings)
            self.labels = np.array(all_labels)
            print(f"Loaded {len(self.labels)} samples with {len(set(self.labels))} unique classes")
            if self.exclude_classes:
                print(f"Excluded classes: {self.exclude_classes}")
        else:
            self.embeddings = np.array([])
            self.labels = np.array([])
            print("No embeddings found in the pickle file")
    
    def split_dataset(self, 
                      test_size: float = 0.2, 
                      val_size: float = 0.2, 
                      random_state: int = 42) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        if len(self.embeddings) == 0:
            raise ValueError("Dataset is empty. Cannot split.")
        
        encoded_labels = self.label_encoder.fit_transform(self.labels)
        
        train_val_emb, test_emb, train_val_labels, test_labels = train_test_split(
            self.embeddings, 
            encoded_labels,
            test_size=test_size,
            random_state=random_state,
            stratify=encoded_labels
        )
        
        if val_size > 0:
            relative_val_size = val_size / (1 - test_size)
            
            train_emb, val_emb, train_labels, val_labels = train_test_split(
                train_val_emb,
                train_val_labels,
                test_size=relative_val_size,
                random_state=random_state,
                stratify=train_val_labels
            )
            
            return {
                'train': (train_emb, train_labels),
                'val': (val_emb, val_labels),
                'test': (test_emb, test_labels)
            }
        else:
            return {
                'train': (train_val_emb, train_val_labels),
                'test': (test_emb, test_labels)
            }