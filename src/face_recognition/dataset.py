from pathlib import Path
from typing import Union, Tuple, Dict
import pickle
import numpy as np
from sklearn.model_selection import train_test_split


class FaceEmbeddingDataset:
    def __init__(self, pickle_path: Union[str, Path]):
        self.pickle_path = Path(pickle_path)
        self.embeddings = []
        self.labels = []
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
                    all_embeddings.append(data['embeddings'])
                    # Make sure labels are stored as strings, not converted to integers
                    if 'labels' in data:
                        all_labels.extend([str(label) for label in data['labels']])
            except EOFError:
                pass  # End of file reached
        
        if all_embeddings:
            self.embeddings = np.vstack(all_embeddings)
            self.labels = np.array(all_labels)
            print(f"Loaded {len(self.labels)} samples with {len(set(self.labels))} unique classes")
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
        
        train_val_emb, test_emb, train_val_labels, test_labels = train_test_split(
            self.embeddings, 
            self.labels,
            test_size=test_size,
            random_state=random_state,
            stratify=self.labels  # Maintain class distribution
        )
        
        if val_size > 0:
            # Adjust validation size relative to train+val size
            relative_val_size = val_size / (1 - test_size)
            
            train_emb, val_emb, train_labels, val_labels = train_test_split(
                train_val_emb,
                train_val_labels,
                test_size=relative_val_size,
                random_state=random_state,
                stratify=train_val_labels  # Maintain class distribution
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
    
    def get_class_distribution(self) -> Dict[str, int]:
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique_labels, counts))

    def __len__(self) -> int:
        return len(self.embeddings)
    
    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.embeddings, self.labels