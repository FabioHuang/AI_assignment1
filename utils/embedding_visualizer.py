import pickle
import numpy as np
from pathlib import Path
from typing import Union, Optional, List
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors

def visualize_embeddings(
    pickle_path: Union[str, Path], 
    method: str = 'tsne', 
    exclude_classes: List[str] = None,
    figsize: tuple = (12, 10),
    save_path: Optional[Union[str, Path]] = None,
    random_state: int = 42
):
    """
    Visualize embeddings using dimensionality reduction techniques (PCA or t-SNE)
    
    Args:
        pickle_path: Path to the pickle file containing embeddings
        method: Dimensionality reduction method ('pca' or 'tsne')
        exclude_classes: List of class names to exclude from visualization
        figsize: Figure size (width, height)
        save_path: Path to save the visualization (if None, will display instead)
        random_state: Random seed for reproducibility
    """
    method = method.lower()
    if method not in ['pca', 'tsne']:
        raise ValueError("Method must be either 'pca' or 'tsne'")
    
    pickle_path = Path(pickle_path)
    if not pickle_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {pickle_path}")
    
    all_embeddings = []
    all_labels = []
    
    with open(pickle_path, "rb") as f:
        try:
            while True:
                data = pickle.load(f)
                embeddings = data['embeddings']
                
                # Process labels
                if 'labels' in data:
                    labels = [str(label) for label in data['labels']]
                    
                    # Filter out excluded classes
                    if exclude_classes:
                        keep_indices = [i for i, label in enumerate(labels) if label not in exclude_classes]
                        if keep_indices:
                            embeddings = embeddings[keep_indices]
                            labels = [labels[i] for i in keep_indices]
                    
                    all_embeddings.append(embeddings)
                    all_labels.extend(labels)
        except EOFError:
            pass
    
    if not all_embeddings:
        raise ValueError("No embeddings found in the pickle file")
    
    embeddings = np.vstack(all_embeddings)
    labels = np.array(all_labels)
    
    print(f"Loaded {len(labels)} samples with {len(set(labels))} unique classes")
    
    # Apply dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=2, random_state=random_state)
        reduced_data = reducer.fit_transform(embeddings)
        title = "PCA Visualization of Face Embeddings"
        explained_var = reducer.explained_variance_ratio_.sum() * 100
        title += f" (Explained variance: {explained_var:.2f}%)"
    else:  # t-SNE
        reducer = TSNE(n_components=2, random_state=random_state, perplexity=min(30, len(embeddings)-1))
        reduced_data = reducer.fit_transform(embeddings)
        title = "t-SNE Visualization of Face Embeddings"
    
    # Plot results
    plt.figure(figsize=figsize)
    
    # Get unique classes and assign colors
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    # Plot each class with a different color
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            reduced_data[mask, 0], 
            reduced_data[mask, 1],
            c=[colors[i]],
            label=label,
            alpha=0.7,
            s=50
        )
    
    plt.title(title, fontsize=14)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    return reduced_data, labels

if __name__ == "__main__":
    pickle_path = "/home/fabio/Repos/embeddings.pkl"
    visualize_embeddings(
        pickle_path, 
        method='tsne', 
        exclude_classes=["unknown"],
        save_path="/home/fabio/Repos/AI_assignment1/results/embedding_tsne.png"
    )
    visualize_embeddings(
        pickle_path, 
        method='pca', 
        exclude_classes=["unknown"],
        save_path="/home/fabio/Repos/AI_assignment1/results/embedding_pca.png"
    )
