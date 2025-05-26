from pathlib import Path
from typing import Union, Optional
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import random
import warnings


def view_dataset(dataset_path: Union[Path, str],
                 samples_per_class: int,
                 save_path: Optional[Union[Path, str]] = None):
    '''
    Generate an image with dataset samples given the sample size per class,
    and if given, save the image in the save_path.

    Args:
        - dataset_path:         Path or string to dataset directory.
        - samples_per_class:    Number of samples per class shown.
        - save_path:            Path or string to save directory.
    '''

    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise ValueError(f"Dataset path not found {dataset_path}")
    
    cls_imgs = []
    cls_labels = []

    # Iterating the folder dataset and selecting random samples for each class
    for cls_dir in dataset_path.iterdir():
        if cls_dir.is_dir():
            cls_labels.append(cls_dir.name)
            _imgs = []

            img_files = list(cls_dir.iterdir())
            random.shuffle(img_files)
            
            for img_file in img_files:
                if len(_imgs) >= samples_per_class:
                    break;
                if (img_file.name).lower().endswith(('.png', '.jpg', '.jpeg')):
                    _imgs.append(img_file)
                else:
                    warnings.warn(f"Not an image file found at {img_file}")
            cls_imgs.append(_imgs)

    n_classes = len(cls_labels)
    n_cols = samples_per_class
    n_rows = n_classes
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 20))
    fig.suptitle("Dataset Samples", fontsize=16)

    for i, (img_list, cls_label) in enumerate(zip(cls_imgs, cls_labels)):
        for j, image_file in enumerate(img_list):
            img = (Image.open(image_file)).resize((224, 224), Image.LANCZOS)
            if n_rows > 1:
                ax = axes[i, j]
            else:
                ax = axes[j]
            ax.imshow(np.array(img))
            ax.set_title(f"{cls_label}", fontsize=8)
            ax.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the title
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    # Update this path to your dataset location
    dataset_path = Path("/home/fabio/Downloads/fotos")
    view_dataset(dataset_path, samples_per_class=5, save_path="../dataset_samples.png")