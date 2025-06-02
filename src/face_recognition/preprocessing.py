import warnings
warnings.filterwarnings("ignore")

from deepface import DeepFace
from pathlib import Path
from typing import Union, List
from tqdm import tqdm
import keras.backend as K
import numpy as np
import pickle


models = [
    "VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace",
    "DeepID", "ArcFace", "Dlib", "SFace", "GhostFaceNet",
    "Buffalo_L",
]


def get_embeddings_from_db(model: str, dataset_path: Union[str, Path], save_path: Union[str, Path]):
    assert model in models, f"{model} is not supported. Models supported: {models}"

    DeepFace.build_model(model)
    

    dataset_path = Path(dataset_path)
    save_path = Path(save_path)

    # Make sure save folder exists, if not, create.
    save_path.mkdir(parents=True, exist_ok=True)

    embeddings = []
    labels     = []

    for person_folder in tqdm(list(dataset_path.iterdir()), desc = "Processing Dataset"):
        if not person_folder.is_dir():
            continue
        
        person_label = person_folder.name

        # Searching for all images.
        for img_path in person_folder.glob('*.*'):
            K.clear_session()
            print(img_path)
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue

            try:
                embedding = DeepFace.represent(
                    img_path=img_path,
                    model_name=model,
                    enforce_detection=False,
                    align=True,
                    detector_backend="yolov8"
                )

                # Deepface.represent may send a list of embeddings vectors, so getting only the highest confidence detection here.
                if isinstance(embedding, list) and len(embedding) > 0:
                    highest_conf_vec = embedding[0]
                    for vector in embedding:
                        if vector['face_confidence'] > highest_conf_vec['face_confidence']:
                            highest_conf_vec = vector
                
                # Some images, when the face is not fully visible, the confidence is zero. But it stills outputs a embedding.
                embeddings.append(highest_conf_vec['embedding'])
                labels.append(person_label)

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        with open(save_path / "embeddings.pkl", "ab") as f:
            pickle.dump({'embeddings': np.array(embeddings), 'labels': labels}, f)

        embeddings = []
        labels = []

    print(f"Saved embeddings ({len(embeddings)} to {save_path / 'embeddings.pkl'})")

def get_image_embeddings(images: Union[str, Path, List[Union[Path, str]]], model: str = "ArcFace"):
    assert model in models, f"{model} is not supported. Models supported: {models}"

    DeepFace.build_model(model)
    
    if isinstance(images, (str, Path)):
        images = [images]
    
    results = []
    
    for img_path in images:
        try:
            embedding = DeepFace.represent(
                    img_path=img_path,
                    model_name=model,
                    enforce_detection=False,
                    align=True,
                    detector_backend="yolov8"
                )
            
            # Handle multiple faces case (select highest confidence)
            if isinstance(embedding, list) and len(embedding) > 0:
                highest_conf_vec = embedding[0]
                for vector in embedding:
                    if vector['face_confidence'] > highest_conf_vec['face_confidence']:
                        highest_conf_vec = vector
                
                results.append(highest_conf_vec['embedding'])
            else:
                # Single face case
                results.append(embedding['embedding'])
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            results.append(None)
    
    # Return single result if input was single image
    if len(results) == 1 and not isinstance(images, list):
        return results[0]
    
    return results

#get_embeddings_from_db("ArcFace", Path("/home/fabio/Documents/Dataset_IA_2025"), Path("/home/fabio/Repos/"))