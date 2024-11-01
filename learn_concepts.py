import os
import argparse
import numpy as np
import torch
from pandas.io import pickle
from sklearn.cluster import DBSCAN
from sklearn.svm import SVC
from skimage.segmentation import slic
from skimage import io
from torchvision import models, transforms
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="SLIC Superpixel Segmentation and DBSCAN Clustering")
    parser.add_argument("--image", required=True, default="image", type=str, help="Path to the input image")
    parser.add_argument("--train-dir", required=True, default="image/label_dataset", type=str, help="Directory containing training images")
    parser.add_argument("--output-dir", required=True, default="Concept_base", type=str, help="Directory to save output results")
    parser.add_argument("--eps", default=0.5, type=float, help="DBSCAN eps parameter")
    parser.add_argument("--min-samples", default=5, type=int, help="DBSCAN min_samples parameter")
    return parser.parse_args()

def superpixel_segmentation(image_path, n_segments=120):
    image = io.imread(image_path)
    segments = slic(image, n_segments=n_segments, compactness=10, start_label=1)
    return image, segments

def extract_features(image, segments, model_path):
    model = torch.load(model_path)
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    features = []
    for segment in np.unique(segments):
        mask = (segments == segment)
        segment_image = np.zeros_like(image)
        if np.sum(mask) > 0:
            segment_image[mask] = image[mask]
            segment_tensor = transform(segment_image).unsqueeze(0)
            with torch.no_grad():
                feature = model(segment_tensor).flatten()
            features.append(feature.numpy())

            # Save the segment as an image
            save_superpixel(segment_image, segment)  # Save the segment image

    return np.vstack(features) if features else np.empty((0, model.fc.in_features))

def save_superpixel(segment_image, segment_number, concept_name, output_dir):
    """Save the superpixel image to the specified directory for a given concept."""
    concept_folder = os.path.join(output_dir, concept_name)
    os.makedirs(concept_folder, exist_ok=True)
    file_path = os.path.join(concept_folder, f'superpixel_{segment_number}.png')
    io.imsave(file_path, segment_image)

def cluster_features(features, eps=0.5, min_samples=5):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(features)
    return db.labels_

def learn_concept_vectors(features, labels):
    concept_vectors = {}
    for label in np.unique(labels):
        if label == -1:
            continue
        concept_features = features[labels == label]
        svm = SVC(kernel="linear").fit(concept_features, np.ones(len(concept_features)))
        concept_vectors[label] = {
            "vector": svm.coef_.flatten(),
            "intercept": svm.intercept_[0],
            "margin": np.abs(svm.decision_function(concept_features)).mean()
        }
    return concept_vectors

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    input_image, segments_image = superpixel_segmentation(args.image)

    features_image = extract_features(input_image, segments_image, model_path='meta_dataset/models/dog(snow).pth')

    labels_image = cluster_features(features_image, eps=args.eps, min_samples=args.min_samples)

    concept_vectors_image = learn_concept_vectors(features_image, labels_image)
    np.save(os.path.join(args.output_dir, "input_image_concept_vectors.npy"), concept_vectors_image)

    concept_bank = defaultdict(list)
    train_images = [os.path.join(args.train_dir, img) for img in os.listdir(args.train_dir)]

    for train_image_path in train_images:
        train_image, segments_train = superpixel_segmentation(train_image_path)

        features_train = extract_features(train_image, segments_train, model_path='meta_dataset/models/dog(snow).pth')

        labels_train = cluster_features(features_train, eps=args.eps, min_samples=args.min_samples)

        concept_vectors_train = learn_concept_vectors(features_train, labels_train)

        for label, vector_info in concept_vectors_train.items():
            concept_bank[label].append(vector_info)

            # Save superpixels for each concept
            for segment in np.unique(segments_train[labels_train == label]):
                mask = (segments_train == segment)
                segment_image = np.zeros_like(train_image)
                if np.sum(mask) > 0:
                    segment_image[mask] = train_image[mask]
                    save_superpixel(segment_image, segment, f'concept_{label}', args.output_dir)  # Save under concept folder

    with open(os.path.join(args.output_dir, "concept_bank.pkl"), "wb") as f:
        pickle.dump(concept_bank, f)
    print("Concept bank has been saved.")

if __name__ == "__main__":
    main()
