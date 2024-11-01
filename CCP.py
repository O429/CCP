import os
import argparse
import torch
import numpy as np
import seaborn as sns
from PIL import Image

from model_utils import get_model, ResNetBottom, ResNetTop
from model_utils import imagenet_resnet_transforms as preprocess
from concept_utils import conceptual_counterfactual, ConceptBank, ListDataset, get_embeddings, learn_concept_bank

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="./mit67/mitmodel.pth", type=str)
    parser.add_argument("--concept-bank", default="Concept_base/resnet18_bank.pkl", type=str)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--image-folder", default="image", type=str)
    parser.add_argument("--explanation-folder", default="Explanation", type=str)
    parser.add_argument("--pos-samples-folder", type=str, help="Folder with positive samples for each concept")
    parser.add_argument("--neg-samples-folder", type=str, help="Folder with negative samples for each concept")
    parser.add_argument("--n-samples", default=50, type=int, help="Number of positive/negative samples")
    parser.add_argument("--C", nargs="+", default=[0.001, 0.01, 0.1, 1.0], type=float,
                        help="SVM regularization parameter")
    return parser.parse_args()

def main(args):
    sns.set_context("poster")
    np.random.seed(args.seed)

    # Load the model
    model = torch.load(args.model_path, map_location=torch.device(args.device))
    model = model.to(args.device)
    model.eval()

    # Define the class mappings
    idx_to_class = {0: "bear", 1: "bird", 2: "cat", 3: "dog", 4: "elephant"}
    cls_to_idx = {v: k for k, v in idx_to_class.items()}

    # Split the model into backbone and predictor layers
    backbone, model_top = ResNetBottom(model), ResNetTop(model)

    # Prepare positive and negative datasets for concept bank creation
    pos_images = [os.path.join(args.pos_samples_folder, img) for img in os.listdir(args.pos_samples_folder)]
    neg_images = [os.path.join(args.neg_samples_folder, img) for img in os.listdir(args.neg_samples_folder)]
    pos_dataset = ListDataset(pos_images, preprocess=preprocess)
    neg_dataset = ListDataset(neg_images, preprocess=preprocess)

    pos_loader = torch.utils.data.DataLoader(pos_dataset, batch_size=32, shuffle=False, num_workers=4)
    neg_loader = torch.utils.data.DataLoader(neg_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Learn the concept bank
    concept_dict = learn_concept_bank(pos_loader, neg_loader, backbone, args.n_samples, args.C, device=args.device)
    concept_bank = ConceptBank(concept_dict, device=args.device, output_dir=args.explanation_folder)  # Save to explanation folder

    os.makedirs(args.explanation_folder, exist_ok=True)

    # Loop through each image in the folder and generate explanations
    for image_path in os.listdir(args.image_folder):
        # Read the image and label
        image = Image.open(os.path.join(args.image_folder, image_path))
        image_tensor = preprocess(image).to(args.device)

        # Assign label (for demonstration purposes, set as "dog")
        label = cls_to_idx["dog"] * torch.ones(1, dtype=torch.long).to(args.device)

        # Get the embedding for the image
        embedding = backbone(image_tensor.unsqueeze(0))

        # Get the model prediction
        pred = model_top(embedding).argmax(dim=1)

        # Only evaluate over mistakes
        if pred.item() == label.item():
            print(f"Warning: {image_path} is correctly classified, but we'll still try to increase the confidence.")

        # Detach the embedding for counterfactual explanation
        embedding = embedding.detach()

        # Run CCP to get explanation
        explanation = conceptual_counterfactual(embedding, label, concept_bank, model_top)

    # Optionally, loop through concept bank and save superpixels for each concept
    for concept_name in concept_bank.concept_names:
        for seg_num, segment_image in enumerate(explanation['selected_superpixels']['pos']):
            concept_bank.save_superpixel(concept_name, segment_image, seg_num, 'pos')
        for seg_num, segment_image in enumerate(explanation['selected_superpixels']['neg']):
            concept_bank.save_superpixel(concept_name, segment_image, seg_num, 'neg')

if __name__ == "__main__":
    args = config()
    main(args)
