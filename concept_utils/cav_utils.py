import os
import torch
from collections import defaultdict
import numpy as np
from sklearn.svm import SVC
from tqdm import tqdm
from PIL import Image

class ListDataset:
    def __init__(self, images, preprocess=None):
        self.images = images
        self.preprocess = preprocess

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.preprocess:
            image = self.preprocess(image)
        return image

class EasyDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class ConceptBank:
    def __init__(self, concept_dict, device, output_dir):
        all_vectors, concept_names, all_intercepts = [], [], []
        all_margin_info = defaultdict(list)

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        for k, (tensor, _, _, intercept, margin_info) in concept_dict.items():
            all_vectors.append(tensor)
            concept_names.append(k)
            all_intercepts.append(np.array(intercept).reshape(1, 1))
            for key, value in margin_info.items():
                if key != "train_margins":
                    all_margin_info[key].append(np.array(value).reshape(1, 1))

        for key, val_list in all_margin_info.items():
            margin_tensor = torch.tensor(np.concatenate(val_list, axis=0), requires_grad=False).float().to(device)
            all_margin_info[key] = margin_tensor

        self.concept_info = EasyDict()
        self.concept_info.margin_info = EasyDict(dict(all_margin_info))
        self.concept_info.vectors = torch.tensor(np.concatenate(all_vectors, axis=0), requires_grad=False).float().to(device)
        self.concept_info.norms = torch.norm(self.concept_info.vectors, p=2, dim=1, keepdim=True).detach()
        self.concept_info.intercepts = torch.tensor(np.concatenate(all_intercepts, axis=0), requires_grad=False).float().to(device)
        self.concept_info.concept_names = concept_names
        print("Concept Bank is initialized.")

    def __getattr__(self, item):
        return self.concept_info[item]

    def save_superpixel(self, concept_name, segment_image, segment_number):
        """Save the superpixel image to the specified concept folder."""
        concept_folder = os.path.join(self.output_dir, f'concept_{concept_name}')
        os.makedirs(concept_folder, exist_ok=True)
        file_path = os.path.join(concept_folder, f'superpixel_{segment_number}.png')
        segment_image.save(file_path)  # Save using PIL's Image save method

@torch.no_grad()
def get_embeddings(loader, model, device="cuda"):
    activations = None
    for image in tqdm(loader):
        image = image.to(device)
        batch_act = model(image).squeeze().detach().cpu().numpy()
        if activations is None:
            activations = batch_act
        else:
            activations = np.concatenate([activations, batch_act], axis=0)
    return activations

def get_cavs(X_train, y_train, X_val, y_val, C):
    svm = SVC(C=C, kernel="linear")
    svm.fit(X_train, y_train)
    train_acc = svm.score(X_train, y_train)
    test_acc = svm.score(X_val, y_val)

    train_margin = ((np.dot(svm.coef_, X_train.T) + svm.intercept_) / np.linalg.norm(svm.coef_)).T
    margin_info = {
        "max": np.max(train_margin),
        "min": np.min(train_margin),
        "pos_mean": np.nanmean(train_margin[train_margin > 0]),
        "pos_std": np.nanstd(train_margin[train_margin > 0]),
        "neg_mean": np.nanmean(train_margin[train_margin < 0]),
        "neg_std": np.nanstd(train_margin[train_margin < 0]),
        "q_90": np.quantile(train_margin, 0.9),
        "q_10": np.quantile(train_margin, 0.1),
        "pos_count": y_train.sum(),
        "neg_count": (1 - y_train).sum(),
    }
    concept_info = (svm.coef_, train_acc, test_acc, svm.intercept_, margin_info)
    return concept_info

def learn_concept_bank(pos_loader, neg_loader, backbone, n_samples, C, output_dir, device="cuda"):
    print("Extracting Embeddings:")
    pos_act = get_embeddings(pos_loader, backbone, device=device)
    neg_act = get_embeddings(neg_loader, backbone, device=device)

    X_train = np.concatenate([pos_act[:n_samples], neg_act[:n_samples]], axis=0)
    X_val = np.concatenate([pos_act[n_samples:], neg_act[n_samples:]], axis=0)
    y_train = np.concatenate([np.ones(pos_act[:n_samples].shape[0]), np.zeros(neg_act[:n_samples].shape[0])], axis=0)
    y_val = np.concatenate([np.ones(pos_act[n_samples:].shape[0]), np.zeros(neg_act[n_samples:].shape[0])], axis=0)

    concept_info = {}
    for c in C:
        concept_info[c] = get_cavs(X_train, y_train, X_val, y_val, c)

    # Initialize the concept bank
    concept_bank = ConceptBank(concept_info, device, output_dir)
    return concept_info
