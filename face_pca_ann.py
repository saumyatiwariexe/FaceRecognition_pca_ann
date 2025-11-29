# face_pca_ann.py
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import joblib

# -----------------------
# Configuration
# -----------------------
IMG_SIZE = (100, 100)          # resize images to this size (h, w)
DATASET_DIR = "dataset"        # dataset root folder (see README)
IMPOSTER_DIRNAME = "imposters" # folder name (inside dataset) containing imposter images
RANDOM_STATE = 42
K_LIST = [5, 10, 20, 30, 40, 50]  # try varying K values
ANN_HIDDEN = (100,)            # hidden layer sizes for MLP
ANN_MAX_ITER = 500
PROB_THRESHOLD = 0.6           # threshold for imposter detection (tune as needed)

# -----------------------
# Utilities
# -----------------------
def load_images_from_folder(root_dir, img_size=IMG_SIZE):
    """
    Loads images and labels.
    Returns:
      images: list of flattened grayscale images (each numpy vector shape (mn,))
      labels: list of class labels (strings)
      label_list: sorted unique label names (excluding imposters)
    """
    images = []
    labels = []
    class_names = []
    for name in sorted(os.listdir(root_dir)):
        path = os.path.join(root_dir, name)
        if not os.path.isdir(path):
            continue
        if name == IMPOSTER_DIRNAME:
            continue
        class_names.append(name)
    class_names = sorted(class_names)
    for label in class_names:
        folder = os.path.join(root_dir, label)
        for fname in sorted(os.listdir(folder)):
            fpath = os.path.join(folder, fname)
            if not os.path.isfile(fpath):
                continue
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (img_size[1], img_size[0]))
            vec = img.flatten().astype(np.float32)
            images.append(vec)
            labels.append(label)
    return np.array(images).T, np.array(labels), class_names  # images as (mn, p) matrix

def load_imposters(root_dir, img_size=IMG_SIZE):
    imp_path = os.path.join(root_dir, IMPOSTER_DIRNAME)
    imp_images = []
    if not os.path.isdir(imp_path):
        return np.array([]).T
    for fname in sorted(os.listdir(imp_path)):
        fpath = os.path.join(imp_path, fname)
        if not os.path.isfile(fpath):
            continue
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (img_size[1], img_size[0]))
        imp_images.append(img.flatten().astype(np.float32))
    if len(imp_images) == 0:
        return np.array([]).T
    return np.array(imp_images).T  # shape (mn, num_imposters)

# -----------------------
# PCA / Eigenfaces functions
# -----------------------
def compute_mean_face(Face_Db):
    # Face_Db shape: (mn, p)
    mean_face = np.mean(Face_Db, axis=1, keepdims=True)  # shape (mn, 1)
    return mean_face

def mean_zero_faces(Face_Db, mean_face):
    return Face_Db - mean_face  # shape (mn, p)

def compute_eigenfaces(delta, p):
    """
    delta: (mn, p) mean-zero faces
    Returns:
      eigenfaces_all: (mn, p) unnormalized eigenfaces (columns)
      eigenvalues: (p,) corresponding eigenvalues
    Implementation uses surrogate covariance trick:
      L = delta.T @ delta  (p x p)
      eigvecs_small = eig(L)
      eigenfaces = delta @ eigvecs_small
    """
    L = delta.T @ delta  # p x p
    # compute eigenvalues and eigenvectors
    eigvals, eigvecs_small = np.linalg.eigh(L)  # eigh returns ascending eigenvalues
    # sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs_small = eigvecs_small[:, idx]
    # compute eigenfaces (unnormalized)
    eigenfaces = delta @ eigvecs_small  # mn x p
    # Normalize eigenfaces
    for i in range(eigenfaces.shape[1]):
        norm = np.linalg.norm(eigenfaces[:, i])
        if norm > 1e-10:
            eigenfaces[:, i] = eigenfaces[:, i] / norm
    return eigenfaces, eigvals

def project_faces(eigenfaces_k, delta):
    # eigenfaces_k: (mn, k), delta: (mn, p)
    # projection (k, p)
    return eigenfaces_k.T @ delta

# -----------------------
# Main routine
# -----------------------
def main():
    # 1) Load dataset
    Face_Db, labels, class_names = load_images_from_folder(DATASET_DIR, IMG_SIZE)
    if Face_Db.size == 0:
        raise RuntimeError("No images found. Check DATASET_DIR and structure.")
    mn, p = Face_Db.shape
    print(f"Loaded dataset with {p} images, image vector length {mn}, classes: {len(class_names)}")

    # 2) Mean face and mean-zero
    mean_face = compute_mean_face(Face_Db)        # (mn,1)
    delta = mean_zero_faces(Face_Db, mean_face)   # (mn,p)

    # 3) Compute eigenfaces (full set)
    eigenfaces_all, eigvals = compute_eigenfaces(delta, p)  # eigenfaces_all: mn x p
    print("Computed eigenfaces. Eigenvalues (top 10):", eigvals[:10])

    # Prepare labels -> numeric
    label_to_idx = {n:i for i,n in enumerate(class_names)}
    y = np.array([label_to_idx[l] for l in labels])

    # Split indices per-image into train/test with 60/40 preserving class balance:
    # We'll split by images globally but ensure reasonable class distribution using stratify
    X_vectors = Face_Db.T  # shape (p, mn) for splitting convenience
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_vectors, y, np.arange(p), test_size=0.4, random_state=RANDOM_STATE, stratify=y
    )
    # Convert back to column format
    Face_Db_train = X_train.T  # (mn, p_train)
    Face_Db_test = X_test.T    # (mn, p_test)
    print(f"Train images: {Face_Db_train.shape[1]}, Test images: {Face_Db_test.shape[1]}")

    # Recompute mean and delta for training set (follow PDF: mean over all faces?
    # PDF suggests using entire face database for mean; but many implementations compute mean on training set only.
    # We will compute mean from training set and use it consistently for test projection.)
    mean_face_train = compute_mean_face(Face_Db_train)
    delta_train = mean_zero_faces(Face_Db_train, mean_face_train)
    # Compute eigenfaces from training data
    eigenfaces_train_all, eigvals_train = compute_eigenfaces(delta_train, delta_train.shape[1])

    acc_list = []
    k_values_used = []
    best_acc = -1
    best_k = None
    best_model = None
    results = {}

    for k in K_LIST:
        if k > eigenfaces_train_all.shape[1]:
            continue
        eigenfaces_k = eigenfaces_train_all[:, :k]  # (mn, k)
        # Signatures for training
        omega_train = project_faces(eigenfaces_k, mean_zero_faces(Face_Db_train, mean_face_train)).T  # (p_train, k)
        # Train ANN
        clf = MLPClassifier(hidden_layer_sizes=ANN_HIDDEN, random_state=RANDOM_STATE, max_iter=ANN_MAX_ITER)
        clf.fit(omega_train, y_train)
        # Prepare test signatures
        delta_test = mean_zero_faces(Face_Db_test, mean_face_train)  # center test using train mean
        omega_test = (eigenfaces_k.T @ delta_test).T  # shape (p_test, k)
        y_pred = clf.predict(omega_test)
        acc = accuracy_score(y_test, y_pred)
        acc_list.append(acc)
        k_values_used.append(k)
        results[k] = {"acc": acc, "clf": clf, "y_pred": y_pred}
        print(f"k={k:3d} -> Accuracy: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_k = k
            best_model = clf

    # Save best model and eigenfaces + mean
    model_out = {
        "model": best_model,
        "eigenfaces_k": eigenfaces_train_all[:, :best_k],
        "mean_face": mean_face_train,
        "label_to_idx": label_to_idx,
        "idx_to_label": {v:k for k,v in label_to_idx.items()},
        "k": best_k
    }
    joblib.dump(model_out, f"ann_model_k{best_k}.pkl")
    print(f"Saved best model (k={best_k}) to ann_model_k{best_k}.pkl")

    # Plot Accuracy vs K
    plt.figure(figsize=(6,4))
    plt.plot(k_values_used, acc_list, marker='o')
    plt.xlabel("K (number of eigenfaces)")
    plt.ylabel("Accuracy (on test set)")
    plt.title("Accuracy vs K")
    plt.grid(True)
    plt.savefig("accuracy_vs_k.png", dpi=200)
    print("Saved plot accuracy_vs_k.png")

    # Print best results
    print(f"Best K: {best_k} with Accuracy: {best_acc:.4f}")
    # Detailed classification report for best_k
    best_clf = results[best_k]["clf"]
    # recreate predictions for best_k to get labels
    eigenfaces_best = eigenfaces_train_all[:, :best_k]
    delta_test = mean_zero_faces(Face_Db_test, mean_face_train)
    omega_test_best = (eigenfaces_best.T @ delta_test).T
    y_pred_best = best_clf.predict(omega_test_best)
    print("Classification report (best K):")
    unique_test_labels = np.unique(y_test)

    print(classification_report(
        y_test,
        y_pred_best,
        labels=unique_test_labels,
        target_names=[class_names[i] for i in unique_test_labels]
    ))

    # -----------------------
    # Imposter detection (if imposters folder exists)
    # Strategy: for each imposter image, compute signature using train mean and eigenfaces,
    # get model.predict_proba and use probability threshold.
    # -----------------------
    impX = load_imposters(DATASET_DIR, IMG_SIZE)  # shape (mn, n_imp)
    if impX.size != 0:
        imp_delta = mean_zero_faces(impX, mean_face_train)  # (mn, n_imp)
        imp_omega = eigenfaces_best.T @ imp_delta  # (k, n_imp)
        imp_omega = imp_omega.T  # (n_imp, k)
        probs = best_clf.predict_proba(imp_omega)
        max_probs = probs.max(axis=1)
        predicted_labels = best_clf.predict(imp_omega)
        # Decide imposter if max_prob < PROB_THRESHOLD
        detected_as_imposter = max_probs < PROB_THRESHOLD
        for i in range(len(max_probs)):
            print(f"Imposter image {i}: max_prob={max_probs[i]:.3f}, predicted={model_out['idx_to_label'][predicted_labels[i]]}, imposter_detected={detected_as_imposter[i]}")
        # Summary:
        num_detected = detected_as_imposter.sum()
        print(f"Imposter detection: {num_detected}/{len(max_probs)} detected as imposters with threshold {PROB_THRESHOLD}")

    # Save accuracy results to text
    with open("results_summary.txt", "w") as f:
        f.write("K,Accuracy\n")
        for k,a in zip(k_values_used, acc_list):
            f.write(f"{k},{a:.6f}\n")
        f.write(f"\nBest K,{best_k},{best_acc:.6f}\n")
    print("Saved results_summary.txt")

if __name__ == "__main__":
    main()
