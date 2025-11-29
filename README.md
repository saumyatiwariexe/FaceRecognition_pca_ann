# **Face Recognition using PCA + ANN**

A complete implementation of a classical **Face Recognition System** using **Principal Component Analysis (PCA)** for Eigenface extraction and **Artificial Neural Networks (ANN)** for classification. This project demonstrates the full workflow of image preprocessing, dimensionality reduction, model training, accuracy evaluation, and imposter detection.

---

## **📌 Features**

* PCA-based Eigenface generation
* ANN classification using Scikit-learn
* Recognition accuracy evaluation for multiple K values
* Imposter detection using probability thresholding
* Mean face & feature vector generation
* Automated outputs: accuracy plot, summary file, saved model
* Fully reproducible machine learning pipeline

---

## **📁 Project Structure**

```
project/
│── face_pca_ann.py
│── accuracy_vs_k.png
│── results_summary.txt
│── ann_model_k50.pkl
│── README.md
│
└── dataset/
    ├── s1/
    ├── s2/
    ├── …
    └── imposters/
```

---

## **🛠 Tools & Technologies**

* Python
* NumPy, SciPy
* OpenCV
* Scikit-learn
* Matplotlib
* Joblib

---

## **📥 Dataset**

This project uses the ORL-style face dataset:

GitHub Dataset:
[https://github.com/robaita/introduction_to_machine_learning/blob/main/dataset.zip](https://github.com/robaita/introduction_to_machine_learning/blob/main/dataset.zip)

Extract and place inside a folder named `dataset/`.

---

## **🚀 How to Run**

1. Install dependencies:

```
pip install numpy scipy opencv-python scikit-learn matplotlib joblib
```

2. Ensure dataset is placed inside `dataset/`.

3. Run the script:

```
python face_pca_ann.py
```

4. Outputs generated:

* `accuracy_vs_k.png`
* `results_summary.txt`
* Classification report in terminal
* Imposter detection summary
* `ann_model_k50.pkl` (saved model)

---

## **📊 Results**

* **Best Accuracy:** 52.2%
* **Best K:** 50
* Performance improves as K increases
* Partial imposter detection success

---

## **📝 Future Improvements**

* Replace ANN with CNN for higher accuracy
* Add real-time face detection & recognition
* Improve lighting robustness
* Expand dataset for better generalization

---

## **📚 References**

* Turk & Pentland – Eigenfaces Research
* OpenCV Documentation
* NumPy Documentation
* Scikit-learn Documentation

