# 🌾 Crop & Fertilizer Recommendation System 

Welcome! 👋  
This repository documents our **machine learning journey** in building a Crop & Fertilizer Recommendation System. It now includes progress from **Week 1**, **Week 2**, and **Week 3** 🚀

---

## 🗂️ Folder Structure

```
├── Dataset
│   ├── Crop_Recommendation.csv
│   └── Fertilizer Prediction.csv
├── Week-1
│   ├── Session_I_Mentoring_Session.docx
│   ├── crop_recom_algorithm.ipynb
├── Week-2
│   └── Crop_Prediction.ipynb
├── Week-3
│   ├── Crop_Prediction.ipynb
│   └── Fertilizer_Recommendation.ipynb
└── README.md
```

---

## 📚 Week 1  
**Mentoring Session 1: ML Basics & Data Loading**

### ✅ Quick Summary:

#### 🧠 Basic Theory:
- Introduced **Machine Learning** with real-world examples.
- Explained **Supervised Learning**, **Labelled Data**, and the difference between:
  - **Classification** → Predicting categories
  - **Regression** → Predicting numeric values

#### 💻 Practical:
- Used **Google Colab** to upload and load data.
- Imported libraries like:
  ```python
  import numpy as np  
  import pandas as pd  
  import matplotlib.pyplot as plt  
  import seaborn as sns
  ```
- Loaded dataset using `pd.read_csv()`
- Explored data using `df.head()` and `df.info()`

⚙️ _Next: Data Processing..._ 🔄

---

## 📚 Week 2  
**Crop Prediction – Initial Code Drafting**

### ✅ Quick Summary:

- Created `Crop_Prediction.ipynb` notebook
- Added initial code snippets relevant to crop prediction logic.
- Served as a **Week 2 submission placeholder**.
- Work-in-progress; further model building and evaluation to follow in upcoming weeks.

---

## 📚 Week 3  
**Final Crop & Fertilizer Recommendation System**

### ✅ Quick Summary:

This week brings together all previous work into a complete, functional system for both crop and fertilizer recommendations. The folder contains two main files:

- **Crop_Prediction.ipynb**  
  - Performed detailed data exploration and visualization on the crop recommendation dataset.
  - Encoded crop labels numerically for model training.
  - Split the data into training and test sets, and applied feature scaling.
  - Trained a Decision Tree Classifier to predict the best crop based on environmental and soil features.
  - Evaluated model performance on both train and test sets.
  - Built a predictive system function to recommend crops for new input conditions.
  - Saved and demonstrated loading the trained model and scaler for future predictions.

- **Fertilizer_Recommendation.ipynb**  
  - Loaded and explored the fertilizer dataset, checking for missing/duplicate values and visualizing feature distributions.
  - Encoded categorical variables (soil type, crop type, fertilizer name) for model compatibility.
  - Split the data, scaled features, and trained a Decision Tree Classifier to recommend the best fertilizer.
  - Evaluated the model's accuracy and built a predictive function for fertilizer recommendation based on input parameters.
  - Saved the trained model and scaler, and included code to reload them for new predictions.

---

🌟 Happy Learning & Coding! 🌱