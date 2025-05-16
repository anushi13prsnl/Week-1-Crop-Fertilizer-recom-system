# <p align="center"> 🌱 Crop & Fertilizer Recommendation System using ML </p>

<p align="center"> <img src="extras/readme.gif" alt="Smart Agriculture GIF"> </p> <p align="center"> <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-v3.7+-blue.svg" alt="Python"></a> <a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/scikit--learn-1.3.0-orange" alt="Scikit-learn"></a> <a href="https://pandas.pydata.org/"><img src="https://img.shields.io/badge/pandas-2.0.3-brightgreen" alt="Pandas"></a> <a href="https://github.com/anushi13prsnl/Week-1-Crop-Fertilizer-recom-system"><img src="https://img.shields.io/badge/status-active-success.svg" alt="Status"></a> <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a> </p>

> An intelligent system that recommends optimal crops and fertilizers based on soil conditions and environmental factors using machine learning.

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Repository Structure](#-current-repository-structure)
- [Recommended Structure](#-recommended-repository-structure)
- [Development Timeline](#-development-timeline)
- [Installation & Usage](#-installation--usage)
- [Models & Performance](#-models--performance)
- [Key Insights](#-key-insights)
- [Required Changes](#-required-changes)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)

## 🔍 Project Overview

This project leverages machine learning to help farmers make data-driven decisions about crop selection and fertilizer application. By analyzing soil composition, climate conditions, and other environmental factors, our system provides personalized recommendations to optimize agricultural yield and sustainability.

## ✨ Features

- **Crop Recommendation:** Suggests the most suitable crops based on N-P-K values, temperature, humidity, pH, and rainfall
- **Fertilizer Recommendation:** Determines the optimal fertilizer type based on soil and crop characteristics
- **Interactive Notebooks:** Fully documented Jupyter notebooks detailing the entire ML pipeline
- **Production-Ready Models:** Trained models saved for direct implementation

## 📁 Current Repository Structure

```
.
├── 📂 Dataset
│ ├── Crop_Recommendation.csv             # Dataset for crop prediction
│ └── Fertilizer_Prediction.csv           # Dataset for fertilizer recommendation
│
├── 📂 extras                            # Additional resources and utilities
│
├── 📂 Week-1                            # Foundation & data exploration
│ ├── Session_I_Mentoring_Session.docx
│ └── crop_recom_algorithm.ipynb
│
├── 📂 Week-2                           # Initial model development
│ └── Crop_Prediction.ipynb
│
├── 📂 Week-3                           # Final system implementation
│ ├── Crop_Prediction.ipynb            
│ └── Fertilizer_Recommendation.ipynb  
│
├── LICENSE
└── README.md                           # Project documentation
```

## 📋 Recommended Repository Structure

```
.
├── 📂 data/
│   ├── raw/                         # Original, immutable data
│   │   ├── Crop_Recommendation.csv
│   │   └── Fertilizer_Prediction.csv
│   └── processed/                   # Cleaned and processed datasets
│
├── 📂 notebooks/
│   ├── 1.0-data-exploration.ipynb
│   ├── 2.0-crop-recommendation-model.ipynb
│   └── 3.0-fertilizer-recommendation-model.ipynb
│
├── 📂 models/                       # Trained and serialized models
│   ├── crop_model.pkl
│   ├── crop_scaler.pkl
│   ├── fertilizer_model.pkl
│   └── fertilizer_scaler.pkl
│
├── 📂 src/                          # Source code for use in this project
│   ├── __init__.py
│   ├── data/                        # Scripts to download or generate data
│   │   └── make_dataset.py
│   ├── features/                    # Scripts for feature engineering
│   │   └── build_features.py
│   └── models/                      # Scripts to train and predict
│       ├── predict_model.py
│       └── train_model.py
│
├── 📂 app/                          # Web application files (future)
│   ├── app.py
│   └── templates/
│
├── requirements.txt                 # Dependencies
├── setup.py                         # Make project pip installable
├── README.md
└── LICENSE
```

## 📅 Development Timeline

### Week 1: Foundation & Exploration
- **Machine Learning Fundamentals**
  - Introduction to supervised learning concepts
  - Classification vs. regression techniques
  - Data loading and exploration workflows
- **Initial Data Analysis**
  - Loaded datasets using pandas
  - Performed exploratory data analysis
  - Identified key features for prediction models

### Week 2: Model Development
- Created initial notebook structure for crop prediction
- Drafted preprocessing pipeline
- Explored algorithm selection criteria
- Established evaluation framework

### Week 3: Complete System Implementation
- **Crop Recommendation System**
  - Comprehensive data visualization
  - Feature engineering and preprocessing
  - Decision Tree Classifier implementation
  - Model evaluation and performance metrics
  - Prediction system deployment

- **Fertilizer Recommendation System**
  - Categorical variable encoding
  - Feature scaling and normalization
  - Model training and validation
  - Predictive function implementation
  - Model serialization for production use

## 🚀 Installation & Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/anushi13prsnl/Week-1-Crop-Fertilizer-recom-system.git
   cd Week-1-Crop-Fertilizer-recom-system
   ```

2. **Install required dependencies**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn pickle-mixin
   ```

3. **Dependencies used in this project**
   
   For Crop Recommendation:
   ```python
   # Import necessary libraries
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   import seaborn as sns
   import warnings
   warnings.filterwarnings('ignore')
   import seaborn as sns
   from sklearn.preprocessing import StandardScaler
   from sklearn.tree import DecisionTreeClassifier 
   from sklearn.metrics import accuracy_score
   import pickle
   ```

   For Fertilizer Recommendation:
   ```python
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   import seaborn as sns
   from sklearn.preprocessing import LabelEncoder, StandardScaler
   from sklearn.model_selection import train_test_split
   from sklearn.tree import DecisionTreeClassifier
   ```

4. **Run the Jupyter notebooks**
   ```bash
   jupyter notebook
   ```

5. **Using the prediction system**
   ```python
   # Example code for crop recommendation
   import pickle
   
   # Load saved model and scaler
   with open('crop_recommendation_model.pkl', 'rb') as f:
       model = pickle.load(f)
   
   with open('crop_scaler.pkl', 'rb') as f:
       scaler = pickle.load(f)
   
   # Input parameters (N, P, K, temperature, humidity, pH, rainfall)
   input_data = [[90, 40, 40, 20, 80, 7, 200]]
   
   # Scale the inputs
   scaled_input = scaler.transform(input_data)
   
   # Get prediction
   prediction = model.predict(scaled_input)
   print(f"Recommended crop: {prediction[0]}")
   ```

## 📊 Models & Performance

| Model | Task | Accuracy | Features Used |
|-------|------|----------|---------------|
| Decision Tree | Crop Recommendation | 97.5% | N, P, K, temperature, humidity, pH, rainfall |
| Decision Tree | Fertilizer Recommendation | 95.2% | Soil type, crop type, N, P, K levels |

## 💡 Key Insights

- **Crop Recommendation**:
  - Nitrogen (N), Phosphorus (P), and Potassium (K) levels are the most influential factors
  - pH value has significant impact on crop suitability
  - Different crops have distinct rainfall requirements
  
- **Fertilizer Recommendation**:
  - Soil type determines which fertilizers are most effective
  - Different crops have unique nutrient profiles
  - Over-application of certain nutrients can be counter-productive
  - Significant correlations between specific soil conditions and optimal fertilizer types

## 📝 Required Changes

Before implementing this system, please ensure you:

1. **Update file paths** in all notebooks to match your local directory structure
2. **Check dataset integrity** - verify column names match those used in code
3. **Create model output directory** for saving trained models
4. **Update model save/load paths** to use your preferred location
5. **Environment considerations** - For deployment, create a `requirements.txt` file:
   ```
   numpy==1.24.3
   pandas==2.0.3
   matplotlib==3.7.2
   seaborn==0.12.2
   scikit-learn==1.3.0
   ```

## 🔮 Future Improvements

- [ ] Deploy as a web application with interactive UI (Flask/Streamlit)
- [ ] Integrate local weather API for real-time data
- [ ] Add soil image analysis capabilities with computer vision
- [ ] Create mobile application for field use
- [ ] Build time-series forecasting for seasonal recommendations
- [ ] Add multilingual support for global accessibility

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

<p align="center">
  Made with ❤️ for sustainable agriculture
</p>
