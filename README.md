# Sonar Signal Classification: Rock vs. Mine Prediction

An end-to-end Machine Learning binary classification pipeline built to distinguish between sonar signals bouncing off metal cylinders (simulated sea mines) and rocks. 

## Project Overview
- **Objective:** Classify sonar target data into Rocks (`R`) or Mines (`M`).
- **Dataset:** 208 samples with 60 numerical sonar energy features.
- **Baseline Model:** Logistic Regression (chosen to prevent overfitting on a small dataset scale).
- **Core Stack:** Python, Pandas, NumPy, Scikit-Learn.

## Key Highlights & Engineering Decisions
- **Data Integrity:** Implemented a stratified train-test split (`stratify=Y`) to precisely maintain class distribution (53% Mines, 47% Rocks) across small sample sets.
- **Model Baseline:** Achieved **83.4% accuracy on training data** and **76.2% accuracy on test data** using Logistic Regression. 
- **Production-Ready Pipeline:** Developed a modular inference block using NumPy reshaping to instantly process and predict individual incoming streaming rows of sonar readings.

## Future Improvements
1. Incorporate **Feature Scaling** (`StandardScaler`) to stabilize logistic boundaries.
2. Experiment with ensemble methods like **Random Forests** or lightweight **Multi-Layer Perceptrons (MLPs)**.
3. Shift optimization metrics from pure Accuracy to **Recall** to prioritize minimizing dangerous False Negatives (missing a mine).
