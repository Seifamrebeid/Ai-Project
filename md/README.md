# COVID-19 Prediction Model

## Quick Start

### Running the Project
1. Open `code/covid19ML.ipynb` in Jupyter Notebook or VS Code
2. Run all cells sequentially (top to bottom)
3. The model will be saved to `model/` directory upon completion

### Expected Runtime
- Sample (400K rows): **4-5 minutes**
- Full dataset (700K rows): **10-15 minutes**

---

## Project Structure

```
Ai Project/
├── code/
│   └── covid19ML.ipynb       # Main notebook - run this
├── data/
│   ├── Covid_Data.csv        # Dataset
│   └── dataMeaning.txt       # Feature descriptions
├── model/
│   └── *.pkl                 # Saved trained models
├── docs/
│   └── *.pdf                 # Project documentation
├── documentation.md          # Detailed methodology & design decisions
└── README.md                 # This file
```

---

## What This Project Does

Implements a complete machine learning pipeline for predicting COVID-19 hospitalization:

1. **Data Cleaning**: Removes leakage, bias, and invalid records
2. **EDA**: Visualizes data distributions and correlations
3. **Model Training**: Trains 7 different ML algorithms
4. **Validation**: Comprehensive testing with cross-validation, ROC curves, confusion matrices
5. **Deployment**: Saves best model for production use

---

## Key Features

✅ **7 ML Models Compared**: Logistic Regression, Decision Tree, Random Forest, KNN, Gradient Boosting (standard & tuned), Voting Ensemble  
✅ **Regularization Applied**: Prevents overfitting with max_depth, min_samples constraints  
✅ **Feature Scaling**: StandardScaler for distance-based algorithms  
✅ **Hyperparameter Tuning**: Fast RandomizedSearchCV optimization  
✅ **Advanced Validation**: Learning curves, 10-fold CV, ROC-AUC analysis  
✅ **Medical Context**: Focus on minimizing False Negatives (missed COVID cases)

---

## Documentation

- **`code/covid19ML.ipynb`**: Complete implementation with code and concise explanations
- **`documentation.md`**: Detailed methodology, design decisions, and best practices

---

## Performance

Typical results (varies by data quality):
- **Accuracy**: 60-75%
- **ROC-AUC**: 0.65-0.80
- **Overfitting Gap**: < 5% (good generalization)

---

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
```

---

## Usage After Training

Load the saved model:
```python
import joblib
model = joblib.load('model/covid_random_forest_complete.pkl')

# Make predictions
predictions = model.predict(X_new)
```

---

## Notes

- The notebook processes 400K rows by default for speed
- Adjust `SAMPLE_SIZE` variable for different dataset sizes
- See [documentation.md](documentation.md) for detailed explanations
- All code is reproducible with `random_state=42`

---

**Last Updated**: December 2025
