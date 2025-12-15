# COVID-19 Disease Analysis - Adapted from Heart Disease Analysis

## 📋 Overview

This project adapts the heart disease prediction analysis to work with COVID-19 patient data. The `covid_disease_analysis.ipynb` notebook is based on `heart_disease_analysis.ipynb` but has been specifically modified to handle COVID-19 dataset characteristics.

## 🎯 Purpose

**Goal**: Build machine learning models to predict COVID-19 diagnosis using patient medical data, symptoms, and comorbidities.

**Dataset**: `data/Covid_Data.csv` - Over 1 million patient records from Mexico's COVID-19 surveillance system

## 📊 Dataset Information

### Source Data
- **Location**: `data/Covid_Data.csv`
- **Size**: 1,048,576 rows × 21 columns
- **Data Dictionary**: See `data/dataMeaning.txt` for detailed column descriptions

### Key Features

#### Demographics
- `AGE` - Patient age (years)

#### Symptoms & Treatment
- `PNEUMONIA` - Whether patient developed pneumonia (1=Yes, 2=No)
- `INTUBED` - Whether patient was intubated (1=Yes, 2=No)
- `PATIENT_TYPE` - 1=Hospitalized, 2=Returned home (outpatient)
- `ICU` - ICU admission (1=Yes, 2=No)

#### Comorbidities (Pre-existing Conditions)
- `DIABETES` - Diabetes (1=Yes, 2=No)
- `COPD` - Chronic Obstructive Pulmonary Disease (1=Yes, 2=No)
- `ASTHMA` - Asthma (1=Yes, 2=No)
- `INMSUPR` - Immunosuppressed (1=Yes, 2=No)
- `HIPERTENSION` - Hypertension/High blood pressure (1=Yes, 2=No)
- `OTHER_DISEASE` - Other chronic diseases (1=Yes, 2=No)
- `CARDIOVASCULAR` - Cardiovascular disease (1=Yes, 2=No)
- `OBESITY` - Obesity (1=Yes, 2=No)
- `RENAL_CHRONIC` - Chronic kidney disease (1=Yes, 2=No)
- `TOBACCO` - Tobacco use (1=Yes, 2=No)

#### Special Population
- `PREGNANT` - Pregnancy status (1=Yes, 2=No)

#### Target Variable (Created During Preprocessing)
- `covid` - COVID-19 diagnosis (1=Positive, 0=Negative)
  - Derived from `CLASIFFICATION_FINAL` where 1-3 = Positive, 4-7 = Negative

### Excluded Columns
The following columns are removed during preprocessing:
- `CLASIFFICATION_FINAL` - (used to create target, then removed to prevent data leakage)
- `DATE_DIED` - (data leakage - occurs after diagnosis)
- `MEDICAL_UNIT` - (administrative, not predictive)
- `USMER` - (administrative, not predictive)
- `SEX` - (removed to avoid demographic bias)

## 🔄 Key Adaptations from Heart Disease Analysis

### 1. Data Loading
```python
# Original (Heart Disease)
df = pd.read_csv('dataset/heart.csv')

# Adapted (COVID-19)
df = pd.read_csv('data/Covid_Data.csv')
```

### 2. Data Preprocessing (NEW - COVID-Specific)
COVID-19 data uses special encoding that requires transformation:

#### Binary Feature Encoding
- **Original**: 1 = Yes, 2 = No, 97/98/99 = Missing/Unknown
- **Transformed**: 1 = Yes, 0 = No, Missing → 0 (conservative approach)

#### Target Variable Creation
```python
# Create binary target from CLASIFFICATION_FINAL
df['covid'] = (df['CLASIFFICATION_FINAL'] <= 3).astype(int)
# 1-3 → COVID Positive (1)
# 4-7 → COVID Negative (0)
```

### 3. Feature Set
- **Heart Disease**: 13 features (cholesterol, blood pressure, chest pain type, etc.)
- **COVID-19**: 14+ features (age, comorbidities, symptoms, treatment indicators)

### 4. Documentation
All markdown cells updated to reflect:
- COVID-19 terminology instead of heart disease
- COVID-specific features and comorbidities
- Mexican healthcare system context
- Larger dataset size considerations

## 🚀 How to Use

### Prerequisites
```bash
# Required Python libraries
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Running the Analysis

1. **Open the notebook**:
   ```bash
   jupyter notebook covid_disease_analysis.ipynb
   ```

2. **Execute all cells**:
   - Click "Run All" in Jupyter
   - Or execute cells sequentially from top to bottom
   - **Note**: Processing time is 20-30 minutes due to large dataset (1M+ rows)

3. **Review outputs**:
   - Data exploration and visualization
   - Model training results (6 different algorithms)
   - Performance metrics and comparisons
   - Final model evaluation and validation

### Important Notes
- ⏱️ **Processing Time**: Expect 20-30 minutes for full execution
- 💾 **Memory Usage**: Large dataset may require 4GB+ RAM
- 📊 **Class Imbalance**: COVID+ and COVID- may not be balanced
- 🔍 **Train/Test Split**: 90/10 due to large dataset size

## 📈 Analysis Workflow

### Phase 1: Data Preparation
1. Import libraries
2. Load COVID-19 dataset
3. **COVID-specific preprocessing** (encoding conversion, target creation)
4. Initial exploration
5. Data cleaning and missing value handling

### Phase 2: Exploratory Data Analysis
6. Target distribution (COVID+ vs COVID-)
7. Correlation analysis with diagnosis
8. Feature distributions (age, comorbidities)
9. Outlier detection

### Phase 3: Data Preprocessing
10. Feature-target split
11. Train-test split (90/10)
12. Feature scaling (StandardScaler)

### Phase 4: Model Building
13. Train 6 models:
    - Logistic Regression
    - Random Forest
    - Support Vector Machine (SVM)
    - K-Nearest Neighbors (KNN)
    - Decision Tree
    - Gradient Boosting
14. Model comparison
15. Best model selection

### Phase 5: Model Improvement
16. Hyperparameter tuning (GridSearchCV)
17. Feature engineering
18. Ensemble methods
19. Learning curves analysis
20. Final comparison of all approaches

### Phase 6: Testing & Validation
21. K-Fold cross-validation
22. Final test set evaluation
23. ROC curve and AUC score
24. Confusion matrix
25. Performance summary

## 📊 Expected Results

After running the complete analysis, you should see:

✅ **Data Preprocessing**
- Binary encoding converted (1/2 → 1/0)
- Target variable created from CLASIFFICATION_FINAL
- Missing values handled appropriately

✅ **Model Training**
- 6 different models trained and compared
- Accuracy, precision, recall, F1-score for each model
- Best performing model identified

✅ **Model Evaluation**
- Cross-validation scores
- Learning curves (training vs validation)
- ROC curve with AUC score
- Confusion matrix showing true/false positives/negatives

✅ **Feature Importance**
- Key risk factors identified (e.g., age, pneumonia, ICU, comorbidities)
- Correlation with COVID-19 diagnosis

## 🔍 Key Differences: Heart Disease vs COVID-19

| Aspect | Heart Disease | COVID-19 |
|--------|---------------|----------|
| **Dataset Size** | ~300 rows | 1,048,576 rows |
| **Features** | 13 (lab values, ECG) | 14+ (comorbidities, symptoms) |
| **Target** | Binary (disease/no disease) | Binary (positive/negative) |
| **Preprocessing** | Minimal | **Extensive** (encoding conversion) |
| **Data Source** | UCI ML Repository | Mexican Health Surveillance |
| **Train/Test Split** | 80/20 | 90/10 (due to size) |
| **Processing Time** | 5-10 minutes | 20-30 minutes |

## 📝 Files in This Project

```
Ai-Project/
├── covid_disease_analysis.ipynb     # Main analysis notebook (COVID-19)
├── heart_disease_analysis.ipynb     # Original analysis notebook (Heart Disease)
├── data/
│   ├── Covid_Data.csv              # COVID-19 patient data (1M+ rows)
│   └── dataMeaning.txt             # Data dictionary for COVID dataset
├── COVID_ANALYSIS_README.md        # This file
└── code/
    └── covid19ML.ipynb             # Alternative COVID analysis (if exists)
```

## 🎓 Educational Value

This adaptation demonstrates:
1. **Data Transformation**: Handling different encoding schemes
2. **Feature Engineering**: Creating binary features from categorical data
3. **Target Variable Creation**: Deriving labels from raw classification codes
4. **Data Leakage Prevention**: Removing post-diagnosis information
5. **Bias Mitigation**: Excluding demographic factors
6. **Scalability**: Working with large datasets (1M+ rows)
7. **Transfer Learning**: Adapting analysis pipelines across domains

## 🤝 Contributing

This notebook was adapted from the heart disease analysis to demonstrate how ML pipelines can be transferred across different medical prediction tasks while respecting domain-specific requirements.

## 📚 References

- **Data Source**: Mexican COVID-19 Surveillance System
- **Data Dictionary**: See `data/dataMeaning.txt`
- **Original Analysis**: `heart_disease_analysis.ipynb`

## ⚠️ Important Notes

1. **Data Encoding**: COVID data uses 1/2 encoding which is automatically converted to 0/1
2. **Missing Values**: Codes 97, 98, 99 indicate missing/unknown and are handled conservatively
3. **Ethical Considerations**: SEX variable removed to prevent demographic bias in predictions
4. **Data Leakage**: DATE_DIED and CLASIFFICATION_FINAL removed after target creation
5. **Large Dataset**: Be patient during execution - 1M+ rows take time to process

## 📞 Support

For questions about:
- **COVID Data**: See `data/dataMeaning.txt`
- **Analysis Methodology**: Review markdown cells in notebook
- **Preprocessing Logic**: Check Cell 6 in the notebook

---

**Last Updated**: December 2024  
**Based On**: heart_disease_analysis.ipynb  
**Adapted For**: COVID-19 patient data analysis
