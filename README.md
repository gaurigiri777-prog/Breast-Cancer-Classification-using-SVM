# 🩺 Breast Cancer Classification using Machine Learning  
A complete end-to-end Machine Learning project that predicts whether a tumor is **Benign (B)** or **Malignant (M)** using clinical features from the Breast Cancer Wisconsin Dataset.

<img width="1671" height="940" alt="Breast cancer classification with machine learning" src="https://github.com/user-attachments/assets/0ce3fc00-6a5e-4a12-85c8-a0203b6c3f46" />


---

## 🚀 Project Overview  
This project builds multiple ML models to classify breast tumors using medical diagnostic measurements such as radius, texture, smoothness, compactness, etc.

The main goals of the project are:
- 🧼 Clean & preprocess medical data  
- 🧠 Apply ML models (Logistic Regression, SVM, Random Forest, etc.)  
- 📊 Compare model performance  
- 🎯 Identify the best-performing classifier  
- 🔍 Evaluate using accuracy, confusion matrix, ROC curve & AUC  

---

## 📁 Repository Structure
```
📂 Breast-Cancer-Classification-ML
│── README.md
│── breast_cancer.ipynb
│── requirements.txt
│── 📂 dataset
│     ├── breast_cancer.csv
│── 📂 models
│     ├── scaler.pkl
│     ├── classifier.pkl
│── 📂 screenshots
│     ├── data_preview.png
│     ├── preprocessing.png
│     ├── correlation_heatmap.png
│     ├── model_training.png
│     ├── confusion_matrix.png
│     ├── roc_curve.png
│     ├── prediction_demo.png
│── 📂 src
│     ├── train.py
│     ├── predict.py
│     ├── utils.py
```

---

## 🧠 Technologies Used  
- Python  
- Pandas & NumPy  
- Scikit-Learn  
- Matplotlib / Seaborn  
- Joblib / Pickle  
- Jupyter Notebook  

---

## 🧹 Data Preprocessing  
The dataset is cleaned and prepared with these steps:

✔ Remove null values  
✔ Encode categorical columns  
✔ Normalize/standardize features  
✔ Train-test split (80/20)  
✔ Correlation analysis  

### 🔹 Sample: Feature Scaling (StandardScaler)
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

📸 **dataset preview**
<img width="1107" height="236" alt="dataset preview" src="https://github.com/user-attachments/assets/f6b04633-d793-4cd8-a99c-3e9b179a3763" />

---

## 🧬Model Performance Benchmarks  
This bar chart compares the accuracy of different SVM kernels, showing that the Best configuration achieves the highest predictive performance over the Linear and RBF models.

```
<img width="608" height="401" alt="Accuracy Comparison" src="https://github.com/user-attachments/assets/d5c6cd2e-c9ab-4c69-902e-4f4eabfd71ed" />

```


---

## 🏗️ Model Building  

The following models were trained & compared:

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 96% |
| **SVM (RBF Kernel)** | **97–98% (Best)** |
| Random Forest | 96% |
| KNN | 94% |
| Decision Tree | 92% |

### ✔ Example: SVM Model  
```python
model = SVC(kernel='rbf', C=1, gamma='scale', probability=True)
model.fit(X_train, y_train)
```

📸

---

## 📊 Model Evaluation  

### Metrics Used:
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  
- ROC Curve  
- AUC Score  

### ✔ Confusion Matrix  
```python
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
```

📸 
<img width="432" height="89" alt="Confusion matrices" src="https://github.com/user-attachments/assets/4067e1dd-8846-4870-b418-4b6467e8ef25" />

<img width="378" height="172" alt="confusion matrices explain" src="https://github.com/user-attachments/assets/1b199f34-230f-4b77-b34b-f5eb9aea76ad" />

---

### ✔ ROC Curve & AUC  
```python
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)
```


---

## 🧪 Making Predictions  
Predict whether a tumor is malignant or benign from feature input:

```python
sample = scaler.transform([[14.5, 19.4, 96.7, 657, ...]])
prediction = model.predict(sample)
print("Malignant" if prediction == 1 else "Benign")
```

📸

<img width="475" height="413" alt="Final Result" src="https://github.com/user-attachments/assets/d14541e2-17bd-4c6c-8351-08b49d13ef4e" />

<img width="762" height="262" alt="Final Result 2" src="https://github.com/user-attachments/assets/d33eaccd-b279-417c-b971-de8b6d28eed5" />

---

## 💾 Saving the Model  
The trained model and scaler are saved for deployment:

```python
joblib.dump(model, "models/classifier.pkl")
joblib.dump(scaler, "models/scaler.pkl")
```

---

## ▶️ How to Run the Project  

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/your-username/Breast-Cancer-Classification-ML.git
cd Breast-Cancer-Classification-ML
```

### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3️⃣ Train the Model**
```bash
python src/train.py
```

### **4️⃣ Make Predictions**
```bash
python src/predict.py
```

---

## ⭐ Future Improvements  
- Add Hyperparameter Tuning (GridSearchCV)  
- Build a Streamlit Web App  
- Deploy model using Flask / FastAPI  
- Add SHAP explainability visualizations  

---

## 🤝 Contributing  
Pull requests are welcome!  
If this project helps you, consider giving it a ⭐ on GitHub.  

---

## 📜 License  
MIT License
