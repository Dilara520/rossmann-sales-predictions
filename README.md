# Retail-Sales-Prediction

<p align="center">
  <img width="460" height="300" src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b2/Ro%C3%9Fmann-Markt_in_Berlin.jpg/1024px-Ro%C3%9Fmann-Markt_in_Berlin.jpg">
</p>

Rossmann operates over 3,000 drug stores in 7 European countries. Currently, Rossmann store managers are tasked with predicting their daily sales for up to six weeks in advance. Store sales are influenced by many factors, including promotions, competition, school and state holidays, seasonality, and locality. With thousands of individual managers predicting sales based on their unique circumstances, the accuracy of results can be quite varied.

My work includes various plots and graphs, visualizations, feature engineering, ensemble techniques, different ML algorithms with their respective parameter tuning, analysis, and trends. Predictions are of 6 weeks of daily sales for 1,115 stores located across Germany.

In this project, the Kaggle Rossmann challenge is being taken on. The goal is to predict the sales of a given store on a given day. Model performance is evaluated on the root mean absolute percentage error (MAPE).

My work includes:
- Data exploration and visualization
- Feature engineering
- Machine learning model development
- Hyperparameter tuning
- Business impact analysis

---

## Project Structure

- `Retail_Sales_Prediction_Capstone_Project.ipynb`: Jupyter notebook containing the full pipeline — from data loading (via DBRepo API) to modeling and evaluation.
- `Retail Sales Prediction.pdf`: Presentation summarizing visualizations and key insights from the analysis.
- `requirements.txt`: Lists all Python packages needed.
- `models/`: Not included for size reasons.
- `data/`: Not included. Datasets are accessed programmatically via **DBRepo API** using dataset identifiers.
- `sample_submission.csv`: A sample submission file. Predictions made on accessed test dataset using Random Forest Model.

---

## 1. Business Problem

Rossmann operates over 3,000 drug stores in 7 European countries. Store managers are tasked with predicting daily sales up to six weeks in advance. Store sales are influenced by promotions, competition, holidays, seasonality, and locality, making predictions challenging and variable.

---

## 2. Solution Strategy

1. **Data Description**: Statistical profiling of features.
2. **Feature Engineering**: Creating new meaningful features.
3. **Exploratory Data Analysis (EDA)**: Understanding trends and relationships.
4. **Feature Selection**: Selecting the most impactful variables.
5. **Machine Learning Modeling**: Training multiple algorithms.
6. **Hyperparameter Tuning**: Optimizing model parameters.
7. **Business Value Translation**: Turning model metrics into actionable business insights.

---

## 3. Machine Learning Model Implementation and Performance

| Model                | Training Score | Testing Score |
|----------------------|----------------|---------------|
| Linear Regression     | 0.783317        | 0.784890      |
| Lasso Regression      | 0.783298        | 0.784866      |
| Decision Tree         | 0.963351        | 0.930715      |
| KNN                   | 0.676349        | 0.650872      |
| Random Forest         | 0.993726        | 0.956172      |

---

## 4. Achievements

- **MAPE**: 5.63%
- **RMSE**: 517

These results show the model is highly accurate for forecasting sales. Insights from EDA and feature importance analyses provide valuable tools for budgeting and inventory decisions.

---

## How to Set Up

1. Clone the repository:

```bash
git clone https://github.com/dilara520/rossmann-sales-prediction.git
cd rossmann-sales-prediction
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Install DBRepo:
```bash
pip install dbrepo
```
4. Authenticate if needed (API key or login).
5. Download datasets using the provided identifiers in the notebook.

## How to Run the Notebook

1. Start Jupyter Lab or Notebook:
   ```bash
   jupyter lab
   ```
2. Open the provided notebook:
   ```bash
   Retail_Sales_Prediction_Capstone_Project.ipynb
   ```
3. Follow the notebook steps
- Load data via DBRepo API.
-  Run the EDA, feature engineering, model training, and evaluation cells.

---

## Dependencies

- dbrepo==1.7.3
- joblib==1.4.2
- matplotlib==3.10.1
- numpy==2.2.5
- pandas==2.2.3
- python-dotenv==1.1.0
- scikit_learn==1.6.1
- seaborn==0.13.2
- statsmodels==0.14.4
