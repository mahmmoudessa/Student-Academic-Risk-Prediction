# Student Risk Prediction System

A machine learning-based web application that predicts student academic risk levels using various demographic and academic features. The system helps identify students who may need additional support to succeed academically.

## 🎯 Project Overview

This project implements a comprehensive student risk assessment system using machine learning algorithms. It analyzes multiple factors including academic performance, demographic information, and behavioral patterns to classify students into risk categories.

## ✨ Features

- **Risk Prediction**: Classifies students into High Risk, Medium Risk, or Low Risk categories
- **Web Interface**: User-friendly Flask web application for easy data input and prediction
- **API Endpoint**: RESTful API for programmatic access to predictions
- **Model Information**: Detailed view of model performance and feature importance
- **Data Processing**: Complete pipeline from raw data to trained models

## 📁 Project Structure

```
student_project_enhanced/
├── data/                          # Data files and processed datasets
│   ├── feature_info.pkl          # Feature information and metadata
│   └── processed_student_data.csv # Processed student dataset
├── flask_app/                     # Flask web application
│   ├── app.py                    # Main Flask application
│   ├── static/                   # Static files (CSS, JS)
│   │   └── css/
│   │       └── style.css
│   └── templates/                # HTML templates
│       ├── base.html
│       ├── index.html           # Main prediction form
│       ├── result.html          # Prediction results
│       ├── model_info.html      # Model information
│       ├── about.html           # About page
│       └── error.html           # Error page
├── models/                        # Trained ML models
│   ├── student_model.pkl         # Main prediction model
│   ├── label_encoder.pkl         # Label encoder for categories
│   └── model_metadata.pkl       # Model metadata and performance
├── notebooks/                     # Jupyter notebooks for analysis
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_evaluation.ipynb
├── requirements.txt               # Python dependencies
├── Student_Risk_ML_REP.pdf       # Project report
└── Peresentation.pptx            # Project presentation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd student_project_enhanced
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask application**
   ```bash
   cd flask_app
   python app.py
   ```

4. **Access the application**
   Open your browser and navigate to `http://127.0.0.1:5000`

## 💻 Usage

### Web Interface

1. Navigate to the main page
2. Fill in the student information form with:
   - Academic grades (G1, G2)
   - Study time and attendance
   - Family information
   - Personal characteristics
3. Click "Predict Risk Level"
4. View the prediction results with confidence scores

### API Usage

Send a POST request to `/api/predict` with JSON data:

```python
import requests

data = {
    "G1": 15,
    "G2": 14,
    "failures": 0,
    "studytime": 3,
    "absences": 2,
    "age": 16,
    # ... other features
}

response = requests.post('http://127.0.0.1:5000/api/predict', json=data)
result = response.json()
print(result)
```

## 🧠 Model Information

- **Algorithm**: Random Forest Classifier
- **Accuracy**: 94.7%
- **Precision**: 91.2%
- **Recall**: 99%
- **F1-Score**: 94.9%

### Key Features Used

**Academic Features:**
- Previous grades (G1, G2)
- Number of failures
- Study time
- Absences

**Demographic Features:**
- Age, gender, address type
- Family size and parents' status
- Parents' education and jobs

**Behavioral Features:**
- Free time activities
- Alcohol consumption
- Romantic relationships
- Internet access

## 📊 Data Processing Pipeline

1. **Data Preprocessing** (`01_data_preprocessing.ipynb`)
   - Data cleaning and validation
   - Feature engineering
   - Handling missing values

2. **Model Development** (`02_modeling.ipynb`)
   - Feature selection
   - Model training and hyperparameter tuning
   - Cross-validation

3. **Model Evaluation** (`03_evaluation.ipynb`)
   - Performance metrics
   - Feature importance analysis
   - Model interpretation

## 🛠️ Technical Details

### Dependencies

- **Flask**: Web framework
- **scikit-learn**: Machine learning algorithms
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **pickle**: Model serialization
- **tensorflow**: Deep learning (optional)

### Model Architecture

The system uses a Random Forest classifier with:
- 100 estimators
- Balanced class weights
- Feature importance ranking
- Fallback model for robustness

## 📈 Performance Metrics

| Metric | Score |
|--------|-------|
| Accuracy | 94.7% |
| Precision (avg) | 91.2% |
| Recall (avg) | 99% |
| F1-Score (avg) | 94.9% |

## 🔧 Configuration

The application includes several configuration options:
- Model fallback mechanisms
- Error handling and logging
- API response formatting
- Web interface customization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📋 Future Enhancements

- [ ] Real-time model retraining
- [ ] Additional visualization features
- [ ] Mobile-responsive design improvements
- [ ] Batch prediction capabilities
- [ ] Integration with school information systems

## 📝 License

This project is developed for academic purposes as part of the MAIM program.

## 👥 Authors

- Student Project - MAIM Program By Mahmmoud A.Essa
- University Project


---
**Note**: This system is designed for educational and research purposes. Predictions should be used as supplementary information alongside professional academic counseling.
