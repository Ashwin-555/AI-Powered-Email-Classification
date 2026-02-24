# AI-Powered Email Classification and Urgency Detection System

## 1. Project Overview

This project is an AI-based email classification and urgency detection system developed using Python, Machine Learning, and Streamlit.

The system automatically classifies customer emails into predefined categories and assigns an urgency level. It also provides a real-time analytics dashboard for monitoring classification trends and priority distribution.

The objective of this project is to reduce manual effort in handling customer emails and improve prioritization efficiency within enterprise environments.

---

## 2. Problem Statement

Enterprises receive a high volume of customer support emails daily. Manually reviewing, categorizing, and identifying the urgency of these emails is time-consuming and inefficient. This often results in delayed responses, improper prioritization, and reduced customer satisfaction.

There is a need for an intelligent automated system that can categorize emails and determine urgency levels accurately and efficiently.

---

## 3. Proposed Solution

The proposed solution is an AI-powered email intelligence system that:

- Classifies emails into four categories:
  - Complaint
  - Request
  - Feedback
  - Spam
- Assigns an urgency level:
  - High
  - Medium
  - Low
- Displays classification results with confidence scores
- Provides a real-time analytics dashboard with visual insights

The system is built using supervised machine learning techniques and deployed using Streamlit with a modern professional interface.

---

## 4. System Architecture

The system consists of the following components:

1. Data Preprocessing  
2. Feature Extraction using TF-IDF  
3. Category Classification Model  
4. Urgency Classification Model  
5. Model Serialization using Pickle  
6. Streamlit Web Application  
7. Analytics Dashboard  

Two independent machine learning models are trained:

- Category Classification Model (4-class classification)
- Urgency Classification Model (3-class classification)

Both models use the same TF-IDF feature representation of email text.

---

## 5. Dataset Structure

The dataset must contain the following columns:

| email_text | label | urgency |
|------------|--------|----------|
| Email content | Complaint/Request/Feedback/Spam | High/Medium/Low |

- `email_text` contains the full email content.
- `label` represents the category of the email.
- `urgency` represents the urgency level.

---

## 6. Machine Learning Approach

### 6.1 Feature Extraction
TF-IDF (Term Frequency–Inverse Document Frequency) is used to convert textual email content into numerical feature vectors.

### 6.2 Classification Models
Logistic Regression is used for both classification tasks:

- Model 1: Predicts email category (Complaint, Request, Feedback, Spam)
- Model 2: Predicts urgency level (High, Medium, Low)

### 6.3 Training Workflow
1. Load dataset  
2. Split into training and testing sets  
3. Apply TF-IDF vectorization  
4. Train category model  
5. Train urgency model  
6. Evaluate using accuracy and classification metrics  
7. Save trained models using Pickle  

---

## 7. Project Structure

```
AI-Email-Classification/
│
├── train_models.py        # Script for training models
├── app.py                 # Streamlit web application
├── emails_dataset.csv     # Dataset file
├── vectorizer.pkl         # Saved TF-IDF vectorizer
├── category_model.pkl     # Trained category classifier
├── urgency_model.pkl      # Trained urgency classifier
├── requirements.txt       # Required Python packages
└── README.md              # Project documentation
```

---

## 8. Features

### 8.1 Email Classification Module

The user provides:
- Email ID
- Subject
- Email Body

The system:
- Combines subject and body
- Vectorizes the text
- Predicts category and urgency
- Displays confidence scores

---

### 8.2 Analytics Dashboard

The dashboard includes:

- Total Emails Processed (KPI)
- High Priority Emails (KPI)
- Spam Emails (KPI)
- Category Distribution (Pie Chart)
- Urgency Distribution (Bar Chart)
- Classification History Table

The dashboard updates dynamically using session-based storage without requiring a database.

---

## 9. Installation and Setup

### Step 1: Clone the Repository

```bash
git clone <repository-link>
cd AI-Email-Classification
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

If the requirements file is not available:

```bash
pip install streamlit pandas scikit-learn matplotlib plotly
```

### Step 3: Train the Models

```bash
python train_models.py
```

This will generate:
- vectorizer.pkl
- category_model.pkl
- urgency_model.pkl

### Step 4: Run the Application

```bash
streamlit run app.py
```

The application will open in a web browser.

---

## 10. Evaluation Metrics

Both models are evaluated independently using:

- Accuracy Score
- Precision
- Recall
- F1-Score
- Classification Report

These metrics ensure reliable performance across all classes.

---

## 11. Technologies Used

- Python
- Scikit-learn
- Pandas
- Streamlit
- Plotly
- Matplotlib
- Pickle

---

## 12. Use Cases

- Customer Support Centers
- IT Helpdesk Systems
- CRM Platforms
- Enterprise Ticketing Systems
- SaaS Support Platforms

---

## 13. Future Enhancements

- Integration with real-time email APIs
- Database support (PostgreSQL or MongoDB)
- Deep learning models such as BERT
- User authentication and role-based access
- Deployment on cloud platforms
- Automated priority queue management

---

## 14. Academic and Technical Significance

This project demonstrates practical implementation of:

- Natural Language Processing
- Multi-class Classification
- Multi-task Learning
- Feature Engineering
- Model Deployment
- Interactive Dashboard Development

It serves as a complete end-to-end machine learning application integrating backend modeling and frontend deployment.

---

## 15. Conclusion

The AI-Powered Email Classification and Urgency Detection System transforms manual email handling into an intelligent automated workflow. By combining machine learning with a modern web interface, the system improves prioritization, enhances operational efficiency, and provides analytical insights for better decision-making.
