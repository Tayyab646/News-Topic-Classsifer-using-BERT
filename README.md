# News-Topic-Classsifer-using-BERT

## 📌 Project Overview

The goal of this project is to build a machine learning model that can automatically classify news headlines into different topic categories using a transformer-based model.
The project uses **BERT (bert-base-uncased)** and is implemented with **Hugging Face Transformers**.

---

## 🎯 Objective

To fine-tune a pre-trained transformer model (BERT) for **news topic classification**, allowing the model to understand text context and predict the correct news category accurately.

---

## 📂 Dataset

* **AG News Dataset**
* Available on **Hugging Face Datasets**
* Contains news headlines and descriptions labeled into multiple topic categories such as World, Sports, Business, and Sci/Tech.

---

## 🛠️ Project Tasks

The following steps are performed in this project:

* Tokenization and preprocessing of text data
* Fine-tuning the **bert-base-uncased** model
* Training the model on the AG News dataset
* Evaluating the model using:

  * Accuracy
  * F1-score
* Deploying the trained model for live interaction using:

  * Streamlit or Gradio

---

## 📊 Model Evaluation

The performance of the model is evaluated using standard text classification metrics:

* **Accuracy**
* **F1-score**

These metrics help measure how well the model predicts the correct news category.

---

## 🚀 Deployment

The trained model is deployed using a lightweight web interface:

* **Streamlit** or **Gradio**
* Users can enter a news headline and get the predicted topic in real time.

---

## 🧠 Skills Gained

Through this project, the following skills were developed:

* Natural Language Processing (NLP) using Transformers
* Transfer learning and fine-tuning pre-trained models
* Evaluation metrics for text classification
* Building and deploying interactive ML applications

---

## 📁 Repository Structure (Optional)

```
├── data/
├── notebooks/
├── app.py
└── README.md


## 📌 Tools & Technologies

* Python
* Hugging Face Transformers
* Hugging Face Datasets
* PyTorch / TensorFlow
* Streamlit or Gradio

---

## ✅ Conclusion

This project demonstrates how transformer-based models like BERT can be effectively used for real-world NLP tasks such as news classification. It highlights the power of transfer learning and modern NLP techniques.
