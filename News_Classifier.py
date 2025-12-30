# ==========================================
# 1. INSTALL DEPENDENCIES
# ==========================================
!pip install -q transformers[torch] datasets evaluate gradio pandas scikit-learn

import pandas as pd
import torch
import gradio as gr
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline
)
import evaluate

# ==========================================
# 2. LOAD & PREPROCESS KAGGLE DATASET
# ==========================================
# Assumes files are named 'train.csv' and 'test.csv' in the root directory
# AG News Kaggle format: Class Index (1-4), Title, Description
def load_and_prep_data():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    # Rename columns for clarity (Class Index is 1-indexed, we need 0-indexed for BERT)
    # 1: World, 2: Sports, 3: Business, 4: Sci/Tech
    for df in [train_df, test_df]:
        df.columns = ['label', 'title', 'description']
        df['text'] = df['title'] + " " + df['description']
        df['label'] = df['label'] - 1  # Convert 1-4 to 0-3

    # Convert to Hugging Face Dataset format
    train_ds = Dataset.from_pandas(train_df[['text', 'label']])
    test_ds = Dataset.from_pandas(test_df[['text', 'label']])

    # We take a subset (e.g., 5000 samples) if you want the run to be fast in Colab
    # For full training, remove the .select() lines below
    train_ds = train_ds.shuffle(seed=42).select(range(5000))
    test_ds = test_ds.shuffle(seed=42).select(range(1000))

    return DatasetDict({"train": train_ds, "test": test_ds})

datasets = load_and_prep_data()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = datasets.map(tokenize_function, batched=True)

# ==========================================
# 3. FINE-TUNE BERT MODEL
# ==========================================
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)

# Evaluation Metric function
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    return {**acc, **f1}

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train()

# Save the model
model.save_pretrained("./fine_tuned_bert")
tokenizer.save_pretrained("./fine_tuned_bert")

# ==========================================
# 4. GRADIO DEPLOYMENT
# ==========================================
# Define labels for the interface
id2label = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
classifier = pipeline("text-classification", model="./fine_tuned_bert", tokenizer="./fine_tuned_bert")

def classify_news(headline):
    result = classifier(headline)[0]
    # Map the internal label (LABEL_X) back to our category names
    label_idx = int(result['label'].split('_')[-1])
    return {id2label[label_idx]: result['score']}

# Create Gradio UI
demo = gr.Interface(
    fn=classify_news,
    inputs=gr.Textbox(lines=2, placeholder="Enter a news headline here..."),
    outputs=gr.Label(num_top_classes=4),
    title="AG News Topic Classifier (BERT)",
    description="Fine-tuned BERT model to classify news into World, Sports, Business, or Sci/Tech."
)

demo.launch(share=True)
