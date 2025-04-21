# Complete IMDb Sentiment Pipeline with Optuna Tuning

# 1.  imports
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments
)
import evaluate
import optuna
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# 2. Load & tokenize IMDb
raw_datasets = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_fn(batch):
    return tokenizer(batch["text"],
                     truncation=True,
                     padding="max_length",
                     max_length=256)

tokenized_datasets = raw_datasets.map(
    tokenize_fn, batched=True, remove_columns=["text"]
)

# 3. Metric
accuracy_metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=preds, references=labels)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 4. Optuna objective
def objective(trial):
    # hyperparameters
    lr     = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    bs     = trial.suggest_categorical("per_device_train_batch_size", [8, 16])
    epochs = trial.suggest_int("num_train_epochs", 1, 3)

    # fresh model
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )

    # training args WITHOUT evaluation_strategy
    args = TrainingArguments(
        output_dir="hp_tuning",
        logging_steps=500,
        save_steps=1000,
        save_total_limit=1,
        learning_rate=lr,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        num_train_epochs=epochs,
        fp16=True,
        disable_tqdm=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"].shuffle(seed=42).select(range(5000)),
        eval_dataset=tokenized_datasets["test"].select(range(1000)),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    res = trainer.evaluate()
    return res["eval_accuracy"]

# 5. Run tuning
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=5)
best = study.best_params
print("Best hyperparameters:", best)

# 6. Final training with best params
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)
args = TrainingArguments(
    output_dir="imdb_final",
    logging_steps=500,
    save_steps=1000,
    save_total_limit=2,
    learning_rate=best["learning_rate"],
    per_device_train_batch_size=best["per_device_train_batch_size"],
    per_device_eval_batch_size=best["per_device_train_batch_size"],
    num_train_epochs=best["num_train_epochs"],
    fp16=True,
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
trainer.train()
results = trainer.evaluate()
print(f"\n Final Test Accuracy: {results['eval_accuracy'] * 100:.2f}%")

# 7. Detailed evaluation
preds = trainer.predict(tokenized_datasets["test"])
y_pred = np.argmax(preds.predictions, axis=1)
y_true = preds.label_ids

print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=["neg","pos"]))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

# 8. Saving  model & tokenizer
model.save_pretrained("imdb-distilbert-best")
tokenizer.save_pretrained("imdb-distilbert-best")

