import gradio as gr
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "tnlanh99/phobert-news-classification"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

num_labels = len(classifier.model.config.id2label)


def classify_news(news):
    output = classifier(news, truncation=True, top_k=num_labels)
    return {pred["label"]: pred["score"] for pred in output}


interface = gr.Interface(
    fn=classify_news, inputs="text", outputs=gr.Label(num_top_classes=num_labels)
)

interface.launch()
