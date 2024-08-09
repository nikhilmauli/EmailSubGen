import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BartForConditionalGeneration, BartTokenizer, pipeline
import nltk
import torch
import re

# Initialize FastAPI
app = FastAPI()

# Initialize NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the BART model and tokenizer
model_name = "maulinikhil/emailsubgen"  
model = BartForConditionalGeneration.from_pretrained(model_name, ignore_mismatched_sizes=True)
tokenizer = BartTokenizer.from_pretrained(model_name)

# Initialize the Named Entity Recognition (NER) pipeline
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", tokenizer="dbmdz/bert-large-cased-finetuned-conll03-english")

def normalize_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r"n't", " not", text)  # Handle contractions
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'s", " is", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'t", " not", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'m", " am", text)
    return text

def remove_names(text):
    entities = ner(text)
    words = text.split()
    for entity in entities:
        if entity['entity'] == 'B-PER' or entity['entity'] == 'I-PER':  # Person names
            word = entity['word']
            text = text.replace(word, "")
    return text

def preprocess_text(text):
    # Normalize the text
    text = normalize_text(text)
    # Remove names
    text = remove_names(text)
    # Tokenization
    words = nltk.word_tokenize(text)
    # Remove stop words
    words = [word for word in words if word not in nltk.corpus.stopwords.words('english')]
    # Lemmatization
    lemmatizer = nltk.WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

def generate_subject(email_body):
    try:
        # Preprocess the email body
        processed_email = preprocess_text(email_body)

        # Tokenize the input
        inputs = tokenizer.encode(processed_email, return_tensors="pt")

        # Generate the subject using enhanced beam search
        outputs = model.generate(
            inputs, 
            max_length=50, 
            num_beams=10,  # Increase beams for better results
            repetition_penalty=2.5,  # Penalize repetitive sequences
            early_stopping=True,
            num_return_sequences=3  # Return multiple candidates
        )
        
        # Decode all generated sequences and pick the best one
        candidates = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        
        # Choose the first generated candidate as the final subject
        final_subject = candidates[0]
        
        return final_subject
    except Exception as e:
        return f"Error: {str(e)}"

# Define the input model for FastAPI
class EmailBody(BaseModel):
    text: str

# FastAPI route
@app.post("/generate_subject/")
def generate_subject_api(email: EmailBody):
    return {"subject": generate_subject(email.text)}

# Gradio interface function
def gradio_interface(email_text):
    return generate_subject(email_text)

# Create Gradio Interface
gradio_app = gr.Interface(fn=gradio_interface, inputs="text", outputs="text", title="Email Subject Generator")

# Run Gradio Interface
if __name__ == "__main__":
    import uvicorn
    gradio_app.launch()  # For launching Gradio
    uvicorn.run(app, host="0.0.0.0", port=8000)  # For launching FastAPI
