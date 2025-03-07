# Email Subject Line Generation

## Problem Description
The task is to generate concise email subject lines from the body of the email. This involves identifying the most salient sentences and summarizing them into a few words.

## Model Architecture 
Here's a comparison of the BART (Bidirectional and Auto-Regressive Transformers) and GPT-2 (Generative Pre-trained Transformer 2) models:

### BART (Bidirectional and Auto-Regressive Transformers)
- **Developed by**: Facebook AI
- **Architecture**: 
  - **Encoder-Decoder structure** similar to the original Transformer architecture introduced by Vaswani et al.
  - The **encoder** is similar to BERT, and the **decoder** is similar to GPT.
  - The encoder processes the input text bidirectionally, allowing the model to consider the full context of a word from both the left and the right.
  - The decoder generates text in an auto-regressive manner, predicting one word at a time from left to right.
- **Training Objectives**:
  - **Denoising Autoencoder**: BART is trained to reconstruct the original text from a corrupted version.
  - Various types of noise are introduced, such as token masking, token deletion, text infilling, and sentence permutation.
- **Use Cases**:
  - Text generation
  - Machine translation
  - Summarization
  - Text completion and correction

### GPT-2 (Generative Pre-trained Transformer 2)
- **Developed by**: OpenAI
- **Architecture**:
  - **Decoder-only Transformer** model.
  - Utilizes a stack of transformer decoders.
  - Operates in an auto-regressive manner, predicting the next word in a sequence based on the previous words.
- **Training Objectives**:
  - **Language Modeling**: GPT-2 is trained to predict the next word in a sequence, making it proficient in generating coherent and contextually relevant text.
- **Use Cases**:
  - Text generation
  - Dialogue systems
  - Content creation
  - Code generation
  - Any application requiring understanding and generating human-like text

### Key Differences
- **Structure**:
  - BART uses an **encoder-decoder** structure, making it versatile for both understanding and generating text.
  - GPT-2 uses a **decoder-only** structure, focusing on generating text based on given prompts.
- **Training Objectives**:
  - BART is trained as a **denoising autoencoder**, which helps it learn to generate coherent text by reconstructing corrupted inputs.
  - GPT-2 is trained solely on the task of **language modeling**, predicting the next word in a sequence.
- **Flexibility**:
  - BART can be fine-tuned for various tasks, including summarization, translation, and question-answering, due to its encoder-decoder structure.
  - GPT-2 excels in tasks requiring text generation and continuation but may need additional modifications or fine-tuning for specific tasks like translation or summarization.

### Performance and Use Cases
- **BART**:
  - Better suited for tasks requiring understanding and transforming text, such as summarization and translation.
  - Effective for tasks where the context from both directions (left and right) is important.
- **GPT-2**:
  - Superior in generating long, coherent, and contextually appropriate text based on a given prompt.
  - Ideal for creative writing, content generation, and dialogue systems.

In summary, while both models are powerful in their own right, BART's encoder-decoder structure makes it more versatile for a range of tasks involving text transformation, whereas GPT-2's decoder-only architecture makes it a robust choice for text generation and continuation tasks.

## Dataset
We have used the dataset from [Github AESLC.](https://github.com/ryanzhumich/AESLC)

The dataset is annotated Enron Subject Line Corpus. It has dev, test, and train email datasets.
* Cleaned, filtered, and deduplicated emails from the Enron Email Corpus. 
* Sizes of train/dev/test splits: 14,436 / 1,960 / 1,906
* Average email length: 75 words
* Average subject length: 4 words

Dev and test datasets have @subject, @ann0,@ann1, and @ann2 to represent the subject lines of the mail.   
The train dataset has @subject as the subject line. 

Here's a summary of the methodologies used in your code along with potential challenges you might face:

## Steps for training model

### 1. **Text Preprocessing**
- **Tokenization**: The text is tokenized using NLTK's `word_tokenize` method.
- **Stopwords Removal**: Common English stopwords are removed using NLTK's stopwords list.
- **Lemmatization**: Words are lemmatized using NLTK's `WordNetLemmatizer` to reduce them to their base forms.

### 2. **Custom Dataset Class**
- **EmailSubjectDataset**: A custom PyTorch `Dataset` class is created to handle the email texts and their corresponding subject lines. This class handles the tokenization and formatting required by the GPT-2 model.

### 3. **Data Loading**
- **Data Loading from Files**: Emails and their subject lines are loaded from files. The subject lines are extracted using regex, and the email bodies are preprocessed.
- **Dataset Splitting**: The dataset is split into training and validation sets using PyTorch's `random_split`.

### 4. **Model and Tokenizer Initialization**
- **GPT-2 Model**: The GPT-2 model and tokenizer from Hugging Face's Transformers library are used. The tokenizer is set to use the EOS token for padding.

### 5. **Training**
- **TrainingArguments**: Training arguments for the Hugging Face `Trainer` are defined, including output directory, number of epochs, batch sizes, logging directory, and evaluation strategy.
- **Trainer**: The `Trainer` class is used to handle the training process, including evaluation on the validation set.

### 6. **Subject Line Generation**
- **Preprocessing**: The email text is preprocessed before subject line generation.
- **Generation**: The model generates a subject line using beam search.
- **Postprocessing**: The generated subject line is truncated to the first four words.

### 7. **Model Saving**
- **Model Saving**: The trained model is saved to a local directory using the `save_model` method of the `Trainer` class.

## Tabular comparison of the two models based on BLEU and ROUGE scores:

![image](https://github.com/user-attachments/assets/e875f404-d6b4-4ebd-88c6-0f5adc7cddd4)

Observations:
BLEU Score: GPT-2 performs better, indicating higher n-gram overlap.
ROUGE-1: BART performs better, suggesting better recall of single-word matches.
ROUGE-2: GPT-2 performs better, indicating better recall of 2-gram matches.
ROUGE-L and ROUGE-Lsum: BART performs slightly better, suggesting better performance in capturing longer sequences and overall summary structure.

# Q&A

## Step 1: Data Preprocessing

Data Cleaning: Remove any irrelevant or noisy data, such as headers and extra symbols from the Q & A pairs.
Tokenization: Tokenize both the questions and answers to feed into the generative model.
Removing Answer Prefix: Strip any "Answer:" prefixes to ensure clean output.
Padding and Truncation: Adjust the length of the sequences to ensure they fit the model input size.

## Step 2: Model Selection

Pre-trained GPT-2 Model: Utilize a pre-trained GPT-2 model, fine-tuned for text generation, particularly for generating answers in response to questions.


## Step 3: Model Fine-tuning

Fine-tuning GPT-2: Fine-tune the GPT-2 model on the Q & A dataset to enable it to generate precise answers.
Loss Function: Use cross-entropy loss to minimize the gap between the generated answer and the actual answer.
Optimization: Use AdamW as an optimizer to adapt the model's weights efficiently during training.


## Step 4: Evaluation

Evaluation Metrics: Use ROUGE score to measure the model’s accuracy in generating correct answers.
Manual Validation: Manually evaluate generated answers for coherence, relevance, and correctness.

![Screenshot 2024-09-26 152628](https://github.com/user-attachments/assets/ab2ef3a1-9329-4502-9c25-032048cf37c8)

## Demonstration 

![Screenshot 2024-09-26 155557](https://github.com/user-attachments/assets/91cd7832-e880-4638-92e8-67b102fb5a74)


![Screenshot 2024-09-26 153945](https://github.com/user-attachments/assets/30eb87c8-ae20-42c2-9c35-6ff8406e0558)






## HuggingFace App 
  - https://huggingface.co/spaces/maulinikhil/email_sub_gen






