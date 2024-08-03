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

## Challenges Faced

### 1. **Data Quality and Preprocessing**
- **Inconsistent Formats**: Emails might have inconsistent formats, making it difficult to accurately extract subject lines.
- **Noise in Text**: Emails could contain signatures, disclaimers, and other noise that can affect model performance.

### 2. **Model and Tokenizer Limitations**
- **Tokenization Issues**: Tokenizer might split words incorrectly or handle special characters poorly.
- **Model Size and Memory**: GPT-2 is a large model and can be memory-intensive, especially for long sequences and large batch sizes.

### 3. **Training and Evaluation**
- **Overfitting**: With a limited dataset, there is a risk of overfitting, where the model performs well on training data but poorly on unseen data.
- **Evaluation Strategy**: The evaluation strategy might need adjustments depending on the dataset's characteristics and the model's performance.

### 4. **Text Generation**
- **Coherence and Relevance**: Generated subject lines might not always be coherent or relevant to the email content.
- **Length Control**: Ensuring the generated subject lines are of appropriate length can be challenging.

### 5. **Generalization**
- **Domain Specificity**: The model trained on a specific dataset (e.g., Enron emails) might not generalize well to other email datasets or domains.

### 6. **Resource Constraints**
- **Computational Resources**: Training large models like GPT-2 requires substantial computational resources, which might be a constraint.

Addressing these challenges involves careful preprocessing, tuning model and training parameters, and potentially incorporating additional techniques such as data augmentation, advanced preprocessing, or using a more domain-specific model.
