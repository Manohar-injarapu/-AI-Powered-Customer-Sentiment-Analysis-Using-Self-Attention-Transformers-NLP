# AI-Powered-Customer-Sentiment-Analysis-Using-Self-Attention-Transformers-NLP

📌 PROBLEM STATEMENT

Given a restaurant review in textual form, build a Transformer-based NLP model that can automatically predict whether the sentiment of the review is Positive or Negative.
The model should learn semantic relationships between words using multi-head self-attention, without relying on recurrent networks such as RNN or LSTM.

🎯 PROJECT OBJECTIVES

->Understand how raw text is processed in NLP
->Learn how Transformers work internally
->Apply self-attention for sentiment classification
->Build an end-to-end NLP pipeline
->Perform real-time sentiment prediction

**conclusion**

The trained attention-based sentiment model was used to predict sentiment on unseen real-world reviews by applying the same preprocessing, tokenization, padding, and probability thresholding pipeline.

So this model can be tested on new data , by passing unseen sentences through the same preprocessing and tokenization pipeline used during training, padded them to a fixed length, and used the trained model to generate probability-based sentiment predictions.

In this project, we developed a sentiment analysis system that automatically classifies customer reviews as positive or negative using Natural Language Processing (NLP) techniques. The text data was first cleaned, tokenized, and converted into numerical form using word embeddings, enabling the model to process human language effectively.

The model uses a Transformer-based self-attention mechanism, which allows it to understand the context of words by analyzing their relationships within a sentence. This helps the system focus on important sentiment-related keywords rather than processing text sequentially, improving overall accuracy compared to traditional models.

After training and validation, the model showed reliable performance on unseen test data, indicating good generalization capability. The results confirm that the attention-based model can effectively capture semantic meaning and sentiment patterns in real-world customer reviews.

Overall, this project demonstrates a practical application of deep learning, self-attention, and NLP pipelines for automated text classification.
