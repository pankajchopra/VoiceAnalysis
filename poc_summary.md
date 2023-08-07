# Sentiment Analysis

Draft - 0.01

**Problem Statemen** t: Explore how sentiment analysis can be used to analyze the audio of calls received by call center agents.

Sentiment analysis is the task of identifying and extracting the emotions, opinions, and attitudes expressed in texts. Sentiment analysis can be applied to various domains and use cases, such as customer feedback, product reviews, social media posts, and more.

We will explore the use of machine learning models to analyze customer audio recordings and extract emotions and overall sentiments. This information can be displayed in real time to help call center agents better understand how customers are feeling during a call. Additionally, the same models can be applied to all call data and analyzed on a monthly or daily basis to identify trends and patterns that can be used to improve the customer experience.

We will also compare the performance and results of four different machine learning models for sentiment analysis: TextBlob, Flair, VADER, and DistilBERT.

Summary of the models:

- TextBlob is a Python library that provides a simple API for common natural language processing tasks, such as sentiment analysis, part-of-speech tagging, noun phrase extraction, and more. TextBlob uses a rule-based approach to sentiment analysis, which relies on a predefined lexicon of words and their polarity scores. TextBlob returns a sentiment object with two attributes: polarity and subjectivity. Polarity is a float value within the range [-1.0, 1.0] where -1.0 is very negative and 1.0 is very positive. Subjectivity is a float value within the range [0.0, 1.0] where 0.0 is very objective and 1.0 is very subjective1.
- Flair is another Python library that leverages state-of-the-art natural language processing models, such as BERT and DistilBERT, to perform various tasks, such as named entity recognition, sentiment analysis, text classification, and more. Flair uses a sequence labeling approach to sentiment analysis, which means that it assigns a sentiment label to each word or token in a text. Flair provides several pre-trained models for sentiment analysis, including one trained on the IMDB dataset. Flair returns a list of labels with two attributes: value and score. Value is the predicted sentiment label, such as POSITIVE or NEGATIVE. Score is a float value within the range [0.0, 1.0] that represents the confidence of the prediction.
- VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool that is specifically designed for social media texts, such as tweets, comments, reviews, and more. VADER uses a combination of lexical features, syntactical features, and semantic features to determine the polarity and intensity of sentiments expressed in texts. VADER returns a dictionary with four keys: neg, neu, pos, and compound. Neg, neu, and pos are float values within the range [0.0, 1.0] that indicate the proportion of negative, neutral, and positive words in the text. Compound is a normalized float value within the range [-1.0, 1.0] that represents the overall sentiment score of the text4.
- DistilBERT is a smaller and faster version of BERT, which is a powerful deep learning model for natural language understanding. DistilBERT retains most of the performance of BERT while reducing its size and computational cost by 40%. DistilBERT can be fine-tuned for various natural language processing tasks, such as sentiment analysis, text classification, question answering, and more. DistilBERT returns a list of logits for each possible class or label in the task. Logits are unnormalized probability scores that can be converted into probabilities using a softmax function.

![](RackMultipart20230807-1-w2vrtb_html_bac47c3b63b1e913.png)

**Conclusion:**

We have seen how sentiment analysis can be used to understand the sentiments of call center agents and employees from their audio and survey responses. We have also tested and compared four different machine learning models for sentiment analysis: TextBlob, Flair, VADER, and DistilBERT. The results showed that:

- TextBlob was the simplest and fastest model to use, but it had the lowest accuracy and consistency among the four models. **It also did not capture the intensity or subjectivity of the sentiments very well.**
- **Flair was the most accurate and consistent model among the four models** , but it was also the slowest and most resource-intensive model to use. It also had some limitations in handling noisy or informal texts, such as abbreviations, slang, or emojis.
- VADER was the best model for analyzing social media texts, as it was designed specifically for that purpose. It also had a good balance between speed and accuracy. However, it did not perform well on texts that were not from social media sources, such as audio transcripts or survey responses.
- DistilBERT was the most versatile and powerful model among the four models, as it could handle various types of texts and tasks. **It also had a good balance between speed** and accuracy. However, it required more fine-tuning and customization to achieve optimal results.

POC Link:

[https://voiceanalysis-pkchopra-2.streamlit.app/](https://voiceanalysis-pkchopra-2.streamlit.app/)

[https://voiceanalysis-pkchopra.streamlit.app/](https://voiceanalysis-pkchopra.streamlit.app/)

References:

1. Sentiment Analysis with TextBlob and Vader - Analytics Vidhya. [https://www.analyticsvidhya.com/blog/2021/10/sentiment-analysis-with-textblob-and-vader/](https://www.analyticsvidhya.com/blog/2021/10/sentiment-analysis-with-textblob-and-vader/).
2. GitHub - flairNLP/flair: A very simple framework for state-of-the-art. [https://github.com/flairNLP/flair](https://github.com/flairNLP/flair).
3. Unleash the Power of NLP with Flair: A Beginner's Guide to Sentiment .... [https://medium.com/@apappascs/unleash-the-power-of-nlp-with-flair-a-beginners-guide-to-sentiment-analysis-42f3565d72](https://medium.com/@apappascs/unleash-the-power-of-nlp-with-flair-a-beginners-guide-to-sentiment-analysis-42f3565d72).
4. GitHub - cjhutto/vaderSentiment: VADER Sentiment Analysis. VADER .... [https://github.com/cjhutto/vaderSentiment](https://github.com/cjhutto/vaderSentiment).
5. VADER Sentiment Analysis Explained | by Pio Calderon | Medium. [https://medium.com/@piocalderon/vader-sentiment-analysis-explained-f1c4f9101cd9](https://medium.com/@piocalderon/vader-sentiment-analysis-explained-f1c4f9101cd9).
6. DistilBERT - Hugging Face. [https://huggingface.co/docs/transformers/model\_doc/distilbert](https://huggingface.co/docs/transformers/model_doc/distilbert).
7. Getting Started with Sentiment Analysis using Python - Hugging Face. [https://huggingface.co/blog/sentiment-analysis-python](https://huggingface.co/blog/sentiment-analysis-python).
8. Fine-Tuning a Hugging Face DistilBERT Model for IMDB Sentiment Analysis .... [https://jamesmccaffrey.wordpress.com/2021/10/29/fine-tuning-a-hugging-face-distilbert-model-for-imdb-sentiment-analysis/](https://jamesmccaffrey.wordpress.com/2021/10/29/fine-tuning-a-hugging-face-distilbert-model-for-imdb-sentiment-analysis/).
9. Sentiment Analysis Comparision https://aashishmehta.com/sentiment-analysis-comparison/