#To run in debug mode
python -m streamlit.cli run code.py

# VoiceAnalysis
https://voiceanalysis-pkchopra.streamlit.app/

https://docs.google.com/presentation/d/e/2PACX-1vTzSLasf4BF4oeAOi66N0fXYzICBlJA3_PyLZAOjqNhJ8GuTm5V2l5EJlknS7Xn2Z7PNkTYa1zNpPMz/pub?start=false&loop=false&delayms=3000

**Natural language programming NLP uses semantic reasoning to try to interpret what a sentence means.**

_Text polarity describes whether it is a positive, neutral, or negative statement._ 
_Text polarity describes whether it is a positive, neutral, or negative statement. Online product reviews are often scored by NLP to get a percentage like 34% positive. Often the words in text are scored where like is +1 but the phrase don’t like is -1._

_Text subjectivity is a measure of how subjective or objective the statement is. An objective statement has presumably true factual information. A subjective statement gives an opinion about something._

---
### Models Used:


***bhadresh-savani/bert-base-uncased-emotion***

* **bert** is a Transformer Bidirectional Encoder based Architecture trained on MLM(Mask Language Modeling) *objective*

* **bert-base-uncased** finetuned on the emotion dataset using HuggingFace Trainer with below training parameters
---
**SamLowe/roberta-base-go_emotions**: _Model trained from **roberta-base** on the **go_emotions** dataset for multi-label classification._

* **roBERTa base model** _Pretrained model on English language using a masked language modeling (MLM) objective. It was introduced in this paper and first released in this repository. This model is case-sensitive: it makes a difference between english and English._
* **go_emotions** _is based on Reddit data and has 28 labels. It is a multi-label dataset where one or multiple labels may apply for any given input text, hence this model is a multi-label classification model with 28 'probability' float outputs for any given input text. Typically a threshold of 0.5 is applied to the probabilities for the prediction for each label._
---
**Flair Model** _[Flair](Link'[https://flairnlp.github.io/), is framework for Natural Language Processing (NLP). It employs pre-trained language models and transfer learning to generate contextual string embeddings for sentiment analysis._


* _Flair provide solution for NER(named entity recognition), Pos(Part of speech tagging), Sentiments, Sense disambiguation and text classification. Here in this POS we are using Text Classifier of Flair_ 
---
**VADER** (Valence Aware Dictionary and sEntiment Reasoner) is **a rule-based model** that uses a 
 sentiment lexicon and grammatical rules to determine the sentiment scores of the text.
* _VADER is a less resource-consuming sentiment analysis model that uses a set of rules to specify a mathematical model without explicitly coding it._ 
* _VADER consumes fewer resources as compared to Machine Learning models as there is no need for vast amounts of training data._
* _Comparing with other it out-perform benchmarks [reference link](https://www.researchgate.net/publication/275828927_VADER_A_Parsimonious_Rule-based_Model_for_Sentiment_Analysis_of_Social_Media_Text)_
---
**TextBlob**(default PatternAnalyzer) is a Python NLP library that uses a natural language toolkit (NLTK).
 aTextblob it gives two outputs, which are polarity and subjectivity. 
 Polarity is the output that lies between [-1,1], where -1 refers to negative 
 sentiment and +1 refers to positive sentiment. Subjectivity is the output that 
 lies within [0,1] and refers to personal opinions and judgments sentiment lexicon and grammatical rules to determine the sentiment scores of the text. More details [PatternAnalysis](https://phdservices.org/pattern-analysis-in-machine-learning/) [Naive Bayes](https://www.machinelearningplus.com/predictive-modeling/how-naive-bayes-algorithm-works-with-example-and-full-code/)