from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import flair


class LoadModules:
    all_modules = None

    def __init__(self, loadAllModule):
        print ('in LoadModules constructor')
        if loadAllModule:
            self.all_modules = self.load_all_models();

    def load_model_vader(self):
        self.vader_obj = SentimentIntensityAnalyzer()
        try:
            self.vader_obj = SentimentIntensityAnalyzer()
        except Exception as ex:
            print("Error occurred during .. load_model_vader")
            print(str(ex))
            return "error", str(ex)

    def load_model_flair(self):
        try:
            self.all_modules['flair'] = flair.models.TextClassifier.load('en-sentiment')
            return self.all_modules['flair']
        except Exception as ex:
            print("Error occurred during .. load_model_flair")
            print(str(ex))
            return "error", str(ex)

    def load_model_distilbert(self):
        try:
            return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", top_k=1)
        except Exception as ex:
            print("Error occurred during .. load_model_distilbert")
            print(str(ex))
            return "error", str(ex)

    def load_model_sam_lowe(self, return_all_score):
        try:
            return pipeline("sentiment-analysis", model="SamLowe/roberta-base-go_emotions", return_all_scores=return_all_score)
        except Exception as ex:
            print("Error occurred during .. load_model_sam_lowe")
            print(str(ex))
            return "error", str(ex)

    def load_model_flair(self):
        try:
            return flair.models.TextClassifier.load('en-sentiment')
        except Exception as ex:
            print("Error occurred during .. load_model_flair")
            print(str(ex))
            return "error", str(ex)

    # Being used multiple times

    def load_model_savani(self):
        try:
            return pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion",
                            return_all_scores=False)
        except Exception as ex:
            print("Error occurred during .. load_model_savani")
            print(str(ex))
            return "error", str(ex)

    def load_all_models(self):
        print ('in load_all_models()')
        result = dict()
        result["flair"] = self.load_model_flair()
        result["distilbert"] = self.load_model_distilbert()
        result["vader"] = self.load_model_sid()
        result["savani"] = self.savani_classification = self.load_model_savani(False)
        result["sam_low"] = self.load_model_sam_lowe()
        return result

    def load_all_models(self, model: []):
        if 'flair' in model:
            return self.load_model_flair()
        if 'distilbert' in model:
            return self.load_model_distilbert()
        if 'savani' in model:
            return self.load_model_savani()
        if 'vader' in model:
            return self.load_model_vader()
        if 'samLowe' in model:
            return self.load_model_sam_lowe()
