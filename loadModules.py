from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import flair


class LoadModules:
    all_modules = dict()

    def __init__(self, loadAllModule):
        print('in LoadModules constructor')
        if loadAllModule:
            self.all_modules = self.load_all_models()

    def load_model_vader(self):
        try:
            LoadModules.all_modules['vader'] = SentimentIntensityAnalyzer()
            return LoadModules.all_modules['vader']
        except Exception as ex:
            print("Error occurred during .. load_model_vader")
            print(str(ex))
            return "error", str(ex)

    def load_model_flair(self):
        try:
            LoadModules.all_modules['flair'] = flair.models.TextClassifier.load('en-sentiment')
            return LoadModules.all_modules['flair']
        except Exception as ex:
            print("Error occurred during .. load_model_flair")
            print(str(ex))
            return "error", str(ex)

    def load_model_distilbert(self):
        try:
            LoadModules.all_modules['distilbert'] = pipeline("sentiment-analysis",
                                                             model="distilbert-base-uncased-finetuned-sst-2-english",
                                                             top_k=1)
            return LoadModules.all_modules['distilbert']
        except Exception as ex:
            print("Error occurred during .. load_model_distilbert")
            print(str(ex))
            return "error", str(ex)

    def load_model_sam_lowe(self, return_all_score):
        try:
            LoadModules.all_modules['sam_lowe'] = pipeline("sentiment-analysis",
                                                           model="SamLowe/roberta-base-go_emotions",
                                                           return_all_scores=return_all_score)
            return LoadModules.all_modules['sam_lowe']
        except Exception as ex:
            print("Error occurred during .. load_model_sam_lowe")
            print(str(ex))
            return "error", str(ex)


    def load_deepset_roberta_base_squad2(self, return_all_score):
        try:
            LoadModules.all_modules['deepset'] = pipeline("question-answering",
                                                           model="deepset/roberta-base-squad2",
                                                           return_all_scores=return_all_score)
            return LoadModules.all_modules['deepset']
        except Exception as ex:
            print("Error occurred during .. load_deepset_roberta_base_squad2")
            print(str(ex))
            return "error", str(ex)


    def load_model_bhadresh_savani(self):
        try:
            LoadModules.all_modules['savani'] = pipeline("text-classification",
                                                         model="bhadresh-savani/bert-base-uncased-emotion",
                                                         return_all_scores=False)
            return LoadModules.all_modules['savani']
        except Exception as ex:
            print("Error occurred during .. load_model_savani")
            print(str(ex))
            return "error", str(ex)

    def load_all_models(self):
        print('in load_all_models()')
        result = dict()
        result["flair"] = self.load_model_flair()
        result["distilbert"] = self.load_model_distilbert()
        result["vader"] = self.load_model_sid()
        result["savani"] = self.savani_classification = self.load_model_savani(False)
        result["sam_lowe"] = self.load_model_sam_lowe()
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
        if 'deepset' in model:
            return self.load_deepset_roberta_base_squad2()
        else:
            print("Nothing to load")
