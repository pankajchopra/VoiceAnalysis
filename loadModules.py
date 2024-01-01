from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import flair


class LoadModules:
    all_modules = dict()

    def __init__(self, loadAllModule):
        print('in LoadModules constructor')
        if loadAllModule:
            self.load_all_models()

    def load_model_vader(self):
        try:
            LoadModules.all_modules['vader'] = SentimentIntensityAnalyzer()
            print ('loaded VADER model')
            return LoadModules.all_modules['vader']
        except Exception as ex:
            print("Error occurred during .. load_model_vader")
            print(str(ex))
            return "error", str(ex)

    def load_model_flair(self):
        try:
            LoadModules.all_modules['flair'] = flair.models.TextClassifier.load('en-sentiment')
            print('loaded Flair model')
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
            print('loaded distilbert-base-uncased-finetuned-sst-2-english model')
            return LoadModules.all_modules['distilbert']
        except Exception as ex:
            print("Error occurred during .. load_model_distilbert")
            print(str(ex))
            return "error", str(ex)

    def load_model_sam_lowe(self, return_all_score=False):
        try:
            LoadModules.all_modules['samLowe'] = pipeline("sentiment-analysis",
                                                           model="SamLowe/roberta-base-go_emotions",
                                                           return_all_scores=return_all_score)
            print('loaded SamLowe/roberta-base-go_emotions model')
            return LoadModules.all_modules['samLowe']
        except Exception as ex:
            print("Error occurred during .. load_model_sam_lowe")
            print(str(ex))
            return "error", str(ex)



    # def load_punctuation_model(self):
    #     try:
    #         LoadModules.all_modules['punctuation'] = PunctuationModel()
    #         print('loaded PunctuationModel model')
    #         return LoadModules.all_modules['punctuation']
    #     except Exception as ex:
    #         print("Error occurred during .. load_punctuation_model")
    #         print(str(ex))
    #         return "error", str(ex)


    def load_model_bhadresh_savani(self):
        try:
            LoadModules.all_modules['savani'] = pipeline("text-classification",
                                                         model="bhadresh-savani/bert-base-uncased-emotion",
                                                         return_all_scores=False)
            print('loaded bhadresh-savani/bert-base-uncased-emotion model')

            return LoadModules.all_modules['savani']
        except Exception as ex:
            print("Error occurred during .. load_model_bhadresh_savani")
            print(str(ex))
            return "error", str(ex)

    def load_all_models(self):
        print('in load_all_models()')
        result = self.all_modules
        result["flair"] = self.load_model_flair()
        result["distilbert"] = self.load_model_distilbert()
        result["vader"] = self.load_model_vader()
        result["savani"] = self.load_model_bhadresh_savani()
        result["samLowe"] = self.load_model_sam_lowe(False)
        # result["punctuation"] = self.load_punctuation_model()


    def load_model(self, model: []):
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
        if 'punctuation' in model:
            return self.load_model_punctuation()
        else:
            print("Nothing to load")

