import os


class MyUtility:

    def __init__(self):
        # Do something
        print('In the constructor of MyUtility')

    def convert_to_new_dictionary(self, list_of_dicts):
        new_dict = {}
        for dictionary in list_of_dicts:
            temp_key = None
            temp_value = None
            for key, value in dictionary.items():
                if (key == 'label'):
                    temp_key = value
                if (key == 'score'):
                    temp_value = value;
                if temp_key != None and temp_value != None:
                    new_dict[temp_key] = temp_value
                    emp_key = temp_value = None
        return new_dict;

    def read_txt_files(self, folder_path: str) -> str:
        all_text = ""
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                with open(os.path.join(folder_path, filename), "r") as file:
                    all_text += file.read() + "\n"
        return all_text

    def print_sentiments(self, sentiment_label: object, sentiment_score: object) -> object:
        sentiment_label = ('' + sentiment_label).upper()
        return f"{self.get_sentiment_emoji(sentiment_label.lower())} {sentiment_label} (Score: {sentiment_score})"

    def get_sentiment_emoji(self, sentiment):
        # Define the emojis corresponding to each sentiment
        emoji_mapping = {
            "disappointment": "😞",
            "sadness": "😢",
            "annoyance": "😠",
            "neutral": "😐",
            "disapproval": "👎",
            "realization": "😮",
            "nervousness": "😬",
            "approval": "👍",
            "joy": "😄",
            "anger": "😡",
            "embarrassment": "😳",
            "caring": "🤗",
            "remorse": "😔",
            "disgust": "🤢",
            "grief": "😥",
            "confusion": "😕",
            "relief": "😌",
            "desire": "😍",
            "admiration": "😌",
            "optimism": "😊",
            "fear": "😨",
            "love": "❤️",
            "excitement": "🎉",
            "curiosity": "🤔",
            "amusement": "😄",
            "surprise": "😲",
            "gratitude": "🙏",
            "pride": "🦁",
            "negative": "👎",
            "positive": "👍",
            "bad_data": "#@#%"
        }
        return emoji_mapping.get(sentiment, "")
