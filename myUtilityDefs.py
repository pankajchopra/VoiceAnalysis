def convert_to_new_dictionary(list_of_dicts):
    new_dict = {}
    for dictionary in list_of_dicts:
        temp_key=None
        temp_value =None
        for key, value in dictionary.items():
            if(key =='label'):
                temp_key = value
            if (key == 'score'):
                temp_value = value;
            if temp_key != None and temp_value !=None:
                new_dict[temp_key] = temp_value
                emp_key = temp_value = None
    return new_dict;


def print_sentiments(sentiment_label, sentiment_score):
    sentiment_label = (''+sentiment_label).upper()
    return f"{get_sentiment_emoji(sentiment_label.lower())} {sentiment_label} (Score: {sentiment_score})"


def get_sentiment_emoji(sentiment):
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
        "positive":"👍",
        "bad_data":"#@#%"
    }
    return emoji_mapping.get(sentiment, "")


def readAMDFileAndReturnBuffer(filename):
    import glob
    import os
    with open(filename, 'r') as f:
        readme_line = f.readlines()
        file_buffer = []
        # resource_files = [os.path.basename(x) for x in glob.glob(f'Resources/*')]
    for line in readme_line:
        file_buffer.append(line)
        # for image in resource_files:
        #     if image in line:
        #         st.markdown(''.join(readme_buffer[:-1]))
        #         st.image(f'Resources/{image}')
        #         readme_buffer.clear()
    return file_buffer