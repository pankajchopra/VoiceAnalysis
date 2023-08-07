import os
from myUtilityDefs import MyUtility
from loadModules import LoadModules

myUtilityDefs = MyUtility()
loadModules = LoadModules()


def ask_question(deepset, text: str, question: str) -> str:
    response = deepset.Completion.create(
        engine="text-davinci-002",
        prompt=f"Answer the following question based on these texts:\n\n{text}\n\nQuestion: {question} \n ",
        temperature=0.5,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    answer = response.choices[0].text.strip()
    return answer


def main():
    myUtilityDefs.read_txt_files()
    folder_path = r"c:\users\panka"

    deepset = loadModules.load_deepset_roberta_base_squad2()
    all_text = myUtilityDefs.read_files("")

    while True:
        question = input("Ask a question based on the text files (type 'quit' to exit): ")
        if question.lower() == "quit":
            break

    answer = ask_question(deepset, all_text, question)
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
