from bs4 import BeautifulSoup
import requests
import json


faq_url = "https://thepowercompany.com/faqs/"

response = requests.get(faq_url)

soup = BeautifulSoup(response.text, "html.parser")

questions = soup.find_all("div", class_="et_pb_toggle")

data = []

for question in questions:
    question_text = question.find("h3").text
    answer = question.find("div", class_="et_pb_toggle_content")

    data.append({
        "question": question_text,
        "answer": answer.text
    })


# export to json file
with open("faq_data.txt", "w") as f:
    for item in data:
        f.write("Question: " + item["question"] + "\n")
        f.write("Answer: " + item["answer"] + "\n\n")    


print("FAQ data exported to faq_data.txt")