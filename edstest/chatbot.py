import ollama

while True:
    question = input("You: ")

    if question.lower() == "exit":
        break

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": question
            }
        ]
    )

    print("Llama:", response["message"]["content"])