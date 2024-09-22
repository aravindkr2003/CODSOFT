def chatbot_response(user_input):
    user_input = user_input.lower()

    if "hello" in user_input:
        return "Hi there! How can I assist you today?"
    elif "how are you" in user_input:
        return "I'm just a bot, but I'm doing great! How about you?"
    elif "your name" in user_input:
        return "I'm your friendly chatbot. What's your name?"
    elif "time" in user_input:
        from datetime import datetime
        now = datetime.now()
        return f"The current time is {now.strftime('%H:%M:%S')}."
    elif "bye" in user_input or "goodbye" in user_input:
        return "Goodbye! Have a great day!"
    else:
        return "I'm sorry, I don't understand that. Can you rephrase?"

while True:
    user_input = input("You: ")
   
    if "bye" in user_input.lower():
        print("Chatbot: Goodbye! Have a great day!")
        break
   
    response = chatbot_response(user_input)
    print(f"Chatbot: {response}")