def test_bot(chatbot):
    print("Zeg 'stop' om de conversatie te stoppen.")
    
    while True:
        user_input = input()
        if user_input == "stop":
            break

        bot_response = chatbot.get_response(user_input)
        print(bot_response)