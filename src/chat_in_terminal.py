import time
from utils.basic_chatbot_v1 import ChatBot
from utils.chatbot_agentic_v2 import ChatBot as ChatBot_v2
from utils.chatbot_agentic_v3 import ChatBot as Chatbot_v3

# If you'd like to chat with a different chatbot, modify the code manually.
# chatbot_version = "basic"
# chatbot_version = "v2"
chatbot_version = "v3"

if __name__ == "__main__":
    chatbot = None  # Initialize chatbot variable
    if chatbot_version == "basic":
        print("Basic chatbot is initialized. Type 'exit' to end the conversation.")
        chatbot = ChatBot()
    elif chatbot_version == "v2":
        print("Chatbot-agentic-v2 is initialized.  Type 'exit' to end the conversation.")
        chatbot = ChatBot_v2()
    elif chatbot_version == "v3":
        print("Chatbot-agentic-v3 is initialized.  Type 'exit' to end the conversation.")
        chatbot = Chatbot_v3()
    else:
        print("Invalid chatbot version. Please set chatbot_version to 'basic', 'v2', or 'v3'.")
        exit(1)
    
    if chatbot is None:
        print("Failed to initialize chatbot.")
        exit(1)

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        # Get response from the chatbot
        print("\nThinking...")
        start_time = time.time()
        response = chatbot.chat(user_input)
        end_time = time.time()

        print(f"\nAssistant ({round(end_time - start_time, 2)}s): {response}")
