import ollama
from .utils import image_to_bytes
class Chatbot:
    def __init__(self, model_name, retries):
        self.client = ollama.Client()
        self.model_name = model_name
        self.retries = retries
        self.history = []

    def add_user_message(self, text_input, image_input, response_style):
        message = {'role': 'user', 'content': text_input.strip()}

        if response_style == "Detailed":
            message['content'] += " Please provide a detailed response."
        elif response_style == "Concise":
            message['content'] += " Keep the response concise."
        else:
            message['content'] += f" Respond in a {response_style.lower()} manner."

        if image_input:
            image_bytes = image_to_bytes(image_input)
            message['images'] = {'data': image_bytes, 'type': 'image/jpeg'}

        self.history.append(message)
        
    def generate_response(self):
        for attempt in range(self.retries):
            try:
                response = self.client.chat(
                    model=self.model_name, messages=self.history
                )
                assistant_message = {'role': 'assistant', 'content': response['message']['content']}
                self.history.append(assistant_message)
                return assistant_message
            except Exception as e:
                if attempt == self.retries - 1:
                    raise e
                continue
    
    def get_history_description(self):
        return "\n".join(
            [f"{msg['role'].capitalize()}: {msg['content']}" for msg in self.history]
        )