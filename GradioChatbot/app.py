import gradio as gr
from businesslayer.config import MODEL_NAME, API_RETRIES, RESPONSE_STYLES
from businesslayer.chatbot import Chatbot

chatbot = Chatbot(model_name=MODEL_NAME, retries=API_RETRIES)

def handle_user_input(text_input, image_input, response_style):
    if not text_input.strip() and not image_input:
        return "Please provide either text or an image.", chatbot.get_history_description()
    
    try:
        chatbot.add_user_message(text_input, image_input, response_style)
        generated_text = chatbot.generate_response()
        return generated_text['content'], chatbot.get_history_description()
    except Exception as e:
        return f"An error occurred: {str(e)}", chatbot.get_history_description()

# Create the Gradio Blocks-based interface
with gr.Blocks() as chatUI:
    gr.Markdown("# Enhanced Multimodal Chatbot with AI Vision")
    gr.Markdown("Upload an image or enter a text prompt, choose a response style, and view the generated response along with the interaction history.")

    with gr.Row():
        text_input = gr.Textbox(lines=2, placeholder="Enter your question here...", label="Text Input")
        image_input = gr.Image(type="pil", label="Image Input (Optional)")
        response_style = gr.Dropdown(RESPONSE_STYLES, label="Response Style", value="Detailed")
    
    with gr.Column():
        generated_response = gr.Textbox(label="Generated Response")
        history_display = gr.Textbox(label="Chat History", interactive=False)
        submit_button = gr.Button("Submit")

    # Set button click event
    submit_button.click(
        fn=handle_user_input,
        inputs=[text_input, image_input, response_style],
        outputs=[generated_response, history_display]
    )

chatUI.launch()