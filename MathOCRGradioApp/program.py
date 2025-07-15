import ollama
import re
import gradio as gr

import torch
from PIL import Image
import easyocr
import requests

reader = easyocr.Reader(['en', 'it'])
basic_prompt = (
    f"You are an AI specialized in solving puzzles. Analyze the following, identify hidden patterns or rules, and provide the missing value with step-by-step reasoning in text format. Do not return an answer in Latex."
    "Format your response strictly as follows:\n"
    "1. **Given Equation**:\n   - (original equations)\n"
    "2. **Pattern Identified**:\n   (explain the hidden logic)\n"
    "3. **Step-by-step Calculation**:\n   - For (input values):\n     (calculation and result)\n"
    "4. **Final Answer**:\n     (Answer = X)"
)


def query_ollama(formatted_prompt):
    # Query Gemma3 using Ollama
    response = ollama.chat(
        model="Gemma3",
        messages=[{"role": "user", "content": formatted_prompt}],
        options={"temperature": 0},
    )

    response_content = response["message"]["content"]

    # Remove content between <think> and </think> tags to remove thinking output
    final_answer = re.sub(r"<think>.*?</think>", "", response_content, flags=re.DOTALL).strip()
    return final_answer


def solve_puzzle(prompt, image):
    """Extracts the puzzle from the image and sends it to Gemma3 for solving."""
    try:
        # 1. Save the uploaded image temporarily; EasyOCR uses file paths
        image_path = "uploaded_image.png"
        image.save(image_path)

        # 2. Extract text from the image using EasyOCR
        results = reader.readtext(image_path)
        extracted_text = " ".join([res[1] for res in results])

        # Standardize the text to avoid misinterpretation of "??" as "2?"
        extracted_text = extracted_text.replace('??', '?')
      
        if "?" not in extracted_text:
            extracted_text += "?"

        print("Extracted Text:", extracted_text)  # Debugging output

        # 3. Refine the extracted text to standardize expressions
        refined_text = extracted_text.replace('x', '*').replace('X', '*').replace('=', ' = ').strip()
        print("Refined Text:", refined_text)  # Debugging output

        # 4. Compose the user message with concise instructions
        puzzle_prompt = (
            f"{prompt}\n"
            f"\nPuzzle:\n{refined_text}\n"
        )

        # 5. Send the request to Gemma3 with a timeout
        response = query_ollama(puzzle_prompt)
        return {response}
    except requests.exceptions.Timeout:
        return "Error: request timed out. Please try again."
    except Exception as e:
        return f"Error: {str(e)}"


# Set up the Gradio interface
interface = gr.Interface(
    fn=solve_puzzle,
    inputs=[
        gr.Textbox(label="Specify a puzzle solving prompt request", placeholder="Specify a puzzle solving prompt request", value=basic_prompt),
        gr.Image(type="pil")
    ],
    outputs="text",
    title="Logic Puzzle Solver with EasyOCR & Gemma3",
    description="Upload an image of a logic puzzle, and the model will solve it for you."
)

interface.launch(debug=True)