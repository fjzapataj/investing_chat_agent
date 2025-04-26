import os


def load_prompt(prompt_path: str):
    print(f"Ruta actual: {os.getcwd()}")
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt no encontrado: {prompt_path}")
    with open(prompt_path) as f:
        prompt_text = f.read()
    return prompt_text
