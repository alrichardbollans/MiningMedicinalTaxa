if __name__ == '__main__':
    # FOllowing
    # https://ai.google.dev/gemini-api/docs/get-started/tutorial?lang=python
    # https://cloud.google.com/vertex-ai/generative-ai/docs/samples/generativeaionvertexai-gemini-pdf#generativeaionvertexai_gemini_pdf-python

    import google.generativeai as genai
    import os
    from dotenv import load_dotenv

    load_dotenv()

    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    genai.configure(api_key=GOOGLE_API_KEY)


    # project_id = "MedPlantMining"

    model = genai.GenerativeModel(model_name="gemini-1.5-flash-001")

    prompt = """
    From the chunk of text below (delineated in curly brackets), you should list all scientific plant and fungal names including authorship.
    For each of the plant or fungi names you should then list any medical conditions they treat or medicinal effects they have.
    """

    with open(os.path.join('example_inputs', '4187756.txt'), "r", encoding="utf8") as f:
        text = f.read()

    prompt += "{"+text+"}"

    print(model.count_tokens(prompt))

    response = model.generate_content(prompt)

    print(response.text)