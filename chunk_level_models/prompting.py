import os

medicinal_condition = " A medical condition refers to a specific health issue, disease, or physical state. Only collect medical conditions that are treated using plants or fungi."
medicinal_effect = " A medicinal effect refers to therapeutic or negative effects induced by plants or fungi, such as 'antitumor', 'anti-inflammatory', or 'digestive stimulant'. Only collect medicinal effects that are produced by plants or fungi."
other_notes = " Your outputs should be direct quotes from the text. If you notice a bibliography or reference section in the text, you should ignore its contents. If a scientific name, medical condition or medicinal effect appears multiple times, it should only appear once in the output under a single entity."
formatting = (
    ' Please provide a response in a structured JSON format that matches the following: {"annotations":[{"entity": "example name 1", "Medical condition": ["example condition 1", "example condition 2"], "Medicinal effect": ["example effect"]}, {"entity": "example name 2", "Medical condition": [], "Medicinal effect": []}]}. ')

example_output = '{"annotations":[{"entity": "Aloe vera L.", "Medical condition": ["eczema", "asthma"], "Medicinal effect": ["reduce inflammation"]},{"entity": "Amanita Pers.", "Medical condition": [], "Medicinal effect": []}]}'

base_prompt = (
        "From the chunk of text below (delineated with ```), you should collect all scientific plant and fungal names. You should include scientific authority in the name if it appears in the text. "
        "For each of the plant or fungi names you should also collect any medical conditions they treat or medicinal effects they have. "
        + medicinal_condition + medicinal_effect
        + other_notes + formatting +
        "As an example, in the following text: "
        "'Aloe vera L. is a very nice plant, it is used for lots of things. Though little evidence exists for it's efficacy, it is widely used to treat eczema and asthma as it is believed to reduce inflammation. I don't only like plants like Aloe vera L., I also like fungi such as Amanita Pers. ' "
        "Your output should be: " + example_output)


def get_full_prompt_for_txt_file(txt_file: str):
    import google.generativeai as genai
    import os
    from dotenv import load_dotenv

    load_dotenv()

    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    print(f'Number of tokens for base prompt: {model.count_tokens(base_prompt)}')

    with open(os.path.join(txt_file), "r", encoding="utf8") as f:
        text = f.read()
    full_prompt = base_prompt + "Text chunk: ```" + text + "```" + " The JSON response:"

    print(f'Number of tokens for full prompt: {model.count_tokens(full_prompt)}')

    return full_prompt
