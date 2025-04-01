from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

medical_condition_def = ("specific health issues, diseases, or physical states that a plant or fungus is used to treat; "
                         "such as 'diabetes', 'cancer', 'high blood pressure' or 'inflammation'.")
medicinal_effect_def = "therapeutic benefits induced by consuming a plant or fungus; such as 'antibiotic', 'anti-inflammatory', 'diuretic' or 'stimulant'."

standard_medicinal_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm tasked with extracting information about plants and fungi from scientific articles. "
            "Only extract relevant direct quotes from the text. Do not alter the extracted text by correcting spellings, expanding abbreviations or summarising the text. "
            "You should extract all scientific plant and fungal names mentioned in the text. "
            "You should include scientific authorities in the plant and fungal names if they appear in the text. "
            "Only extract scientific names. Do not extract common or vernacular names. "
            "For each of the plant or fungi names in the text, you should also extract mentions of any medical conditions they treat or medicinal effects they have. "
            f"Medical conditions we want to extract are {medical_condition_def} Medicinal effects we want to extract are {medicinal_effect_def}"
            "If you do not know the value of an attribute asked to extract, return null for the attribute's value. "
        ),
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        MessagesPlaceholder("examples"),  # <-- EXAMPLES!
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        ("human", "{text}"),
    ]
)
