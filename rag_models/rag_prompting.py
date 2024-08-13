from langchain_core.prompts import ChatPromptTemplate

medical_condition_def = ("specific health issues, diseases, or physical states that a plant or fungus is used to treat, "
                         "such as 'diabetes', 'high blood pressure' or 'inflammation'.")
medicinal_effect_def = ("therapeutic effects induced by consuming a plant or fungus, such as 'antitumor', 'anti-inflammatory', or 'digestive "
                        "stimulant'.")

standard_medicinal_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm tasked with extracting information about plants and fungi from scientific articles. "
            "Only extract relevant direct quotes from the text. "
            "Do not extract text contained in bibliography or reference sections. "
            "If you do not know the value of an attribute asked to extract, return null for the attribute's value. "
            "You should extract all scientific plant and fungal names mentioned in the text. "
            "You should include scientific authorities in the names if they appear in the text. "
            "Only extract scientific names. Do not extract common or vernacular names. "
            "For each of the plant or fungi names in the text, you should also extract mentions of any medical conditions they treat or medicinal effects they have. "
            "Medical conditions are " + medical_condition_def + " Medicinal effects are " + medicinal_effect_def
        ),
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # MessagesPlaceholder("examples"),  # <-- EXAMPLES!
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        ("human", "{text}"),
    ]
)
