import os

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI
from rag_models.schema import TaxaData
from rag_models.structured_prompting import standard_medicinal_prompt


def get_txt_from_file(txt_file: str):
    import os

    with open(os.path.join(txt_file), "r", encoding="utf8") as f:
        text = f.read()

    return text


def query_a_model(model, text_file: str) -> TaxaData:
    text = get_txt_from_file(text_file)
    runnable = standard_medicinal_prompt | model.with_structured_output(schema=TaxaData, include_raw=False)
    print(len(text))
    output = runnable.invoke({'text': text})
    print(output)
    return output


if __name__ == '__main__':
    # TODO: Arrange chunking specific to models: https://python.langchain.com/v0.1/docs/use_cases/extraction/how_to/handle_long_text/
    # Get API keys
    from dotenv import load_dotenv

    load_dotenv()

    # model1 = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    # gpt3_outputs = query_a_model(model1, os.path.join('example_inputs', '4187756.txt'))
    #
    # model2 = ChatOpenAI(model="gpt-4o", temperature=0)
    # gpt4_outputs = query_a_model(model2, os.path.join('example_inputs', '4187756.txt'))

    # TODO: fix auth for this
    # import vertexai
    #
    # PROJECT_ID = "[medplantmining]"  # @param {type:"string"}
    # REGION = "europe-west2"  # @param {type:"string"}
    #
    # # Initialize Vertex AI SDK
    # vertexai.init(project=PROJECT_ID, location=REGION)
    # model3 = ChatVertexAI(model="gemini-pro", temperature=0)
    # gemini_outputs = query_a_model(model3, os.path.join('example_inputs', '4187756.txt'))

    model4 = ChatAnthropic(model="claude-3-opus-20240229", temperature=0)
    claude_outputs = query_a_model(model4, os.path.join('example_inputs', '4187756.txt'))

    # TODO: install and set up api key for this
    model5 = ChatMistralAI(model="mistral-large-latest")
    pass
