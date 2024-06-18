# Following: https://python.langchain.com/v0.1/docs/use_cases/extraction/how_to/examples/
import uuid
from typing import Dict, List, TypedDict
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.pydantic_v1 import BaseModel, Field
from rag_models.schema import Taxon


class Example(TypedDict):
    """A representation of an example consisting of text input and expected tool calls.

    For extraction, the tool calls are represented as instances of pydantic model.
    """

    input: str  # This is the example text
    tool_calls: List[BaseModel]  # Instances of pydantic model that should be extracted


def _tool_example_to_messages(example: Example) -> List[BaseMessage]:
    """Convert an example into a list of messages that can be fed into an LLM.

    This code is an adapter that converts our example to a list of messages
    that can be fed into a chat model.

    The list of messages per example corresponds to:

    1) HumanMessage: contains the content from which content should be extracted.
    2) AIMessage: contains the extracted information from the model
    3) ToolMessage: contains confirmation to the model that the model requested a tool correctly.

    The ToolMessage is required because some of the chat models are hyper-optimized for agents
    rather than for an extraction use case.
    """
    messages: List[BaseMessage] = [HumanMessage(content=example["input"])]
    openai_tool_calls = []
    for tool_call in example["tool_calls"]:
        openai_tool_calls.append(
            {
                "id": str(uuid.uuid4()),
                "type": "function",
                "function": {
                    # The name of the function right now corresponds
                    # to the name of the pydantic model
                    # This is implicit in the API right now,
                    # and will be improved over time.
                    "name": tool_call.__class__.__name__,
                    "arguments": tool_call.json(),
                },
            }
        )
    messages.append(
        AIMessage(content="", additional_kwargs={"tool_calls": openai_tool_calls})
    )
    tool_outputs = example.get("tool_outputs") or [
        "You have correctly called this tool."
    ] * len(openai_tool_calls)
    for output, tool_call in zip(tool_outputs, openai_tool_calls):
        messages.append(ToolMessage(content=output, tool_call_id=tool_call["id"]))
    return messages


_examples = [
    (
        "'Aloe vera L. is a very nice plant, it is used for lots of things. Though little evidence exists for it's efficacy, it is widely used to treat eczema and asthma as it is believed to reduce inflammation.'",
        Taxon(scientific_name='Aloe vera L.', medical_conditions=["eczema", "asthma"], medicinal_effects=["reduce inflammation"]),
    ),
    (
        "The traditional medicine 'toadstool' fungus, Amanita Pers., probably shouldn't be eaten but is used as an analgesic in the treatment of severe pain.",
        Taxon(scientific_name='Amanita Pers.', medical_conditions=["severe pain"], medicinal_effects=["analgesic"]),
    ),
    (
        "I eat fungi all the time, especially Pleurotus ostreatus",
        Taxon(scientific_name='Pleurotus ostreatus', medical_conditions=None, medicinal_effects=None),
    ),
]
example_messages = []

for text, tool_call in _examples:
    example_messages.extend(
        _tool_example_to_messages({"input": text, "tool_calls": [tool_call]})
    )
