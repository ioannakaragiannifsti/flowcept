"""Generic decision-provenance capture example."""

import json

from langchain_openai import ChatOpenAI

from flowcept import DecisionCapture, Flowcept


def main():
    """Generate and capture a decision using an OpenAI-compatible configured model."""
    llm = ChatOpenAI(
        model="llama3.1:latest",
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    )
    with (
        Flowcept(
            workflow_name="Generic Decision Provenance",
            start_persistence=False,
        ),
        DecisionCapture(
            decision_type="selection",
            context="Select the best output for the supplied request",
            agent_id="example-agent",
            input_entity_ids=["request-001"],
            output_entity_ids=["result-001"],
            llm=llm,
        ) as decision,
    ):
        decision.invoke("Write a formal greeting.")

    print(json.dumps(decision.record.to_dict(), indent=2))


if __name__ == "__main__":
    main()
