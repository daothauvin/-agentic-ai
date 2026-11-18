"""Adaptateur LangChain vers Ollama avec sorties structurées."""

from typing import TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from .config import AgentConfig
from .logger import get_logger, timed

logger = get_logger("llm")
SchemaT = TypeVar("SchemaT", bound=BaseModel)


class LocalLLM:
    """Encapsule ChatOllama et les sorties structurées Pydantic."""

    def __init__(self, config: AgentConfig) -> None:
        """Configure le modèle Ollama utilisé par LangChain."""

        self.model = ChatOllama(
            base_url=config.ollama_url,
            model=config.model,
            temperature=0.,
        )
        

    def structured(
        self,
        schema: type[SchemaT],
        system_prompt: str,
        user_prompt: str,
    ) -> SchemaT:
        """Exécute le LLM et valide sa réponse avec un schéma Pydantic."""
        
        structured_model = self.model.with_structured_output(schema)
        logger.debug(
            "LLM %s | system=%d car. | user=%d car.",
            schema.__name__, len(system_prompt), len(user_prompt),
        )
        with timed(f"LLM {schema.__name__}", logger):
            result = structured_model.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )

        if isinstance(result, schema):
            return result

        return schema.model_validate(result)

    def structured_batch(
        self,
        schema: type[SchemaT],
        system_prompt: str,
        user_prompts: list[str],
        max_concurrency: int | None = None,
    ) -> list[SchemaT]:
        """Exécute plusieurs requêtes LLM avec une sortie structurée Pydantic."""

        structured_model = self.model.with_structured_output(schema)

        inputs = [
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
            for user_prompt in user_prompts
        ]

        config = (
            {"max_concurrency": max_concurrency}
            if max_concurrency is not None
            else None
        )

        with timed(f"LLM batch {schema.__name__} x{len(inputs)}", logger):
            results = structured_model.batch(
                inputs,
                config=config,
            )

        return [
            result if isinstance(result, schema) else schema.model_validate(result)
            for result in results
        ]
