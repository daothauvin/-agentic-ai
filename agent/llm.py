"""Adaptateur LangChain vers Ollama avec sorties structurées."""

from typing import TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from .config import AgentConfig

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class LocalLLM:
    """Encapsule ChatOllama et les sorties structurées Pydantic."""

    def __init__(self, config: AgentConfig) -> None:
        """Configure le modèle Ollama utilisé par LangChain."""

        self.model = ChatOllama(
            base_url=config.ollama_url,
            model=config.model,
            temperature=0.1,
        )

    def structured(
        self,
        schema: type[SchemaT],
        system_prompt: str,
        user_prompt: str,
    ) -> SchemaT:
        """Exécute le LLM et valide sa réponse avec un schéma Pydantic."""
        
        structured_model = self.model.with_structured_output(schema)
        result = structured_model.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )

        if isinstance(result, schema):
            return result

        return schema.model_validate(result)
