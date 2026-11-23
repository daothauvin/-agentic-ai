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
        # Agent de code : modèle dédié optionnel (``--code-model``).
        self.code_model_name = config.code_model or config.model
        self.code_model = (
            ChatOllama(base_url=config.ollama_url, model=config.code_model, temperature=0.)
            if config.code_model and config.code_model != config.model
            else self.model
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

    def text(
        self,
        system_prompt: str,
        user_prompt: str,
        label: str = "texte",
        code: bool = False,
    ) -> str:
        """Exécute le LLM sans schéma et renvoie sa réponse brute.

        Utilisé pour la génération de code : avec le décodage JSON contraint,
        ``qwen2.5:3b`` produit du code tronqué ou invalide (préfixe ``->``,
        imports oubliés), alors qu'en texte libre le même prompt donne un
        script complet.

        Args:
            system_prompt: Prompt système.
            user_prompt: Prompt utilisateur.
            label: Nom de l'appel dans les logs de chronométrage.
            code: Utilise le modèle de l'agent de code (``code_model``).

        Returns:
            Contenu texte de la réponse.
        """

        model = self.code_model if code else self.model
        logger.debug(
            "LLM %s (%s) | system=%d car. | user=%d car.",
            label, model.model, len(system_prompt), len(user_prompt),
        )
        with timed(f"LLM {label}", logger):
            result = model.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
        return str(result.content)

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
