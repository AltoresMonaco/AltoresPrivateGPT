"""This file should be imported if and only if you want to run the UI locally."""

import base64
import logging
import time
import hashlib 
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Any

import gradio as gr  # type: ignore
from fastapi import FastAPI
from gradio.themes.utils.colors import slate  # type: ignore
from injector import inject, singleton
from llama_index.core.llms import ChatMessage, ChatResponse, MessageRole
from llama_index.core.types import TokenGen
from pydantic import BaseModel

from private_gpt.constants import PROJECT_ROOT_PATH
from private_gpt.di import global_injector
from private_gpt.open_ai.extensions.context_filter import ContextFilter
from private_gpt.server.chat.chat_service import ChatService, CompletionGen
from private_gpt.server.chunks.chunks_service import Chunk, ChunksService
from private_gpt.server.ingest.ingest_service import IngestService
from private_gpt.server.recipes.summarize.summarize_service import SummarizeService
from private_gpt.settings.settings import settings
from private_gpt.ui.images import logo_svg

logger = logging.getLogger(__name__)

THIS_DIRECTORY_RELATIVE = Path(__file__).parent.relative_to(PROJECT_ROOT_PATH)
# Should be "private_gpt/ui/avatar-bot.ico"
AVATAR_BOT = THIS_DIRECTORY_RELATIVE / "monacoTelecom-bot.ico"
ALTORES_ICON = THIS_DIRECTORY_RELATIVE / "altores_favicon.ico"

UI_TAB_TITLE = "Private Altores Intelligence"

SOURCES_SEPARATOR = "<hr>Sources: \n"

# Configuration sécurité
ADMIN_PASSWORD = "&cByzq@G88KE5D"  # TODO: À changer !
ADMIN_PASSWORD_HASH = hashlib.sha256(ADMIN_PASSWORD.encode()).hexdigest()

class Modes(str, Enum):
    RAG_MODE = "RAG"
    SEARCH_MODE = "Search"
    BASIC_CHAT_MODE = "Basic"
    SUMMARIZE_MODE = "Summarize"


MODES: list[Modes] = [
    Modes.RAG_MODE,
    Modes.SEARCH_MODE,
    Modes.BASIC_CHAT_MODE,
    Modes.SUMMARIZE_MODE,
]


class Source(BaseModel):
    file: str
    page: str
    text: str

    class Config:
        frozen = True

    @staticmethod
    def curate_sources(sources: list[Chunk]) -> list["Source"]:
        curated_sources = []

        for chunk in sources:
            doc_metadata = chunk.document.doc_metadata

            file_name = doc_metadata.get("file_name", "-") if doc_metadata else "-"
            page_label = doc_metadata.get("page_label", "-") if doc_metadata else "-"

            source = Source(file=file_name, page=page_label, text=chunk.text)
            curated_sources.append(source)
            curated_sources = list(
                dict.fromkeys(curated_sources).keys()
            )  # Unique sources only

        return curated_sources


@singleton
class PrivateGptUi:
    @inject
    def __init__(
        self,
        ingest_service: IngestService,
        chat_service: ChatService,
        chunks_service: ChunksService,
        summarizeService: SummarizeService,
    ) -> None:
        self._ingest_service = ingest_service
        self._chat_service = chat_service
        self._chunks_service = chunks_service
        self._summarize_service = summarizeService

        # Cache the UI blocks
        self._ui_block = None

        self._selected_filename = None

        # Initialize system prompt based on default mode
        default_mode_map = {mode.value: mode for mode in Modes}
        self._default_mode = default_mode_map.get(
            settings().ui.default_mode, Modes.RAG_MODE
        )
        self._system_prompt = self._get_default_system_prompt(self._default_mode)
        
        # État d'authentification
        self._admin_authenticated = False

    def _chat(
        self, message: str, history: list[list[str]], mode: Modes, *_: Any
    ) -> Any:
        def yield_deltas(completion_gen: CompletionGen) -> Iterable[str]:
                """Yield les deltas en gérant le raisonnement <think>."""
                
                class ReasoningParser:
                    def __init__(self):
                        self.is_in_reasoning = False
                        self.has_reasoning = False
                        self.reasoning_content = ""
                        self.response_content = ""
                        self.start_time = None
                        self.elapsed_time = 0
                        self.buffer = ""
                        self.think_start = "<think>"
                        self.think_end = "</think>"
                        
                    def process_chunk(self, text: str):
                        """Traite un chunk de texte."""
                        self.buffer += text
                        
                        while True:
                            if not self.is_in_reasoning:
                                # Chercher le début du raisonnement
                                start_idx = self.buffer.find(self.think_start)
                                if start_idx != -1:
                                    # Ajouter le texte avant la balise à la réponse
                                    self.response_content += self.buffer[:start_idx]
                                    self.buffer = self.buffer[start_idx + len(self.think_start):]
                                    self.is_in_reasoning = True
                                    self.has_reasoning = True
                                    self.start_time = time.time()
                                else:
                                    # Garder les derniers caractères au cas où la balise est coupée
                                    if len(self.buffer) > len(self.think_start):
                                        self.response_content += self.buffer[:-len(self.think_start)]
                                        self.buffer = self.buffer[-len(self.think_start):]
                                    break
                            else:
                                # En mode raisonnement, chercher la fin
                                end_idx = self.buffer.find(self.think_end)
                                if end_idx != -1:
                                    self.reasoning_content += self.buffer[:end_idx]
                                    self.buffer = self.buffer[end_idx + len(self.think_end):]
                                    self.is_in_reasoning = False
                                    # Calculer le temps SEULEMENT ici
                                    self.elapsed_time = time.time() - self.start_time
                                else:
                                    # Continuer à accumuler le raisonnement
                                    if len(self.buffer) > len(self.think_end):
                                        self.reasoning_content += self.buffer[:-len(self.think_end)]
                                        self.buffer = self.buffer[-len(self.think_end):]
                                    break
                                    
                    def flush(self):
                        """Vide le buffer restant."""
                        if self.is_in_reasoning:
                            self.reasoning_content += self.buffer
                            if self.start_time:
                                self.elapsed_time = time.time() - self.start_time
                        else:
                            self.response_content += self.buffer
                        self.buffer = ""
                        
                    def get_formatted_output(self) -> str:
                        """Retourne le contenu formaté pour Gradio."""
                        # Si on n'a jamais eu de raisonnement, retourner juste la réponse
                        if not self.has_reasoning:
                            return self.response_content
                        
                        # Format Markdown Gradio avec séparateur HTML
                        if self.is_in_reasoning:
                            elapsed = time.time() - self.start_time if self.start_time else 0
                            # Section de raisonnement en cours
                            reasoning_section = f"""<div style="border: 1px solid var(--border-color-primary); border-radius: 8px; padding: 16px; margin-bottom: 16px; background: var(--panel-background-fill);">
        <div style="font-weight: bold; margin-bottom: 8px;">🧠 Raisonnement en cours... (⏱️ {elapsed:.1f}s)</div>
        <pre style="margin: 0; white-space: pre-wrap; font-family: monospace; color: var(--body-text-color-subdued);">{self.reasoning_content}</pre>
        </div>"""
                        else:
                            # Section de raisonnement terminé - utilisons un style accordéon simple
                            reasoning_section = f"""<div style="border: 1px solid var(--border-color-primary); border-radius: 8px; padding: 16px; margin-bottom: 16px; background: var(--panel-background-fill);">
        <details>
        <summary style="cursor: pointer; font-weight: bold;">🧠 Afficher le raisonnement (⏱️ {self.elapsed_time:.2f}s)</summary>
        <pre style="margin-top: 8px; white-space: pre-wrap; font-family: monospace; color: var(--body-text-color-subdued);">{self.reasoning_content}</pre>
        </details>
        </div>"""
                        
                        # Séparer clairement le raisonnement de la réponse
                        return reasoning_section + "\n\n" + self.response_content
                
                # Initialiser le parser
                parser = ReasoningParser()
                
                # Traiter le stream
                stream = completion_gen.response
                for delta in stream:
                    if isinstance(delta, str):
                        text = str(delta)
                    elif isinstance(delta, ChatResponse):
                        text = delta.delta or ""
                    else:
                        continue
                        
                    parser.process_chunk(text)
                    yield parser.get_formatted_output()
                    time.sleep(0.02)
                
                # Vider le buffer final
                parser.flush()
                
                # Ajouter les sources si présentes
                if completion_gen.sources:
                    parser.response_content += SOURCES_SEPARATOR
                    cur_sources = Source.curate_sources(completion_gen.sources)
                    sources_text = "\n\n\n"
                    used_files = set()
                    for index, source in enumerate(cur_sources, start=1):
                        if f"{source.file}-{source.page}" not in used_files:
                            sources_text = (
                                sources_text
                                + f"{index}. {source.file} (page {source.page}) \n\n"
                            )
                            used_files.add(f"{source.file}-{source.page}")
                    sources_text += "<hr>\n\n"
                    parser.response_content += sources_text
                
                yield parser.get_formatted_output()

        def yield_tokens(token_gen: TokenGen) -> Iterable[str]:
            full_response: str = ""
            for token in token_gen:
                full_response += str(token)
                yield full_response

        def build_history() -> list[ChatMessage]:
            history_messages: list[ChatMessage] = []

            for interaction in history:
                history_messages.append(
                    ChatMessage(content=interaction[0], role=MessageRole.USER)
                )
                if len(interaction) > 1 and interaction[1] is not None:
                    history_messages.append(
                        ChatMessage(
                            # Remove from history content the Sources information
                            content=interaction[1].split(SOURCES_SEPARATOR)[0],
                            role=MessageRole.ASSISTANT,
                        )
                    )

            # max 20 messages to try to avoid context overflow
            return history_messages[:20]

        new_message = ChatMessage(content=message, role=MessageRole.USER)
        all_messages = [*build_history(), new_message]
        # If a system prompt is set, add it as a system message
        if self._system_prompt:
            all_messages.insert(
                0,
                ChatMessage(
                    content=self._system_prompt,
                    role=MessageRole.SYSTEM,
                ),
            )
        match mode:
            case Modes.RAG_MODE:
                # Use only the selected file for the query
                context_filter = None
                if self._selected_filename is not None:
                    docs_ids = []
                    for ingested_document in self._ingest_service.list_ingested():
                        if (
                            ingested_document.doc_metadata["file_name"]
                            == self._selected_filename
                        ):
                            docs_ids.append(ingested_document.doc_id)
                    context_filter = ContextFilter(docs_ids=docs_ids)

                query_stream = self._chat_service.stream_chat(
                    messages=all_messages,
                    use_context=True,
                    context_filter=context_filter,
                )
                yield from yield_deltas(query_stream)
            case Modes.BASIC_CHAT_MODE:
                llm_stream = self._chat_service.stream_chat(
                    messages=all_messages,
                    use_context=False,
                )
                yield from yield_deltas(llm_stream)

            case Modes.SEARCH_MODE:
                response = self._chunks_service.retrieve_relevant(
                    text=message, limit=4, prev_next_chunks=0
                )

                sources = Source.curate_sources(response)

                yield "\n\n\n".join(
                    f"{index}. **{source.file} "
                    f"(page {source.page})**\n "
                    f"{source.text}"
                    for index, source in enumerate(sources, start=1)
                )
            case Modes.SUMMARIZE_MODE:
                # Summarize the given message, optionally using selected files
                context_filter = None
                if self._selected_filename:
                    docs_ids = []
                    for ingested_document in self._ingest_service.list_ingested():
                        if (
                            ingested_document.doc_metadata["file_name"]
                            == self._selected_filename
                        ):
                            docs_ids.append(ingested_document.doc_id)
                    context_filter = ContextFilter(docs_ids=docs_ids)

                summary_stream = self._summarize_service.stream_summarize(
                    use_context=True,
                    context_filter=context_filter,
                    instructions=message,
                )
                yield from yield_tokens(summary_stream)

    # On initialization and on mode change, this function set the system prompt
    # to the default prompt based on the mode (and user settings).
    @staticmethod
    def _get_default_system_prompt(mode: Modes) -> str:
        p = ""
        match mode:
            # For query chat mode, obtain default system prompt from settings
            case Modes.RAG_MODE:
                p = settings().ui.default_query_system_prompt
            # For chat mode, obtain default system prompt from settings
            case Modes.BASIC_CHAT_MODE:
                p = settings().ui.default_chat_system_prompt
            # For summarization mode, obtain default system prompt from settings
            case Modes.SUMMARIZE_MODE:
                p = settings().ui.default_summarization_system_prompt
            # For any other mode, clear the system prompt
            case _:
                p = ""
        return p

    @staticmethod
    def _get_default_mode_explanation(mode: Modes) -> str:
        match mode:
            case Modes.RAG_MODE:
                return "Get contextualized answers from selected files."
            case Modes.SEARCH_MODE:
                return "Find relevant chunks of text in selected files."
            case Modes.BASIC_CHAT_MODE:
                return "Chat with the LLM using its training data. Files are ignored."
            case Modes.SUMMARIZE_MODE:
                return "Generate a summary of the selected files. Prompt to customize the result."
            case _:
                return ""

    def _set_system_prompt(self, system_prompt_input: str) -> None:
        logger.info(f"Setting system prompt to: {system_prompt_input}")
        self._system_prompt = system_prompt_input

    def _set_explanatation_mode(self, explanation_mode: str) -> None:
        self._explanation_mode = explanation_mode

    def _set_current_mode(self, mode: Modes) -> Any:
        self.mode = mode
        self._set_system_prompt(self._get_default_system_prompt(mode))
        self._set_explanatation_mode(self._get_default_mode_explanation(mode))
        interactive = self._system_prompt is not None
        return [
            gr.update(placeholder=self._system_prompt, interactive=interactive),
            gr.update(value=self._explanation_mode),
        ]

    def _verify_password(self, password: str) -> bool:
        """Vérifie le mot de passe admin"""
        if not password:
            return False
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        return password_hash == ADMIN_PASSWORD_HASH

    def _toggle_admin_access(self, password: str) -> tuple[Any, Any, Any, Any, Any]:
        """Toggle l'accès admin avec vérification mot de passe"""
        if self._admin_authenticated:
            # Déconnexion
            self._admin_authenticated = False
            return [
                gr.update(visible=True),   # auth_section (réapparaît)
                gr.update(visible=False),  # admin_section (disparaît)
                gr.update(value="🔒"),     # unlock_button
                gr.update(value=""),       # password_input (clear)
                gr.update(label=self._get_chatbot_label()),  # chatbot label update
            ]
        else:
            # Tentative de connexion
            if self._verify_password(password):
                self._admin_authenticated = True
                return [
                    gr.update(visible=False),  # auth_section (disparaît)
                    gr.update(visible=True),   # admin_section (apparaît)
                    gr.update(value="🔓"),     # unlock_button  
                    gr.update(value=""),       # password_input (clear)
                    gr.update(label=self._get_chatbot_label()),  # chatbot label update
                ]
            else:
                return [
                    gr.update(visible=True),   # auth_section (reste visible)
                    gr.update(visible=False),  # admin_section (reste cachée)
                    gr.update(value="🔒"),     # unlock_button
                    gr.update(value="❌ Mot de passe incorrect"),  # password_input
                    gr.update(),  # chatbot pas de changement
                ]

    def _logout_admin(self) -> tuple[Any, Any, Any, Any, Any]:
        """Déconnexion admin"""
        self._admin_authenticated = False
        return [
            gr.update(visible=True),   # auth_section (réapparaît)
            gr.update(visible=False),  # admin_section (disparaît)
            gr.update(value="🔒"),     # unlock_button
            gr.update(value=""),       # password_input (clear)
            gr.update(label=self._get_chatbot_label()),  # chatbot label update
        ]

    def _get_chatbot_label(self) -> str:
        """Retourne le label approprié selon l'état d'authentification"""
        if not self._admin_authenticated:
            # Utilisateur non connecté - afficher le nom générique
            return "Altores Intelligence Performance"
        else:
            # Utilisateur admin connecté - afficher le vrai modèle
            def get_model_label() -> str | None:
                """Get model label from llm mode setting YAML."""
                config_settings = settings()
                if config_settings is None:
                    raise ValueError("Settings are not configured.")

                llm_mode = config_settings.llm.mode
                model_mapping = {
                    "llamacpp": config_settings.llamacpp.llm_hf_model_file,
                    "openai": config_settings.openai.model,
                    "openailike": config_settings.openai.model,
                    "azopenai": config_settings.azopenai.llm_model,
                    "sagemaker": config_settings.sagemaker.llm_endpoint_name,
                    "mock": llm_mode,
                    "ollama": config_settings.ollama.llm_model,
                    "gemini": config_settings.gemini.model,
                }

                if llm_mode not in model_mapping:
                    print(f"Invalid 'llm mode': {llm_mode}")
                    return None

                return model_mapping[llm_mode]

            model_label = get_model_label()
            if model_label is not None:
                return f"LLM: {settings().llm.mode} | Model: {model_label}"
            else:
                return f"LLM: {settings().llm.mode}"

    def _list_ingested_files(self) -> list[list[str]]:
        files = set()
        for ingested_document in self._ingest_service.list_ingested():
            if ingested_document.doc_metadata is None:
                # Skipping documents without metadata
                continue
            file_name = ingested_document.doc_metadata.get(
                "file_name", "[FILE NAME MISSING]"
            )
            files.add(file_name)
        return [[row] for row in files]

    def _upload_file(self, files: list[str]) -> None:
        logger.debug("Loading count=%s files", len(files))
        paths = [Path(file) for file in files]

        # remove all existing Documents with name identical to a new file upload:
        file_names = [path.name for path in paths]
        doc_ids_to_delete = []
        for ingested_document in self._ingest_service.list_ingested():
            if (
                ingested_document.doc_metadata
                and ingested_document.doc_metadata["file_name"] in file_names
            ):
                doc_ids_to_delete.append(ingested_document.doc_id)
        if len(doc_ids_to_delete) > 0:
            logger.info(
                "Uploading file(s) which were already ingested: %s document(s) will be replaced.",
                len(doc_ids_to_delete),
            )
            for doc_id in doc_ids_to_delete:
                self._ingest_service.delete(doc_id)

        self._ingest_service.bulk_ingest([(str(path.name), path) for path in paths])

    def _delete_all_files(self) -> Any:
        ingested_files = self._ingest_service.list_ingested()
        logger.debug("Deleting count=%s files", len(ingested_files))
        for ingested_document in ingested_files:
            self._ingest_service.delete(ingested_document.doc_id)
        return [
            gr.List(self._list_ingested_files()),
            gr.components.Button(interactive=False),
            gr.components.Button(interactive=False),
            gr.components.Textbox("All files"),
        ]

    def _delete_selected_file(self) -> Any:
        logger.debug("Deleting selected %s", self._selected_filename)
        # Note: keep looping for pdf's (each page became a Document)
        for ingested_document in self._ingest_service.list_ingested():
            if (
                ingested_document.doc_metadata
                and ingested_document.doc_metadata["file_name"]
                == self._selected_filename
            ):
                self._ingest_service.delete(ingested_document.doc_id)
        return [
            gr.List(self._list_ingested_files()),
            gr.components.Button(interactive=False),
            gr.components.Button(interactive=False),
            gr.components.Textbox("All files"),
        ]

    def _deselect_selected_file(self) -> Any:
        self._selected_filename = None
        return [
            gr.components.Button(interactive=False),
            gr.components.Button(interactive=False),
            gr.components.Textbox("All files"),
        ]

    def _selected_a_file(self, select_data: gr.SelectData) -> Any:
        self._selected_filename = select_data.value
        return [
            gr.components.Button(interactive=True),
            gr.components.Button(interactive=True),
            gr.components.Textbox(self._selected_filename),
        ]

    def _build_ui_blocks(self) -> gr.Blocks:
        logger.debug("Creating the UI blocks")
        with gr.Blocks(
            title=UI_TAB_TITLE,
            theme=gr.themes.Soft(primary_hue=slate),
            css=".logo { "
            "display:flex;"
            "background: linear-gradient(135deg, #ffd0e0 0%, #a7c1ff 100%);"
            "height: 80px;"
            "border-radius: 8px;"
            "align-content: center;"
            "justify-content: center;"
            "align-items: center;"
            "}"
            ".logo img { height: 25% }"
            ".contain { display: flex !important; flex-direction: column !important; }"
            "#component-0, #component-3, #component-10, #component-8  { height: 100% !important; }"
            "#chatbot { flex-grow: 1 !important; overflow: auto !important;}"
            "#col { height: calc(100vh - 112px - 16px) !important; }"
            "hr { margin-top: 1em; margin-bottom: 1em; border: 0; border-top: 1px solid #FFF; }"
            ".avatar-container { display: contents !important; }"
            ".avatar-image { height: 45px !important; width: 45px !important; background-color: transparent !important; border-radius: 0 !important; padding: 0 !important; margin: 0 !important; object-fit: contain; }"
            ".footer { text-align: center; margin-top: 20px; font-size: 14px; display: flex; align-items: center; justify-content: center; }"
            ".footer-zylon-link { display:flex; margin-left: 5px; text-decoration: auto; color: var(--body-text-color); }"
            ".footer-zylon-link:hover { color: #C7BAFF; }"
            ".footer-zylon-ico { height: 20px; margin-left: 5px; background-color: antiquewhite; border-radius: 2px; }",
        ) as blocks:
            with gr.Row():
                gr.HTML(f"<div class='logo'/><img src={logo_svg} alt=PrivateGPT></div")

            with gr.Row(equal_height=False):
                # Section d'authentification (visible quand non connecté)
                with gr.Column(scale=2, visible=True) as auth_section:
                    gr.Markdown("### 🔐 Admin Access")
                    password_input = gr.Textbox(
                        type="password", 
                        placeholder="Enter admin password",
                        label="Password",
                        scale=3
                    )
                    unlock_button = gr.Button("🔒", scale=1, size="sm")
                
                # Section admin (cachée par défaut)  
                with gr.Column(scale=3, visible=False) as admin_section:
                    default_mode = self._default_mode
                    mode = gr.Radio(
                        [mode.value for mode in MODES],
                        label="Mode",
                        value=default_mode,
                    )
                    explanation_mode = gr.Textbox(
                        placeholder=self._get_default_mode_explanation(default_mode),
                        show_label=False,
                        max_lines=3,
                        interactive=False,
                    )
                    upload_button = gr.components.UploadButton(
                        "Upload File(s)",
                        type="filepath",
                        file_count="multiple",
                        size="sm",
                    )
                    ingested_dataset = gr.List(
                        self._list_ingested_files,
                        headers=["File name"],
                        label="Ingested Files",
                        height=235,
                        interactive=False,
                        render=False,  # Rendered under the button
                    )
                    upload_button.upload(
                        self._upload_file,
                        inputs=upload_button,
                        outputs=ingested_dataset,
                    )
                    ingested_dataset.change(
                        self._list_ingested_files,
                        outputs=ingested_dataset,
                    )
                    ingested_dataset.render()
                    deselect_file_button = gr.components.Button(
                        "De-select selected file", size="sm", interactive=False
                    )
                    selected_text = gr.components.Textbox(
                        "All files", label="Selected for Query or Deletion", max_lines=1
                    )
                    delete_file_button = gr.components.Button(
                        "🗑️ Delete selected file",
                        size="sm",
                        visible=settings().ui.delete_file_button_enabled,
                        interactive=False,
                    )
                    delete_files_button = gr.components.Button(
                        "⚠️ Delete ALL files",
                        size="sm",
                        visible=settings().ui.delete_all_files_button_enabled,
                    )
                    deselect_file_button.click(
                        self._deselect_selected_file,
                        outputs=[
                            delete_file_button,
                            deselect_file_button,
                            selected_text,
                        ],
                    )
                    ingested_dataset.select(
                        fn=self._selected_a_file,
                        outputs=[
                            delete_file_button,
                            deselect_file_button,
                            selected_text,
                        ],
                    )
                    delete_file_button.click(
                        self._delete_selected_file,
                        outputs=[
                            ingested_dataset,
                            delete_file_button,
                            deselect_file_button,
                            selected_text,
                        ],
                    )
                    delete_files_button.click(
                        self._delete_all_files,
                        outputs=[
                            ingested_dataset,
                            delete_file_button,
                            deselect_file_button,
                            selected_text,
                        ],
                    )
                    system_prompt_input = gr.Textbox(
                        placeholder=self._system_prompt,
                        label="System Prompt",
                        lines=2,
                        interactive=True,
                        render=False,
                    )
                    
                    # Bouton de déconnexion en bas de la section admin
                    with gr.Row():
                        logout_button = gr.Button("🚪 Déconnexion", size="sm", variant="secondary")
                    
                    # When mode changes, set default system prompt, and other stuffs
                    mode.change(
                        self._set_current_mode,
                        inputs=mode,
                        outputs=[system_prompt_input, explanation_mode],
                    )
                    # On blur, set system prompt to use in queries
                    system_prompt_input.blur(
                        self._set_system_prompt,
                        inputs=system_prompt_input,
                    )

                with gr.Column(scale=7, elem_id="col"):
                    # Utiliser le label approprié selon l'état d'authentification
                    chatbot_component = gr.Chatbot(
                        label=self._get_chatbot_label(),
                        show_copy_button=True,
                        elem_id="chatbot",
                        render=False,
                        avatar_images=(
                            None,
                            AVATAR_BOT,
                        ),
                    )

                    _ = gr.ChatInterface(
                        self._chat,
                        chatbot=chatbot_component,
                        additional_inputs=[mode, upload_button, system_prompt_input],
                    )
            
            # Événements d'authentification (à l'extérieur des sections)
            unlock_button.click(
                self._toggle_admin_access,
                inputs=[password_input],
                outputs=[auth_section, admin_section, unlock_button, password_input, chatbot_component]
            )
            
            # Authentification sur Enter dans le champ password
            password_input.submit(
                self._toggle_admin_access,
                inputs=[password_input], 
                outputs=[auth_section, admin_section, unlock_button, password_input, chatbot_component]
            )
            
            # Événement de déconnexion
            logout_button.click(
                self._logout_admin,
                outputs=[auth_section, admin_section, unlock_button, password_input, chatbot_component]
            )

            with gr.Row():
                avatar_byte = ALTORES_ICON.read_bytes()
                f_base64 = f"data:image/png;base64,{base64.b64encode(avatar_byte).decode('utf-8')}"
                gr.HTML(
                    f"<div class='footer'><a class='footer-zylon-link' href='https://altores.app'>Powered by Altores <img class='footer-zylon-ico' src='{f_base64}' alt=Zylon></a></div>"
                )

        return blocks

    def get_ui_blocks(self) -> gr.Blocks:
        if self._ui_block is None:
            self._ui_block = self._build_ui_blocks()
        return self._ui_block

    def mount_in_app(self, app: FastAPI, path: str) -> None:
        blocks = self.get_ui_blocks()
        blocks.queue()
        logger.info("Mounting the gradio UI, at path=%s", path)
        gr.mount_gradio_app(app, blocks, path=path, favicon_path=ALTORES_ICON)


if __name__ == "__main__":
    ui = global_injector.get(PrivateGptUi)
    _blocks = ui.get_ui_blocks()
    _blocks.queue()
    _blocks.launch(debug=False, show_api=False)
