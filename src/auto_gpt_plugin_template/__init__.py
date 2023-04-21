"""This is a template for Auto-GPT plugins."""
import abc
from llama_cpp import Llama
from llm_utils import create_chat_completion

from typing import Any, Dict, List, Optional, Tuple, TypeVar, TypedDict

from abstract_singleton import AbstractSingleton, Singleton

PromptGenerator = TypeVar("PromptGenerator")

llm = Llama(model_path="path/to/llm")


class Message(TypedDict):
    role: str
    content: str


class AutoGPTPluginTemplate(AbstractSingleton, metaclass=Singleton):
    """
    This is an attepmpt to make Auto-GPT use local LLMs .
    """

    def __init__(self,model_path, n_ctx=512, n_parts=-1, seed=1337, f16_kv=True, logits_all=False, vocab_only=False, use_mmap=True, use_mlock=False, embedding=False, n_threads=None, n_batch=8, last_n_tokens_size=64, verbose=True):
        super().__init__()
        self._name = "Auto-GPT-local-llm-plugin"
        self._version = "0.0.1"
        self._description = "This is an Auto-GPT plugin that allows using local LLMs with help of alpaca.cpp."
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_parts=n_parts, seed=seed, f16_kv=f16_kv, logits_all=logits_all, vocab_only=vocab_only, use_mmap=use_mmap, use_mlock=use_mlock, embedding=embedding, n_threads=n_threads, n_batch=n_batch, last_n_tokens_size=last_n_tokens_size, verbose=verbose)
    
    
        
        


    @abc.abstractmethod
    def can_handle_on_response(self) -> bool:
        """This method is called to check that the plugin can
        handle the on_response method.

        Returns:
            bool: True if the plugin can handle the on_response method."""
        return False

    @abc.abstractmethod
    def on_response(self, response: str, *args, **kwargs) -> str:
        """This method is called when a response is received from the model."""
        pass

    @abc.abstractmethod
    def can_handle_post_prompt(self) -> bool:
        """This method is called to check that the plugin can
        handle the post_prompt method.

        Returns:
            bool: True if the plugin can handle the post_prompt method."""
        return False

    @abc.abstractmethod
    def post_prompt(self, prompt: PromptGenerator) -> PromptGenerator:
        """This method is called just after the generate_prompt is called,
            but actually before the prompt is generated.

        Args:
            prompt (PromptGenerator): The prompt generator.

        Returns:
            PromptGenerator: The prompt generator.
        """
        pass

    @abc.abstractmethod
    def can_handle_on_planning(self) -> bool:
        """This method is called to check that the plugin can
        handle the on_planning method.

        Returns:
            bool: True if the plugin can handle the on_planning method."""
        return False

    @abc.abstractmethod
    def on_planning(
        self, prompt: PromptGenerator, messages: List[Message]
    ) -> Optional[str]:
        """This method is called before the planning chat completion is done.

        Args:
            prompt (PromptGenerator): The prompt generator.
            messages (List[str]): The list of messages.
        """
        pass

    @abc.abstractmethod
    def can_handle_post_planning(self) -> bool:
        """This method is called to check that the plugin can
        handle the post_planning method.

        Returns:
            bool: True if the plugin can handle the post_planning method."""
        return False

    @abc.abstractmethod
    def post_planning(self, response: str) -> str:
        """This method is called after the planning chat completion is done.

        Args:
            response (str): The response.

        Returns:
            str: The resulting response.
        """
        pass

    @abc.abstractmethod
    def can_handle_pre_instruction(self) -> bool:
        """This method is called to check that the plugin can
        handle the pre_instruction method.

        Returns:
            bool: True if the plugin can handle the pre_instruction method."""
        return False

    @abc.abstractmethod
    def pre_instruction(self, messages: List[Message]) -> List[Message]:
        """This method is called before the instruction chat is done.

        Args:
            messages (List[Message]): The list of context messages.

        Returns:
            List[Message]: The resulting list of messages.
        """
        pass

    @abc.abstractmethod
    def can_handle_on_instruction(self) -> bool:
        """This method is called to check that the plugin can
        handle the on_instruction method.

        Returns:
            bool: True if the plugin can handle the on_instruction method."""
        return False

    @abc.abstractmethod
    def on_instruction(self, messages: List[Message]) -> Optional[str]:
        """This method is called when the instruction chat is done.

        Args:
            messages (List[Message]): The list of context messages.

        Returns:
            Optional[str]: The resulting message.
        """
        pass

    @abc.abstractmethod
    def can_handle_post_instruction(self) -> bool:
        """This method is called to check that the plugin can
        handle the post_instruction method.

        Returns:
            bool: True if the plugin can handle the post_instruction method."""
        return False

    @abc.abstractmethod
    def post_instruction(self, response: str) -> str:
        """This method is called after the instruction chat is done.

        Args:
            response (str): The response.

        Returns:
            str: The resulting response.
        """
        pass

    @abc.abstractmethod
    def can_handle_pre_command(self) -> bool:
        """This method is called to check that the plugin can
        handle the pre_command method.

        Returns:
            bool: True if the plugin can handle the pre_command method."""
        return False

    @abc.abstractmethod
    def pre_command(
        self, command_name: str, arguments: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """This method is called before the command is executed.

        Args:
            command_name (str): The command name.
            arguments (Dict[str, Any]): The arguments.

        Returns:
            Tuple[str, Dict[str, Any]]: The command name and the arguments.
        """
        pass

    @abc.abstractmethod
    def can_handle_post_command(self) -> bool:
        """This method is called to check that the plugin can
        handle the post_command method.

        Returns:
            bool: True if the plugin can handle the post_command method."""
        return False

    @abc.abstractmethod
    def post_command(self, command_name: str, response: str) -> str:
        """This method is called after the command is executed.

        Args:
            command_name (str): The command name.
            response (str): The response.

        Returns:
            str: The resulting response.
        """
        pass

    @abc.abstractmethod
    def can_handle_chat_completion(
        self, messages: Dict[Any, Any], model: str, temperature: float, max_tokens: int
    ) -> bool:
        """This method is called to check that the plugin can
          handle the chat_completion method.

        Args:
            messages (List[Message]): The messages.
            model (str): The model name.
            temperature (float): The temperature.
            max_tokens (int): The max tokens.

          Returns:
              bool: True if the plugin can handle the chat_completion method."""
        return False

    @abc.abstractmethod
    def handle_chat_completion(
        self, messages: List[Message], model: None, temperature: float, max_tokens: int
    ) -> str:
        """This method is called when the chat completion is done.

        Args:
            messages (List[Message]): The messages.
            model (str): The model name.
            temperature (float): The temperature.
            max_tokens (int): The max tokens.

        Returns:
            str: The resulting response.
        """
        if model is None:
            model = llm
            # For each arg, if any are None, convert to "None":
            args = [str(arg) if arg is not None else "None" for arg in args]
            # parse args to comma seperated string
            args = ", ".join(args)
            messages = [
            {
                "role": "system",
                "content": f"You are now the following python function: ```# {description}\n{function}```\n\nOnly respond with your `return` value.",
            },
            {"role": "user", "content": args},
        ]

        response = create_chat_completion(
            model=model, messages=messages, temperature=0
        )

        return response



      
