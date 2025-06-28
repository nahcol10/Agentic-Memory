import tiktoken
from pydantic import create_model
import inspect
from inspect import Parameter
from typing import Callable,Dict,Any

class Utils:
    @staticmethod
    def count_number_of_tokens(text:str) -> int:
        """
        Count the number of tokens in text using GPT-4o-mini encoding.
        :param text: The text to tokenize
        :return: The number of tokens in text
        """
        encoding = tiktoken.encoding_for_model("gpt-4o-mini")
        tokens = encoding.encode(text)
        return len(tokens)

    @staticmethod
    def count_number_of_character(text:str) -> int:
        """
        Count the number of characters in a given text
        :param text: The text whose number character to count
        :return: The number of characters the given text
        """
        return len(text)

    @staticmethod
    def jsonschema(f: Callable) -> Dict[str,Any]:
        """
        Generate a JSON schema for the input parameters of the given function.
        Parameters:
        f (FunctionType): The function for which to generate the JSON schema.
        Returns:
        Dict: A dictionary containing the function name, description, and parameters schema.
        """
        kw = {n: (o.annotation, ... if o.default == Parameter.empty else o.default)
              for n, o in inspect.signature(f).parameters.items()}
        s = create_model(f'Input for `{f.__name__}`', **kw).model_json_schema()

        json_format = dict(
            type="function",
            function=dict(
                name=f.__name__,
                description=f.__doc__,
                parameters=s
            )
        )
        return json_format

# if __name__ == "__main__":
    # def add(a:int,b:int) -> int:
    #     """ Add two given numbers"""
    #     return a + b
    #
    # result = Utils.jsonschema(add)
    # print(result)

    # print(Utils.count_number_of_tokens("hello"))
    # print(Utils.count_number_of_character("hello"))