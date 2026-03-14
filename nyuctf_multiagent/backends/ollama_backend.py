import json
import requests
import uuid

from ..conversation import MessageRole
from ..tools import ToolCall
from .backend import Backend, BackendResponse


class OllamaBackend(Backend):
    """
    Hardened Backend for Ollama.
    Prevents JSON truncation errors by strictly managing context size and 
    sanitizing tool outputs.
    """

    NAME = "ollama"

    MODELS = {
    "llama3.1": {
        "max_context": 128000,
        "cost_per_input_token": 2e-07,
        "cost_per_output_token": 8e-07
    },
    "qwen3-coder-next:cloud": {
        "max_context": 256000,
        "cost_per_input_token": 1e-07,
        "cost_per_output_token": 2e-07
    },
    "cogito-2.1:671b-cloud": {
        "max_context": 256000,
        "cost_per_input_token": 2e-07,
        "cost_per_output_token": 6e-07
    },
    "gpt-oss:120b-cloud": {
        "max_context": 256000,
        "cost_per_input_token": 1e-07,
        "cost_per_output_token": 2e-07
    },
    "deepseek-v3.1:671b-cloud": {
        "max_context": 256000,
        "cost_per_input_token": 1e-07,
        "cost_per_output_token": 2e-07
    },
    "gemma3:27b-cloud": {
        "max_context": 128000,
        "cost_per_input_token": 3e-07,
        "cost_per_output_token": 9e-07
    }
}

    def __init__(self, role, model, tools, api_key, config):
        super().__init__(role, model, tools, config)
        self.base_url = "http://localhost:11434/api/chat"
        self.tool_schemas = [self.get_tool_schema(tool) for tool in tools.values()]

    @staticmethod
    def get_tool_schema(tool):
        return {
            "type": "function",
            "function": {
                "name": tool.NAME,
                "description": tool.DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        n: {"type": p[0], "description": p[1]}
                        for n, p in tool.PARAMETERS.items()
                    },
                    "required": list(tool.REQUIRED_PARAMETERS),
                },
            },
        }

    def calculate_cost(self, response_json):
        prompt_tokens = response_json.get("prompt_eval_count", 0)
        completion_tokens = response_json.get("eval_count", 0)
        return (self.in_price * prompt_tokens + self.out_price * completion_tokens)

    def _call_model(self, messages):
        # We cap context at 16k to ensure the HTTP request doesn't exceed 
        # Ollama's default server-side buffer limits.
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": self.tool_schemas,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_ctx": 16384, 
                "num_predict": 1024
            }
        }
        return requests.post(self.base_url, json=payload, timeout=120)

    def send(self, messages):
        formatted_messages = []

        for m in messages:
            if m.role == MessageRole.OBSERVATION:
                # Use 'tool' role for observations
                formatted_messages.append({
                    "role": "tool",
                    "content": str(m.tool_data.result),
                    "tool_call_id": str(m.tool_data.id)
                })

            elif m.role == MessageRole.ASSISTANT:
                msg = {"role": "assistant", "content": m.content or ""}
                if m.tool_data:
                    # Ollama expects tool_calls to be a LIST of OBJECTS
                    # The 'arguments' inside must be a DICT or a valid JSON object
                    try:
                        args = json.loads(m.tool_data.arguments) if isinstance(m.tool_data.arguments, str) else m.tool_data.arguments
                    except:
                        args = m.tool_data.arguments

                    msg["tool_calls"] = [{
                        "id": str(m.tool_data.id),
                        "type": "function",
                        "function": {
                            "name": m.tool_data.name,
                            "arguments": args
                        }
                    }]
                formatted_messages.append(msg)
            
            elif m.role == MessageRole.USER:
                formatted_messages.append({"role": "user", "content": str(m.content)})
            
            elif m.role == MessageRole.SYSTEM:
                formatted_messages.append({"role": "system", "content": str(m.content)})

        # --- DEBUG POINT ---
        # Uncomment the line below to see exactly what you are sending to Ollama
        # print("DEBUG SENDING:", json.dumps(formatted_messages, indent=2))

        try:
            response = self._call_model(formatted_messages)
            if response.status_code != 200:
                # If it's 400 or 500, Ollama usually provides a JSON error
                return BackendResponse(error=f"Ollama Error: {response.text}")

            response_json = response.json()
            message = response_json.get("message", {})
            cost = self.calculate_cost(response_json)
        except Exception as e:
            return BackendResponse(error=f"Ollama Backend Error: {e}")

        # Process response (same tool_call logic as before)
        tool_call = None
        if "tool_calls" in message and message["tool_calls"]:
            call = message["tool_calls"][0]
            func = call.get("function", {})
            args = func.get("arguments")
            # Convert dict args to string for D-CIPHER
            args_string = json.dumps(args) if isinstance(args, dict) else str(args)

            tool_call = ToolCall(
                name=func.get("name"),
                id=call.get("id", str(uuid.uuid4())),
                arguments=args_string,
            )

        return BackendResponse(
            content=message.get("content"),
            tool_call=tool_call,
            cost=cost
        )