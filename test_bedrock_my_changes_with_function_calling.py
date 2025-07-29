import asyncio
import json
import base64
import logging
import time
import litellm
from litellm import acompletion
from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig
from litellm.llms.bedrock.common_utils import BedrockModelInfo
from litellm.integrations.custom_logger import CustomLogger

# Enable debug logging
litellm._turn_on_debug()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
logger.setLevel(logging.DEBUG)


class LiteLLMLXSCustomHandler(CustomLogger):
    def __init__(self):
        self.output_file = f"bedrock_input_payload_{int(time.time())}.json"

    def log_pre_api_call(self, model, messages, kwargs):
        # Get the input payload
        input_payload = kwargs["additional_args"]["complete_input_dict"]

        # If it's a JSON string, parse it to an object
        if isinstance(input_payload, str):
            try:
                input_payload = json.loads(input_payload)
            except json.JSONDecodeError:
                print("Warning: Could not parse input_payload as JSON")

        # Write to file
        try:
            with open(self.output_file, 'w') as f:
                json.dump(input_payload, f, indent=2, default=str)
            print(f"Input payload written to {self.output_file}")
        except Exception as e:
            print(f"Error writing to file: {e}")

    def log_post_api_call(self, kwargs, response_obj, start_time, end_time):
        pass


# Registering custom callback handlers
custom_handler = LiteLLMLXSCustomHandler()
litellm.callbacks = [custom_handler]
print(f"Input payload will be saved to: {custom_handler.output_file}")


async def test_bedrock_computer_use():
    # Test parameter support
    config = AmazonConverseConfig()
    model = "bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0"

    # Debug base model resolution
    base_model = BedrockModelInfo.get_base_model(model)
    print(f"Model: {model}")
    print(f"Base model: {base_model}")
    print(f"Base model starts with 'anthropic': {base_model.startswith('anthropic')}")

    supported_params = config.get_supported_openai_params(model)
    print(f"Supported params for {model}: {supported_params}")

    tools = [
        # Builtin tools - go to additionalModelRequestFields.tools
        {
            "type": "computer_20241022",
            "function": {
                "name": "computer",
                "parameters": {
                    "display_height_px": 768,
                    "display_width_px": 1024,
                    "display_number": 0,
                },
            },
        },
        {
            "type": "bash_20241022",
            "name": "bash",
        },
        {
            "type": "text_editor_20241022",
            "name": "str_replace_editor",
        },
        # Function tool - goes to toolConfig.tools with toolSpec
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            }
        }
    ]

    # Read the image file
    try:
        with open('console2.png', 'rb') as f:
            png_bytes = f.read()
            # Convert to base64 for LiteLLM
            png_base64 = base64.b64encode(png_bytes).decode('utf-8')
    except FileNotFoundError:
        print("console2.png not found, creating a placeholder...")
        # Create a minimal 1x1 PNG as placeholder
        png_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="

    messages = [
        {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': 'Go to the bedrock console and click purchase provisioned throughput'
                },
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': f'data:image/png;base64,{png_base64}'
                    }
                }
            ]
        }
    ]

    try:
        resp = await acompletion(
            model=model,
            messages=messages,
            tools=tools,
            # Note: anthropic_beta should be automatically added
        )

        print("Success! Response:")
        print(json.dumps(resp.model_dump(), indent=2))

    except Exception as e:
        print(f"Error: {e}")
        print(f"Error type: {type(e)}")


if __name__ == "__main__":
    asyncio.run(test_bedrock_computer_use())
    print(f"\nInput payload has been saved to: {custom_handler.output_file}")