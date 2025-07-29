import asyncio
import json
import base64
import litellm
from litellm import acompletion
from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig
from litellm.llms.bedrock.common_utils import BedrockModelInfo

# Enable debug logging
litellm._turn_on_debug()


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
        }
    ]

    # Read the image file
    with open('console2.png', 'rb') as f:
            png_bytes = f.read()
            # Convert to base64 for LiteLLM
            png_base64 = base64.b64encode(png_bytes).decode('utf-8')
    # except FileNotFoundError:
    #     print("console2.png not found, creating a placeholder...")
    #     # Create a minimal 1x1 PNG as placeholder
    #     png_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="

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