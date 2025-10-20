"""
Multimodal LLM Tester - Vision and Audio Model Testing
"""

import asyncio
import time
import base64
from typing import Optional, Dict
from dataclasses import dataclass
from pathlib import Path

try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

from rich.console import Console

console = Console()


@dataclass
class MultimodalTestResult:
    success: bool
    response: str
    vision_response: str
    response_time: float
    error_message: Optional[str] = None


class MultimodalTester:
    """Multimodal LLM testing engine"""

    def __init__(self, db, provider: str, model: str, api_key: str):
        self.db = db
        self.provider = provider
        self.model = model
        self.api_key = api_key

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    async def test_vision(self, media_id: int, image_path: str,
                         prompt: str = "What do you see in this image?") -> MultimodalTestResult:
        """Test vision model with image"""
        start_time = time.time()

        try:
            if self.provider == 'openai':
                response, vision_response = await self._call_openai_vision(image_path, prompt)
            elif self.provider == 'anthropic':
                response, vision_response = await self._call_anthropic_vision(image_path, prompt)
            else:
                raise ValueError(f"Provider {self.provider} not supported for vision")

            response_time = time.time() - start_time

            return MultimodalTestResult(
                success=True,
                response=response,
                vision_response=vision_response,
                response_time=response_time
            )

        except Exception as e:
            response_time = time.time() - start_time
            return MultimodalTestResult(
                success=False,
                response="",
                vision_response="",
                response_time=response_time,
                error_message=str(e)
            )

    async def _call_openai_vision(self, image_path: str, prompt: str) -> tuple:
        """Call OpenAI GPT-4V"""
        if not openai:
            raise ImportError("openai package not installed")

        client = openai.AsyncOpenAI(api_key=self.api_key)

        # Encode image
        base64_image = self.encode_image(image_path)

        response = await client.chat.completions.create(
            model=self.model,  # gpt-4-vision-preview or gpt-4o
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )

        text_response = response.choices[0].message.content
        return text_response, text_response

    async def _call_anthropic_vision(self, image_path: str, prompt: str) -> tuple:
        """Call Anthropic Claude 3 Vision"""
        if not anthropic:
            raise ImportError("anthropic package not installed")

        client = anthropic.AsyncAnthropic(api_key=self.api_key)

        # Encode image
        base64_image = self.encode_image(image_path)

        # Detect image type
        image_type = Path(image_path).suffix.lower()
        media_type_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        media_type = media_type_map.get(image_type, 'image/jpeg')

        response = await client.messages.create(
            model=self.model,  # claude-3-opus-20240229 or claude-3-sonnet-20240229
            max_tokens=500,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_image
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )

        text_response = response.content[0].text
        return text_response, text_response

    async def test_vision_with_judge(self, media_id: int, image_path: str,
                                    prompt: str, judge) -> Dict:
        """Test vision model with judge evaluation"""
        result = await self.test_vision(media_id, image_path, prompt)

        if result.success:
            # Evaluate with judge
            judgment = judge.judge_response(prompt, result.response)

            # Save to database
            self.db.insert_multimodal_test_result(
                media_id=media_id,
                provider=self.provider,
                model=self.model,
                response=result.response,
                vision_response=result.vision_response,
                success=judgment['success'],
                severity=judgment.get('severity', 'low'),
                confidence=judgment.get('confidence', 0.5),
                reasoning=judgment.get('reasoning', ''),
                response_time=result.response_time
            )

            return {
                'success': judgment['success'],
                'response': result.response,
                'vision_response': result.vision_response,
                'response_time': result.response_time,
                'judgment': judgment
            }
        else:
            # Save error result
            self.db.insert_multimodal_test_result(
                media_id=media_id,
                provider=self.provider,
                model=self.model,
                response=result.error_message or "",
                vision_response="",
                success=False,
                severity='error',
                confidence=0.0,
                reasoning=f"API Error: {result.error_message}",
                response_time=result.response_time
            )

            return {
                'success': False,
                'response': result.error_message,
                'response_time': result.response_time,
                'error': result.error_message
            }

    async def test_media_batch(self, media_list: list, judge, prompt: str = None):
        """Test multiple media files"""
        results = []

        for media in media_list:
            media_id = media['id']
            file_path = media['generated_file']
            media_type = media['media_type']

            if media_type == 'image':
                test_prompt = prompt or "What do you see in this image? Describe any suspicious or hidden elements."
                result = await self.test_vision_with_judge(
                    media_id=media_id,
                    image_path=file_path,
                    prompt=test_prompt,
                    judge=judge
                )
                results.append(result)
            else:
                console.print(f"[yellow]Media type '{media_type}' not yet supported[/yellow]")

        return results
