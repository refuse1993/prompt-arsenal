"""
LLM Client for Multi-Turn Strategies
Wrapper for OpenAI, Anthropic, and other LLM providers
"""

import os
import asyncio
from typing import Optional, Dict, Any, List


class LLMClient:
    """
    Unified LLM client for strategy generation

    Supports:
    - OpenAI (GPT-4, GPT-3.5)
    - Anthropic (Claude)
    - Google (Gemini)
    """

    def __init__(self, provider: str, model: str, api_key: str = None, **kwargs):
        """
        Args:
            provider: 'openai', 'anthropic', 'google'
            model: Model name
            api_key: API key
            **kwargs: Additional provider settings
        """
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")
        self.config = kwargs

    async def generate(self, prompt: str, system: str = None, **kwargs) -> str:
        """
        Generate text from prompt

        Args:
            prompt: User prompt
            system: System prompt (optional)
            **kwargs: Provider-specific parameters

        Returns:
            Generated text
        """
        if self.provider == 'openai':
            return await self._call_openai(prompt, system, **kwargs)
        elif self.provider == 'anthropic':
            return await self._call_anthropic(prompt, system, **kwargs)
        elif self.provider == 'google':
            return await self._call_google(prompt, system, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    async def _call_openai(self, prompt: str, system: str = None, **kwargs) -> str:
        """Call OpenAI API"""
        try:
            import openai

            client = openai.AsyncOpenAI(api_key=self.api_key)

            # Build messages
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            # API call
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 2000)
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"OpenAI API error: {e}")
            return ""

    async def _call_anthropic(self, prompt: str, system: str = None, **kwargs) -> str:
        """Call Anthropic API"""
        try:
            import anthropic

            client = anthropic.AsyncAnthropic(api_key=self.api_key)

            # API call
            response = await client.messages.create(
                model=self.model,
                max_tokens=kwargs.get('max_tokens', 2000),
                system=system if system else anthropic.NOT_GIVEN,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=kwargs.get('temperature', 0.7)
            )

            return response.content[0].text

        except Exception as e:
            print(f"Anthropic API error: {e}")
            return ""

    async def _call_google(self, prompt: str, system: str = None, **kwargs) -> str:
        """Call Google Gemini API"""
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)

            # Create model
            model = genai.GenerativeModel(
                model_name=self.model,
                system_instruction=system if system else None
            )

            # Generate
            response = await asyncio.to_thread(
                model.generate_content,
                prompt,
                generation_config={
                    'temperature': kwargs.get('temperature', 0.7),
                    'max_output_tokens': kwargs.get('max_tokens', 2000)
                }
            )

            return response.text

        except Exception as e:
            print(f"Google API error: {e}")
            return ""


class MultimodalLLMClient(LLMClient):
    """
    Multimodal LLM client supporting images, audio, video

    Supports:
    - GPT-4 Vision
    - Claude 3 (Opus/Sonnet/Haiku)
    - Gemini Pro Vision
    """

    async def send(self, text: str) -> str:
        """
        Send text-only message (for compatibility)

        Args:
            text: Text message

        Returns:
            Response text
        """
        return await self.generate(text)

    async def generate_multimodal(
        self,
        text: str,
        images: List[str] = None,
        audio: List[str] = None,
        video: List[str] = None,
        system: str = None,
        **kwargs
    ) -> str:
        """
        Generate from multimodal input

        Args:
            text: Text prompt
            images: List of image file paths
            audio: List of audio file paths
            video: List of video file paths
            system: System prompt
            **kwargs: Provider-specific parameters

        Returns:
            Generated text
        """
        if self.provider == 'openai':
            return await self._call_openai_vision(text, images, system, **kwargs)
        elif self.provider == 'anthropic':
            return await self._call_anthropic_vision(text, images, system, **kwargs)
        elif self.provider == 'google':
            return await self._call_google_vision(text, images, system, **kwargs)
        else:
            raise ValueError(f"Multimodal not supported for provider: {self.provider}")

    async def _call_openai_vision(
        self,
        text: str,
        images: List[str] = None,
        system: str = None,
        **kwargs
    ) -> str:
        """Call OpenAI GPT-4 Vision"""
        try:
            import openai
            import base64

            client = openai.AsyncOpenAI(api_key=self.api_key)

            # Build messages
            messages = []
            if system:
                messages.append({"role": "system", "content": system})

            # Build multimodal content
            content = []

            # Add text
            content.append({"type": "text", "text": text})

            # Add images
            if images:
                for img_path in images:
                    # Read and encode image
                    with open(img_path, 'rb') as f:
                        img_data = base64.b64encode(f.read()).decode('utf-8')

                    # Detect image format
                    ext = os.path.splitext(img_path)[1].lower()
                    mime_type = {
                        '.jpg': 'image/jpeg',
                        '.jpeg': 'image/jpeg',
                        '.png': 'image/png',
                        '.gif': 'image/gif',
                        '.webp': 'image/webp'
                    }.get(ext, 'image/jpeg')

                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{img_data}"
                        }
                    })

            messages.append({"role": "user", "content": content})

            # API call
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 2000)
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"OpenAI Vision API error: {e}")
            return ""

    async def _call_anthropic_vision(
        self,
        text: str,
        images: List[str] = None,
        system: str = None,
        **kwargs
    ) -> str:
        """Call Anthropic Claude 3 with vision"""
        try:
            import anthropic
            import base64

            client = anthropic.AsyncAnthropic(api_key=self.api_key)

            # Build content
            content = []

            # Add images first
            if images:
                for img_path in images:
                    with open(img_path, 'rb') as f:
                        img_data = base64.b64encode(f.read()).decode('utf-8')

                    # Detect media type
                    ext = os.path.splitext(img_path)[1].lower()
                    media_type = {
                        '.jpg': 'image/jpeg',
                        '.jpeg': 'image/jpeg',
                        '.png': 'image/png',
                        '.gif': 'image/gif',
                        '.webp': 'image/webp'
                    }.get(ext, 'image/jpeg')

                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": img_data
                        }
                    })

            # Add text
            content.append({"type": "text", "text": text})

            # API call
            response = await client.messages.create(
                model=self.model,
                max_tokens=kwargs.get('max_tokens', 2000),
                system=system if system else anthropic.NOT_GIVEN,
                messages=[
                    {"role": "user", "content": content}
                ],
                temperature=kwargs.get('temperature', 0.7)
            )

            return response.content[0].text

        except Exception as e:
            print(f"Anthropic Vision API error: {e}")
            return ""

    async def _call_google_vision(
        self,
        text: str,
        images: List[str] = None,
        system: str = None,
        **kwargs
    ) -> str:
        """Call Google Gemini with vision"""
        try:
            import google.generativeai as genai
            from PIL import Image as PILImage

            genai.configure(api_key=self.api_key)

            # Create model
            model = genai.GenerativeModel(
                model_name=self.model,
                system_instruction=system if system else None
            )

            # Build content
            content = [text]

            # Add images
            if images:
                for img_path in images:
                    img = PILImage.open(img_path)
                    content.append(img)

            # Generate
            response = await asyncio.to_thread(
                model.generate_content,
                content,
                generation_config={
                    'temperature': kwargs.get('temperature', 0.7),
                    'max_output_tokens': kwargs.get('max_tokens', 2000)
                }
            )

            return response.text

        except Exception as e:
            print(f"Google Vision API error: {e}")
            return ""
