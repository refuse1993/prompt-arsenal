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
        elif self.provider == 'ollama':
            return await self._call_ollama(prompt, system, **kwargs)
        elif self.provider == 'cohere':
            return await self._call_cohere(prompt, system, **kwargs)
        elif self.provider == 'together':
            return await self._call_together(prompt, system, **kwargs)
        elif self.provider == 'huggingface':
            return await self._call_huggingface(prompt, system, **kwargs)
        elif self.provider == 'replicate':
            return await self._call_replicate(prompt, system, **kwargs)
        elif self.provider == 'local':
            return await self._call_local(prompt, system, **kwargs)
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

    async def _call_ollama(self, prompt: str, system: str = None, **kwargs) -> str:
        """Call Ollama (local) API"""
        try:
            import aiohttp

            base_url = self.config.get('base_url', 'http://localhost:11434')

            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "temperature": kwargs.get('temperature', 0.7),
                            "num_predict": kwargs.get('max_tokens', 2000)
                        }
                    }
                ) as response:
                    result = await response.json()
                    return result['message']['content']

        except Exception as e:
            print(f"Ollama API error: {e}")
            return ""

    async def _call_cohere(self, prompt: str, system: str = None, **kwargs) -> str:
        """Call Cohere API"""
        try:
            import cohere

            client = cohere.AsyncClient(api_key=self.api_key)

            # Cohere uses preamble instead of system
            response = await client.chat(
                model=self.model,
                message=prompt,
                preamble=system if system else None,
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 2000)
            )

            return response.text

        except Exception as e:
            print(f"Cohere API error: {e}")
            return ""

    async def _call_together(self, prompt: str, system: str = None, **kwargs) -> str:
        """Call Together AI (OpenAI-compatible)"""
        try:
            import openai

            client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url="https://api.together.xyz/v1"
            )

            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 2000)
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"Together AI error: {e}")
            return ""

    async def _call_huggingface(self, prompt: str, system: str = None, **kwargs) -> str:
        """Call Hugging Face Inference API"""
        try:
            import aiohttp

            # Combine system and user prompt
            full_prompt = f"{system}\n\n{prompt}" if system else prompt

            headers = {"Authorization": f"Bearer {self.api_key}"}

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"https://api-inference.huggingface.co/models/{self.model}",
                    headers=headers,
                    json={
                        "inputs": full_prompt,
                        "parameters": {
                            "temperature": kwargs.get('temperature', 0.7),
                            "max_new_tokens": kwargs.get('max_tokens', 2000)
                        }
                    }
                ) as response:
                    result = await response.json()
                    if isinstance(result, list) and len(result) > 0:
                        return result[0].get('generated_text', '')
                    return ""

        except Exception as e:
            print(f"Hugging Face API error: {e}")
            return ""

    async def _call_replicate(self, prompt: str, system: str = None, **kwargs) -> str:
        """Call Replicate API"""
        try:
            import replicate

            # Combine system and user prompt
            full_prompt = f"{system}\n\n{prompt}" if system else prompt

            # Run model
            output = await asyncio.to_thread(
                replicate.run,
                self.model,
                input={
                    "prompt": full_prompt,
                    "temperature": kwargs.get('temperature', 0.7),
                    "max_length": kwargs.get('max_tokens', 2000)
                }
            )

            # Handle different output formats
            if isinstance(output, str):
                return output
            elif isinstance(output, list):
                return ''.join(output)
            else:
                return str(output)

        except Exception as e:
            print(f"Replicate API error: {e}")
            return ""

    async def _call_local(self, prompt: str, system: str = None, **kwargs) -> str:
        """Call local OpenAI-compatible API"""
        try:
            import openai

            base_url = self.config.get('base_url', 'http://localhost:8000/v1')

            client = openai.AsyncOpenAI(
                api_key=self.api_key or "dummy-key",
                base_url=base_url
            )

            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 2000)
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"Local API error: {e}")
            return ""


class MultimodalLLMClient(LLMClient):
    """
    Multimodal LLM client supporting images, audio, video

    Supports:
    - GPT-4 Vision
    - Claude 3 (Opus/Sonnet/Haiku)
    - Gemini Pro Vision
    """

    def __init__(self, provider: str, model: str, api_key: str, base_url: str = None):
        super().__init__(provider, model, api_key, base_url=base_url)
        self.conversation_history = []  # Track conversation for multi-turn

    async def send(self, text: str) -> str:
        """
        Send text-only message (for compatibility)

        Args:
            text: Text message

        Returns:
            Response text
        """
        return await self.generate(text)

    def add_to_history(self, role: str, content: str):
        """Add message to conversation history"""
        self.conversation_history.append({"role": role, "content": content})

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []

    def get_history(self) -> List[Dict]:
        """Get conversation history"""
        return self.conversation_history.copy()

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
        """Call OpenAI GPT-4 Vision with conversation history"""
        try:
            import openai
            import base64

            client = openai.AsyncOpenAI(api_key=self.api_key)

            # Build messages from conversation history
            messages = []

            # Add system message
            if system:
                messages.append({"role": "system", "content": system})

            # Add conversation history (excluding system messages)
            for msg in self.conversation_history:
                if msg.get("role") != "system":
                    messages.append(msg)

            # Build multimodal content for current turn
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

            # Add current user message
            messages.append({"role": "user", "content": content})

            # API call
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 2000)
            )

            response_text = response.choices[0].message.content

            # Add to conversation history
            self.conversation_history.append({"role": "user", "content": content})
            self.conversation_history.append({"role": "assistant", "content": response_text})

            return response_text

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
        """Call Anthropic Claude 3 with vision and conversation history"""
        try:
            import anthropic
            import base64

            client = anthropic.AsyncAnthropic(api_key=self.api_key)

            # Build messages from conversation history
            messages = []

            # Add conversation history (excluding system messages)
            for msg in self.conversation_history:
                if msg.get("role") != "system":
                    messages.append(msg)

            # Build multimodal content for current turn
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

            # Add current user message
            messages.append({"role": "user", "content": content})

            # API call
            response = await client.messages.create(
                model=self.model,
                max_tokens=kwargs.get('max_tokens', 2000),
                system=system if system else anthropic.NOT_GIVEN,
                messages=messages,
                temperature=kwargs.get('temperature', 0.7)
            )

            response_text = response.content[0].text

            # Add to conversation history
            self.conversation_history.append({"role": "user", "content": content})
            self.conversation_history.append({"role": "assistant", "content": response_text})

            return response_text

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
        """Call Google Gemini with vision and conversation history"""
        try:
            import google.generativeai as genai
            from PIL import Image as PILImage

            genai.configure(api_key=self.api_key)

            # Create model
            model = genai.GenerativeModel(
                model_name=self.model,
                system_instruction=system if system else None
            )

            # Convert conversation history to Gemini format
            gemini_history = []
            for msg in self.conversation_history:
                role = msg.get("role")
                content = msg.get("content")

                # Convert role name (Gemini uses "model" instead of "assistant")
                gemini_role = "model" if role == "assistant" else role

                # Handle different content formats
                if isinstance(content, str):
                    # Simple text message
                    gemini_history.append({
                        "role": gemini_role,
                        "parts": [content]
                    })
                elif isinstance(content, list):
                    # Multimodal content (from OpenAI/Anthropic format)
                    parts = []
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                parts.append(item.get("text", ""))
                            # Note: Images are already in history as base64,
                            # Gemini doesn't support this in history reconstruction
                            # Skip image parts in history
                        elif isinstance(item, str):
                            parts.append(item)
                    if parts:
                        gemini_history.append({
                            "role": gemini_role,
                            "parts": parts
                        })

            # Build content for current turn
            content = [text]

            # Add images
            if images:
                for img_path in images:
                    img = PILImage.open(img_path)
                    content.append(img)

            # Start chat with history
            chat = model.start_chat(history=gemini_history)

            # Send current message
            response = await asyncio.to_thread(
                chat.send_message,
                content,
                generation_config={
                    'temperature': kwargs.get('temperature', 0.7),
                    'max_output_tokens': kwargs.get('max_tokens', 2000)
                }
            )

            response_text = response.text

            # Add to conversation history (store in unified format)
            # For images, we store the text + image reference
            user_content = [text]
            if images:
                user_content.extend([f"[Image: {os.path.basename(img)}]" for img in images])

            self.conversation_history.append({"role": "user", "content": " ".join(user_content)})
            self.conversation_history.append({"role": "assistant", "content": response_text})

            return response_text

        except Exception as e:
            print(f"Google Vision API error: {e}")
            return ""
