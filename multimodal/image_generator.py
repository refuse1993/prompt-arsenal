"""
Image Generator
Supports DALL-E, Stable Diffusion, and other image generation APIs
"""

import os
import base64
import asyncio
import aiohttp
from typing import Optional, Dict, Any
from datetime import datetime


class ImageGenerator:
    """
    Unified image generator supporting multiple backends
    """

    def __init__(self, provider: str, api_key: str = None, model: str = None, **kwargs):
        """
        Args:
            provider: 'dalle', 'stable-diffusion', 'replicate'
            api_key: API key for the service
            model: Model name (provider-specific)
            **kwargs: Additional provider-specific settings
        """
        self.provider = provider.lower()
        self.api_key = api_key
        self.model = model
        self.config = kwargs

        # Default models
        if not self.model:
            if self.provider == 'dalle':
                self.model = 'dall-e-3'
            elif self.provider == 'stable-diffusion':
                self.model = 'stable-diffusion-xl-1024-v1-0'

    async def generate(self, prompt: str, output_path: str, **kwargs) -> Optional[str]:
        """
        Generate image from text prompt

        Args:
            prompt: Text description of image
            output_path: Where to save the generated image
            **kwargs: Provider-specific parameters

        Returns:
            Path to generated image file (or None on failure)
        """
        if self.provider == 'dalle':
            return await self._generate_dalle(prompt, output_path, **kwargs)
        elif self.provider == 'stable-diffusion':
            return await self._generate_stable_diffusion(prompt, output_path, **kwargs)
        elif self.provider == 'replicate':
            return await self._generate_replicate(prompt, output_path, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    async def _generate_dalle(self, prompt: str, output_path: str, **kwargs) -> Optional[str]:
        """
        Generate image using OpenAI DALL-E

        Supports: DALL-E 3, DALL-E 2
        """
        try:
            import openai

            # Create async OpenAI client
            client = openai.AsyncOpenAI(api_key=self.api_key)

            # DALL-E 3 parameters
            size = kwargs.get('size', '1024x1024')  # 1024x1024, 1792x1024, 1024x1792
            quality = kwargs.get('quality', 'standard')  # standard, hd
            style = kwargs.get('style', 'natural')  # natural, vivid

            # Generate image
            response = await client.images.generate(
                model=self.model,
                prompt=prompt,
                size=size,
                quality=quality,
                style=style,
                n=1
            )

            # Get image URL
            image_url = response.data[0].url

            # Download and save image
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as resp:
                    if resp.status == 200:
                        # Ensure output directory exists
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)

                        # Save image
                        with open(output_path, 'wb') as f:
                            f.write(await resp.read())

                        return output_path

            return None

        except Exception as e:
            print(f"DALL-E generation failed: {e}")
            return None

    async def _generate_stable_diffusion(self, prompt: str, output_path: str, **kwargs) -> Optional[str]:
        """
        Generate image using Stable Diffusion

        Supports:
        - Stability AI API
        - Local Stable Diffusion (via API endpoint)
        """
        try:
            # Stability AI API endpoint
            api_host = self.config.get('api_host', 'https://api.stability.ai')
            engine_id = self.model

            # Request parameters
            steps = kwargs.get('steps', 30)
            cfg_scale = kwargs.get('cfg_scale', 7.0)
            width = kwargs.get('width', 1024)
            height = kwargs.get('height', 1024)
            samples = kwargs.get('samples', 1)

            # API request
            url = f"{api_host}/v1/generation/{engine_id}/text-to-image"

            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            data = {
                "text_prompts": [
                    {
                        "text": prompt,
                        "weight": 1.0
                    }
                ],
                "cfg_scale": cfg_scale,
                "height": height,
                "width": width,
                "samples": samples,
                "steps": steps
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data, headers=headers) as resp:
                    if resp.status == 200:
                        result = await resp.json()

                        # Get image data (base64 encoded)
                        image_data = result['artifacts'][0]['base64']

                        # Decode and save
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        with open(output_path, 'wb') as f:
                            f.write(base64.b64decode(image_data))

                        return output_path
                    else:
                        error = await resp.text()
                        print(f"Stable Diffusion API error: {error}")
                        return None

        except Exception as e:
            print(f"Stable Diffusion generation failed: {e}")
            return None

    async def _generate_replicate(self, prompt: str, output_path: str, **kwargs) -> Optional[str]:
        """
        Generate image using Replicate API

        Supports various SD models on Replicate
        """
        try:
            # Replicate API
            url = "https://api.replicate.com/v1/predictions"

            headers = {
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "application/json"
            }

            # Model version (SDXL by default)
            version = self.config.get(
                'version',
                'stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b'
            )

            data = {
                "version": version,
                "input": {
                    "prompt": prompt,
                    "width": kwargs.get('width', 1024),
                    "height": kwargs.get('height', 1024),
                    "num_inference_steps": kwargs.get('steps', 30)
                }
            }

            async with aiohttp.ClientSession() as session:
                # Create prediction
                async with session.post(url, json=data, headers=headers) as resp:
                    if resp.status != 201:
                        error = await resp.text()
                        print(f"Replicate API error: {error}")
                        return None

                    prediction = await resp.json()
                    prediction_id = prediction['id']

                # Poll for result
                while True:
                    async with session.get(
                        f"{url}/{prediction_id}",
                        headers=headers
                    ) as resp:
                        result = await resp.json()

                        if result['status'] == 'succeeded':
                            # Get output image URL
                            output_url = result['output'][0]

                            # Download image
                            async with session.get(output_url) as img_resp:
                                if img_resp.status == 200:
                                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                                    with open(output_path, 'wb') as f:
                                        f.write(await img_resp.read())
                                    return output_path

                        elif result['status'] == 'failed':
                            print(f"Replicate generation failed: {result.get('error')}")
                            return None

                        # Wait before polling again
                        await asyncio.sleep(1)

        except Exception as e:
            print(f"Replicate generation failed: {e}")
            return None
