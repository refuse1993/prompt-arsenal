"""
Audio Generator
Supports OpenAI TTS and other audio generation APIs
"""

import os
import asyncio
import aiohttp
from typing import Optional, Dict, Any
from datetime import datetime


class AudioGenerator:
    """
    Unified audio generator supporting multiple backends
    """

    def __init__(self, provider: str, api_key: str = None, model: str = None, **kwargs):
        """
        Args:
            provider: 'openai', 'google', 'elevenlabs', 'azure'
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
            if self.provider == 'openai':
                self.model = 'tts-1'  # tts-1, tts-1-hd
            elif self.provider == 'google':
                self.model = 'en-US-Neural2-C'  # Google Cloud TTS voice

    async def generate(self, text: str, output_path: str, **kwargs) -> Optional[str]:
        """
        Generate audio from text

        Args:
            text: Text to convert to speech
            output_path: Where to save the generated audio
            **kwargs: Provider-specific parameters

        Returns:
            Path to generated audio file (or None on failure)
        """
        if self.provider == 'openai':
            return await self._generate_openai(text, output_path, **kwargs)
        elif self.provider == 'google':
            return await self._generate_google(text, output_path, **kwargs)
        elif self.provider == 'elevenlabs':
            return await self._generate_elevenlabs(text, output_path, **kwargs)
        elif self.provider == 'azure':
            return await self._generate_azure(text, output_path, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    async def _generate_openai(self, text: str, output_path: str, **kwargs) -> Optional[str]:
        """
        Generate audio using OpenAI TTS

        Supports: tts-1, tts-1-hd
        Voices: alloy, echo, fable, onyx, nova, shimmer
        """
        try:
            import openai

            # Create async OpenAI client
            client = openai.AsyncOpenAI(api_key=self.api_key)

            # TTS parameters
            voice = kwargs.get('voice', 'alloy')  # alloy, echo, fable, onyx, nova, shimmer
            response_format = kwargs.get('response_format', 'mp3')  # mp3, opus, aac, flac
            speed = kwargs.get('speed', 1.0)  # 0.25 to 4.0

            # Generate audio
            response = await client.audio.speech.create(
                model=self.model,
                voice=voice,
                input=text,
                response_format=response_format,
                speed=speed
            )

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save audio
            await response.astream_to_file(output_path)

            return output_path

        except Exception as e:
            print(f"OpenAI TTS generation failed: {e}")
            return None

    async def _generate_google(self, text: str, output_path: str, **kwargs) -> Optional[str]:
        """
        Generate audio using Google Cloud Text-to-Speech

        Supports: Neural2 voices, Wavenet voices, Standard voices
        """
        try:
            from google.cloud import texttospeech
            import asyncio

            # Create client (synchronous)
            def _generate():
                client = texttospeech.TextToSpeechClient()

                # Set input text
                synthesis_input = texttospeech.SynthesisInput(text=text)

                # Configure voice
                voice = texttospeech.VoiceSelectionParams(
                    language_code=kwargs.get('language_code', 'en-US'),
                    name=self.model
                )

                # Configure audio
                audio_config = texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.MP3,
                    speaking_rate=kwargs.get('speaking_rate', 1.0),
                    pitch=kwargs.get('pitch', 0.0)
                )

                # Generate audio
                response = client.synthesize_speech(
                    input=synthesis_input,
                    voice=voice,
                    audio_config=audio_config
                )

                # Save audio
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'wb') as f:
                    f.write(response.audio_content)

                return output_path

            # Run in executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, _generate)

        except Exception as e:
            print(f"Google TTS generation failed: {e}")
            return None

    async def _generate_elevenlabs(self, text: str, output_path: str, **kwargs) -> Optional[str]:
        """
        Generate audio using ElevenLabs API

        High-quality voice synthesis
        """
        try:
            # ElevenLabs API endpoint
            voice_id = kwargs.get('voice_id', '21m00Tcm4TlvDq8ikWAM')  # Default voice
            api_url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.api_key
            }

            data = {
                "text": text,
                "model_id": self.model or "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": kwargs.get('stability', 0.5),
                    "similarity_boost": kwargs.get('similarity_boost', 0.5)
                }
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(api_url, json=data, headers=headers) as resp:
                    if resp.status == 200:
                        # Ensure output directory exists
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)

                        # Save audio
                        with open(output_path, 'wb') as f:
                            f.write(await resp.read())

                        return output_path
                    else:
                        error = await resp.text()
                        print(f"ElevenLabs API error: {error}")
                        return None

        except Exception as e:
            print(f"ElevenLabs generation failed: {e}")
            return None

    async def _generate_azure(self, text: str, output_path: str, **kwargs) -> Optional[str]:
        """
        Generate audio using Azure Cognitive Services TTS

        Microsoft's text-to-speech service
        """
        try:
            # Azure TTS endpoint
            region = self.config.get('region', 'eastus')
            endpoint = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"

            headers = {
                "Ocp-Apim-Subscription-Key": self.api_key,
                "Content-Type": "application/ssml+xml",
                "X-Microsoft-OutputFormat": "audio-16khz-128kbitrate-mono-mp3"
            }

            # SSML format
            voice_name = kwargs.get('voice', 'en-US-JennyNeural')
            ssml = f"""
            <speak version='1.0' xml:lang='en-US'>
                <voice xml:lang='en-US' name='{voice_name}'>
                    {text}
                </voice>
            </speak>
            """

            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, data=ssml.encode('utf-8'), headers=headers) as resp:
                    if resp.status == 200:
                        # Ensure output directory exists
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)

                        # Save audio
                        with open(output_path, 'wb') as f:
                            f.write(await resp.read())

                        return output_path
                    else:
                        error = await resp.text()
                        print(f"Azure TTS API error: {error}")
                        return None

        except Exception as e:
            print(f"Azure TTS generation failed: {e}")
            return None
