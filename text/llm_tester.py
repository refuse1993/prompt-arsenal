"""
LLM Tester - Text Prompt Testing
"""

import asyncio
import time
from typing import Optional, List, Dict
from dataclasses import dataclass

try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None

try:
    import cohere
except ImportError:
    cohere = None

import aiohttp
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


@dataclass
class TestResult:
    success: bool
    response: str
    response_time: float
    error_message: Optional[str] = None


class LLMTester:
    """LLM testing engine for text prompts"""

    def __init__(self, db, provider: str, model: str, api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        self.db = db
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    async def test_prompt(self, prompt: str) -> TestResult:
        """Test single prompt"""
        start_time = time.time()

        try:
            if self.provider == 'openai':
                response = await self._call_openai(prompt)
            elif self.provider == 'anthropic':
                response = await self._call_anthropic(prompt)
            elif self.provider == 'google':
                response = await self._call_google(prompt)
            elif self.provider == 'xai':
                response = await self._call_xai(prompt)
            elif self.provider == 'local':
                response = await self._call_local(prompt)
            elif self.provider == 'huggingface':
                response = await self._call_huggingface(prompt)
            elif self.provider == 'ollama':
                response = await self._call_ollama(prompt)
            elif self.provider == 'together':
                response = await self._call_together(prompt)
            elif self.provider == 'replicate':
                response = await self._call_replicate(prompt)
            elif self.provider == 'cohere':
                response = await self._call_cohere(prompt)
            else:
                raise ValueError(f"Unknown provider: {self.provider}")

            response_time = time.time() - start_time

            return TestResult(
                success=True,
                response=response,
                response_time=response_time
            )

        except Exception as e:
            response_time = time.time() - start_time
            return TestResult(
                success=False,
                response="",
                response_time=response_time,
                error_message=str(e)
            )

    def _get_openai_token_param(self, max_tokens: int) -> dict:
        """Get appropriate token parameter for OpenAI model"""
        # GPT-5, o1 series use max_completion_tokens
        if self.model.startswith(('gpt-5', 'o1')):
            return {"max_completion_tokens": max_tokens}
        # Older models use max_tokens
        else:
            return {"max_tokens": max_tokens}

    async def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API"""
        if not openai:
            raise ImportError("openai package not installed")

        client = openai.AsyncOpenAI(api_key=self.api_key)

        # Get appropriate token parameter
        token_param = self._get_openai_token_param(500)

        response = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            **token_param
        )

        return response.choices[0].message.content

    async def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API"""
        if not anthropic:
            raise ImportError("anthropic package not installed")

        client = anthropic.AsyncAnthropic(api_key=self.api_key)

        response = await client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text

    async def _call_google(self, prompt: str) -> str:
        """Call Google Gemini API"""
        if not genai:
            raise ImportError("google-generativeai package not installed. Install with: pip install google-generativeai")

        # Configure API key
        genai.configure(api_key=self.api_key)

        # Create model
        model = genai.GenerativeModel(self.model)

        # Generate response (synchronous, so we need to run it in executor)
        import asyncio
        loop = asyncio.get_event_loop()

        def _generate():
            response = model.generate_content(prompt)
            return response.text

        response_text = await loop.run_in_executor(None, _generate)
        return response_text

    async def _call_xai(self, prompt: str) -> str:
        """Call xAI Grok API (OpenAI-compatible)"""
        if not openai:
            raise ImportError("openai package not installed")

        # xAI uses OpenAI-compatible API
        base_url = self.base_url or "https://api.x.ai/v1"
        client = openai.AsyncOpenAI(api_key=self.api_key, base_url=base_url)

        response = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )

        return response.choices[0].message.content

    async def _call_local(self, prompt: str) -> str:
        """Call local LLM API (OpenAI compatible)"""
        if not self.base_url:
            raise ValueError("base_url required for local provider")

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 500
                }
            ) as response:
                data = await response.json()
                return data['choices'][0]['message']['content']

    async def _call_huggingface(self, prompt: str) -> str:
        """Call Hugging Face Inference API"""
        if not InferenceClient:
            raise ImportError("huggingface_hub package not installed. Install with: pip install huggingface_hub")

        if not self.api_key:
            raise ValueError("API key required for Hugging Face")

        # Use async executor for sync HF client
        import asyncio
        loop = asyncio.get_event_loop()

        def _generate():
            client = InferenceClient(token=self.api_key)
            response = client.text_generation(
                prompt,
                model=self.model,
                max_new_tokens=500,
                temperature=0.7
            )
            return response

        response_text = await loop.run_in_executor(None, _generate)
        return response_text

    async def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API (local, OpenAI-compatible)"""
        base_url = self.base_url or "http://localhost:11434"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base_url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 500
                }
            ) as response:
                data = await response.json()
                return data['choices'][0]['message']['content']

    async def _call_together(self, prompt: str) -> str:
        """Call Together AI API (OpenAI-compatible)"""
        if not openai:
            raise ImportError("openai package not installed")

        if not self.api_key:
            raise ValueError("API key required for Together AI")

        # Together AI uses OpenAI-compatible API
        base_url = self.base_url or "https://api.together.xyz/v1"
        client = openai.AsyncOpenAI(api_key=self.api_key, base_url=base_url)

        response = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )

        return response.choices[0].message.content

    async def _call_replicate(self, prompt: str) -> str:
        """Call Replicate API"""
        if not self.api_key:
            raise ValueError("API key required for Replicate")

        # Use Replicate HTTP API directly
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "application/json"
            }

            # Create prediction
            async with session.post(
                "https://api.replicate.com/v1/predictions",
                headers=headers,
                json={
                    "version": self.model,
                    "input": {"prompt": prompt, "max_length": 500}
                }
            ) as response:
                prediction = await response.json()
                prediction_id = prediction['id']

            # Poll for result
            import asyncio
            for _ in range(30):  # 30 attempts, 1 second each
                await asyncio.sleep(1)
                async with session.get(
                    f"https://api.replicate.com/v1/predictions/{prediction_id}",
                    headers=headers
                ) as response:
                    result = await response.json()
                    if result['status'] == 'succeeded':
                        output = result['output']
                        return ''.join(output) if isinstance(output, list) else str(output)
                    elif result['status'] == 'failed':
                        raise Exception(f"Replicate prediction failed: {result.get('error')}")

            raise Exception("Replicate prediction timed out")

    async def _call_cohere(self, prompt: str) -> str:
        """Call Cohere API"""
        if not cohere:
            raise ImportError("cohere package not installed. Install with: pip install cohere")

        if not self.api_key:
            raise ValueError("API key required for Cohere")

        # Use async executor for sync Cohere client
        import asyncio
        loop = asyncio.get_event_loop()

        def _generate():
            co = cohere.Client(self.api_key)
            response = co.generate(
                model=self.model,
                prompt=prompt,
                max_tokens=500,
                temperature=0.7
            )
            return response.generations[0].text

        response_text = await loop.run_in_executor(None, _generate)
        return response_text

    async def test_prompt_with_judge(self, prompt_id: int, prompt: str, judge) -> Dict:
        """Test prompt with judge evaluation"""
        result = await self.test_prompt(prompt)

        if result.success:
            # Evaluate with judge (supports both sync and async judge)
            if hasattr(judge, 'judge') and asyncio.iscoroutinefunction(judge.judge):
                # HybridJudge (async)
                judgment_result = await judge.judge(prompt, result.response)
                judgment = {
                    'success': judgment_result.success,
                    'severity': judgment_result.severity,
                    'confidence': judgment_result.confidence,
                    'reasoning': judgment_result.reasoning
                }
            else:
                # JudgeSystem (sync)
                judgment_result = judge.evaluate(prompt, result.response)
                judgment = {
                    'success': judgment_result.success,
                    'severity': judgment_result.severity.value,
                    'confidence': judgment_result.confidence,
                    'reasoning': judgment_result.reasoning
                }

            # Save to database
            self.db.insert_test_result(
                prompt_id=prompt_id,
                provider=self.provider,
                model=self.model,
                response=result.response,
                success=judgment['success'],
                severity=judgment.get('severity', 'low'),
                confidence=judgment.get('confidence', 0.5),
                reasoning=judgment.get('reasoning', ''),
                response_time=result.response_time,
                used_input=prompt
            )

            return {
                'success': judgment['success'],
                'response': result.response,
                'response_time': result.response_time,
                'judgment': judgment
            }
        else:
            # Save error result
            self.db.insert_test_result(
                prompt_id=prompt_id,
                provider=self.provider,
                model=self.model,
                response=result.error_message or "",
                success=False,
                severity='error',
                confidence=0.0,
                reasoning=f"API Error: {result.error_message}",
                response_time=result.response_time,
                used_input=prompt
            )

            return {
                'success': False,
                'response': result.error_message,
                'response_time': result.response_time,
                'error': result.error_message
            }

    async def test_batch(self, prompts: List[Dict], judge, max_concurrent: int = 5):
        """Test multiple prompts with concurrency control"""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def test_with_semaphore(prompt_data):
            async with semaphore:
                return await self.test_prompt_with_judge(
                    prompt_data['id'],
                    prompt_data['payload'],
                    judge
                )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(
                f"Testing {len(prompts)} prompts...",
                total=len(prompts)
            )

            results = []
            for prompt_data in prompts:
                result = await test_with_semaphore(prompt_data)
                results.append(result)
                progress.update(task, advance=1)

        return results

    async def test_category(self, category: str, limit: int = 100, judge = None,
                           max_concurrent: int = 5):
        """Test all prompts in a category"""
        prompts = self.db.get_prompts(category=category, limit=limit)

        if not prompts:
            console.print(f"[yellow]No prompts found in category '{category}'[/yellow]")
            return []

        console.print(f"[cyan]Testing {len(prompts)} prompts from category '{category}'[/cyan]")

        results = await self.test_batch(prompts, judge, max_concurrent)

        # Calculate statistics
        successful = sum(1 for r in results if r.get('success', False))
        success_rate = (successful / len(results) * 100) if results else 0

        console.print(f"\n[bold green]Results:[/bold green]")
        console.print(f"  Total: {len(results)}")
        console.print(f"  Successful: {successful}")
        console.print(f"  Success Rate: {success_rate:.2f}%")

        return results
