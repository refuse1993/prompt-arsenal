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

    async def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API"""
        if not openai:
            raise ImportError("openai package not installed")

        client = openai.AsyncOpenAI(api_key=self.api_key)

        response = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
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

    async def test_prompt_with_judge(self, prompt_id: int, prompt: str, judge) -> Dict:
        """Test prompt with judge evaluation"""
        result = await self.test_prompt(prompt)

        if result.success:
            # Evaluate with judge
            judgment = judge.judge_response(prompt, result.response)

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
