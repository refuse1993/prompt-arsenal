"""
LLM Reasoner - CTF 문제 분석 및 전략 수립
"""

import json
from typing import Dict, List, Optional
from dataclasses import dataclass

try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None


@dataclass
class CTFAnalysis:
    """CTF 문제 분석 결과"""
    category: str  # web, pwn, crypto, forensics, reversing, misc
    difficulty: str  # easy, medium, hard
    vulnerability_type: str
    description: str
    strategy: List[str]
    required_tools: List[str]
    expected_flag_format: str
    confidence: float


class LLMReasoner:
    """LLM 기반 CTF 문제 추론 엔진"""

    def __init__(self, provider: str, model: str, api_key: str):
        """
        Args:
            provider: 'openai' or 'anthropic'
            model: 모델 이름
            api_key: API 키
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key

    async def analyze_challenge(self, challenge_data: Dict) -> CTFAnalysis:
        """
        CTF 문제 분석

        Args:
            challenge_data: {
                'title': '문제 제목',
                'description': '문제 설명',
                'url': 'URL (선택)',
                'files': ['file1.txt', 'file2.bin'],
                'hints': ['힌트1', '힌트2']
            }

        Returns:
            분석 결과
        """
        prompt = self._build_analysis_prompt(challenge_data)
        response = await self._call_llm(prompt)
        return self._parse_analysis(response)

    def _build_analysis_prompt(self, challenge_data: Dict) -> str:
        """CTF 문제 분석 프롬프트 생성"""
        prompt = """You are an expert CTF player and cybersecurity specialist. Analyze this CTF challenge and provide a detailed strategy.

## Challenge Information

"""
        if challenge_data.get('title'):
            prompt += f"**Title**: {challenge_data['title']}\n\n"

        if challenge_data.get('description'):
            prompt += f"**Description**:\n{challenge_data['description']}\n\n"

        if challenge_data.get('url'):
            prompt += f"**URL**: {challenge_data['url']}\n\n"

        if challenge_data.get('files'):
            prompt += f"**Files**: {', '.join(challenge_data['files'])}\n\n"

        if challenge_data.get('hints'):
            prompt += "**Hints**:\n"
            for hint in challenge_data['hints']:
                prompt += f"- {hint}\n"
            prompt += "\n"

        prompt += """
Analyze this challenge and respond in JSON format:

{
  "category": "web/pwn/crypto/forensics/reversing/misc",
  "difficulty": "easy/medium/hard",
  "vulnerability_type": "specific vulnerability type (e.g., SQL Injection, Buffer Overflow)",
  "description": "detailed analysis of what the challenge is testing",
  "strategy": [
    "Step 1: First action to take",
    "Step 2: Next action",
    "Step 3: ...",
  ],
  "required_tools": ["tool1", "tool2", "tool3"],
  "expected_flag_format": "expected flag format (e.g., flag{...}, CTF{...})",
  "confidence": 0.85
}

**IMPORTANT**: Respond ONLY with valid JSON. No explanations outside JSON.
"""
        return prompt

    async def _call_llm(self, prompt: str) -> str:
        """LLM 호출"""
        if self.provider == 'openai':
            return await self._call_openai(prompt)
        elif self.provider == 'anthropic':
            return await self._call_anthropic(prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    async def _call_openai(self, prompt: str) -> str:
        """OpenAI API 호출"""
        if not openai:
            raise ImportError("openai package not installed")

        client = openai.AsyncOpenAI(api_key=self.api_key)

        token_param = {}
        if self.model.startswith(('gpt-5', 'o1')):
            token_param = {"max_completion_tokens": 2000}
        else:
            token_param = {"max_tokens": 2000}

        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert CTF player. Always respond in valid JSON format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            **token_param
        )

        return response.choices[0].message.content

    async def _call_anthropic(self, prompt: str) -> str:
        """Anthropic API 호출"""
        if not anthropic:
            raise ImportError("anthropic package not installed")

        client = anthropic.AsyncAnthropic(api_key=self.api_key)

        response = await client.messages.create(
            model=self.model,
            max_tokens=2000,
            temperature=0.3,
            system="You are an expert CTF player. Always respond in valid JSON format.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response.content[0].text

    def _parse_analysis(self, response: str) -> CTFAnalysis:
        """LLM 응답 파싱"""
        try:
            # JSON 추출
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.startswith('```'):
                response = response[3:]
            if response.endswith('```'):
                response = response[:-3]
            response = response.strip()

            data = json.loads(response)

            return CTFAnalysis(
                category=data.get('category', 'misc'),
                difficulty=data.get('difficulty', 'medium'),
                vulnerability_type=data.get('vulnerability_type', 'Unknown'),
                description=data.get('description', ''),
                strategy=data.get('strategy', []),
                required_tools=data.get('required_tools', []),
                expected_flag_format=data.get('expected_flag_format', 'flag{...}'),
                confidence=data.get('confidence', 0.5)
            )

        except json.JSONDecodeError as e:
            print(f"  ⚠️  LLM 응답 파싱 실패: {e}")
            print(f"  응답: {response[:500]}")

            # Fallback
            return CTFAnalysis(
                category='misc',
                difficulty='medium',
                vulnerability_type='Unknown',
                description='분석 실패',
                strategy=['수동 분석 필요'],
                required_tools=[],
                expected_flag_format='flag{...}',
                confidence=0.0
            )

    async def generate_exploit(self, analysis: CTFAnalysis, context: str) -> str:
        """
        분석 결과를 기반으로 exploit 코드 생성

        Args:
            analysis: 문제 분석 결과
            context: 추가 컨텍스트 (파일 내용, 도구 출력 등)

        Returns:
            exploit 코드 또는 명령어
        """
        prompt = f"""Based on the CTF challenge analysis, generate an exploit or solution.

## Challenge Analysis
- Category: {analysis.category}
- Vulnerability: {analysis.vulnerability_type}
- Description: {analysis.description}

## Strategy
"""
        for i, step in enumerate(analysis.strategy, 1):
            prompt += f"{i}. {step}\n"

        prompt += f"\n## Additional Context\n{context}\n\n"

        prompt += """
Generate the exploit code or command sequence. Include:
1. Complete working code/commands
2. Comments explaining each step
3. Expected output

Respond in markdown format with code blocks.
"""

        response = await self._call_llm(prompt)
        return response

    async def analyze_output(self, command_output: str, expected_result: str) -> Dict:
        """
        도구 실행 결과 분석

        Args:
            command_output: 명령어 실행 결과
            expected_result: 기대하는 결과 설명

        Returns:
            분석 결과 딕셔너리
        """
        prompt = f"""Analyze the output of a CTF tool and determine next steps.

## Expected Result
{expected_result}

## Actual Output
```
{command_output}
```

Respond in JSON format:
{{
  "success": true/false,
  "flag_found": "flag value if found, else null",
  "findings": ["key finding 1", "key finding 2"],
  "next_steps": ["next action 1", "next action 2"],
  "confidence": 0.9
}}
"""

        response = await self._call_llm(prompt)

        try:
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.startswith('```'):
                response = response[3:]
            if response.endswith('```'):
                response = response[:-3]
            response = response.strip()

            return json.loads(response)
        except json.JSONDecodeError:
            return {
                'success': False,
                'flag_found': None,
                'findings': [],
                'next_steps': ['수동 분석 필요'],
                'confidence': 0.0
            }
