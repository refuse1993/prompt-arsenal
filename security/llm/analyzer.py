"""
LLM Security Analyzer
LLM을 활용한 코드 취약점 검증 및 탐지
"""

import json
import asyncio
from typing import List, Tuple, Optional
from pathlib import Path

from ..models import Finding
from multimodal.llm_client import LLMClient
from core.config import Config


class LLMSecurityAnalyzer:
    """
    LLM 기반 보안 분석기

    기능:
    1. verify_finding() - 도구가 발견한 취약점 검증 (False Positive 필터링)
    2. detect_vulnerabilities() - 코드에서 직접 취약점 탐지
    """

    VERIFY_PROMPT = """당신은 정적 분석 도구가 발견한 잠재적 취약점을 분석하는 보안 전문가입니다.

**취약점 보고서:**
- CWE ID: {cwe_id}
- CWE 이름: {cwe_name}
- 심각도: {severity}
- 도구 신뢰도: {confidence}
- 파일: {file_path}:{line_number}
- 설명: {description}

**코드 컨텍스트:**
```
{code_snippet}
```

**당신의 임무:**
이것이 진짜 취약점(TRUE)인지 오탐(FALSE POSITIVE)인지 판단하세요.

고려사항:
1. 취약한 코드가 실제로 프로덕션에서 실행 가능한가?
2. 완화 조치가 있는가?
3. 데이터가 실제로 사용자가 제어 가능한가?
4. 진짜 보안 위험인가, 아니면 코드 품질 문제인가?

JSON 형식으로만 응답하세요:
{{
    "is_valid": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "판단 근거를 한글로 간략히 설명",
    "severity": "Critical/High/Medium/Low",
    "attack_scenario": "공격자가 어떻게 악용할 수 있는지 한글로 설명 (진짜 취약점인 경우)"
}}"""

    DETECT_PROMPT = """당신은 코드 리뷰를 수행하는 보안 전문가입니다.

**분석할 코드:**
파일: {file_path}

```{language}
{code}
```

**당신의 임무:**
이 코드에서 CWE Top 25 취약점을 중심으로 모든 잠재적인 보안 취약점을 찾으세요.

찾아야 할 취약점:
- SQL Injection (CWE-89)
- XSS (CWE-79)
- Path Traversal (CWE-22)
- Command Injection (CWE-78)
- Insecure Deserialization (CWE-502)
- Authentication Issues (CWE-287)
- Weak Cryptography (CWE-327)
- Hard-coded Credentials (CWE-798)
- CSRF (CWE-352)
- 기타 CWE Top 25 취약점

발견된 각 취약점에 대해 JSON 배열 형식으로 응답하세요:
[
    {{
        "cwe_id": "CWE-XXX",
        "cwe_name": "취약점 이름 (한글)",
        "severity": "Critical/High/Medium/Low",
        "confidence": 0.0-1.0,
        "line_number": 줄번호,
        "title": "간단한 제목 (한글)",
        "description": "상세한 설명 (한글, 무엇이 문제인지 명확하게)",
        "attack_scenario": "공격자가 어떻게 악용할 수 있는지 (한글, 구체적으로)",
        "remediation": "어떻게 수정해야 하는지 (한글, 단계별로)",
        "remediation_code": "수정된 코드 예시 (원본과 동일한 언어)"
    }},
    ...
]

취약점이 발견되지 않으면 빈 배열 반환: []

JSON 배열만 응답하세요. 다른 텍스트는 포함하지 마세요."""

    def __init__(self, profile_name: str, db=None):
        """
        Args:
            profile_name: API 프로필 이름 (config.json에서 로드)
            db: ArsenalDB 인스턴스 (선택)
        """
        self.profile_name = profile_name
        self.db = db
        self.config = Config()

        # Load profile
        profile = self.config.get_profile(profile_name)
        if not profile:
            raise ValueError(f"Profile not found: {profile_name}")

        # Initialize LLM client
        self.llm_client = LLMClient(
            provider=profile['provider'],
            model=profile['model'],
            api_key=profile.get('api_key'),
            base_url=profile.get('base_url')
        )

        self.provider = profile['provider']
        self.model = profile['model']

        # LLM 호출 통계
        self.call_count = 0
        self.total_cost = 0.0

    async def verify_finding(self, finding: Finding) -> Tuple[bool, str]:
        """
        도구가 발견한 취약점 검증 (False Positive 필터링)

        Args:
            finding: 도구가 발견한 Finding 객체

        Returns:
            (is_valid, reasoning) 튜플
        """
        # Read code snippet if not provided
        code_snippet = finding.code_snippet
        if not code_snippet and finding.file_path:
            code_snippet = self._read_code_context(
                finding.file_path,
                finding.line_number
            )

        # Build prompt
        prompt = self.VERIFY_PROMPT.format(
            cwe_id=finding.cwe_id,
            cwe_name=finding.cwe_name,
            severity=finding.severity,
            confidence=finding.confidence,
            file_path=finding.file_path,
            line_number=finding.line_number or "unknown",
            description=finding.description,
            code_snippet=code_snippet or "(code not available)"
        )

        # Call LLM
        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                temperature=0.3,  # Low temperature for consistent results
                max_tokens=1000
            )

            self.call_count += 1
            self._estimate_cost(prompt, response)

            # Parse JSON
            result = self._parse_json_response(response)

            if result:
                # Update finding with LLM analysis
                finding.llm_reasoning = result.get('reasoning', '')
                if 'severity' in result:
                    finding.severity = result['severity']
                if 'attack_scenario' in result:
                    finding.attack_scenario = result['attack_scenario']

                return result.get('is_valid', False), result.get('reasoning', '')
            else:
                # JSON parsing failed - conservative approach
                return True, "LLM response parsing failed - keeping finding"

        except Exception as e:
            print(f"LLM verification error: {e}")
            # On error, keep the finding (conservative)
            return True, f"LLM error: {str(e)}"

    async def detect_vulnerabilities(self, file_path: str) -> List[Finding]:
        """
        LLM으로 코드에서 직접 취약점 탐지

        Args:
            file_path: 분석할 파일 경로

        Returns:
            발견된 Finding 리스트
        """
        # Read file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
        except Exception as e:
            print(f"File read error: {e}")
            return []

        # Determine language
        language = self._detect_language(file_path)

        # Build prompt
        prompt = self.DETECT_PROMPT.format(
            file_path=file_path,
            language=language,
            code=code
        )

        # Call LLM
        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=3000
            )

            self.call_count += 1
            self._estimate_cost(prompt, response)

            # Parse JSON array
            vulnerabilities = self._parse_json_response(response)

            if not isinstance(vulnerabilities, list):
                return []

            # Convert to Finding objects
            findings = []
            for vuln in vulnerabilities:
                finding = Finding(
                    cwe_id=vuln.get('cwe_id', 'CWE-Unknown'),
                    cwe_name=vuln.get('cwe_name', ''),
                    severity=vuln.get('severity', 'Medium'),
                    confidence=vuln.get('confidence', 0.7),
                    file_path=file_path,
                    line_number=vuln.get('line_number'),
                    title=vuln.get('title', ''),
                    description=vuln.get('description', ''),
                    attack_scenario=vuln.get('attack_scenario', ''),
                    remediation=vuln.get('remediation', ''),
                    remediation_code=vuln.get('remediation_code', ''),
                    verified_by='llm'
                )
                findings.append(finding)

            return findings

        except Exception as e:
            print(f"LLM detection error: {e}")
            return []

    def _read_code_context(self, file_path: str, line_number: Optional[int], context_lines: int = 5) -> str:
        """파일에서 코드 컨텍스트 읽기"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            if line_number is None:
                # Return first few lines if line number unknown
                return ''.join(lines[:context_lines * 2])

            start = max(0, line_number - context_lines - 1)
            end = min(len(lines), line_number + context_lines)

            return ''.join(lines[start:end])

        except Exception as e:
            return f"(Error reading file: {e})"

    def _detect_language(self, file_path: str) -> str:
        """파일 확장자로 언어 감지"""
        ext = Path(file_path).suffix.lower()

        lang_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'jsx',
            '.tsx': 'tsx',
            '.vue': 'vue',
            '.svelte': 'svelte',
            '.java': 'java',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.go': 'go',
            '.rs': 'rust',
            '.c': 'c',
            '.cpp': 'cpp',
            '.cc': 'cpp',
            '.h': 'c',
            '.hpp': 'cpp',
            '.php': 'php',
            '.rb': 'ruby',
            '.sh': 'bash',
            '.bash': 'bash',
            '.cs': 'csharp',
            '.swift': 'swift',
            '.m': 'objective-c',
            '.mm': 'objective-c++',
            '.html': 'html',
            '.xml': 'xml',
            '.sql': 'sql'
        }

        return lang_map.get(ext, 'code')

    def _parse_json_response(self, response: str):
        """LLM 응답에서 JSON 추출 및 파싱"""
        try:
            # Clean response
            response = response.strip()

            # Extract from code blocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            # Parse JSON
            return json.loads(response)

        except Exception as e:
            print(f"JSON parsing error: {e}")
            return None

    def _estimate_cost(self, prompt: str, response: str):
        """비용 추정 (간단한 토큰 기반)"""
        # Rough token estimation (1 token ≈ 4 characters)
        input_tokens = len(prompt) / 4
        output_tokens = len(response) / 4

        # Cost estimation (rough averages)
        cost_per_1k = {
            'gpt-4o': (0.0025, 0.010),      # (input, output)
            'gpt-4o-mini': (0.00015, 0.0006),
            'claude-3-5-sonnet': (0.003, 0.015),
            'claude-3-haiku': (0.00025, 0.00125),
            'gemini-pro': (0.0005, 0.0015)
        }

        # Get cost for model (default to cheap model if unknown)
        model_lower = self.model.lower()
        costs = cost_per_1k.get(model_lower, (0.0001, 0.0005))

        # Calculate
        input_cost = (input_tokens / 1000) * costs[0]
        output_cost = (output_tokens / 1000) * costs[1]

        self.total_cost += (input_cost + output_cost)

    def get_stats(self) -> dict:
        """LLM 호출 통계 반환"""
        return {
            'calls': self.call_count,
            'total_cost': round(self.total_cost, 4),
            'provider': self.provider,
            'model': self.model
        }
