"""
LLM-based Vulnerability Analyzer
LLM을 활용한 취약점 분석 및 Remediation 제안
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
class AnalysisResult:
    """LLM 분석 결과"""
    summary: str
    risk_assessment: str
    priority_vulnerabilities: List[Dict]
    attack_scenarios: List[str]
    remediation_steps: List[str]
    security_recommendations: List[str]


class LLMAnalyzer:
    """LLM 기반 취약점 분석기"""

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

    async def analyze_scan_results(self, scan_data: Dict) -> AnalysisResult:
        """
        스캔 결과를 LLM으로 분석

        Args:
            scan_data: 스캔 결과 딕셔너리 (ports, services, cves)

        Returns:
            분석 결과
        """
        # LLM 프롬프트 생성
        prompt = self._build_analysis_prompt(scan_data)

        # LLM 호출
        response = await self._call_llm(prompt)

        # 응답 파싱
        return self._parse_response(response)

    def _build_analysis_prompt(self, scan_data: Dict) -> str:
        """분석 프롬프트 생성"""
        target = scan_data.get('target', 'Unknown')
        open_ports = scan_data.get('open_ports', [])
        services = scan_data.get('services', [])
        findings = scan_data.get('findings', [])

        prompt = f"""당신은 사이버 보안 전문가입니다. 다음 시스템 취약점 스캔 결과를 분석해주세요.

## 스캔 대상
- Target: {target}

## 열린 포트
"""
        for port in open_ports:
            prompt += f"- Port {port['port']}/{port['protocol']}: {port['service']} ({port.get('version', 'Unknown version')})\n"

        prompt += "\n## 발견된 취약점\n"
        for finding in findings:
            prompt += f"- [{finding['severity'].upper()}] {finding['title']}\n"
            prompt += f"  {finding['description']}\n"
            if finding.get('cve_id'):
                prompt += f"  CVE: {finding['cve_id']}\n"

        prompt += """
다음 형식으로 분석 결과를 JSON으로 출력해주세요:

{
  "summary": "전체 보안 상태 요약 (2-3문장)",
  "risk_assessment": "위험도 평가 (Critical/High/Medium/Low) 및 근거",
  "priority_vulnerabilities": [
    {
      "title": "취약점 제목",
      "severity": "critical/high/medium/low",
      "priority": 1,
      "reason": "우선순위 이유"
    }
  ],
  "attack_scenarios": [
    "공격 시나리오 1: 구체적인 공격 방법 설명",
    "공격 시나리오 2: ..."
  ],
  "remediation_steps": [
    "Step 1: 즉시 조치할 사항",
    "Step 2: 단기 조치 (1주일 이내)",
    "Step 3: 중장기 조치"
  ],
  "security_recommendations": [
    "보안 권장사항 1",
    "보안 권장사항 2"
  ]
}

**중요**: 반드시 유효한 JSON 형식으로만 응답하세요. 다른 설명은 포함하지 마세요.
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

        # GPT-4/5/o1 모델은 max_completion_tokens 사용
        token_param = {}
        if self.model.startswith(('gpt-5', 'o1')):
            token_param = {"max_completion_tokens": 2000}
        else:
            token_param = {"max_tokens": 2000}

        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a cybersecurity expert. Always respond in valid JSON format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # 일관된 분석을 위해 낮은 temperature
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
            system="You are a cybersecurity expert. Always respond in valid JSON format.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response.content[0].text

    def _parse_response(self, response: str) -> AnalysisResult:
        """LLM 응답 파싱"""
        try:
            # JSON 추출 (코드 블록 제거)
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.startswith('```'):
                response = response[3:]
            if response.endswith('```'):
                response = response[:-3]
            response = response.strip()

            data = json.loads(response)

            return AnalysisResult(
                summary=data.get('summary', ''),
                risk_assessment=data.get('risk_assessment', ''),
                priority_vulnerabilities=data.get('priority_vulnerabilities', []),
                attack_scenarios=data.get('attack_scenarios', []),
                remediation_steps=data.get('remediation_steps', []),
                security_recommendations=data.get('security_recommendations', [])
            )

        except json.JSONDecodeError as e:
            print(f"  ⚠️  LLM 응답 파싱 실패: {e}")
            print(f"  응답: {response[:500]}")

            # Fallback: 기본 분석 결과 반환
            return AnalysisResult(
                summary="LLM 분석을 완료하지 못했습니다.",
                risk_assessment="분석 실패",
                priority_vulnerabilities=[],
                attack_scenarios=[],
                remediation_steps=["취약점을 수동으로 검토하세요."],
                security_recommendations=[]
            )

    def format_analysis_report(self, analysis: AnalysisResult) -> str:
        """분석 결과를 읽기 쉬운 리포트로 포맷"""
        report = "=" * 80 + "\n"
        report += "🔍 LLM 취약점 분석 리포트\n"
        report += "=" * 80 + "\n\n"

        report += "## 📊 요약\n"
        report += f"{analysis.summary}\n\n"

        report += "## ⚠️  위험도 평가\n"
        report += f"{analysis.risk_assessment}\n\n"

        if analysis.priority_vulnerabilities:
            report += "## 🎯 우선순위 취약점\n"
            for vuln in analysis.priority_vulnerabilities:
                priority = vuln.get('priority', '?')
                title = vuln.get('title', 'Unknown')
                severity = vuln.get('severity', 'unknown').upper()
                reason = vuln.get('reason', '')
                report += f"{priority}. [{severity}] {title}\n"
                report += f"   이유: {reason}\n\n"

        if analysis.attack_scenarios:
            report += "## 💥 공격 시나리오\n"
            for i, scenario in enumerate(analysis.attack_scenarios, 1):
                report += f"{i}. {scenario}\n\n"

        if analysis.remediation_steps:
            report += "## 🔧 조치 방법\n"
            for step in analysis.remediation_steps:
                report += f"- {step}\n"
            report += "\n"

        if analysis.security_recommendations:
            report += "## 💡 보안 권장사항\n"
            for rec in analysis.security_recommendations:
                report += f"- {rec}\n"
            report += "\n"

        report += "=" * 80 + "\n"

        return report

    async def analyze_single_vulnerability(self, cve_data: Dict) -> str:
        """단일 CVE에 대한 심화 분석"""
        prompt = f"""다음 CVE 취약점에 대해 상세히 분석해주세요:

CVE ID: {cve_data.get('cve_id', 'Unknown')}
Severity: {cve_data.get('severity', 'Unknown')}
CVSS Score: {cve_data.get('cvss_score', 0.0)}
Affected: {cve_data.get('affected_software', 'Unknown')}
Description: {cve_data.get('description', 'No description')}

다음 항목을 포함하여 분석해주세요:
1. 취약점의 기술적 원리
2. 실제 공격 시나리오 (구체적인 예시)
3. 즉시 적용 가능한 완화 방법
4. 장기적인 해결 방안
"""

        response = await self._call_llm(prompt)
        return response
