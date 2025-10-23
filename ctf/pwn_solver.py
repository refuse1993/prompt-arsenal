"""
Pwn Solver - CTF Pwnable 자동 풀이
"""

import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass

from .llm_reasoner import LLMReasoner, CTFAnalysis
from .tool_executor import ToolExecutor, ToolResult


@dataclass
class PwnResult:
    """Pwn 결과"""
    exploit_type: str
    success: bool
    flag: Optional[str]
    exploit_code: str
    confidence: float


class PwnSolver:
    """CTF Pwnable 자동 풀이 엔진"""

    def __init__(self, llm: LLMReasoner, executor: ToolExecutor):
        self.llm = llm
        self.executor = executor

    async def solve(self, binary_path: str, target_info: Dict) -> Dict:
        """
        Pwnable 문제 자동 풀이

        Args:
            binary_path: 바이너리 파일 경로
            target_info: {
                'host': 'ctf.example.com',
                'port': 1337,
                'description': '문제 설명',
                'hints': ['힌트1', '힌트2']
            }

        Returns:
            풀이 결과
        """
        # 1. LLM으로 문제 분석
        analysis = await self.llm.analyze_challenge({
            'title': target_info.get('title', ''),
            'description': target_info.get('description', ''),
            'files': [binary_path],
            'hints': target_info.get('hints', [])
        })

        if analysis.category != 'pwn':
            return {
                'success': False,
                'error': f'Not a pwn challenge: {analysis.category}'
            }

        # 2. 바이너리 보안 기능 확인
        print(f"  🔍 Checksec 실행: {binary_path}")
        checksec_result = await self.executor.run_checksec(binary_path)

        if not checksec_result.success:
            return {
                'success': False,
                'error': 'Failed to run checksec'
            }

        security_features = self._parse_checksec(checksec_result.output)

        # 3. 취약점 유형별 처리
        vuln_type = analysis.vulnerability_type.lower()

        if 'buffer overflow' in vuln_type or 'bof' in vuln_type:
            result = await self._solve_buffer_overflow(
                binary_path, target_info, analysis, security_features
            )
        elif 'format string' in vuln_type or 'fmt' in vuln_type:
            result = await self._solve_format_string(
                binary_path, target_info, analysis, security_features
            )
        elif 'rop' in vuln_type or 'return oriented programming' in vuln_type:
            result = await self._solve_rop(
                binary_path, target_info, analysis, security_features
            )
        elif 'heap' in vuln_type or 'use after free' in vuln_type:
            result = await self._solve_heap_exploit(
                binary_path, target_info, analysis, security_features
            )
        else:
            # 알 수 없는 유형 → LLM에게 exploit 생성 요청
            result = await self._generic_pwn_exploit(
                binary_path, target_info, analysis, security_features
            )

        return {
            'success': result.success,
            'flag': result.flag,
            'exploit_type': result.exploit_type,
            'exploit_code': result.exploit_code,
            'confidence': result.confidence,
            'security_features': security_features,
            'analysis': analysis
        }

    # === Buffer Overflow ===

    async def _solve_buffer_overflow(
        self,
        binary_path: str,
        target_info: Dict,
        analysis: CTFAnalysis,
        security: Dict
    ) -> PwnResult:
        """Buffer Overflow 자동 풀이"""
        print(f"  💥 Buffer Overflow 공격 시도")

        # 1. LLM으로 exploit 코드 생성
        context = f"""
Binary: {binary_path}
Security Features: {security}
Target: {target_info.get('host')}:{target_info.get('port')}

Generate a pwntools-based buffer overflow exploit.
Consider NX, PIE, ASLR, Canary settings.
"""

        exploit_code = await self.llm.generate_exploit(analysis, context)

        # 2. LLM이 생성한 Python exploit 코드 추출
        python_code = self._extract_python_code(exploit_code)

        if not python_code:
            return PwnResult(
                exploit_type='Buffer Overflow',
                success=False,
                flag=None,
                exploit_code=exploit_code,
                confidence=0.0
            )

        # 3. Exploit 실행 (실제 구현 필요)
        # 실제 구현에서는 pwntools를 사용하여 exploit 실행
        # from pwn import *
        # p = remote(host, port)
        # p.sendline(payload)
        # flag = p.recvuntil(b'}').decode()

        # 현재는 더미 구현
        return PwnResult(
            exploit_type='Buffer Overflow',
            success=False,
            flag=None,
            exploit_code=python_code,
            confidence=0.6
        )

    # === Format String ===

    async def _solve_format_string(
        self,
        binary_path: str,
        target_info: Dict,
        analysis: CTFAnalysis,
        security: Dict
    ) -> PwnResult:
        """Format String 자동 풀이"""
        print(f"  📝 Format String 공격 시도")

        context = f"""
Binary: {binary_path}
Security Features: {security}
Target: {target_info.get('host')}:{target_info.get('port')}

Generate a pwntools-based format string exploit.
Use %p, %n, %s to leak/write memory.
"""

        exploit_code = await self.llm.generate_exploit(analysis, context)
        python_code = self._extract_python_code(exploit_code)

        return PwnResult(
            exploit_type='Format String',
            success=False,
            flag=None,
            exploit_code=python_code or exploit_code,
            confidence=0.6
        )

    # === ROP ===

    async def _solve_rop(
        self,
        binary_path: str,
        target_info: Dict,
        analysis: CTFAnalysis,
        security: Dict
    ) -> PwnResult:
        """ROP (Return Oriented Programming) 자동 풀이"""
        print(f"  🔗 ROP 체인 생성 시도")

        # 1. Readelf로 바이너리 분석
        readelf_result = await self.executor.run_readelf(binary_path, option="-a")

        # 2. Objdump로 가젯 찾기
        objdump_result = await self.executor.run_objdump(binary_path, option="-d")

        context = f"""
Binary: {binary_path}
Security Features: {security}
Target: {target_info.get('host')}:{target_info.get('port')}

Readelf output (first 2000 chars):
{readelf_result.output[:2000]}

Objdump output (first 2000 chars):
{objdump_result.output[:2000]}

Generate a ROP chain exploit using pwntools.
Find gadgets and construct ROP chain.
"""

        exploit_code = await self.llm.generate_exploit(analysis, context)
        python_code = self._extract_python_code(exploit_code)

        return PwnResult(
            exploit_type='ROP',
            success=False,
            flag=None,
            exploit_code=python_code or exploit_code,
            confidence=0.5
        )

    # === Heap Exploit ===

    async def _solve_heap_exploit(
        self,
        binary_path: str,
        target_info: Dict,
        analysis: CTFAnalysis,
        security: Dict
    ) -> PwnResult:
        """Heap Exploit 자동 풀이"""
        print(f"  🗑️  Heap Exploit 시도")

        context = f"""
Binary: {binary_path}
Security Features: {security}
Target: {target_info.get('host')}:{target_info.get('port')}

Generate a heap exploitation exploit.
Consider techniques: UAF, Double Free, Heap Overflow, Fastbin Attack, etc.
"""

        exploit_code = await self.llm.generate_exploit(analysis, context)
        python_code = self._extract_python_code(exploit_code)

        return PwnResult(
            exploit_type='Heap Exploit',
            success=False,
            flag=None,
            exploit_code=python_code or exploit_code,
            confidence=0.4
        )

    # === Generic Pwn ===

    async def _generic_pwn_exploit(
        self,
        binary_path: str,
        target_info: Dict,
        analysis: CTFAnalysis,
        security: Dict
    ) -> PwnResult:
        """알 수 없는 Pwn 취약점 → LLM에게 전체 풀이 요청"""
        print(f"  🤖 LLM 기반 일반 Pwn 공격 시도")

        # 1. ltrace로 라이브러리 호출 추적
        ltrace_result = await self.executor.run_ltrace(binary_path)

        # 2. strace로 시스템 호출 추적
        strace_result = await self.executor.run_strace(binary_path)

        context = f"""
Binary: {binary_path}
Security Features: {security}
Target: {target_info.get('host')}:{target_info.get('port')}

Ltrace output (first 1000 chars):
{ltrace_result.output[:1000]}

Strace output (first 1000 chars):
{strace_result.output[:1000]}

Generate a complete pwntools exploit.
Analyze the binary behavior and find vulnerability.
"""

        exploit_code = await self.llm.generate_exploit(analysis, context)
        python_code = self._extract_python_code(exploit_code)

        return PwnResult(
            exploit_type='Generic Pwn',
            success=False,
            flag=None,
            exploit_code=python_code or exploit_code,
            confidence=0.3
        )

    # === Helper Methods ===

    def _parse_checksec(self, checksec_output: str) -> Dict:
        """Checksec 출력 파싱"""
        security = {
            'RELRO': 'Unknown',
            'Stack': 'Unknown',
            'NX': 'Unknown',
            'PIE': 'Unknown',
            'RPATH': 'Unknown',
            'RUNPATH': 'Unknown',
            'Symbols': 'Unknown',
            'FORTIFY': 'Unknown',
            'Fortified': 'Unknown',
            'Canary': 'Unknown'
        }

        lines = checksec_output.split('\n')

        for line in lines:
            line = line.strip()

            if 'RELRO' in line:
                if 'Full RELRO' in line:
                    security['RELRO'] = 'Full'
                elif 'Partial RELRO' in line:
                    security['RELRO'] = 'Partial'
                elif 'No RELRO' in line:
                    security['RELRO'] = 'No'

            if 'Stack' in line or 'Canary' in line:
                if 'Canary found' in line or 'Stack: Canary found' in line:
                    security['Canary'] = 'Yes'
                elif 'No canary' in line:
                    security['Canary'] = 'No'

            if 'NX' in line:
                if 'NX enabled' in line:
                    security['NX'] = 'Enabled'
                elif 'NX disabled' in line:
                    security['NX'] = 'Disabled'

            if 'PIE' in line:
                if 'PIE enabled' in line:
                    security['PIE'] = 'Enabled'
                elif 'No PIE' in line:
                    security['PIE'] = 'No'

        return security

    def _extract_python_code(self, llm_output: str) -> Optional[str]:
        """LLM 출력에서 Python 코드 추출"""
        import re

        # ```python ... ``` 블록 찾기
        pattern = r'```python\n(.*?)\n```'
        matches = re.findall(pattern, llm_output, re.DOTALL)

        if matches:
            return matches[0]

        # ``` ... ``` 블록 찾기
        pattern = r'```\n(.*?)\n```'
        matches = re.findall(pattern, llm_output, re.DOTALL)

        if matches:
            # Python 코드인지 확인
            code = matches[0]
            if 'from pwn import' in code or 'import pwn' in code:
                return code

        return None

    def _extract_flag(self, text: str) -> Optional[str]:
        """텍스트에서 플래그 추출"""
        import re

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
