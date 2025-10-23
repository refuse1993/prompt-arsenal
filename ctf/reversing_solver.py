"""
Reversing Solver - CTF 리버싱 자동 풀이
"""

import asyncio
import os
from typing import Dict, List, Optional
from dataclasses import dataclass

from .llm_reasoner import LLMReasoner, CTFAnalysis
from .tool_executor import ToolExecutor, ToolResult


@dataclass
class ReversingResult:
    """리버싱 결과"""
    analysis_type: str
    success: bool
    flag: Optional[str]
    findings: List[str]
    decompiled_code: Optional[str]
    confidence: float


class ReversingSolver:
    """CTF 리버싱 자동 풀이 엔진"""

    def __init__(self, llm: LLMReasoner, executor: ToolExecutor):
        self.llm = llm
        self.executor = executor

    async def solve(self, binary_path: str, challenge_info: Dict) -> Dict:
        """
        리버싱 문제 자동 풀이

        Args:
            binary_path: 바이너리 파일 경로
            challenge_info: 문제 정보

        Returns:
            풀이 결과
        """
        if not os.path.exists(binary_path):
            return {
                'success': False,
                'error': f'File not found: {binary_path}'
            }

        # 1. LLM으로 문제 분석
        analysis = await self.llm.analyze_challenge({
            'title': challenge_info.get('title', ''),
            'description': challenge_info.get('description', ''),
            'files': [binary_path],
            'hints': challenge_info.get('hints', [])
        })

        if analysis.category != 'reversing':
            return {
                'success': False,
                'error': f'Not a reversing challenge: {analysis.category}'
            }

        # 2. 바이너리 분석
        results = []

        # 기본 분석
        results.append(await self._basic_analysis(binary_path))

        # 정적 분석
        results.append(await self._static_analysis(binary_path))

        # 동적 분석
        results.append(await self._dynamic_analysis(binary_path))

        # 디컴파일 분석
        results.append(await self._decompile_analysis(binary_path))

        # 3. 결과 통합
        successful_results = [r for r in results if r.success]

        if successful_results:
            best_result = max(successful_results, key=lambda r: r.confidence)
            return {
                'success': True,
                'flag': best_result.flag,
                'analysis_type': best_result.analysis_type,
                'findings': best_result.findings,
                'decompiled_code': best_result.decompiled_code,
                'confidence': best_result.confidence,
                'all_results': results
            }

        # 4. LLM 기반 심화 분석
        print("  🤖 LLM 기반 심화 분석 시도")
        llm_result = await self._llm_based_analysis(binary_path, analysis, results)

        return {
            'success': llm_result.success,
            'flag': llm_result.flag,
            'analysis_type': llm_result.analysis_type,
            'findings': llm_result.findings,
            'confidence': llm_result.confidence
        }

    # === 기본 분석 ===

    async def _basic_analysis(self, binary_path: str) -> ReversingResult:
        """기본 바이너리 분석 (strings, file)"""
        print(f"  🔍 기본 분석 시작: {binary_path}")

        findings = []

        # 1. File 타입 확인
        file_result = await self.executor.run_file(binary_path)

        if file_result.success:
            findings.append(f"File type: {file_result.output.strip()}")

        # 2. Strings 추출
        strings_result = await self.executor.run_strings(binary_path, min_length=6)

        if strings_result.success:
            flag = self._extract_flag(strings_result.output)

            if flag:
                return ReversingResult(
                    analysis_type='Strings Analysis',
                    success=True,
                    flag=flag,
                    findings=['Flag found in strings'],
                    decompiled_code=None,
                    confidence=0.9
                )

            # 흥미로운 문자열 추출
            interesting = self._find_interesting_strings(strings_result.output)
            findings.extend(interesting)

        return ReversingResult(
            analysis_type='Basic Analysis',
            success=False,
            flag=None,
            findings=findings,
            decompiled_code=None,
            confidence=0.0
        )

    # === 정적 분석 ===

    async def _static_analysis(self, binary_path: str) -> ReversingResult:
        """정적 분석 (readelf, objdump)"""
        print(f"  📊 정적 분석 시작: {binary_path}")

        findings = []

        # 1. Readelf로 ELF 구조 분석
        readelf_result = await self.executor.run_readelf(binary_path, option="-a")

        if readelf_result.success:
            findings.append("Readelf analysis completed")

            # 심볼 테이블에서 흥미로운 함수 찾기
            symbols = self._extract_symbols(readelf_result.output)

            if symbols:
                findings.append(f"Found {len(symbols)} symbols")

                for symbol in symbols:
                    if any(keyword in symbol.lower() for keyword in ['flag', 'check', 'password', 'secret']):
                        findings.append(f"Interesting symbol: {symbol}")

        # 2. Objdump로 디스어셈블
        objdump_result = await self.executor.run_objdump(binary_path, option="-d")

        if objdump_result.success:
            findings.append("Objdump disassembly completed")

            # 어셈블리에서 플래그 찾기
            flag = self._extract_flag(objdump_result.output)

            if flag:
                return ReversingResult(
                    analysis_type='Static Analysis',
                    success=True,
                    flag=flag,
                    findings=findings,
                    decompiled_code=None,
                    confidence=0.85
                )

            # 주요 함수 추출
            functions = self._extract_functions(objdump_result.output)

            if functions:
                findings.append(f"Found {len(functions)} functions")

        return ReversingResult(
            analysis_type='Static Analysis',
            success=False,
            flag=None,
            findings=findings,
            decompiled_code=None,
            confidence=0.0
        )

    # === 동적 분석 ===

    async def _dynamic_analysis(self, binary_path: str) -> ReversingResult:
        """동적 분석 (ltrace, strace)"""
        print(f"  🏃 동적 분석 시작: {binary_path}")

        findings = []

        # 1. Ltrace로 라이브러리 호출 추적
        ltrace_result = await self.executor.run_ltrace(binary_path)

        if ltrace_result.success:
            findings.append("Ltrace analysis completed")

            # 플래그 찾기
            flag = self._extract_flag(ltrace_result.output)

            if flag:
                return ReversingResult(
                    analysis_type='Dynamic Analysis (ltrace)',
                    success=True,
                    flag=flag,
                    findings=['Flag found in library calls'],
                    decompiled_code=None,
                    confidence=0.9
                )

            # 흥미로운 함수 호출
            interesting_calls = self._find_interesting_calls(ltrace_result.output)
            findings.extend(interesting_calls)

        # 2. Strace로 시스템 호출 추적
        strace_result = await self.executor.run_strace(binary_path)

        if strace_result.success:
            findings.append("Strace analysis completed")

            # 플래그 찾기
            flag = self._extract_flag(strace_result.output)

            if flag:
                return ReversingResult(
                    analysis_type='Dynamic Analysis (strace)',
                    success=True,
                    flag=flag,
                    findings=['Flag found in system calls'],
                    decompiled_code=None,
                    confidence=0.9
                )

        return ReversingResult(
            analysis_type='Dynamic Analysis',
            success=False,
            flag=None,
            findings=findings,
            decompiled_code=None,
            confidence=0.0
        )

    # === 디컴파일 분석 ===

    async def _decompile_analysis(self, binary_path: str) -> ReversingResult:
        """디컴파일 분석 (Ghidra)"""
        print(f"  🔧 디컴파일 분석 시도: {binary_path}")

        findings = []

        # Ghidra 설치 확인
        if not self.executor.installed_tools.get('ghidra'):
            findings.append("Ghidra not installed, skipping decompilation")

            # Objdump로 대체
            objdump_result = await self.executor.run_objdump(binary_path, option="-d")

            if objdump_result.success:
                # main 함수 찾기
                main_asm = self._extract_main_function(objdump_result.output)

                if main_asm:
                    findings.append("Found main function assembly")

                    return ReversingResult(
                        analysis_type='Decompile Analysis',
                        success=False,
                        flag=None,
                        findings=findings,
                        decompiled_code=main_asm[:2000],
                        confidence=0.3
                    )

        return ReversingResult(
            analysis_type='Decompile Analysis',
            success=False,
            flag=None,
            findings=findings,
            decompiled_code=None,
            confidence=0.0
        )

    # === LLM 기반 분석 ===

    async def _llm_based_analysis(
        self,
        binary_path: str,
        analysis: CTFAnalysis,
        previous_results: List[ReversingResult]
    ) -> ReversingResult:
        """LLM 기반 심화 분석"""
        print(f"  🤖 LLM 기반 심화 분석")

        # 이전 결과 요약
        findings_summary = []
        decompiled_code = None

        for result in previous_results:
            if result.findings:
                findings_summary.extend(result.findings)

            if result.decompiled_code:
                decompiled_code = result.decompiled_code

        context = f"""
Binary: {binary_path}
Previous findings:
{chr(10).join(findings_summary[:20])}

Decompiled code (if available):
{decompiled_code[:2000] if decompiled_code else 'N/A'}
"""

        # LLM에게 추가 분석 전략 요청
        exploit_code = await self.llm.generate_exploit(analysis, context)

        # LLM 응답 분석
        flag = self._extract_flag(exploit_code)

        if flag:
            return ReversingResult(
                analysis_type='LLM Analysis',
                success=True,
                flag=flag,
                findings=['LLM found flag in analysis'],
                decompiled_code=None,
                confidence=0.7
            )

        return ReversingResult(
            analysis_type='LLM Analysis',
            success=False,
            flag=None,
            findings=['LLM analysis completed but no flag found'],
            decompiled_code=None,
            confidence=0.0
        )

    # === Helper Methods ===

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

    def _find_interesting_strings(self, strings_output: str) -> List[str]:
        """흥미로운 문자열 추출"""
        interesting = []

        lines = strings_output.split('\n')

        for line in lines[:100]:
            line = line.strip()

            # 플래그 관련 키워드
            if any(keyword in line.lower() for keyword in ['flag', 'password', 'secret', 'key']):
                interesting.append(f"Keyword found: {line[:100]}")

            # Base64 패턴
            if len(line) > 20 and line.replace('=', '').replace('+', '').replace('/', '').isalnum():
                interesting.append(f"Possible Base64: {line[:50]}")

            # Hex 패턴
            if len(line) > 20 and all(c in '0123456789abcdefABCDEF' for c in line):
                interesting.append(f"Possible Hex: {line[:50]}")

        return interesting[:10]

    def _extract_symbols(self, readelf_output: str) -> List[str]:
        """심볼 테이블 추출"""
        symbols = []

        lines = readelf_output.split('\n')

        for line in lines:
            # 심볼 테이블 라인 파싱
            if 'FUNC' in line or 'OBJECT' in line:
                parts = line.split()
                if len(parts) > 7:
                    symbol_name = parts[7]
                    symbols.append(symbol_name)

        return symbols

    def _extract_functions(self, objdump_output: str) -> List[str]:
        """함수 목록 추출"""
        import re

        functions = []

        # <function_name>: 패턴 찾기
        pattern = r'<([a-zA-Z_][a-zA-Z0-9_]*)>:'

        matches = re.findall(pattern, objdump_output)

        return list(set(matches))[:50]  # 최대 50개

    def _find_interesting_calls(self, ltrace_output: str) -> List[str]:
        """흥미로운 함수 호출 추출"""
        interesting = []

        lines = ltrace_output.split('\n')

        for line in lines[:50]:
            line = line.strip()

            # 흥미로운 함수
            if any(func in line for func in ['strcmp', 'strncmp', 'memcmp', 'printf', 'scanf']):
                interesting.append(f"Call: {line[:100]}")

        return interesting[:10]

    def _extract_main_function(self, objdump_output: str) -> Optional[str]:
        """main 함수 어셈블리 추출"""
        lines = objdump_output.split('\n')

        main_start = -1
        main_end = -1

        for i, line in enumerate(lines):
            if '<main>:' in line:
                main_start = i
            elif main_start > 0 and (line.strip().startswith('00') or '<' in line):
                # 다음 함수 시작
                main_end = i
                break

        if main_start > 0:
            if main_end > 0:
                return '\n'.join(lines[main_start:main_end])
            else:
                return '\n'.join(lines[main_start:main_start+100])

        return None
