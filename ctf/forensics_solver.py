"""
Forensics Solver - CTF 포렌식 자동 풀이
"""

import asyncio
import os
from typing import Dict, List, Optional
from dataclasses import dataclass

from .llm_reasoner import LLMReasoner, CTFAnalysis
from .tool_executor import ToolExecutor, ToolResult


@dataclass
class ForensicsResult:
    """포렌식 분석 결과"""
    analysis_type: str
    success: bool
    flag: Optional[str]
    findings: List[str]
    confidence: float


class ForensicsSolver:
    """CTF 포렌식 자동 풀이 엔진"""

    def __init__(self, llm: LLMReasoner, executor: ToolExecutor):
        self.llm = llm
        self.executor = executor

    async def solve(self, file_path: str, challenge_info: Dict) -> Dict:
        """
        포렌식 문제 자동 풀이

        Args:
            file_path: 분석 대상 파일
            challenge_info: 문제 정보

        Returns:
            풀이 결과
        """
        if not os.path.exists(file_path):
            return {
                'success': False,
                'error': f'File not found: {file_path}'
            }

        # 1. LLM으로 문제 분석
        analysis = await self.llm.analyze_challenge({
            'title': challenge_info.get('title', ''),
            'description': challenge_info.get('description', ''),
            'files': [file_path],
            'hints': challenge_info.get('hints', [])
        })

        if analysis.category != 'forensics':
            return {
                'success': False,
                'error': f'Not a forensics challenge: {analysis.category}'
            }

        # 2. 파일 타입 확인
        file_type_result = await self.executor.run_file(file_path)

        if not file_type_result.success:
            return {
                'success': False,
                'error': 'Failed to determine file type'
            }

        file_type = file_type_result.output.lower()

        # 3. 파일 타입별 분석
        results = []

        # 기본 분석
        results.append(await self._basic_analysis(file_path, file_type))

        # 이미지 파일
        if any(ext in file_type for ext in ['image', 'jpeg', 'png', 'gif', 'bmp']):
            results.append(await self._analyze_image(file_path))

        # 압축 파일
        if any(ext in file_type for ext in ['zip', 'tar', 'gzip', 'rar', '7-zip']):
            results.append(await self._analyze_archive(file_path))

        # 실행 파일
        if any(ext in file_type for ext in ['executable', 'elf', 'pe32']):
            results.append(await self._analyze_executable(file_path))

        # 메모리 덤프
        if 'memory' in file_type or 'dump' in file_type:
            results.append(await self._analyze_memory_dump(file_path))

        # 네트워크 캡처
        if 'pcap' in file_type or 'tcpdump' in file_type:
            results.append(await self._analyze_pcap(file_path))

        # 디스크 이미지
        if any(ext in file_type for ext in ['filesystem', 'ext4', 'ntfs']):
            results.append(await self._analyze_disk_image(file_path))

        # 4. 결과 통합
        successful_results = [r for r in results if r.success]

        if successful_results:
            best_result = max(successful_results, key=lambda r: r.confidence)
            return {
                'success': True,
                'flag': best_result.flag,
                'analysis_type': best_result.analysis_type,
                'findings': best_result.findings,
                'confidence': best_result.confidence,
                'all_results': results
            }

        # 5. 실패 시 LLM에게 추가 분석 요청
        print("  🤖 기본 분석 실패, LLM 기반 심화 분석 시도")
        llm_result = await self._llm_based_analysis(file_path, file_type, analysis, results)

        return {
            'success': llm_result.success,
            'flag': llm_result.flag,
            'analysis_type': llm_result.analysis_type,
            'findings': llm_result.findings,
            'confidence': llm_result.confidence
        }

    # === 기본 분석 ===

    async def _basic_analysis(self, file_path: str, file_type: str) -> ForensicsResult:
        """기본 파일 분석 (strings, exiftool)"""
        print(f"  🔍 기본 분석 시작: {file_path}")

        findings = []

        # 1. Strings 추출
        strings_result = await self.executor.run_strings(file_path, min_length=6)

        if strings_result.success:
            flag = self._extract_flag(strings_result.output)
            if flag:
                return ForensicsResult(
                    analysis_type='Strings Analysis',
                    success=True,
                    flag=flag,
                    findings=['Flag found in strings output'],
                    confidence=0.9
                )

            # 흥미로운 문자열 추출
            interesting = self._find_interesting_strings(strings_result.output)
            findings.extend(interesting)

        # 2. Exiftool 메타데이터
        exif_result = await self.executor.run_exiftool(file_path)

        if exif_result.success:
            flag = self._extract_flag(exif_result.output)
            if flag:
                return ForensicsResult(
                    analysis_type='EXIF Metadata',
                    success=True,
                    flag=flag,
                    findings=['Flag found in EXIF metadata'],
                    confidence=0.95
                )

            findings.append(f"EXIF data extracted: {len(exif_result.output)} bytes")

        return ForensicsResult(
            analysis_type='Basic Analysis',
            success=False,
            flag=None,
            findings=findings,
            confidence=0.0
        )

    # === 이미지 분석 ===

    async def _analyze_image(self, file_path: str) -> ForensicsResult:
        """이미지 파일 분석 (스테가노그래피)"""
        print(f"  🖼️  이미지 분석 시작: {file_path}")

        findings = []

        # 1. Binwalk로 숨겨진 파일 탐지
        binwalk_result = await self.executor.run_binwalk(file_path, extract=True)

        if binwalk_result.success:
            findings.append("Binwalk analysis completed")

            # 추출된 파일 확인
            flag = self._extract_flag(binwalk_result.output)
            if flag:
                return ForensicsResult(
                    analysis_type='Steganography (Binwalk)',
                    success=True,
                    flag=flag,
                    findings=findings,
                    confidence=0.9
                )

            # 추출된 파일들 검색
            extracted_files = self._find_extracted_files(file_path)
            for extracted in extracted_files:
                # 각 파일에서 strings 실행
                strings_result = await self.executor.run_strings(extracted)
                if strings_result.success:
                    flag = self._extract_flag(strings_result.output)
                    if flag:
                        return ForensicsResult(
                            analysis_type='Extracted File Analysis',
                            success=True,
                            flag=flag,
                            findings=[f'Flag found in extracted file: {extracted}'],
                            confidence=0.85
                        )

        # 2. LSB Steganography (실제 구현 필요)
        # 실제 구현에서는 PIL/OpenCV로 LSB 분석

        return ForensicsResult(
            analysis_type='Image Analysis',
            success=False,
            flag=None,
            findings=findings,
            confidence=0.0
        )

    # === 압축 파일 분석 ===

    async def _analyze_archive(self, file_path: str) -> ForensicsResult:
        """압축 파일 분석"""
        print(f"  📦 압축 파일 분석 시작: {file_path}")

        findings = []

        # 1. Binwalk로 압축 파일 추출
        binwalk_result = await self.executor.run_binwalk(file_path, extract=True)

        if binwalk_result.success:
            findings.append("Archive extracted")

            # 추출된 파일들 검색
            extracted_files = self._find_extracted_files(file_path)

            for extracted in extracted_files:
                # 각 파일 분석
                file_result = await self.executor.run_file(extracted)

                if 'text' in file_result.output.lower():
                    # 텍스트 파일 → strings 실행
                    strings_result = await self.executor.run_strings(extracted)
                    if strings_result.success:
                        flag = self._extract_flag(strings_result.output)
                        if flag:
                            return ForensicsResult(
                                analysis_type='Archive Analysis',
                                success=True,
                                flag=flag,
                                findings=[f'Flag found in {extracted}'],
                                confidence=0.9
                            )

        return ForensicsResult(
            analysis_type='Archive Analysis',
            success=False,
            flag=None,
            findings=findings,
            confidence=0.0
        )

    # === 실행 파일 분석 ===

    async def _analyze_executable(self, file_path: str) -> ForensicsResult:
        """실행 파일 분석"""
        print(f"  💾 실행 파일 분석 시작: {file_path}")

        findings = []

        # 1. Strings 분석
        strings_result = await self.executor.run_strings(file_path, min_length=8)

        if strings_result.success:
            flag = self._extract_flag(strings_result.output)
            if flag:
                return ForensicsResult(
                    analysis_type='Executable Strings',
                    success=True,
                    flag=flag,
                    findings=['Flag found in executable strings'],
                    confidence=0.85
                )

            interesting = self._find_interesting_strings(strings_result.output)
            findings.extend(interesting)

        # 2. Binwalk (숨겨진 데이터 확인)
        binwalk_result = await self.executor.run_binwalk(file_path)

        if binwalk_result.success:
            findings.append("Binwalk scan completed")

        return ForensicsResult(
            analysis_type='Executable Analysis',
            success=False,
            flag=None,
            findings=findings,
            confidence=0.0
        )

    # === 메모리 덤프 분석 ===

    async def _analyze_memory_dump(self, file_path: str) -> ForensicsResult:
        """메모리 덤프 분석 (Volatility)"""
        print(f"  🧠 메모리 덤프 분석 시작: {file_path}")

        findings = []

        # Volatility는 설치 확인 필요
        if not self.executor.installed_tools.get('volatility'):
            findings.append("Volatility not installed, using basic analysis")

            # Strings 분석으로 대체
            strings_result = await self.executor.run_strings(file_path, min_length=10)

            if strings_result.success:
                flag = self._extract_flag(strings_result.output)
                if flag:
                    return ForensicsResult(
                        analysis_type='Memory Dump (Strings)',
                        success=True,
                        flag=flag,
                        findings=['Flag found in memory dump strings'],
                        confidence=0.7
                    )

        return ForensicsResult(
            analysis_type='Memory Dump Analysis',
            success=False,
            flag=None,
            findings=findings,
            confidence=0.0
        )

    # === 네트워크 캡처 분석 ===

    async def _analyze_pcap(self, file_path: str) -> ForensicsResult:
        """PCAP 파일 분석"""
        print(f"  🌐 네트워크 캡처 분석 시작: {file_path}")

        findings = []

        # tshark 사용 (실제 구현 필요)
        # 예: tshark -r file.pcap -Y "http" -T fields -e http.request.uri

        # 간단한 strings 분석
        strings_result = await self.executor.run_strings(file_path, min_length=6)

        if strings_result.success:
            flag = self._extract_flag(strings_result.output)
            if flag:
                return ForensicsResult(
                    analysis_type='PCAP Analysis',
                    success=True,
                    flag=flag,
                    findings=['Flag found in network traffic'],
                    confidence=0.8
                )

            # HTTP 트래픽 흔적 확인
            if 'HTTP' in strings_result.output or 'GET' in strings_result.output:
                findings.append("HTTP traffic detected")

        return ForensicsResult(
            analysis_type='PCAP Analysis',
            success=False,
            flag=None,
            findings=findings,
            confidence=0.0
        )

    # === 디스크 이미지 분석 ===

    async def _analyze_disk_image(self, file_path: str) -> ForensicsResult:
        """디스크 이미지 분석"""
        print(f"  💿 디스크 이미지 분석 시작: {file_path}")

        findings = []

        # Foremost로 파일 복구
        foremost_result = await self.executor.run_foremost(
            file_path,
            output_dir=f"foremost_output_{os.path.basename(file_path)}"
        )

        if foremost_result.success:
            findings.append("File carving completed")

            # 복구된 파일들 검색
            output_dir = f"foremost_output_{os.path.basename(file_path)}"

            if os.path.exists(output_dir):
                for root, dirs, files in os.walk(output_dir):
                    for file in files:
                        recovered_path = os.path.join(root, file)

                        strings_result = await self.executor.run_strings(recovered_path)
                        if strings_result.success:
                            flag = self._extract_flag(strings_result.output)
                            if flag:
                                return ForensicsResult(
                                    analysis_type='Disk Image Analysis',
                                    success=True,
                                    flag=flag,
                                    findings=[f'Flag found in recovered file: {file}'],
                                    confidence=0.85
                                )

        return ForensicsResult(
            analysis_type='Disk Image Analysis',
            success=False,
            flag=None,
            findings=findings,
            confidence=0.0
        )

    # === LLM 기반 분석 ===

    async def _llm_based_analysis(
        self,
        file_path: str,
        file_type: str,
        analysis: CTFAnalysis,
        previous_results: List[ForensicsResult]
    ) -> ForensicsResult:
        """LLM 기반 심화 분석"""
        print(f"  🤖 LLM 기반 심화 분석 시도")

        # 이전 결과 요약
        findings_summary = []
        for result in previous_results:
            if result.findings:
                findings_summary.extend(result.findings)

        context = f"""
File: {file_path}
Type: {file_type}
Previous findings:
{chr(10).join(findings_summary)}
"""

        # LLM에게 추가 분석 전략 요청
        exploit_code = await self.llm.generate_exploit(analysis, context)

        # LLM 응답 분석
        # 실제 구현에서는 LLM이 제안한 명령어를 실행

        return ForensicsResult(
            analysis_type='LLM Analysis',
            success=False,
            flag=None,
            findings=['LLM analysis completed but no flag found'],
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

        for line in lines[:100]:  # 처음 100줄만
            line = line.strip()

            # URL
            if 'http://' in line or 'https://' in line:
                interesting.append(f"URL found: {line[:100]}")

            # 파일 경로
            if '/' in line and len(line) > 10:
                interesting.append(f"Path found: {line[:100]}")

            # Base64 패턴
            if len(line) > 20 and line.replace('=', '').isalnum():
                interesting.append(f"Possible Base64: {line[:50]}")

        return interesting[:10]  # 최대 10개

    def _find_extracted_files(self, original_file: str) -> List[str]:
        """추출된 파일 목록 찾기"""
        extracted_files = []

        # Binwalk는 _extracted 디렉토리에 저장
        base_name = os.path.basename(original_file)
        extracted_dir = f"_{base_name}.extracted"

        if os.path.exists(extracted_dir):
            for root, dirs, files in os.walk(extracted_dir):
                for file in files:
                    extracted_files.append(os.path.join(root, file))

        return extracted_files
