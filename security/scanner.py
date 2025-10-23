"""
Security Scanner - Main Orchestrator
정적 분석 도구 + LLM 검증을 통한 CWE 기반 취약점 스캔
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

from .models import Finding, SecurityReport, ScanConfig
from .static.tool_runner import ToolRunner


class SecurityScanner:
    """
    보안 스캐너 메인 클래스

    지원 모드:
    - rule_only: 정적 분석 도구만 사용
    - verify_with_llm: 도구 결과 → LLM 검증
    - llm_detect: LLM 탐지 → 도구 교차 검증
    - hybrid: 신뢰도 기반 선택적 LLM 검증
    """

    def __init__(self, config: ScanConfig, db=None):
        """
        Args:
            config: 스캔 설정
            db: ArsenalDB 인스턴스 (선택)
        """
        self.config = config
        self.db = db
        self.tool_runner = ToolRunner()

        # LLM 분석기는 필요 시에만 초기화
        self.llm_analyzer = None
        if config.enable_llm_verification and config.profile_name:
            self._init_llm_analyzer()

    def _init_llm_analyzer(self):
        """LLM 분석기 초기화"""
        try:
            from .llm.analyzer import LLMSecurityAnalyzer
            self.llm_analyzer = LLMSecurityAnalyzer(self.config.profile_name, self.db)
            print(f"✅ LLM analyzer initialized: {self.llm_analyzer.provider}/{self.llm_analyzer.model}")
        except Exception as e:
            print(f"⚠️  LLM analyzer initialization failed: {e}")
            self.llm_analyzer = None

    async def scan(self) -> SecurityReport:
        """
        메인 스캔 함수

        Returns:
            SecurityReport with findings and statistics
        """
        start_time = time.time()
        report = SecurityReport(
            target=self.config.target,
            mode=self.config.mode,
            scan_type="static"
        )

        # 모드별 스캔 실행
        if self.config.mode == "rule_only":
            findings = await self._scan_rule_only()
        elif self.config.mode == "verify_with_llm":
            findings = await self._scan_verify_with_llm()
        elif self.config.mode == "llm_detect":
            findings = await self._scan_llm_detect()
        elif self.config.mode == "hybrid":
            findings = await self._scan_hybrid()
        else:
            raise ValueError(f"Unknown mode: {self.config.mode}")

        # 결과 저장
        report.findings = findings
        report.scan_duration = time.time() - start_time

        # LLM 통계 추가
        if self.llm_analyzer:
            stats = self.llm_analyzer.get_stats()
            report.llm_calls = stats['calls']
            report.llm_cost = stats['total_cost']

        report.calculate_stats()

        return report

    async def _scan_rule_only(self) -> List[Finding]:
        """
        Mode 1: 정적 분석 도구만 사용
        가장 빠르지만 False Positive 가능성 있음
        """
        # 모든 도구 병렬 실행
        tool_results = await self.tool_runner.run_all_tools(self.config.target)

        # 결과 병합 및 중복 제거
        all_findings = []
        for tool_name, findings in tool_results.items():
            all_findings.extend(findings)

        # 중복 제거 (같은 파일, 같은 라인, 같은 CWE)
        unique_findings = self._deduplicate_findings(all_findings)

        return unique_findings

    async def _scan_verify_with_llm(self) -> List[Finding]:
        """
        Mode 2: 도구 결과 → LLM 검증
        모든 발견사항을 LLM으로 검증 (False Positive 필터링)
        """
        # 1. 도구 실행
        initial_findings = await self._scan_rule_only()

        if not self.llm_analyzer:
            print("⚠️  LLM analyzer not available, falling back to rule_only mode")
            return initial_findings

        print(f"🤖 Verifying {len(initial_findings)} findings with LLM...")

        # 2. LLM 검증
        verified_findings = []
        false_positives = 0

        for i, finding in enumerate(initial_findings, 1):
            print(f"  [{i}/{len(initial_findings)}] Verifying {finding.cwe_id} in {finding.file_path}:{finding.line_number}")

            is_valid, reasoning = await self.llm_analyzer.verify_finding(finding)

            if is_valid:
                finding.verified_by = f"{finding.verified_by}+llm"
                finding.llm_reasoning = reasoning
                verified_findings.append(finding)
            else:
                finding.is_false_positive = True
                false_positives += 1
                print(f"    ✗ False positive: {reasoning[:80]}...")

        print(f"✅ Verification complete: {len(verified_findings)} valid, {false_positives} false positives")

        return verified_findings

    async def _scan_llm_detect(self) -> List[Finding]:
        """
        Mode 3: LLM 탐지 → 도구 교차 검증
        LLM이 먼저 취약점을 찾고, 도구로 확인
        """
        if not self.llm_analyzer:
            print("⚠️  LLM analyzer required for llm_detect mode")
            return []

        # 1. Get code files to scan
        from pathlib import Path
        target_path = Path(self.config.target)

        if target_path.is_file():
            files_to_scan = [str(target_path)]
        else:
            # Get all supported code files in directory
            supported_extensions = [
                '.py', '.js', '.ts', '.jsx', '.tsx', '.vue', '.svelte',
                '.java', '.kt', '.scala',
                '.go', '.rs',
                '.c', '.cpp', '.cc', '.h', '.hpp',
                '.php', '.rb', '.sh', '.bash',
                '.cs', '.swift', '.m', '.mm',
                '.html', '.xml', '.sql'
            ]
            files_to_scan = [
                str(f) for f in target_path.rglob("*")
                if f.suffix.lower() in supported_extensions
            ]

        print(f"🤖 LLM detecting vulnerabilities in {len(files_to_scan)} files...")

        # 2. LLM detection for each file
        llm_findings = []
        for i, file_path in enumerate(files_to_scan, 1):
            print(f"  [{i}/{len(files_to_scan)}] Analyzing {file_path}")
            findings = await self.llm_analyzer.detect_vulnerabilities(file_path)
            llm_findings.extend(findings)
            print(f"    Found {len(findings)} potential issues")

        print(f"📊 LLM found {len(llm_findings)} total findings")

        # 3. Cross-verify with tool results
        print("🔧 Cross-verifying with static analysis tools...")
        tool_findings = await self._scan_rule_only()

        # 4. Cross-verify
        cross_verified = self._cross_verify(llm_findings, tool_findings)

        print(f"✅ Cross-verification complete: {len(cross_verified)} findings")

        return cross_verified

    async def _scan_hybrid(self) -> List[Finding]:
        """
        Mode 4: Hybrid - 신뢰도 기반 선택적 LLM 검증

        전략:
        - 도구 confidence >= threshold → 자동 확정
        - 도구 confidence < threshold → LLM 검증

        목표: 80% 비용 절감 + 95% 정확도 유지
        """
        # 1. 도구 실행
        all_findings = await self._scan_rule_only()

        if not self.llm_analyzer:
            print("⚠️  LLM analyzer not available, using rule_only results")
            return all_findings

        # 2. 신뢰도 기반 분류
        high_confidence = []
        low_confidence = []

        threshold = self.config.llm_confidence_threshold
        for finding in all_findings:
            if finding.confidence >= threshold:
                high_confidence.append(finding)
            else:
                low_confidence.append(finding)

        print(f"\n📊 신뢰도 기반 분류 완료:")
        print(f"  ✅ High confidence: {len(high_confidence)}개 (자동 확정)")
        if high_confidence:
            for finding in high_confidence:
                print(f"    • {finding.severity}: {finding.cwe_id} - {finding.file_path}:{finding.line_number}")
        print(f"  🔍 Low confidence: {len(low_confidence)}개 (LLM 검증 필요)")

        # 3. Low confidence만 LLM 검증 (80% 비용 절감 목표)
        verified_findings = high_confidence.copy()
        false_positives = 0

        if low_confidence:
            print(f"🤖 Verifying {len(low_confidence)} low-confidence findings with LLM...")

            for i, finding in enumerate(low_confidence, 1):
                print(f"  [{i}/{len(low_confidence)}] Verifying {finding.cwe_id} in {finding.file_path}:{finding.line_number}")

                is_valid, reasoning = await self.llm_analyzer.verify_finding(finding)

                if is_valid:
                    finding.verified_by = f"{finding.verified_by}+llm"
                    finding.llm_reasoning = reasoning
                    finding.confidence = 0.9  # LLM 검증 후 신뢰도 상승
                    verified_findings.append(finding)
                    print(f"    ✓ Valid - {finding.severity}: {finding.cwe_id} ({finding.file_path}:{finding.line_number})")
                else:
                    finding.is_false_positive = True
                    false_positives += 1
                    print(f"    ✗ False positive: {reasoning[:80]}...")

            print(f"✅ Hybrid scan complete: {len(high_confidence)} auto-confirmed, {len(low_confidence) - false_positives} LLM-verified, {false_positives} false positives")

        return verified_findings

    def _deduplicate_findings(self, findings: List[Finding]) -> List[Finding]:
        """
        중복 제거 로직

        같은 파일, 같은 라인, 같은 CWE → 가장 높은 confidence만 유지
        """
        unique = {}

        for finding in findings:
            # 키: (파일경로, 라인번호, CWE ID)
            key = (finding.file_path, finding.line_number, finding.cwe_id)

            if key not in unique:
                unique[key] = finding
            else:
                # 더 높은 confidence를 가진 것으로 대체
                if finding.confidence > unique[key].confidence:
                    unique[key] = finding

        return list(unique.values())

    def _cross_verify(
        self,
        llm_findings: List[Finding],
        tool_findings: List[Finding]
    ) -> List[Finding]:
        """
        LLM 결과와 도구 결과 교차 검증

        양쪽에서 모두 발견된 것 → 매우 높은 신뢰도
        한쪽만 발견 → 낮은 신뢰도 (추가 검토 필요)
        """
        # 도구 발견사항을 딕셔너리로 변환
        tool_dict = {}
        for finding in tool_findings:
            key = (finding.file_path, finding.line_number, finding.cwe_id)
            tool_dict[key] = finding

        # 교차 검증
        verified = []
        for llm_finding in llm_findings:
            key = (llm_finding.file_path, llm_finding.line_number, llm_finding.cwe_id)

            if key in tool_dict:
                # 양쪽에서 발견 → 높은 신뢰도
                tool_finding = tool_dict[key]
                llm_finding.confidence = 0.95
                llm_finding.verified_by = f"llm+{tool_finding.verified_by}"
                verified.append(llm_finding)
            else:
                # LLM만 발견 → 낮은 신뢰도
                llm_finding.confidence = 0.6
                verified.append(llm_finding)

        return verified

    async def save_to_db(self, report: SecurityReport) -> int:
        """
        스캔 결과를 DB에 저장

        Returns:
            scan_id
        """
        if not self.db:
            raise ValueError("Database not available")

        # Insert scan report
        scan_id = self.db.insert_security_scan(report)

        # Insert all findings
        for finding in report.findings:
            self.db.insert_security_finding(scan_id, finding)

        return scan_id
