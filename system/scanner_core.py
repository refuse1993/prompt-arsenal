"""
System Scanner Core
모든 스캔 모듈을 통합하는 핵심 엔진
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from .port_scanner import PortScanner, PortInfo
from .cve_matcher import CVEMatcher, CVEInfo
from .llm_analyzer import LLMAnalyzer, AnalysisResult


@dataclass
class ScanFinding:
    """스캔 발견 사항"""
    severity: str  # critical, high, medium, low
    category: str  # port, service, cve, config
    title: str
    description: str
    affected_service: str
    port: Optional[int] = None
    cve_id: Optional[str] = None
    recommendation: str = ""
    exploit_available: bool = False


class SystemScanner:
    """시스템 취약점 스캐너 통합 엔진"""

    def __init__(self, db, vulners_api_key: Optional[str] = None):
        """
        Args:
            db: ArsenalDB 인스턴스
            vulners_api_key: Vulners API 키 (선택)
        """
        self.db = db
        self.port_scanner = PortScanner()
        self.cve_matcher = CVEMatcher(api_key=vulners_api_key)
        self.llm_analyzer = None  # LLM은 스캔 시 동적으로 초기화

    async def scan(self, target: str, scan_type: str = "standard",
                   use_llm: bool = False, llm_config: Optional[Dict] = None) -> Dict:
        """
        시스템 취약점 스캔 실행

        Args:
            target: IP 주소 또는 도메인
            scan_type: quick, standard, full
            use_llm: LLM 분석 사용 여부
            llm_config: LLM 설정 {'provider': 'openai', 'model': 'gpt-4', 'api_key': '...'}

        Returns:
            스캔 결과 딕셔너리
        """
        print(f"\n{'='*80}")
        print(f"🔍 System Vulnerability Scan")
        print(f"{'='*80}")
        print(f"Target: {target}")
        print(f"Scan Type: {scan_type}")
        print(f"LLM Analysis: {'Enabled' if use_llm else 'Disabled'}")
        print(f"{'='*80}\n")

        start_time = datetime.now()
        findings = []

        # 1단계: 포트 스캔
        print("[1/4] 🌐 포트 스캔 및 서비스 탐지...")
        ports = await self.port_scanner.scan(target, scan_type)
        print(f"  ✓ {len(ports)}개 열린 포트 발견\n")

        # 2단계: 위험한 서비스 체크
        print("[2/4] ⚠️  위험한 서비스 분석...")
        dangerous_services = self.port_scanner.check_dangerous_services(ports)
        for svc in dangerous_services:
            findings.append(ScanFinding(
                severity=svc['severity'],
                category='service',
                title=f"Insecure service: {svc['service']}",
                description=svc['reason'],
                affected_service=svc['service'],
                port=svc['port'],
                recommendation=f"Disable {svc['service']} and use secure alternatives"
            ))
        print(f"  ✓ {len(dangerous_services)}개 위험한 서비스 발견\n")

        # 3단계: CVE 매칭
        print("[3/4] 🔎 CVE 취약점 매칭...")
        cve_count = 0
        for port_info in ports:
            if port_info.version:
                # 서비스 버전으로 CVE 검색
                cves = await self.cve_matcher.match_vulnerabilities(
                    port_info.service,
                    port_info.version
                )

                # 우선순위 정렬
                cves = self.cve_matcher.prioritize_vulnerabilities(cves)

                for cve in cves:
                    findings.append(ScanFinding(
                        severity=cve.severity,
                        category='cve',
                        title=f"{cve.cve_id} in {port_info.service} {port_info.version}",
                        description=cve.description,
                        affected_service=f"{port_info.service} {port_info.version}",
                        port=port_info.port,
                        cve_id=cve.cve_id,
                        recommendation=f"Update {port_info.service} to the latest version",
                        exploit_available=cve.exploit_available
                    ))
                    cve_count += 1

        print(f"  ✓ {cve_count}개 CVE 취약점 발견\n")

        # 위험도 점수 계산
        risk_score = self._calculate_risk_score(findings)

        # 스캔 결과 구성
        end_time = datetime.now()
        scan_duration = (end_time - start_time).total_seconds()

        scan_result = {
            'target': target,
            'scan_type': scan_type,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration': scan_duration,
            'open_ports': [asdict(p) for p in ports],
            'findings': [asdict(f) for f in findings],
            'risk_score': risk_score,
            'summary': self._generate_summary(ports, findings, risk_score)
        }

        # 4단계: LLM 분석 (선택)
        llm_analysis_text = None
        if use_llm and llm_config:
            print("[4/4] 🤖 LLM 취약점 분석...")
            try:
                self.llm_analyzer = LLMAnalyzer(
                    provider=llm_config['provider'],
                    model=llm_config['model'],
                    api_key=llm_config['api_key']
                )

                analysis = await self.llm_analyzer.analyze_scan_results(scan_result)
                llm_analysis_text = self.llm_analyzer.format_analysis_report(analysis)
                scan_result['llm_analysis'] = asdict(analysis)
                print(f"  ✓ LLM 분석 완료\n")

            except Exception as e:
                print(f"  ⚠️  LLM 분석 실패: {e}\n")
        else:
            print("[4/4] ⏭️  LLM 분석 건너뛰기\n")

        # 데이터베이스 저장
        scan_id = self.db.insert_system_scan(
            target=target,
            scan_type=scan_type,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            open_ports=json.dumps(scan_result['open_ports']),
            services=json.dumps([{'port': p.port, 'service': p.service, 'version': p.version} for p in ports]),
            findings=json.dumps(scan_result['findings']),
            risk_score=risk_score,
            llm_analysis=llm_analysis_text
        )

        scan_result['scan_id'] = scan_id

        # 결과 출력
        self._print_results(scan_result)

        return scan_result

    def _calculate_risk_score(self, findings: List[ScanFinding]) -> int:
        """위험도 점수 계산 (0-100)"""
        weights = {
            'critical': 40,
            'high': 20,
            'medium': 5,
            'low': 1
        }

        score = 0
        for finding in findings:
            base_score = weights.get(finding.severity, 0)

            # Exploit 가능한 경우 가중치 추가
            if finding.exploit_available:
                base_score *= 1.5

            score += base_score

        return min(int(score), 100)

    def _generate_summary(self, ports: List[PortInfo], findings: List[ScanFinding], risk_score: int) -> Dict:
        """스캔 요약 생성"""
        severity_counts = {
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0
        }

        for finding in findings:
            severity_counts[finding.severity] += 1

        return {
            'total_open_ports': len(ports),
            'total_findings': len(findings),
            'severity_counts': severity_counts,
            'risk_score': risk_score,
            'risk_level': self._get_risk_level(risk_score)
        }

    def _get_risk_level(self, risk_score: int) -> str:
        """위험도 레벨 결정"""
        if risk_score >= 80:
            return 'Critical'
        elif risk_score >= 60:
            return 'High'
        elif risk_score >= 30:
            return 'Medium'
        else:
            return 'Low'

    def _print_results(self, scan_result: Dict):
        """스캔 결과 출력"""
        summary = scan_result['summary']
        risk_score = summary['risk_score']
        risk_level = summary['risk_level']

        print(f"\n{'='*80}")
        print(f"📊 SCAN RESULTS")
        print(f"{'='*80}\n")

        # 위험도 표시
        risk_emoji = {
            'Critical': '🔴',
            'High': '🟠',
            'Medium': '🟡',
            'Low': '🟢'
        }

        print(f"Risk Score: {risk_score}/100 {risk_emoji.get(risk_level, '')} ({risk_level})")
        print(f"Open Ports: {summary['total_open_ports']}")
        print(f"Total Findings: {summary['total_findings']}")
        print()

        # Severity 분포
        severity_counts = summary['severity_counts']
        if severity_counts['critical'] > 0:
            print(f"  🔴 Critical: {severity_counts['critical']}")
        if severity_counts['high'] > 0:
            print(f"  🟠 High: {severity_counts['high']}")
        if severity_counts['medium'] > 0:
            print(f"  🟡 Medium: {severity_counts['medium']}")
        if severity_counts['low'] > 0:
            print(f"  🟢 Low: {severity_counts['low']}")
        print()

        # 주요 발견 사항
        if scan_result['findings']:
            print("Top Findings:")
            for i, finding in enumerate(scan_result['findings'][:5], 1):
                severity_symbol = {
                    'critical': '🔴',
                    'high': '🟠',
                    'medium': '🟡',
                    'low': '🟢'
                }.get(finding['severity'], '⚪')

                print(f"  {i}. {severity_symbol} [{finding['severity'].upper()}] {finding['title']}")
                if finding.get('cve_id'):
                    print(f"     CVE: {finding['cve_id']}")

        print(f"\n{'='*80}")
        print(f"✅ Scan completed in {scan_result['duration']:.2f} seconds")
        print(f"💾 Results saved to database (ID: {scan_result.get('scan_id', 'N/A')})")
        print(f"{'='*80}\n")

    def get_scan_history(self, target: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """스캔 이력 조회"""
        scans = self.db.get_system_scans(limit=limit)

        if target:
            scans = [s for s in scans if s['target'] == target]

        return scans

    def get_scan_details(self, scan_id: int) -> Optional[Dict]:
        """특정 스캔 상세 조회"""
        scan = self.db.get_system_scan_by_id(scan_id)

        if scan:
            # JSON 문자열을 파싱
            scan['open_ports'] = json.loads(scan['open_ports'])
            scan['services'] = json.loads(scan['services'])
            scan['findings'] = json.loads(scan['findings'])

        return scan

    def export_report(self, scan_id: int, format: str = 'json') -> str:
        """스캔 결과 리포트 내보내기"""
        scan = self.get_scan_details(scan_id)

        if not scan:
            return ""

        if format == 'json':
            return json.dumps(scan, indent=2)
        elif format == 'markdown':
            return self._generate_markdown_report(scan)
        else:
            return ""

    def _generate_markdown_report(self, scan: Dict) -> str:
        """마크다운 리포트 생성"""
        report = f"# System Vulnerability Scan Report\n\n"
        report += f"**Target**: {scan['target']}\n"
        report += f"**Scan Type**: {scan['scan_type']}\n"
        report += f"**Date**: {scan['start_time']}\n"
        report += f"**Risk Score**: {scan['risk_score']}/100\n\n"

        report += f"## Summary\n\n"
        report += f"- Open Ports: {len(scan['open_ports'])}\n"
        report += f"- Total Findings: {len(scan['findings'])}\n\n"

        report += f"## Open Ports\n\n"
        for port in scan['open_ports']:
            report += f"- Port {port['port']}/{port['protocol']}: {port['service']}"
            if port.get('version'):
                report += f" ({port['version']})"
            report += "\n"

        report += f"\n## Findings\n\n"
        for finding in scan['findings']:
            severity_icon = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}.get(finding['severity'], '⚪')
            report += f"### {severity_icon} [{finding['severity'].upper()}] {finding['title']}\n\n"
            report += f"**Description**: {finding['description']}\n\n"
            if finding.get('cve_id'):
                report += f"**CVE**: {finding['cve_id']}\n\n"
            report += f"**Recommendation**: {finding['recommendation']}\n\n"

        if scan.get('llm_analysis'):
            report += f"\n## LLM Analysis\n\n"
            report += f"```\n{scan['llm_analysis']}\n```\n\n"

        return report
