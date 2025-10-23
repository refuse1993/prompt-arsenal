"""
CVE Matcher using Vulners API
서비스 버전에서 CVE 취약점 매칭
"""

import re
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class CVEInfo:
    """CVE 취약점 정보"""
    cve_id: str
    severity: str  # critical, high, medium, low
    cvss_score: float
    description: str
    published_date: str
    affected_software: str
    exploit_available: bool = False


class CVEMatcher:
    """CVE 매칭 엔진"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: Vulners API 키 (선택, 없으면 로컬 DB만 사용)
        """
        self.api_key = api_key
        self.vulners_available = False

        if api_key:
            self._check_vulners_api()

    def _check_vulners_api(self):
        """Vulners API 사용 가능 여부 확인"""
        try:
            import vulners
            self.vulners_api = vulners.Vulners(api_key=self.api_key)
            self.vulners_available = True
        except ImportError:
            print("  ℹ️  vulners 패키지 미설치 - 로컬 DB만 사용")
            self.vulners_available = False
        except Exception as e:
            print(f"  ⚠️  Vulners API 초기화 실패: {e}")
            self.vulners_available = False

    async def match_vulnerabilities(self, service: str, version: str) -> List[CVEInfo]:
        """
        서비스와 버전으로 CVE 매칭

        Args:
            service: 서비스 이름 (예: 'OpenSSH', 'Apache')
            version: 버전 (예: '7.4', '2.4.49')

        Returns:
            CVE 정보 리스트
        """
        cves = []

        # Vulners API 사용
        if self.vulners_available:
            cves.extend(await self._query_vulners_api(service, version))

        # 로컬 알려진 취약점 DB
        cves.extend(self._query_local_db(service, version))

        # 중복 제거 (cve_id 기준)
        unique_cves = {}
        for cve in cves:
            if cve.cve_id not in unique_cves:
                unique_cves[cve.cve_id] = cve

        return list(unique_cves.values())

    async def _query_vulners_api(self, service: str, version: str) -> List[CVEInfo]:
        """Vulners API 조회"""
        cves = []

        try:
            # 소프트웨어 검색 쿼리
            query = f"{service} {version}"
            results = self.vulners_api.softwareVulnerabilities(service, version)

            # 결과 파싱
            if results and isinstance(results, dict):
                for cve_id, cve_data in results.get('vulnerabilities', {}).items():
                    # CVSS 점수 추출
                    cvss_score = cve_data.get('cvss', {}).get('score', 0.0)

                    # Severity 계산
                    severity = self._calculate_severity(cvss_score)

                    cves.append(CVEInfo(
                        cve_id=cve_id,
                        severity=severity,
                        cvss_score=cvss_score,
                        description=cve_data.get('description', ''),
                        published_date=cve_data.get('published', ''),
                        affected_software=f"{service} {version}",
                        exploit_available=cve_data.get('exploit', False)
                    ))

        except Exception as e:
            print(f"  ⚠️  Vulners API 조회 실패: {e}")

        return cves

    def _query_local_db(self, service: str, version: str) -> List[CVEInfo]:
        """로컬 알려진 취약점 DB 조회"""
        cves = []

        # 알려진 주요 취약점 데이터베이스
        known_vulnerabilities = {
            'openssh': {
                '7.4': [
                    {
                        'cve_id': 'CVE-2018-15473',
                        'severity': 'medium',
                        'cvss_score': 5.3,
                        'description': 'OpenSSH 사용자명 열거 취약점',
                        'published_date': '2018-08-17'
                    }
                ],
                '7.7': [
                    {
                        'cve_id': 'CVE-2018-20685',
                        'severity': 'high',
                        'cvss_score': 7.8,
                        'description': 'OpenSSH 권한 상승 취약점',
                        'published_date': '2019-01-10'
                    }
                ]
            },
            'apache': {
                '2.4.49': [
                    {
                        'cve_id': 'CVE-2021-41773',
                        'severity': 'critical',
                        'cvss_score': 9.8,
                        'description': 'Apache HTTP Server 경로 순회 및 RCE 취약점',
                        'published_date': '2021-10-05',
                        'exploit_available': True
                    }
                ],
                '2.4.50': [
                    {
                        'cve_id': 'CVE-2021-42013',
                        'severity': 'critical',
                        'cvss_score': 9.8,
                        'description': 'Apache HTTP Server 경로 순회 및 RCE (CVE-2021-41773 우회)',
                        'published_date': '2021-10-07',
                        'exploit_available': True
                    }
                ]
            },
            'nginx': {
                '1.18.0': [
                    {
                        'cve_id': 'CVE-2021-23017',
                        'severity': 'high',
                        'cvss_score': 8.1,
                        'description': 'Nginx DNS resolver off-by-one 버퍼 오버플로우',
                        'published_date': '2021-06-01'
                    }
                ]
            },
            'mysql': {
                '5.7': [
                    {
                        'cve_id': 'CVE-2021-2144',
                        'severity': 'medium',
                        'cvss_score': 6.5,
                        'description': 'MySQL Server 취약점',
                        'published_date': '2021-04-20'
                    }
                ]
            }
        }

        # 서비스명 정규화
        service_lower = service.lower()

        # 정확한 매치
        if service_lower in known_vulnerabilities:
            if version in known_vulnerabilities[service_lower]:
                for vuln in known_vulnerabilities[service_lower][version]:
                    cves.append(CVEInfo(
                        cve_id=vuln['cve_id'],
                        severity=vuln['severity'],
                        cvss_score=vuln['cvss_score'],
                        description=vuln['description'],
                        published_date=vuln['published_date'],
                        affected_software=f"{service} {version}",
                        exploit_available=vuln.get('exploit_available', False)
                    ))

        # 버전 범위 매칭 (예: 2.4.x)
        version_major = self._extract_major_version(version)
        if service_lower in known_vulnerabilities and version_major:
            for vuln_version in known_vulnerabilities[service_lower].keys():
                if self._version_match(version, vuln_version):
                    for vuln in known_vulnerabilities[service_lower][vuln_version]:
                        cves.append(CVEInfo(
                            cve_id=vuln['cve_id'],
                            severity=vuln['severity'],
                            cvss_score=vuln['cvss_score'],
                            description=vuln['description'],
                            published_date=vuln['published_date'],
                            affected_software=f"{service} {version}",
                            exploit_available=vuln.get('exploit_available', False)
                        ))

        return cves

    def _calculate_severity(self, cvss_score: float) -> str:
        """CVSS 점수로 severity 계산"""
        if cvss_score >= 9.0:
            return 'critical'
        elif cvss_score >= 7.0:
            return 'high'
        elif cvss_score >= 4.0:
            return 'medium'
        else:
            return 'low'

    def _extract_major_version(self, version: str) -> Optional[str]:
        """메이저 버전 추출 (예: '2.4.49' -> '2.4')"""
        parts = version.split('.')
        if len(parts) >= 2:
            return f"{parts[0]}.{parts[1]}"
        return None

    def _version_match(self, version: str, pattern: str) -> bool:
        """버전 패턴 매칭"""
        # 정확한 매치
        if version == pattern:
            return True

        # 와일드카드 매치 (예: 2.4.x)
        if 'x' in pattern:
            pattern_parts = pattern.split('.')
            version_parts = version.split('.')

            for i, part in enumerate(pattern_parts):
                if part == 'x':
                    continue
                if i >= len(version_parts) or version_parts[i] != part:
                    return False
            return True

        return False

    def get_exploit_db_url(self, cve_id: str) -> str:
        """Exploit-DB URL 생성"""
        return f"https://www.exploit-db.com/search?cve={cve_id}"

    def get_nvd_url(self, cve_id: str) -> str:
        """NVD URL 생성"""
        return f"https://nvd.nist.gov/vuln/detail/{cve_id}"

    def prioritize_vulnerabilities(self, cves: List[CVEInfo]) -> List[CVEInfo]:
        """취약점 우선순위 정렬 (심각도 + Exploit 가능성)"""
        severity_order = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}

        def sort_key(cve: CVEInfo):
            return (
                severity_order.get(cve.severity, 0),
                1 if cve.exploit_available else 0,
                cve.cvss_score
            )

        return sorted(cves, key=sort_key, reverse=True)
