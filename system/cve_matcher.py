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
    cvss_v3_score: Optional[float] = None  # CVSS v3.x 점수
    cvss_vector: Optional[str] = None  # CVSS 벡터 문자열
    description: str = ""
    published_date: str = ""
    modified_date: Optional[str] = None
    affected_software: str = ""
    exploit_available: bool = False
    cwe_ids: List[str] = None  # CWE IDs (예: ['CWE-79', 'CWE-89'])
    references: List[str] = None  # 참조 링크들

    def __post_init__(self):
        """Initialize mutable default values"""
        if self.cwe_ids is None:
            self.cwe_ids = []
        if self.references is None:
            self.references = []


class CVEMatcher:
    """CVE 매칭 엔진"""

    def __init__(self, api_key: Optional[str] = None, nvd_api_key: Optional[str] = None):
        """
        Args:
            api_key: Vulners API 키 (선택)
            nvd_api_key: NVD API 키 (선택, 없어도 동작하지만 rate limit 있음)
        """
        self.api_key = api_key
        self.nvd_api_key = nvd_api_key
        self.vulners_available = False
        self.nvd_base_url = "https://services.nvd.nist.gov/rest/json/cves/2.0"

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

        # 1. 로컬 알려진 취약점 DB (우선)
        local_cves = self._query_local_db(service, version)
        cves.extend(local_cves)

        # 2. Vulners API 사용
        if self.vulners_available:
            try:
                vulners_cves = await self._query_vulners_api(service, version)
                cves.extend(vulners_cves)
            except Exception as e:
                print(f"  ⚠️  Vulners API 오류 (계속 진행): {e}")

        # 3. NVD API 사용
        try:
            nvd_cves = await self._query_nvd_api(service, version)
            cves.extend(nvd_cves)
        except Exception as e:
            print(f"  ⚠️  NVD API 오류 (계속 진행): {e}")

        # 중복 제거 (cve_id 기준) - CVSS v3 점수가 있는 것 우선
        unique_cves = {}
        for cve in cves:
            if cve.cve_id not in unique_cves:
                unique_cves[cve.cve_id] = cve
            else:
                # 더 상세한 정보 병합 (NVD가 더 상세함)
                existing = unique_cves[cve.cve_id]
                if cve.cvss_v3_score and not existing.cvss_v3_score:
                    unique_cves[cve.cve_id] = cve
                elif cve.cwe_ids and not existing.cwe_ids:
                    existing.cwe_ids = cve.cwe_ids
                if cve.references and not existing.references:
                    existing.references = cve.references

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

    async def _query_nvd_api(self, service: str, version: str) -> List[CVEInfo]:
        """
        NVD API를 통한 CVE 조회
        https://nvd.nist.gov/developers/vulnerabilities
        """
        cves = []

        try:
            import httpx

            # 키워드 쿼리 생성
            keyword = f"{service} {version}"

            # NVD API 요청 헤더
            headers = {}
            if self.nvd_api_key:
                headers['apiKey'] = self.nvd_api_key

            # API 요청 (keyword search)
            params = {
                'keywordSearch': keyword,
                'resultsPerPage': 10  # 상위 10개만
            }

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    self.nvd_base_url,
                    params=params,
                    headers=headers
                )

                if response.status_code == 200:
                    data = response.json()

                    # CVE 데이터 파싱
                    for item in data.get('vulnerabilities', []):
                        cve_item = item.get('cve', {})
                        cve_id = cve_item.get('id', '')

                        # 설명 추출 (영어)
                        descriptions = cve_item.get('descriptions', [])
                        description = ''
                        for desc in descriptions:
                            if desc.get('lang') == 'en':
                                description = desc.get('value', '')
                                break

                        # CVSS v3.x 점수 추출
                        cvss_v3_score = None
                        cvss_vector = None
                        metrics = cve_item.get('metrics', {})

                        # CVSS v3.1 우선
                        if 'cvssMetricV31' in metrics and metrics['cvssMetricV31']:
                            cvss_data = metrics['cvssMetricV31'][0]['cvssData']
                            cvss_v3_score = cvss_data.get('baseScore', 0.0)
                            cvss_vector = cvss_data.get('vectorString', '')
                        # CVSS v3.0 대체
                        elif 'cvssMetricV30' in metrics and metrics['cvssMetricV30']:
                            cvss_data = metrics['cvssMetricV30'][0]['cvssData']
                            cvss_v3_score = cvss_data.get('baseScore', 0.0)
                            cvss_vector = cvss_data.get('vectorString', '')
                        # CVSS v2 fallback
                        elif 'cvssMetricV2' in metrics and metrics['cvssMetricV2']:
                            cvss_v3_score = metrics['cvssMetricV2'][0]['cvssData'].get('baseScore', 0.0)

                        # Severity 계산
                        severity = self._calculate_severity(cvss_v3_score or 0.0)

                        # CWE IDs 추출
                        cwe_ids = []
                        weaknesses = cve_item.get('weaknesses', [])
                        for weakness in weaknesses:
                            for desc in weakness.get('description', []):
                                cwe_id = desc.get('value', '')
                                if cwe_id.startswith('CWE-'):
                                    cwe_ids.append(cwe_id)

                        # References 추출
                        references = []
                        refs = cve_item.get('references', [])
                        for ref in refs[:5]:  # 최대 5개
                            url = ref.get('url', '')
                            if url:
                                references.append(url)

                        # 발행일 및 수정일
                        published_date = cve_item.get('published', '')[:10]  # YYYY-MM-DD
                        modified_date = cve_item.get('lastModified', '')[:10]

                        cves.append(CVEInfo(
                            cve_id=cve_id,
                            severity=severity,
                            cvss_score=cvss_v3_score or 0.0,
                            cvss_v3_score=cvss_v3_score,
                            cvss_vector=cvss_vector,
                            description=description,
                            published_date=published_date,
                            modified_date=modified_date,
                            affected_software=f"{service} {version}",
                            exploit_available=False,  # NVD doesn't provide exploit info
                            cwe_ids=cwe_ids,
                            references=references
                        ))

                elif response.status_code == 403:
                    print(f"  ℹ️  NVD API rate limit (API 키 권장)")
                else:
                    print(f"  ⚠️  NVD API 오류: {response.status_code}")

        except ImportError:
            print("  ℹ️  httpx 패키지 미설치 - NVD API 스킵")
        except Exception as e:
            print(f"  ⚠️  NVD API 조회 실패: {e}")

        return cves

    def _query_local_db(self, service: str, version: str) -> List[CVEInfo]:
        """로컬 알려진 취약점 DB 조회"""
        cves = []

        # 알려진 주요 취약점 데이터베이스 (확장됨)
        known_vulnerabilities = {
            'openssh': {
                '7.4': [
                    {
                        'cve_id': 'CVE-2018-15473',
                        'severity': 'medium',
                        'cvss_score': 5.3,
                        'cvss_v3_score': 5.3,
                        'description': 'OpenSSH 사용자명 열거 취약점',
                        'published_date': '2018-08-17',
                        'cwe_ids': ['CWE-200'],
                        'references': ['https://www.exploit-db.com/exploits/45233']
                    }
                ],
                '7.7': [
                    {
                        'cve_id': 'CVE-2018-20685',
                        'severity': 'high',
                        'cvss_score': 7.8,
                        'cvss_v3_score': 7.8,
                        'description': 'OpenSSH 권한 상승 취약점',
                        'published_date': '2019-01-10',
                        'cwe_ids': ['CWE-269'],
                        'references': ['https://nvd.nist.gov/vuln/detail/CVE-2018-20685']
                    }
                ],
                '8.2': [
                    {
                        'cve_id': 'CVE-2020-14145',
                        'severity': 'medium',
                        'cvss_score': 5.9,
                        'cvss_v3_score': 5.9,
                        'description': 'OpenSSH 정보 유출 취약점',
                        'published_date': '2020-06-29',
                        'cwe_ids': ['CWE-327'],
                        'references': ['https://nvd.nist.gov/vuln/detail/CVE-2020-14145']
                    }
                ]
            },
            'apache': {
                '2.4.49': [
                    {
                        'cve_id': 'CVE-2021-41773',
                        'severity': 'critical',
                        'cvss_score': 9.8,
                        'cvss_v3_score': 9.8,
                        'description': 'Apache HTTP Server 경로 순회 및 RCE 취약점',
                        'published_date': '2021-10-05',
                        'exploit_available': True,
                        'cwe_ids': ['CWE-22'],
                        'references': [
                            'https://www.exploit-db.com/exploits/50383',
                            'https://nvd.nist.gov/vuln/detail/CVE-2021-41773'
                        ]
                    }
                ],
                '2.4.50': [
                    {
                        'cve_id': 'CVE-2021-42013',
                        'severity': 'critical',
                        'cvss_score': 9.8,
                        'cvss_v3_score': 9.8,
                        'description': 'Apache HTTP Server 경로 순회 및 RCE (CVE-2021-41773 우회)',
                        'published_date': '2021-10-07',
                        'exploit_available': True,
                        'cwe_ids': ['CWE-22'],
                        'references': [
                            'https://www.exploit-db.com/exploits/50406',
                            'https://nvd.nist.gov/vuln/detail/CVE-2021-42013'
                        ]
                    }
                ]
            },
            'nginx': {
                '1.18.0': [
                    {
                        'cve_id': 'CVE-2021-23017',
                        'severity': 'high',
                        'cvss_score': 8.1,
                        'cvss_v3_score': 8.1,
                        'description': 'Nginx DNS resolver off-by-one 버퍼 오버플로우',
                        'published_date': '2021-06-01',
                        'cwe_ids': ['CWE-193'],
                        'references': ['https://nvd.nist.gov/vuln/detail/CVE-2021-23017']
                    }
                ],
                '1.20.0': [
                    {
                        'cve_id': 'CVE-2021-23017',
                        'severity': 'high',
                        'cvss_score': 8.1,
                        'cvss_v3_score': 8.1,
                        'description': 'Nginx DNS resolver off-by-one 버퍼 오버플로우',
                        'published_date': '2021-06-01',
                        'cwe_ids': ['CWE-193'],
                        'references': ['https://nvd.nist.gov/vuln/detail/CVE-2021-23017']
                    }
                ]
            },
            'mysql': {
                '5.7': [
                    {
                        'cve_id': 'CVE-2021-2144',
                        'severity': 'medium',
                        'cvss_score': 6.5,
                        'cvss_v3_score': 6.5,
                        'description': 'MySQL Server 취약점',
                        'published_date': '2021-04-20',
                        'cwe_ids': ['CWE-noinfo'],
                        'references': ['https://nvd.nist.gov/vuln/detail/CVE-2021-2144']
                    }
                ],
                '8.0': [
                    {
                        'cve_id': 'CVE-2021-2307',
                        'severity': 'medium',
                        'cvss_score': 4.9,
                        'cvss_v3_score': 4.9,
                        'description': 'MySQL Server 취약점',
                        'published_date': '2021-04-22',
                        'cwe_ids': ['CWE-noinfo'],
                        'references': ['https://nvd.nist.gov/vuln/detail/CVE-2021-2307']
                    }
                ]
            },
            'postgresql': {
                '13.0': [
                    {
                        'cve_id': 'CVE-2021-32027',
                        'severity': 'high',
                        'cvss_score': 8.8,
                        'cvss_v3_score': 8.8,
                        'description': 'PostgreSQL 버퍼 오버플로우',
                        'published_date': '2021-05-13',
                        'cwe_ids': ['CWE-120'],
                        'references': ['https://nvd.nist.gov/vuln/detail/CVE-2021-32027']
                    }
                ]
            },
            'redis': {
                '6.0': [
                    {
                        'cve_id': 'CVE-2021-32625',
                        'severity': 'high',
                        'cvss_score': 8.8,
                        'cvss_v3_score': 8.8,
                        'description': 'Redis 정수 오버플로우',
                        'published_date': '2021-10-04',
                        'cwe_ids': ['CWE-190'],
                        'references': ['https://nvd.nist.gov/vuln/detail/CVE-2021-32625']
                    }
                ]
            },
            'tomcat': {
                '9.0': [
                    {
                        'cve_id': 'CVE-2021-33037',
                        'severity': 'medium',
                        'cvss_score': 5.3,
                        'cvss_v3_score': 5.3,
                        'description': 'Apache Tomcat HTTP Request Smuggling',
                        'published_date': '2021-07-12',
                        'cwe_ids': ['CWE-444'],
                        'references': ['https://nvd.nist.gov/vuln/detail/CVE-2021-33037']
                    }
                ]
            },
            'wordpress': {
                '5.8': [
                    {
                        'cve_id': 'CVE-2021-39200',
                        'severity': 'high',
                        'cvss_score': 7.5,
                        'cvss_v3_score': 7.5,
                        'description': 'WordPress XXE 취약점',
                        'published_date': '2021-09-09',
                        'exploit_available': True,
                        'cwe_ids': ['CWE-611'],
                        'references': ['https://nvd.nist.gov/vuln/detail/CVE-2021-39200']
                    }
                ]
            },
            'php': {
                '7.4': [
                    {
                        'cve_id': 'CVE-2021-21703',
                        'severity': 'high',
                        'cvss_score': 7.5,
                        'cvss_v3_score': 7.5,
                        'description': 'PHP 로컬 파일 포함',
                        'published_date': '2021-10-21',
                        'cwe_ids': ['CWE-829'],
                        'references': ['https://nvd.nist.gov/vuln/detail/CVE-2021-21703']
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
                        cvss_v3_score=vuln.get('cvss_v3_score'),
                        cvss_vector=vuln.get('cvss_vector'),
                        description=vuln['description'],
                        published_date=vuln['published_date'],
                        affected_software=f"{service} {version}",
                        exploit_available=vuln.get('exploit_available', False),
                        cwe_ids=vuln.get('cwe_ids', []),
                        references=vuln.get('references', [])
                    ))

        # 버전 범위 매칭 (예: 2.4.x)
        version_major = self._extract_major_version(version)
        if service_lower in known_vulnerabilities and version_major:
            for vuln_version in known_vulnerabilities[service_lower].keys():
                if self._version_match(version, vuln_version):
                    for vuln in known_vulnerabilities[service_lower][vuln_version]:
                        # 중복 방지
                        if not any(c.cve_id == vuln['cve_id'] for c in cves):
                            cves.append(CVEInfo(
                                cve_id=vuln['cve_id'],
                                severity=vuln['severity'],
                                cvss_score=vuln['cvss_score'],
                                cvss_v3_score=vuln.get('cvss_v3_score'),
                                cvss_vector=vuln.get('cvss_vector'),
                                description=vuln['description'],
                                published_date=vuln['published_date'],
                                affected_software=f"{service} {version}",
                                exploit_available=vuln.get('exploit_available', False),
                                cwe_ids=vuln.get('cwe_ids', []),
                                references=vuln.get('references', [])
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
