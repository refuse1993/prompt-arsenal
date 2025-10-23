# 🔒 Security Scanner - 코드 취약점 분석 시스템 기획서

## 📋 개요

**목적**: 코드 파일의 보안 취약점을 CWE 기준으로 자동 탐지하고, 배포된 서버의 취약점을 분석하는 통합 보안 스캐너

**대상**:
- 정적 분석: 로컬 코드 파일 (Python, JavaScript, Java, C/C++, Go, PHP, etc.)
- 동적 분석: 배포된 웹 서버/API (HTTP/HTTPS)
- AI 모델 분석: LLM을 활용한 복잡한 로직 취약점 탐지

---

## ⭐ 핵심 차별점: API 프로필 통합

**기존 보안 스캐너의 문제점**:
- LLM 사용 시 별도 API 키 설정 필요
- 모델 선택 유연성 부족
- 로컬 LLM 활용 어려움

**Prompt Arsenal Security Scanner의 해결책**:
```yaml
핵심_기능:
  - 기존 config.json의 API 프로필 재사용 ✅
  - 10개 LLM 프로바이더 즉시 지원 (OpenAI, Anthropic, Ollama, etc.) ✅
  - 로컬 모델로 무료 보안 스캔 ✅
  - Hybrid 모드: 규칙 → LLM 검증 (80% 비용 절감) ✅

사용_예시:
  # 사용자는 프로필만 선택
  python interactive_cli.py --security-scan static --dir ./src --mode hybrid --profile ollama

  # Ollama 로컬 모델 → 완전 무료, API 키 불필요
  # GPT-4/Claude 선택 가능 → 이미 설정된 API 키 재사용
  # Hybrid 모드 → 규칙 빠르게 + LLM 정확하게
```

**3가지 분석 모드**:
1. **규칙 기반** (빠름, 무료): 패턴 매칭으로 빠른 스캔
2. **LLM 검증** (정확, 저비용): 규칙 결과를 LLM이 검증
3. **Hybrid** (최적, 추천): 규칙 먼저 → 불확실하면 LLM

---

## 🎯 핵심 기능

### 1. 정적 코드 분석 (SAST - Static Application Security Testing)

**1.1 CWE 기반 취약점 탐지**

| CWE ID | 취약점 유형 | 탐지 방법 | 우선순위 |
|--------|------------|-----------|----------|
| **CWE-79** | XSS (Cross-Site Scripting) | 사용자 입력 → HTML 출력 경로 추적 | Critical |
| **CWE-89** | SQL Injection | SQL 쿼리 문자열 조합 패턴 | Critical |
| **CWE-78** | OS Command Injection | `os.system()`, `exec()`, `subprocess` 사용 | Critical |
| **CWE-22** | Path Traversal | `../` 패턴, 파일 경로 조작 | High |
| **CWE-502** | Deserialization | `pickle.loads()`, `eval()`, `yaml.load()` | Critical |
| **CWE-798** | Hardcoded Credentials | 소스코드 내 비밀번호/API 키 | High |
| **CWE-327** | Weak Crypto | MD5, SHA1, DES 사용 | Medium |
| **CWE-306** | Missing Authentication | 인증 없는 중요 엔드포인트 | High |
| **CWE-862** | Missing Authorization | 권한 검사 누락 | High |
| **CWE-200** | Information Exposure | 민감 정보 로깅, 에러 메시지 노출 | Medium |

**1.2 탐지 엔진 (최신 도구 활용)**

```yaml
Python_정적_분석_도구:
  Semgrep_★추천:
    설명: "최신 정적 분석 도구, 다중 언어 지원"
    장점:
      - YAML 기반 커스텀 규칙 작성 가능
      - 3,000+ 내장 규칙 (OWASP, CWE)
      - 빠른 속도 (AST 기반)
      - CI/CD 통합 쉬움
    설치: "pip install semgrep"
    실행: "semgrep --config=auto --json ."

  Bandit:
    설명: "Python 보안 전용 스캐너"
    장점:
      - Python 특화
      - 간단한 사용법
      - JSON 출력 지원
    설치: "pip install bandit"
    실행: "bandit -r . -f json -o report.json"

  Ruff:
    설명: "초고속 Python linter (Rust 기반)"
    장점:
      - 매우 빠름 (10-100x)
      - 보안 규칙 포함
      - 최신 Python 지원
    설치: "pip install ruff"
    실행: "ruff check --select S --output-format json ."

JavaScript_도구:
  ESLint_Security:
    설치: "npm install eslint-plugin-security"
    실행: "eslint . --format json"

통합_전략:
  1단계: Semgrep/Bandit/Ruff 실행 (빠름, 무료)
  2단계: 도구 결과를 통합/정규화
  3단계: LLM이 결과 검증 및 보완 (API 프로필 사용)
  4단계: False Positive 필터링
  5단계: 추가 비즈니스 로직 분석 (LLM)
```

**도구 실행 예시**:

```python
import subprocess
import json

class StaticAnalyzer:
    """기존 도구들을 통합하여 실행"""

    def run_semgrep(self, target: str) -> List[Finding]:
        """Semgrep 실행 (최신, 추천)"""
        result = subprocess.run(
            ["semgrep", "--config=auto", "--json", target],
            capture_output=True,
            text=True
        )

        semgrep_results = json.loads(result.stdout)
        findings = []

        for r in semgrep_results['results']:
            findings.append(Finding(
                cwe_id=r['extra'].get('metadata', {}).get('cwe'),
                file_path=r['path'],
                line_number=r['start']['line'],
                code_snippet=r['extra']['lines'],
                severity=r['extra']['severity'],
                title=r['check_id'],
                confidence=0.8,  # Semgrep 결과는 높은 신뢰도
                verified_by='semgrep'
            ))

        return findings

    def run_bandit(self, target: str) -> List[Finding]:
        """Bandit 실행 (Python 전용)"""
        result = subprocess.run(
            ["bandit", "-r", target, "-f", "json"],
            capture_output=True,
            text=True
        )

        bandit_results = json.loads(result.stdout)
        findings = []

        for r in bandit_results['results']:
            findings.append(Finding(
                cwe_id=r['test_id'],  # CWE mapping 필요
                file_path=r['filename'],
                line_number=r['line_number'],
                code_snippet=r['code'],
                severity=r['issue_severity'],
                confidence=self._map_confidence(r['issue_confidence']),
                verified_by='bandit'
            ))

        return findings

    def run_ruff(self, target: str) -> List[Finding]:
        """Ruff 실행 (초고속)"""
        result = subprocess.run(
            ["ruff", "check", "--select", "S", "--output-format", "json", target],
            capture_output=True,
            text=True
        )

        ruff_results = json.loads(result.stdout)
        # Ruff 결과 파싱 및 Finding 변환
        ...

    async def run_all_tools(self, target: str) -> List[Finding]:
        """모든 도구 병렬 실행"""
        semgrep_task = asyncio.create_task(self.run_semgrep(target))
        bandit_task = asyncio.create_task(self.run_bandit(target))
        ruff_task = asyncio.create_task(self.run_ruff(target))

        results = await asyncio.gather(semgrep_task, bandit_task, ruff_task)

        # 결과 병합 및 중복 제거
        all_findings = []
        for findings in results:
            all_findings.extend(findings)

        # 중복 제거 (같은 위치, 같은 CWE)
        deduplicated = self._deduplicate_findings(all_findings)

        return deduplicated
```

**1.3 분석 프로세스 (도구 통합 + LLM)**

```python
class SecurityScanner:
    """Semgrep/Bandit/Ruff + LLM 통합 스캐너"""

    def __init__(self, db: ArsenalDB, profile_name: str = None):
        self.db = db
        self.static_analyzer = StaticAnalyzer()  # Semgrep/Bandit/Ruff 실행

        # API 프로필 선택 시 LLM 분석기 활성화
        if profile_name:
            self.llm_analyzer = LLMSecurityAnalyzer(profile_name)
        else:
            self.llm_analyzer = None

    async def scan_directory(self, target: str, mode: str = "hybrid") -> SecurityReport:
        """
        디렉토리 스캔

        모드:
        - rule_only: Semgrep/Bandit/Ruff만 (빠름, 무료)
        - verify_with_llm: 도구 결과 → LLM 검증
        - llm_detect: LLM 탐지 → 도구 교차검증
        - hybrid: 도구 신뢰도 ≥0.8 → 확정, <0.8 → LLM 검증
        """

        # 1단계: Semgrep/Bandit/Ruff 병렬 실행
        console.print("[bold blue]🔍 Running static analysis tools...[/]")
        tool_findings = await self.static_analyzer.run_all_tools(target)

        console.print(f"  Semgrep: {len(tool_findings['semgrep'])} findings")
        console.print(f"  Bandit:  {len(tool_findings['bandit'])} findings")
        console.print(f"  Ruff:    {len(tool_findings['ruff'])} findings")

        # 2단계: 결과 통합 및 중복 제거
        all_findings = self._merge_findings(tool_findings)

        if mode == "rule_only":
            # 도구 결과만 사용
            return SecurityReport(findings=all_findings)

        # 3단계: LLM 검증/분석
        if not self.llm_analyzer:
            console.print("[yellow]⚠️ LLM analyzer not configured. Using rule-only mode.[/]")
            return SecurityReport(findings=all_findings)

        verified_findings = []

        if mode == "verify_with_llm":
            # 모든 결과를 LLM이 검증
            console.print("[bold blue]🤖 Verifying findings with LLM...[/]")
            for finding in all_findings:
                is_valid = await self.llm_analyzer.verify_finding(finding)
                if is_valid:
                    finding.verified_by = f"{finding.verified_by}+llm"
                    verified_findings.append(finding)

        elif mode == "hybrid":
            # Hybrid: 신뢰도 높으면 바로 확정, 낮으면 LLM 검증
            console.print("[bold blue]⚡ Hybrid mode: Smart verification[/]")
            for finding in all_findings:
                if finding.confidence >= 0.8:
                    # 도구 신뢰도 높음 → 바로 확정
                    verified_findings.append(finding)
                else:
                    # 불확실 → LLM 검증
                    is_valid = await self.llm_analyzer.verify_finding(finding)
                    if is_valid:
                        finding.verified_by = f"{finding.verified_by}+llm"
                        verified_findings.append(finding)

        # 4단계: 추가 LLM 분석 (비즈니스 로직 취약점)
        # Semgrep/Bandit이 못 찾는 복잡한 취약점
        if mode == "llm_detect":
            console.print("[bold blue]🧠 LLM deep analysis...[/]")
            for file in self._get_python_files(target):
                code = open(file).read()
                llm_findings = await self.llm_analyzer.detect_vulnerabilities(code, file)

                # LLM 발견 → 도구로 교차검증
                for llm_finding in llm_findings:
                    if self._is_confirmed_by_tools(llm_finding, all_findings):
                        llm_finding['confidence'] = min(1.0, llm_finding['confidence'] + 0.2)
                        llm_finding['verified_by'] = 'llm+tools'
                    verified_findings.append(llm_finding)

        # 5단계: DB 저장
        scan_id = self.db.insert_security_scan(
            scan_type="static",
            target=target,
            status="completed",
            total_findings=len(verified_findings)
        )

        for finding in verified_findings:
            self.db.insert_security_finding(scan_id=scan_id, **finding)

        return SecurityReport(findings=verified_findings, scan_id=scan_id)
```

### 2. 동적 서버 분석 (DAST - Dynamic Application Security Testing)

**2.1 웹 서버 취약점 스캔**

```yaml
스캔_대상:
  웹_취약점:
    - SQL Injection (모든 파라미터)
    - XSS (Reflected, Stored, DOM-based)
    - CSRF (토큰 검증)
    - Path Traversal (파일 다운로드)
    - SSRF (Server-Side Request Forgery)
    - XXE (XML External Entity)

  설정_취약점:
    - 안전하지 않은 HTTP 헤더
    - CORS 설정 오류
    - TLS/SSL 취약점
    - 디렉토리 리스팅
    - 민감 파일 노출 (.git, .env, backup)

  인증_권한:
    - 약한 세션 관리
    - JWT 설정 오류
    - API 키 노출
    - 권한 상승 (Privilege Escalation)
```

**2.2 자동 페이로드 생성**

```python
# Prompt Arsenal의 payload_utils.py 활용
class DynamicScanner:
    def __init__(self, db: ArsenalDB):
        self.db = db
        self.encoder = PayloadEncoder()
        self.generator = PayloadGenerator()

    async def scan_endpoint(self, url: str, params: dict) -> List[Finding]:
        findings = []

        # 1. SQL Injection 테스트
        sql_payloads = self.db.get_prompts(category="sql-injection", limit=100)
        for payload in sql_payloads:
            # Base64, URL encoding 등 변형
            variants = self.generator.generate_variants(payload)

            for variant in variants:
                response = await self.test_payload(url, params, variant)
                if self.is_vulnerable(response):
                    findings.append(Finding(
                        cwe_id="CWE-89",
                        severity="Critical",
                        payload=variant,
                        evidence=response
                    ))

        return findings
```

**2.3 Playwright 통합**

```python
# 브라우저 기반 XSS/CSRF 테스트
async def test_xss_with_browser(url: str, payloads: List[str]):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()

        for payload in payloads:
            await page.goto(f"{url}?q={payload}")

            # alert() 팝업 감지
            page.on("dialog", lambda dialog: dialog.accept())

            # DOM 변조 감지
            if await page.evaluate("() => document.body.innerHTML.includes('<script>')"):
                return Finding(cwe_id="CWE-79", payload=payload)
```

### 3. LLM 기반 고급 분석 (API 프로필 통합)

**3.1 API 프로필 선택 시스템**

```python
# 기존 Prompt Arsenal의 config.json 활용
{
  "profiles": {
    "gpt4": {
      "provider": "openai",
      "model": "gpt-4o",
      "api_key": "sk-..."
    },
    "claude": {
      "provider": "anthropic",
      "model": "claude-3-5-sonnet-20241022",
      "api_key": "sk-ant-..."
    },
    "ollama": {
      "provider": "ollama",
      "model": "llama3.1:8b",
      "base_url": "http://localhost:11434"
    }
  }
}
```

**3.2 LLM 활용 방식 (3가지 모드)**

```python
class LLMSecurityAnalyzer:
    """
    기존 LLMClient + API 프로필 통합
    """

    def __init__(self, profile_name: str, config_path: str = "config.json"):
        # 기존 API 프로필 로드
        from multimodal.llm_client import LLMClient

        with open(config_path) as f:
            config = json.load(f)

        profile = config['profiles'][profile_name]
        self.llm = LLMClient(
            provider=profile['provider'],
            model=profile['model'],
            api_key=profile.get('api_key'),
            base_url=profile.get('base_url')
        )

    # 모드 1: 규칙 기반 → LLM 검증 (False Positive 필터)
    async def verify_finding(self, finding: Finding) -> bool:
        """규칙 기반 탐지 결과를 LLM이 검증"""
        prompt = f"""
보안 스캐너가 다음 취약점을 탐지했습니다. 실제 취약점인지 검증하세요.

CWE-{finding.cwe_id}: {finding.cwe_name}
위치: {finding.file_path}:{finding.line_number}

취약 코드:
```
{finding.code_snippet}
```

주변 컨텍스트:
```
{finding.context_code}
```

질문:
1. 이것이 실제 취약점인가요? (True/False)
2. 이유는 무엇인가요?
3. 오탐(False Positive)인 경우 왜 그런가요?

JSON 형식으로 응답:
{{
  "is_vulnerable": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "설명",
  "attack_scenario": "공격 시나리오 (취약한 경우)",
  "false_positive_reason": "오탐 이유 (오탐인 경우)"
}}
"""

        response = await self.llm.complete(prompt)
        result = json.loads(response.content)

        # HybridJudge처럼 confidence 기반 결정
        if result['confidence'] >= 0.8:
            return result['is_vulnerable']
        else:
            # 불확실하면 보수적으로 취약점으로 간주 (수동 검토 필요)
            return True

    # 모드 2: LLM 직접 탐지 → 규칙으로 확인
    async def detect_vulnerabilities(self, code: str, file_path: str) -> List[Finding]:
        """LLM이 직접 취약점 탐지"""
        prompt = f"""
당신은 보안 전문가입니다. 다음 코드를 분석하여 CWE 기준 보안 취약점을 찾으세요.

파일: {file_path}
코드:
```
{code}
```

분석 항목:
1. CWE-79 (XSS): 사용자 입력이 HTML에 출력되는가?
2. CWE-89 (SQL Injection): SQL 쿼리가 문자열 조합으로 생성되는가?
3. CWE-78 (Command Injection): 외부 명령 실행에 사용자 입력이 사용되는가?
4. CWE-22 (Path Traversal): 파일 경로에 '../' 필터링이 없는가?
5. CWE-502 (Deserialization): pickle/yaml.load 등 안전하지 않은 역직렬화?
6. CWE-798 (Hardcoded Credentials): 소스코드에 비밀번호/API 키가 하드코딩?
7. CWE-327 (Weak Crypto): MD5, SHA1, DES 등 약한 암호화?
8. CWE-306 (Missing Auth): 인증 검사 없는 중요 함수?
9. CWE-862 (Missing Authorization): 권한 검사 누락?
10. CWE-200 (Info Exposure): 민감 정보가 로그에 출력?

각 발견 사항:
{{
  "cwe_id": "CWE-XXX",
  "cwe_name": "취약점 이름",
  "severity": "Critical/High/Medium/Low",
  "confidence": 0.0-1.0,
  "line_number": 숫자,
  "column_number": 숫자,
  "title": "간단한 제목",
  "description": "상세 설명",
  "attack_scenario": "공격 시나리오",
  "remediation": "수정 방안",
  "code_snippet": "취약 코드 라인"
}}

JSON 배열로 응답하세요.
"""

        response = await self.llm.complete(prompt)
        llm_findings = json.loads(response.content)

        # 규칙 기반 스캐너로 교차 검증
        rule_findings = await self.rule_scanner.scan(code)

        # LLM + 규칙 모두 탐지한 것만 확실한 취약점
        verified_findings = []
        for llm_finding in llm_findings:
            if self._is_confirmed_by_rules(llm_finding, rule_findings):
                llm_finding['verified_by'] = 'llm+rule'
                llm_finding['confidence'] = min(1.0, llm_finding['confidence'] + 0.2)
                verified_findings.append(llm_finding)
            elif llm_finding['confidence'] >= 0.9:
                # LLM 신뢰도 높으면 규칙 없이도 포함
                llm_finding['verified_by'] = 'llm_only'
                verified_findings.append(llm_finding)

        return verified_findings

    # 모드 3: Hybrid (빠른 규칙 → 불확실하면 LLM)
    async def hybrid_analyze(self, code: str, file_path: str) -> List[Finding]:
        """HybridJudge 패턴 적용"""
        # 1단계: 규칙 기반 빠른 스캔
        rule_findings = await self.rule_scanner.scan(code)

        verified_findings = []

        for finding in rule_findings:
            # 규칙의 신뢰도가 높으면 바로 포함
            if finding.confidence >= 0.9:
                finding.verified_by = 'rule'
                verified_findings.append(finding)
            else:
                # 불확실하면 LLM 검증
                is_valid = await self.verify_finding(finding)
                if is_valid:
                    finding.verified_by = 'rule+llm'
                    verified_findings.append(finding)

        return verified_findings
```

**3.2 False Positive 필터링**

```python
async def filter_false_positives(self, findings: List[Finding]) -> List[Finding]:
    """
    LLM을 활용한 오탐 제거
    """
    for finding in findings:
        prompt = f"""
다음 보안 취약점 탐지 결과가 실제 취약점인지 검증하세요.

발견사항:
- CWE: {finding.cwe_id}
- 위치: {finding.location}
- 코드: {finding.code_snippet}

이것이 실제 취약점인지, 오탐인지 판단하고 이유를 설명하세요.
"""

        judgment = await self.llm_client.complete(prompt)
        finding.is_valid = judgment.is_vulnerable
        finding.reasoning = judgment.reasoning

    return [f for f in findings if f.is_valid]
```

---

## 🏗️ 시스템 아키텍처

### 디렉토리 구조

```
prompt_arsenal/
├── security/                      # 🆕 Security Scanner Module
│   ├── __init__.py
│   ├── scanner.py                 # 메인 스캐너 인터페이스
│   │
│   ├── static/                    # 정적 분석
│   │   ├── __init__.py
│   │   ├── ast_analyzer.py        # AST 기반 분석
│   │   ├── rule_engine.py         # 규칙 기반 탐지
│   │   ├── taint_analysis.py      # 데이터 흐름 추적
│   │   ├── language_parsers/      # 언어별 파서
│   │   │   ├── python_parser.py
│   │   │   ├── javascript_parser.py
│   │   │   └── java_parser.py
│   │   └── rules/                 # CWE 탐지 규칙
│   │       ├── cwe_79_xss.yaml
│   │       ├── cwe_89_sqli.yaml
│   │       └── cwe_798_secrets.yaml
│   │
│   ├── dynamic/                   # 동적 분석
│   │   ├── __init__.py
│   │   ├── web_scanner.py         # 웹 취약점 스캔
│   │   ├── api_scanner.py         # API 취약점 스캔
│   │   ├── payload_generator.py   # 페이로드 생성 (기존 payload_utils 활용)
│   │   └── browser_tester.py      # Playwright 기반 테스트
│   │
│   ├── llm/                       # LLM 분석
│   │   ├── __init__.py
│   │   ├── code_analyzer.py       # 코드 심층 분석
│   │   ├── logic_analyzer.py      # 비즈니스 로직 분석
│   │   └── false_positive_filter.py
│   │
│   ├── cwe/                       # CWE 데이터베이스
│   │   ├── __init__.py
│   │   ├── cwe_database.py        # CWE 정보 관리
│   │   └── cwe_data.json          # CWE 목록 및 설명
│   │
│   └── reporters/                 # 리포트 생성
│       ├── __init__.py
│       ├── html_reporter.py       # HTML 리포트
│       ├── json_reporter.py       # JSON 리포트
│       └── sarif_reporter.py      # SARIF 포맷 (GitHub 호환)
│
├── dashboard/
│   ├── security_dashboard.html    # 🆕 보안 스캔 결과 대시보드
│   └── api.py                     # API 확장 (보안 스캔 엔드포인트)
│
└── core/
    └── database.py                # 🆕 security_scans 테이블 추가
```

### 데이터베이스 스키마

```sql
-- 보안 스캔 메타데이터
CREATE TABLE security_scans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_type TEXT NOT NULL,              -- 'static', 'dynamic', 'llm'
    target TEXT NOT NULL,                 -- 파일 경로 또는 URL
    scan_config TEXT,                     -- JSON 설정
    status TEXT DEFAULT 'running',        -- 'running', 'completed', 'failed'
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    duration_seconds REAL,
    total_findings INTEGER DEFAULT 0,
    critical_count INTEGER DEFAULT 0,
    high_count INTEGER DEFAULT 0,
    medium_count INTEGER DEFAULT 0,
    low_count INTEGER DEFAULT 0
);

-- 취약점 발견 사항
CREATE TABLE security_findings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id INTEGER NOT NULL,
    cwe_id TEXT NOT NULL,                 -- 'CWE-79', 'CWE-89', etc.
    cwe_name TEXT,                        -- 'Cross-Site Scripting', 'SQL Injection'
    severity TEXT NOT NULL,               -- 'Critical', 'High', 'Medium', 'Low'
    confidence REAL DEFAULT 1.0,          -- 0.0 - 1.0 (LLM 분석 시)

    -- 위치 정보
    file_path TEXT,
    line_number INTEGER,
    column_number INTEGER,
    function_name TEXT,

    -- 상세 정보
    title TEXT NOT NULL,
    description TEXT,
    attack_scenario TEXT,                 -- 공격 시나리오
    remediation TEXT,                     -- 수정 방안

    -- 증거
    code_snippet TEXT,                    -- 취약 코드
    payload TEXT,                         -- 사용된 페이로드 (동적 분석)
    response_evidence TEXT,               -- 응답 증거

    -- 상태
    status TEXT DEFAULT 'open',           -- 'open', 'confirmed', 'false_positive', 'fixed'
    verified_by TEXT,                     -- 'rule', 'llm', 'manual'

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (scan_id) REFERENCES security_scans(id)
);

-- CWE 정보 캐시
CREATE TABLE cwe_database (
    cwe_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    extended_description TEXT,
    common_consequences TEXT,             -- JSON
    likelihood TEXT,                      -- 'High', 'Medium', 'Low'
    mitigation_strategies TEXT,           -- JSON
    related_cwes TEXT,                    -- JSON array
    owasp_top10_mapping TEXT,             -- 'A03:2021', etc.

    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 인덱스
CREATE INDEX idx_findings_scan ON security_findings(scan_id);
CREATE INDEX idx_findings_severity ON security_findings(severity);
CREATE INDEX idx_findings_cwe ON security_findings(cwe_id);
CREATE INDEX idx_scans_target ON security_scans(target);
```

---

## 🔄 워크플로우

### 워크플로우 1: 정적 코드 분석

```
1. 사용자 입력
   ├─ 단일 파일: /path/to/app.py
   ├─ 디렉토리: /path/to/project/
   └─ GitHub URL: https://github.com/user/repo

2. 스캔 설정
   ├─ CWE 카테고리 선택 (전체 / Injection / XSS / Auth / Crypto)
   ├─ 분석 레벨 (빠름 / 표준 / 심층)
   └─ LLM 분석 활성화 (예/아니오)

3. 분석 실행
   ├─ 규칙 기반 스캔 (1-10초)
   ├─ AST 분석 (10-60초)
   ├─ 데이터 흐름 추적 (30-120초)
   └─ LLM 심층 분석 (60-300초) [옵션]

4. 결과 처리
   ├─ False Positive 필터링
   ├─ 심각도 우선순위 정렬
   └─ DB 저장

5. 리포트 생성
   ├─ 터미널 출력 (Rich 테이블)
   ├─ HTML 리포트
   ├─ JSON/SARIF 내보내기
   └─ 대시보드 연동
```

### 워크플로우 2: 동적 서버 스캔

```
1. 서버 정보 입력
   ├─ URL: https://example.com
   ├─ 인증: API Key / Session Cookie / JWT
   └─ 크롤링 깊이 (1-5)

2. 크롤링 & 엔드포인트 수집
   ├─ Sitemap.xml 파싱
   ├─ 링크 크롤링 (Playwright)
   └─ API 엔드포인트 자동 탐지

3. 취약점 테스트
   ├─ SQL Injection (모든 파라미터)
   ├─ XSS (Reflected, Stored)
   ├─ Path Traversal
   ├─ SSRF
   └─ 설정 취약점 (Headers, TLS, CORS)

4. 페이로드 변형
   ├─ Base64 인코딩
   ├─ URL 인코딩
   ├─ Unicode 변환
   └─ 대소문자 변형

5. 결과 검증
   ├─ 응답 코드 분석
   ├─ 에러 메시지 탐지
   ├─ 페이지 변조 확인
   └─ LLM 기반 검증

6. 리포트 생성
```

---

## 📊 사용 예시

### 예시 1: Interactive CLI 사용 (추천)

```
$ python interactive_cli.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔒 Prompt Arsenal - Main Menu
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🛡️ Security Scanning
  s1. 정적 코드 분석 (SAST)
  s2. 동적 서버 스캔 (DAST)
  s3. 스캔 결과 조회
  s4. 보안 대시보드

선택 > s1
```

**시나리오 A: 빠른 스캔 (도구만, 무료)**

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
정적 코드 분석 (SAST)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

스캔 대상 선택:
  1. 단일 파일
  2. 디렉토리
  3. GitHub 리포지토리

선택 > 2

디렉토리 경로: ./src

분석 모드 선택:
  1. 도구만 (빠름, 무료) ← 이것 선택
  2. 도구 + LLM 검증 (정확, API 비용)
  3. LLM 직접 탐지 (고급, API 비용 높음)
  4. Hybrid 모드 (추천, 최적 균형)

선택 > 1

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
스캔 시작 (도구만)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔍 Running static analysis tools...
  [██████████----------] Semgrep (50%)
  [████████████████----] Bandit (80%)
  [████████████████████] Ruff (100%)

Results:
  Semgrep: 12 findings
  Bandit:   8 findings
  Ruff:     5 findings

Merged: 18 unique findings (7 duplicates removed)

총 발견 사항: 18개
  Critical: 3
  High: 6
  Medium: 7
  Low: 2

스캔 시간: 8초
비용: $0 (무료)

리포트: /tmp/security_scan_20251023_143045.html
```

**시나리오 B: Hybrid 모드 (추천, Ollama 무료)**

```
선택 > 4 (Hybrid 모드)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
API 프로필 선택
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

사용 가능한 프로필:
  1. gpt4         (OpenAI GPT-4o)        - 정확도 최고, $0.03/파일
  2. gpt4_mini    (OpenAI GPT-4o-mini)   - 빠르고 저렴, $0.01/파일
  3. claude       (Anthropic Claude 3.5) - 코드 분석 강함, $0.02/파일
  4. ollama       (Ollama llama3.1:8b)   - 로컬 무료 ← 이것 선택
  5. local_llama  (Local LLM)            - 완전 무료

선택 > 4

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
스캔 시작 (Hybrid 모드)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔍 Running static analysis tools...
  Semgrep: 12 findings
  Bandit:   8 findings
  Ruff:     5 findings

⚡ Hybrid mode: Smart verification

[1/18] CWE-89: SQL Injection (src/api/users.py:45)
  도구 신뢰도: 95% → ✅ 확정 (LLM 검증 불필요)

[2/18] CWE-798: Hardcoded Secret (src/config.py:12)
  도구 신뢰도: 70% → 🤖 LLM 검증 중...
  Ollama 응답: ✅ 실제 취약점 (신뢰도 92%)

[3/18] CWE-306: Missing Auth (src/auth.py:23)
  도구 신뢰도: 65% → 🤖 LLM 검증 중...
  Ollama 응답: ❌ False Positive (테스트 코드)

...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
스캔 완료
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

총 발견 사항: 15개 (3개 오탐 제거됨)
  Critical: 2
  High: 5
  Medium: 6
  Low: 2

스캔 시간: 2분 18초
LLM 호출: 5회 (신뢰도 낮은 것만)
비용: $0 (Ollama 로컬)

검증 통계:
  도구만으로 확정: 13개 (72%)
  LLM 검증 필요: 5개 (28%)
  False Positive 제거: 3개

리포트: /tmp/security_scan_20251023_143045.html
```

**시나리오 C: LLM 직접 탐지 (고급, GPT-4)**

```
선택 > 3 (LLM 직접 탐지)
프로필 선택 > 1 (gpt4)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
스캔 시작 (LLM 직접 탐지)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔍 Running static analysis tools... (병렬)
🧠 LLM deep analysis... (병렬)

GPT-4 분석 중:
  [1/45] src/api/users.py
    → CWE-89: SQL Injection (Line 45)
    → CWE-862: Missing Authorization (Line 67)
    → CWE-201: Info Leak in Error Message (Line 89)

  [2/45] src/auth.py
    → CWE-306: Missing Session Timeout (Line 123)
    → 비즈니스 로직: Race Condition (Line 156) ← 도구가 못 찾음!

교차 검증:
  LLM+도구 모두 탐지: 18개 (높은 신뢰도)
  LLM만 탐지: 5개 (비즈니스 로직 취약점)
  도구만 탐지: 3개 (LLM이 오탐으로 판단)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
스캔 완료
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

총 발견 사항: 23개
  Critical: 4
  High: 8
  Medium: 8
  Low: 3

비즈니스 로직 취약점 (LLM 전용): 5개
  - Race Condition in payment processing
  - Weak session management
  - Missing rate limiting
  - Improper error handling revealing DB schema
  - TOCTOU vulnerability in file upload

스캔 시간: 5분 42초
비용: $1.35 (GPT-4, 45 files)

리포트: /tmp/security_scan_20251023_143045.html
```

**출력 예시**:
```
🔍 Security Scan Started
Target: /path/to/project/src
Type: Static Analysis
CWE Categories: ALL

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Scan Progress
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[████████████████████----] 80% (40/50 files)
Current: src/api/auth.py (CWE-306 detected)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚨 Critical Findings (3)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[CWE-89] SQL Injection
  File: src/api/users.py:45
  Code: f"SELECT * FROM users WHERE id = {user_id}"
  Risk: Attacker can execute arbitrary SQL queries
  Fix:  Use parameterized queries

[CWE-798] Hardcoded Credentials
  File: src/config.py:12
  Code: API_KEY = "sk-1234567890abcdef"
  Risk: Exposed API key in source code
  Fix:  Use environment variables

[CWE-502] Unsafe Deserialization
  File: src/utils.py:89
  Code: pickle.loads(user_data)
  Risk: Remote code execution via crafted pickle
  Fix:  Use JSON instead of pickle

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Files:     50
Total Findings:  27
  Critical:      3
  High:          8
  Medium:        12
  Low:           4

Scan Duration: 45.2s
Report: /tmp/security_report_20251023_142530.html
```

### 예시 2: 동적 서버 스캔 (Interactive)

```
선택 > s2 (동적 서버 스캔)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
동적 서버 스캔 (DAST)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

대상 URL: https://example.com

인증 방법:
  1. 없음
  2. Bearer Token
  3. Session Cookie
  4. API Key

선택 > 2

Bearer Token: sk-...

크롤링 깊이 (1-5): 2

테스트 카테고리:
  1. 전체
  2. Injection만 (SQL, XSS, Command)
  3. 인증/권한
  4. 설정 취약점

선택 > 1

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
스캔 시작
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🌐 크롤링 중...
  발견된 엔드포인트: 47개
  발견된 폼: 12개

🔍 취약점 테스트 중...
  [1/47] GET /api/users?id=1
    → SQL Injection 테스트 (100개 페이로드)
    → XSS 테스트 (50개 페이로드)

  [2/47] POST /api/login
    → 인증 우회 테스트
    → Brute Force 방어 테스트

...

총 발견 사항: 8개
  Critical: 2 (SQL Injection, Missing Auth)
  High: 3 (XSS, CSRF)
  Medium: 2 (Info Leak)
  Low: 1 (Weak Headers)

스캔 시간: 12분 35초
```

### 예시 3: Python API (API 프로필 활용)

```python
from security.scanner import SecurityScanner
from security.llm.code_analyzer import LLMSecurityAnalyzer

# 1. 규칙 기반 빠른 스캔
scanner = SecurityScanner(db=db)

report = await scanner.scan_static(
    target="./src",
    mode="rule_only",  # 규칙만
    cwe_categories=["CWE-79", "CWE-89", "CWE-22"]
)

# 2. LLM 검증 모드 (API 프로필 사용)
report = await scanner.scan_static(
    target="./src",
    mode="verify_with_llm",  # 규칙 → LLM 검증
    profile_name="gpt4"      # config.json의 프로필
)

# 3. LLM 직접 탐지 모드
report = await scanner.scan_static(
    target="./src/auth.py",
    mode="llm_detect",       # LLM 탐지 → 규칙 교차검증
    profile_name="claude"
)

# 4. Hybrid 모드 (HybridJudge 패턴)
report = await scanner.scan_static(
    target="./src",
    mode="hybrid",           # 규칙 먼저, 불확실하면 LLM
    profile_name="ollama"    # 로컬 모델로 비용 절감
)

# 5. 수동으로 LLM 분석기 생성
llm_analyzer = LLMSecurityAnalyzer(
    profile_name="claude",   # API 프로필 자동 로드
    config_path="config.json"
)

findings = await llm_analyzer.detect_vulnerabilities(
    code=open("app.py").read(),
    file_path="app.py"
)

for finding in findings:
    print(f"[{finding['cwe_id']}] {finding['title']}")
    print(f"  신뢰도: {finding['confidence']:.0%}")
    print(f"  검증: {finding['verified_by']}")  # 'llm+rule', 'llm_only', 'rule'
    print(f"  수정: {finding['remediation']}")

# 6. False Positive 필터링
verified = await llm_analyzer.verify_finding(finding)
if verified:
    print("✅ 실제 취약점")
else:
    print("❌ 오탐 (False Positive)")

# 7. 동적 스캔 (기존 페이로드 재사용)
dynamic_report = await scanner.scan_dynamic(
    url="https://example.com",
    auth_token="Bearer xyz",
    test_payloads=db.get_prompts(category="sql-injection"),
    verify_with_llm=True,    # LLM으로 결과 검증
    profile_name="gpt4"
)
```

### 예시 4: Interactive CLI 메뉴 (API 프로필 통합)

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔒 Prompt Arsenal - Security Scanner
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🛡️ Security Scanning
  s1. 정적 코드 분석 (SAST)
  s2. 동적 서버 스캔 (DAST)
  s3. 스캔 결과 조회
  s4. 보안 대시보드

🔍 취약점 관리
  v1. 발견 사항 목록
  v2. CWE 데이터베이스
  v3. 취약점 상태 변경
  v4. False Positive 관리

📊 리포트
  r1. HTML 리포트 생성
  r2. JSON/SARIF 내보내기
  r3. 통계 보기

선택 > s1

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
정적 코드 분석 (SAST)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

스캔 대상 선택:
  1. 단일 파일
  2. 디렉토리
  3. GitHub 리포지토리

선택 > 2

디렉토리 경로: ./src

분석 모드 선택:
  1. 규칙 기반 (빠름, 무료)
  2. 규칙 + LLM 검증 (정확, API 비용 발생)
  3. LLM 직접 탐지 (고급, API 비용 높음)
  4. Hybrid 모드 (추천, 최적 균형)

선택 > 4

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
API 프로필 선택 (Hybrid 모드)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

사용 가능한 프로필:
  1. gpt4         (OpenAI GPT-4o)           - 정확도 높음
  2. gpt4_mini    (OpenAI GPT-4o-mini)      - 빠르고 저렴
  3. claude       (Anthropic Claude 3.5)    - 코드 분석 강함
  4. ollama       (Ollama llama3.1:8b)      - 로컬, 무료
  5. local_llama  (Local LLM)               - 완전 무료

  n. 새 프로필 추가

선택 > 4 (ollama)

CWE 카테고리 선택 (전체/선택):
  1. 전체 (Top 25)
  2. Injection (SQL, Command, XSS)
  3. Authentication & Authorization
  4. Cryptography
  5. Data Exposure

선택 > 1

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
스캔 시작
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Target: ./src (45 files)
Mode: Hybrid (Rule → Ollama llama3.1:8b)
CWE: Top 25

[████████████████████----] 80% (36/45 files)

규칙 기반 탐지: 15개
LLM 검증 필요: 5개
  → LLM 검증 중: src/auth.py (CWE-306)
  → ✅ 실제 취약점 (신뢰도 95%)

현재 파일: src/api/users.py

[발견] CWE-89: SQL Injection
  Line 45: f"SELECT * FROM users WHERE id = {user_id}"
  규칙 신뢰도: 70% → LLM 검증 요청 중...
  LLM 응답: ✅ 취약 (신뢰도 98%)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
스캔 완료
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

총 발견 사항: 18개
  Critical: 3
  High: 6
  Medium: 7
  Low: 2

False Positive 필터: 2개 제거됨 (LLM 검증)

스캔 시간: 2분 15초
LLM 비용: $0.08 (Ollama 로컬이라 무료)

리포트: /tmp/security_scan_20251023_143045.html

다음 작업:
  1. 발견 사항 상세보기
  2. HTML 리포트 열기
  3. False Positive 관리
  4. 새 스캔 시작

선택 >
```

---

## 🎨 대시보드 UI 설계

### Security Dashboard (security_dashboard.html)

```html
<!DOCTYPE html>
<html>
<head>
    <title>Security Scanner Dashboard</title>
</head>
<body>
    <!-- 상단: 전체 통계 -->
    <div class="summary-cards">
        <div class="card critical">
            <h2>3</h2>
            <p>Critical</p>
        </div>
        <div class="card high">
            <h2>8</h2>
            <p>High</p>
        </div>
        <div class="card medium">
            <h2>12</h2>
            <p>Medium</p>
        </div>
        <div class="card low">
            <h2>4</h2>
            <p>Low</p>
        </div>
    </div>

    <!-- 좌측: CWE 분포 차트 -->
    <div class="left-panel">
        <h3>Top 10 CWE Categories</h3>
        <canvas id="cwe-chart"></canvas>
    </div>

    <!-- 중앙: 발견 사항 테이블 -->
    <div class="main-panel">
        <h3>Recent Findings</h3>
        <table id="findings-table">
            <thead>
                <tr>
                    <th>Severity</th>
                    <th>CWE</th>
                    <th>Title</th>
                    <th>Location</th>
                    <th>Status</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
                <!-- Dynamic content -->
            </tbody>
        </table>
    </div>

    <!-- 우측: 스캔 히스토리 -->
    <div class="right-panel">
        <h3>Scan History</h3>
        <div id="scan-timeline">
            <!-- Timeline of scans -->
        </div>
    </div>

    <!-- 하단: 트렌드 차트 -->
    <div class="bottom-panel">
        <h3>Security Trend</h3>
        <canvas id="trend-chart"></canvas>
    </div>
</body>
</html>
```

---

## 🚀 구현 로드맵

### Phase 1: 정적 분석 기본 (2-3주)

**Week 1**: 도구 통합 및 기본 구조
- [ ] 프로젝트 구조 생성 (`security/` 디렉토리)
- [ ] **Semgrep 통합**: JSON 출력 파싱, Finding 객체 변환
- [ ] **Bandit 통합**: JSON 출력 파싱, CWE 매핑
- [ ] **Ruff 통합**: 보안 규칙 (--select S) 실행
- [ ] 도구 결과 병합 및 중복 제거 로직
- [ ] CWE 데이터베이스 구축 (Top 25)

**Week 2**: DB 및 CLI 통합
- [ ] DB 스키마 추가 (security_scans, security_findings, cwe_database)
- [ ] Interactive CLI 메뉴 통합 (s1, s2, s3)
- [ ] 스캔 대상 선택 (파일/디렉토리/GitHub)
- [ ] 분석 모드 선택 (도구만/Hybrid/LLM)
- [ ] API 프로필 선택 UI

**Week 3**: 리포팅
- [ ] Rich 테이블 출력 (진행 상황, 결과 요약)
- [ ] HTML 리포트 생성
- [ ] JSON/SARIF 내보내기
- [ ] CWE 상세 정보 표시

**결과물**: Interactive CLI에서 `s1` 선택 → Semgrep/Bandit/Ruff 스캔 동작

---

### Phase 2: LLM 통합 (2-3주)

**Week 4-5**: LLM 분석기 구현
- [ ] **LLMSecurityAnalyzer 클래스**: API 프로필 자동 로드
- [ ] **verify_finding()**: 도구 결과 → LLM 검증 (False Positive 필터)
- [ ] **detect_vulnerabilities()**: LLM 직접 탐지 → 도구 교차검증
- [ ] **hybrid_analyze()**: HybridJudge 패턴 적용
- [ ] 비즈니스 로직 취약점 탐지 프롬프트 설계
- [ ] Ollama 로컬 모델 테스트 (무료 스캔)

**Week 6**: Hybrid 모드 최적화
- [ ] 신뢰도 임계값 조정 (0.8 기준)
- [ ] LLM 호출 최소화 로직 (비용 절감)
- [ ] 검증 통계 수집 (도구 확정 비율, LLM 호출 횟수)
- [ ] 성능 측정 (스캔 시간, 비용)

**결과물**: Hybrid 모드로 80% 비용 절감 + 95% 정확도 달성

---

### Phase 3: 동적 스캔 & 대시보드 (2-3주)

**Week 7-8**: 동적 스캐너
- [ ] 웹 크롤러 (Playwright 통합)
- [ ] SQL Injection 페이로드 테스트 (기존 payload_utils 활용)
- [ ] XSS 페이로드 테스트 (DOM 기반)
- [ ] 기존 프롬프트 DB 활용 (40,000+ 페이로드)
- [ ] 동적 스캔 결과 DB 저장

**Week 9**: 보안 대시보드
- [ ] `security_dashboard.html` 생성
- [ ] API 엔드포인트 (`/api/security/scans`, `/api/security/findings`)
- [ ] 실시간 스캔 진행 상황 표시
- [ ] CWE 분포 차트, 트렌드 분석

**결과물**: 완전한 SAST + DAST + 대시보드 시스템

---

### Phase 4 (옵션): 고급 기능 (1-2주)

- [ ] JavaScript/Java 지원 (ESLint Security 통합)
- [ ] GitHub 리포지토리 직접 스캔
- [ ] CI/CD 통합 (GitHub Actions)
- [ ] 커스텀 Semgrep 규칙 작성

---

## 📈 성능 목표

| 지표 | 목표 | 측정 방법 |
|------|------|----------|
| **정확도** | False Positive <20% | 수동 검증 100개 샘플 |
| **커버리지** | CWE Top 25 중 80% | MITRE CWE 기준 |
| **속도** | 1,000 LOC당 <10초 | 규칙 기반 스캔 |
| **확장성** | 10,000+ 파일 지원 | 메모리 사용량 <2GB |
| **LLM 비용** | 파일당 <$0.01 | GPT-4o-mini 기준 |

---

## 🔌 기존 시스템 통합

### Prompt Arsenal과의 완벽한 시너지

Security Scanner는 기존 Prompt Arsenal의 모든 인프라를 재사용합니다.

```python
# 1. API 프로필 시스템 (★ 핵심 통합)
# config.json의 프로필을 그대로 사용
from security.llm.code_analyzer import LLMSecurityAnalyzer

analyzer = LLMSecurityAnalyzer(
    profile_name="gpt4",     # config.json에서 자동 로드
    config_path="config.json"
)

# 사용자는 이미 설정한 프로필을 선택만 하면 됨
# → OpenAI, Anthropic, Ollama, 로컬 등 10개 프로바이더 모두 지원

# 2. LLM Client 공유 (10개 프로바이더)
from multimodal.llm_client import LLMClient

llm = LLMClient(
    provider="openai",      # or anthropic, ollama, etc.
    model="gpt-4o",
    api_key=profile['api_key']
)

code_analysis = await llm.complete(security_prompt)
vulnerability_check = await llm.complete(verification_prompt)

# 3. Hybrid Judge 패턴 재사용
from core.llm_judge import HybridJudge

# 규칙 기반 → LLM 검증 패턴 동일
security_judge = HybridJudge(
    rule_based_judge=RuleScanner(),
    llm_judge=LLMSecurityAnalyzer(profile_name="claude")
)

# confidence 기반 자동 전환
if rule_confidence >= 0.8:
    # 규칙만으로 충분
    verified = True
else:
    # LLM 검증 요청
    verified = await security_judge.verify(finding)

# 4. 페이로드 시스템 재사용
from payload_utils import PayloadGenerator, PayloadEncoder

# 동적 스캔에서 기존 페이로드 활용
generator = PayloadGenerator()
sql_payloads = generator.generate_variants(
    base_payload="' OR 1=1--",
    strategies=['base64', 'url', 'unicode', 'hex']
)

# 웹 서버 취약점 테스트
for payload in sql_payloads:
    response = await test_endpoint(url, payload)
    if is_vulnerable(response):
        save_finding(cwe="CWE-89", payload=payload)

# 5. Dashboard 통합
# 기존 dashboard/api.py에 보안 엔드포인트 추가
@app.route('/api/security/scans')
def get_security_scans():
    return db.get_security_scans()

@app.route('/api/security/findings')
def get_security_findings():
    return db.get_security_findings()

# 6. 데이터베이스 공유
# 기존 ArsenalDB 확장
class ArsenalDB:
    # 기존 메서드들...

    # 보안 스캔 메서드 추가
    def insert_security_scan(self, ...):
        pass

    def insert_security_finding(self, ...):
        pass

# 7. Interactive CLI 통합
# 메뉴 추가만으로 통합 완료
# s1-s5: Security Scanner
# 기존 메뉴와 동일한 사용성
```

### 통합의 장점

**1. API 프로필 재사용 (★ 핵심)**
```yaml
장점:
  - 사용자가 이미 설정한 API 키/프로필 활용
  - 10개 프로바이더 모두 즉시 지원
  - 로컬 LLM (Ollama) → 무료 보안 스캔
  - 설정 중복 없음

기존_시스템:
  config.json: ✅ 이미 있음
  LLMClient: ✅ 10개 프로바이더 지원
  API_키_관리: ✅ 안전하게 저장됨

추가_작업:
  - 없음! (기존 것 그대로 사용)
```

**2. False Positive 필터링 최적화**
```yaml
규칙_기반:
  속도: 매우 빠름 (<10ms)
  정확도: 70-80%
  비용: 무료

LLM_검증:
  속도: 느림 (2-5초)
  정확도: 95%+
  비용: 파일당 $0.01-0.05

Hybrid_모드:
  규칙_신뢰도_≥0.9: 바로 확정 (빠름, 무료)
  규칙_신뢰도_<0.9: LLM 검증 (정확)

  결과: 80% 비용 절감 + 95% 정확도
```

**3. 모델 선택의 유연성**
```yaml
고정밀_분석:
  provider: anthropic
  model: claude-3-5-sonnet-20241022
  비용: 높음
  정확도: 최고

일반_분석:
  provider: openai
  model: gpt-4o-mini
  비용: 저렴
  정확도: 우수

무료_분석:
  provider: ollama
  model: llama3.1:8b
  비용: 0원
  정확도: 양호 (로컬)
```

**4. 기존 페이로드 활용**
```python
# Prompt Arsenal의 40,000+ 프롬프트 활용
sql_injection_prompts = db.get_prompts(
    category="sql-injection",
    limit=100,
    order_by_success_rate=True  # 성공률 높은 것부터
)

# 동적 스캔에 바로 사용
for prompt in sql_injection_prompts:
    test_endpoint(url, prompt.payload)
```

**5. HybridJudge 패턴 확장**
```python
# Multi-turn Jailbreak의 HybridJudge
from core.llm_judge import HybridJudge

# Security Scanner의 HybridScanner
class HybridSecurityScanner:
    def __init__(self, profile_name: str):
        # 동일한 패턴 적용
        self.rule_scanner = RuleBasedScanner()
        self.llm_analyzer = LLMSecurityAnalyzer(profile_name)

    async def scan(self, code: str):
        # 1. 규칙 기반 빠른 스캔
        rule_findings = self.rule_scanner.scan(code)

        verified = []
        for finding in rule_findings:
            if finding.confidence >= 0.9:
                verified.append(finding)
            else:
                # 2. LLM 검증 (HybridJudge 패턴)
                is_valid = await self.llm_analyzer.verify(finding)
                if is_valid:
                    verified.append(finding)

        return verified
```

---

## 🎓 학습 자료 & 참고

### CWE 참고
- MITRE CWE Top 25: https://cwe.mitre.org/top25/
- OWASP Top 10: https://owasp.org/www-project-top-ten/

### 도구 참고
- Bandit (Python): https://github.com/PyCQA/bandit
- Semgrep: https://semgrep.dev/
- SARIF 포맷: https://sarifweb.azurewebsites.net/

### 기존 코드베이스
- `payload_utils.py`: 페이로드 생성/변형
- `multimodal/llm_client.py`: LLM 통합
- `core/llm_judge.py`: Hybrid Judge
- `dashboard/`: 웹 UI

---

## 🔐 보안 고려사항

**스캐너 자체 보안**:
1. 악의적 코드 실행 방지 (샌드박스)
2. 원격 서버 스캔 시 Rate Limiting
3. API 키/토큰 안전한 저장
4. 스캔 결과 민감 정보 마스킹

**윤리적 고려**:
1. 소유자 허가 없는 서버 스캔 금지
2. 취약점 발견 시 책임 있는 공개 (Responsible Disclosure)
3. 스캔 로그 보안 저장

---

## 📝 다음 단계

이 기획서를 기반으로 구현할 우선순위:

1. **지금 바로 시작**: Phase 1 - 정적 분석 기본
   - `security/` 디렉토리 생성
   - CWE 데이터베이스 구축
   - Python AST 파서

2. **다음 단계**: DB 스키마 추가
   - `core/database.py`에 security_scans, security_findings 테이블

3. **프로토타입**: 간단한 SQL Injection 탐지기
   - 규칙 기반으로 `f"SELECT * FROM {table}"` 패턴 탐지
   - 결과를 Rich 테이블로 출력

**시작할까요?** 어떤 부분부터 구현하고 싶으신가요?
