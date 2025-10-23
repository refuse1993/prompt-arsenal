# CTF Auto Solver Framework - 완전 기획 문서

## 📋 프로젝트 개요

**목표**: AI 기반 CTF(Capture The Flag) 문제 자동 풀이 시스템 구축

**핵심 기능**:
- LLM을 활용한 문제 자동 분석 및 전략 수립
- 카테고리별 자동화 도구 통합 (Web, Pwn, Crypto, Forensics, Reversing)
- 자동 exploit 생성 및 실행
- 풀이 과정 학습 및 데이터베이스 저장

---

## 🏗️ 시스템 아키텍처

### 계층 구조
```
┌─────────────────────────────────────────────┐
│         Interactive CLI (사용자)             │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│          CTF Core Engine                     │
│  (문제 접수 → 분석 → 실행 → 검증)            │
└─────────────────┬───────────────────────────┘
                  │
        ┌─────────┼─────────┐
        │         │         │
┌───────▼──┐ ┌───▼────┐ ┌─▼──────┐
│ LLM      │ │ Tool   │ │ Solver │
│ Reasoner │ │ Executor│ │ Modules│
└──────────┘ └────────┘ └────────┘
     │            │          │
     └────────────┼──────────┘
                  │
         ┌────────▼────────┐
         │   Arsenal DB    │
         │  (결과 저장)     │
         └─────────────────┘
```

### 모듈 구성

```
prompt-arsenal/
├── ctf/                           # 🆕 CTF 자동 솔버
│   ├── __init__.py
│   ├── ctf_core.py                # 통합 엔진 (오케스트레이션)
│   ├── llm_reasoner.py            # LLM 문제 분석 및 전략 수립
│   ├── tool_executor.py           # 도구 자동 실행 래퍼
│   │
│   ├── web_solver.py              # Web 취약점 자동 공격
│   ├── pwn_solver.py              # Pwnable 자동 exploit
│   ├── crypto_solver.py           # 암호학 자동 해독
│   ├── forensics_solver.py        # 포렌식 자동 분석
│   ├── reversing_solver.py        # 리버싱 자동 분석
│   └── misc_solver.py             # 기타 문제 해결
│
├── core/
│   └── database.py                # ctf_challenges 테이블 추가
```

---

## 📊 데이터베이스 스키마

### ctf_challenges (CTF 문제 및 결과)
```sql
CREATE TABLE ctf_challenges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT,
    category TEXT NOT NULL,  -- web, pwn, crypto, forensics, reversing, misc
    difficulty TEXT,         -- easy, medium, hard
    url TEXT,
    files TEXT,              -- JSON array
    hints TEXT,              -- JSON array

    -- 분석 결과
    vulnerability_type TEXT,
    llm_analysis TEXT,       -- JSON
    strategy TEXT,           -- JSON array

    -- 풀이 결과
    status TEXT DEFAULT 'pending',  -- pending, solving, solved, failed
    flag TEXT,
    exploit_code TEXT,
    solution_steps TEXT,     -- JSON array

    -- 메타데이터
    solve_time REAL,         -- 풀이 소요 시간 (초)
    attempts INTEGER DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    solved_at TEXT
);
```

### ctf_execution_logs (실행 로그)
```sql
CREATE TABLE ctf_execution_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    challenge_id INTEGER NOT NULL,
    step_number INTEGER,
    tool_name TEXT,
    command TEXT,
    output TEXT,
    success BOOLEAN,
    duration REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (challenge_id) REFERENCES ctf_challenges(id)
);
```

---

## 🤖 LLM Reasoner (문제 분석 엔진)

### 기능
1. **문제 분석**: 제목, 설명, 파일, 힌트 → 카테고리 + 취약점 유형 추론
2. **전략 수립**: 단계별 공략 전략 생성
3. **도구 추천**: 필요한 도구 목록 제안
4. **Exploit 생성**: 분석 결과 기반 코드 자동 생성
5. **결과 해석**: 도구 출력 분석 및 다음 단계 결정

### 프롬프트 전략

#### 1. 문제 분석 프롬프트
```
You are an expert CTF player. Analyze this challenge:

Title: {title}
Description: {description}
Files: {files}
Hints: {hints}

Respond in JSON:
{
  "category": "web/pwn/crypto/forensics/reversing/misc",
  "difficulty": "easy/medium/hard",
  "vulnerability_type": "구체적 취약점",
  "strategy": ["Step 1", "Step 2", ...],
  "required_tools": ["tool1", "tool2"],
  "confidence": 0.9
}
```

#### 2. Exploit 생성 프롬프트
```
Generate exploit code for:
- Category: {category}
- Vulnerability: {vulnerability_type}
- Strategy: {strategy}

Include complete working code with comments.
```

#### 3. 결과 분석 프롬프트
```
Analyze tool output:

Expected: {expected_result}
Actual Output: {command_output}

Respond in JSON:
{
  "success": true/false,
  "flag_found": "flag{...}" or null,
  "findings": ["finding 1", "finding 2"],
  "next_steps": ["action 1", "action 2"]
}
```

---

## 🛠️ Tool Executor (도구 자동 실행)

### 지원 도구

#### Web
| 도구 | 용도 | 명령어 예시 |
|------|------|------------|
| sqlmap | SQL Injection | `sqlmap -u URL --batch` |
| nikto | 웹 취약점 스캔 | `nikto -h URL` |
| dirb | 디렉토리 브루트포스 | `dirb URL` |
| gobuster | 디렉토리/파일 발견 | `gobuster dir -u URL -w wordlist` |

#### Pwn
| 도구 | 용도 | 명령어 예시 |
|------|------|------------|
| checksec | 바이너리 보호 기법 확인 | `checksec --file=binary` |
| gdb | 디버깅 | `gdb binary` |
| pwntools | Exploit 개발 | Python library |
| radare2 | 리버싱/디버깅 | `r2 binary` |

#### Forensics
| 도구 | 용도 | 명령어 예시 |
|------|------|------------|
| binwalk | 파일 분석/추출 | `binwalk -e file` |
| foremost | 파일 복구 | `foremost -i file` |
| strings | 문자열 추출 | `strings file` |
| exiftool | 메타데이터 추출 | `exiftool file` |
| volatility | 메모리 덤프 분석 | `volatility -f mem.dump pslist` |

#### Crypto
| 도구 | 용도 | 명령어 예시 |
|------|------|------------|
| hashcat | 해시 크래킹 | `hashcat -m 0 hash.txt wordlist` |
| john | 해시 크래킹 | `john hash.txt` |
| openssl | 암호화/복호화 | `openssl enc -d -aes256` |

#### Reversing
| 도구 | 용도 | 명령어 예시 |
|------|------|------------|
| ghidra | 디컴파일러 | GUI |
| objdump | 디스어셈블 | `objdump -d binary` |
| readelf | ELF 분석 | `readelf -a binary` |
| ltrace | 라이브러리 호출 추적 | `ltrace binary` |
| strace | 시스템 호출 추적 | `strace binary` |

### 자동 설치 확인
```python
# 설치된 도구 자동 감지
installed_tools = {
    'sqlmap': True,
    'nikto': False,  # 미설치
    'binwalk': True,
    ...
}

# 미설치 도구 안내
if not installed_tools['sqlmap']:
    print("sqlmap이 필요합니다: pip install sqlmap")
```

---

## 🌐 Web Solver

### 자동화 공격 시나리오

#### 1. SQL Injection
```python
# 단계 1: SQLMap 자동 스캔
result = await tool_executor.run_sqlmap(
    url="http://target.com/login?id=1",
    options=['--batch', '--dbs']
)

# 단계 2: LLM으로 결과 분석
analysis = await llm_reasoner.analyze_output(
    result.output,
    expected="데이터베이스 목록"
)

# 단계 3: 테이블 덤프
if analysis['success']:
    result = await tool_executor.run_sqlmap(
        url="http://target.com/login?id=1",
        options=['--batch', '-D', db_name, '--tables']
    )
```

#### 2. XSS (Cross-Site Scripting)
```python
# LLM이 페이로드 생성
payloads = await llm_reasoner.generate_xss_payloads(
    context="입력 필드, 필터링 우회"
)

# 자동 테스트
for payload in payloads:
    response = await test_xss(url, payload)
    if is_reflected(response, payload):
        return {"flag": extract_flag(response)}
```

#### 3. LFI/RFI (Local/Remote File Inclusion)
```python
# 일반적인 LFI 페이로드
payloads = [
    "../../../etc/passwd",
    "....//....//....//etc/passwd",
    "php://filter/convert.base64-encode/resource=index.php"
]

# 자동 테스트 및 플래그 추출
```

---

## 💣 Pwn Solver

### 자동화 Exploit 생성

#### 1. Buffer Overflow
```python
# 1단계: 바이너리 분석
checksec_result = await tool_executor.run_checksec(binary_path)
# → NX: disabled, PIE: disabled, Stack Canary: disabled

# 2단계: LLM에게 exploit 전략 요청
strategy = await llm_reasoner.generate_exploit(
    analysis=CTFAnalysis(
        category='pwn',
        vulnerability_type='Buffer Overflow',
        ...
    ),
    context=checksec_result.output
)

# 3단계: pwntools 코드 자동 생성
exploit_code = """
from pwn import *

p = remote('target.com', 1337)
payload = b'A' * 64  # offset
payload += p64(0xdeadbeef)  # return address
p.sendline(payload)
p.interactive()
"""

# 4단계: 실행 및 플래그 캡처
```

#### 2. Format String
```python
# 자동 offset 탐지
offset = find_format_offset(binary_path)

# GOT overwrite exploit 생성
exploit = generate_format_string_exploit(offset, target_address)
```

#### 3. ROP (Return-Oriented Programming)
```python
# ROPgadget으로 가젯 수집
gadgets = await tool_executor.execute("ROPgadget --binary binary")

# LLM이 ROP chain 구성
rop_chain = await llm_reasoner.build_rop_chain(gadgets, goal="execve /bin/sh")
```

---

## 🔐 Crypto Solver

### 자동화 해독

#### 1. 고전 암호 (Classical Ciphers)
```python
# Caesar, Vigenère, Substitution 자동 탐지 및 해독
def detect_cipher_type(ciphertext):
    # LLM이 암호 타입 추론
    analysis = llm_reasoner.analyze_cipher(ciphertext)
    return analysis['cipher_type']

def crack_classical(ciphertext, cipher_type):
    if cipher_type == 'caesar':
        return brute_force_caesar(ciphertext)
    elif cipher_type == 'vigenere':
        return crack_vigenere(ciphertext)
```

#### 2. RSA 공격
```python
# 약한 RSA 키 자동 탐지
def attack_rsa(n, e, c):
    attacks = [
        ('small_e', small_e_attack),
        ('wiener', wiener_attack),
        ('fermat', fermat_factorization),
        ('common_modulus', common_modulus_attack)
    ]

    for name, attack_func in attacks:
        result = attack_func(n, e, c)
        if result:
            return result
```

#### 3. 해시 크래킹
```python
# Hashcat/John 자동 실행
result = await tool_executor.run_hashcat(
    hash_file='hash.txt',
    wordlist='rockyou.txt',
    hash_type=0  # MD5
)
```

---

## 🔍 Forensics Solver

### 자동화 분석

#### 1. 파일 분석
```python
# 1단계: 파일 타입 확인
file_type = await tool_executor.run_file(file_path)

# 2단계: 메타데이터 추출
metadata = await tool_executor.run_exiftool(file_path)

# 3단계: 숨겨진 파일 추출
await tool_executor.run_binwalk(file_path, extract=True)

# 4단계: 문자열 추출
strings = await tool_executor.run_strings(file_path)

# 5단계: LLM이 플래그 패턴 찾기
flag = llm_reasoner.find_flag_pattern(strings.output)
```

#### 2. 메모리 덤프 분석
```python
# Volatility로 프로세스 목록
pslist = await tool_executor.execute(
    "volatility -f mem.dump --profile=Win7SP1x64 pslist"
)

# LLM이 의심스러운 프로세스 식별
suspicious = llm_reasoner.identify_suspicious_process(pslist.output)

# 메모리 덤프
memdump = await tool_executor.execute(
    f"volatility -f mem.dump --profile=Win7SP1x64 memdump -p {pid} -D dump/"
)
```

#### 3. 네트워크 패킷 분석
```python
# pcap 파일 분석
tshark_result = await tool_executor.execute(
    "tshark -r capture.pcap -Y 'http' -T fields -e http.request.uri"
)

# LLM이 패턴 분석
analysis = llm_reasoner.analyze_network_traffic(tshark_result.output)
```

---

## 🔄 Reversing Solver

### 자동화 리버싱

#### 1. 정적 분석
```python
# 1단계: 보호 기법 확인
checksec = await tool_executor.run_checksec(binary_path)

# 2단계: 문자열 추출
strings = await tool_executor.run_strings(binary_path)

# 3단계: 디스어셈블
disasm = await tool_executor.run_objdump(binary_path, option='-d')

# 4단계: LLM이 코드 분석
analysis = llm_reasoner.analyze_disassembly(disasm.output)
```

#### 2. 동적 분석
```python
# ltrace로 라이브러리 호출 추적
ltrace_result = await tool_executor.run_ltrace(binary_path)

# strace로 시스템 호출 추적
strace_result = await tool_executor.run_strace(binary_path)

# LLM이 동작 분석
behavior = llm_reasoner.analyze_behavior(
    ltrace_result.output,
    strace_result.output
)
```

#### 3. 난독화 해제
```python
# LLM이 난독화 패턴 식별 및 해제 전략 제시
deobfuscation_strategy = llm_reasoner.analyze_obfuscation(code)

# 자동 스크립트 생성 및 실행
```

---

## 🎯 CTF Core Engine (통합 오케스트레이션)

### 자동 풀이 워크플로우

```python
class CTFSolver:
    async def solve(self, challenge_data: Dict) -> Dict:
        """
        CTF 문제 자동 풀이

        Workflow:
        1. LLM 문제 분석
        2. 카테고리별 Solver 선택
        3. 자동 도구 실행
        4. 결과 검증 및 플래그 추출
        5. 실패 시 재시도 (다른 전략)
        """

        # 1단계: 문제 분석
        analysis = await self.llm_reasoner.analyze_challenge(challenge_data)

        # 2단계: Solver 선택
        solver = self._select_solver(analysis.category)

        # 3단계: 자동 풀이 시도
        max_attempts = 3
        for attempt in range(max_attempts):
            result = await solver.solve(challenge_data, analysis)

            if result['flag']:
                # 성공!
                return result

            # 실패 시 LLM에게 다른 전략 요청
            analysis = await self.llm_reasoner.refine_strategy(
                analysis,
                failed_attempt=result
            )

        # 모든 시도 실패
        return {'flag': None, 'status': 'failed'}
```

### Solver 인터페이스
```python
class BaseSolver:
    async def solve(self, challenge_data: Dict, analysis: CTFAnalysis) -> Dict:
        """
        공통 인터페이스

        Returns:
            {
                'flag': 'flag{...}' or None,
                'status': 'solved/failed/partial',
                'exploit_code': '...',
                'solution_steps': [...],
                'execution_logs': [...]
            }
        """
        raise NotImplementedError
```

---

## 🎨 CLI 사용 예시

### 메뉴 구조
```
🚩 CTF AUTO SOLVER
  [green]c[/green]. CTF 문제 풀이 (자동)
  [green]l[/green]. CTF 문제 목록 조회
  [green]h[/green]. 풀이 이력 조회
```

### 사용 시나리오

#### 시나리오 1: Web Challenge
```bash
> c (CTF 문제 풀이)

문제 제목: Easy SQL
문제 설명: Find the admin password
URL: http://ctf.example.com/login
파일: (없음)
힌트: SQL Injection

[1/5] 🤖 LLM 문제 분석 중...
  ✓ 카테고리: Web
  ✓ 취약점: SQL Injection
  ✓ 전략: SQLMap 자동 스캔 → DB 덤프 → 플래그 추출

[2/5] 🔧 SQLMap 실행 중...
  ✓ 취약점 발견: UNION-based SQLi
  ✓ 데이터베이스: ctf_db

[3/5] 💉 테이블 덤프 중...
  ✓ 테이블: users
  ✓ admin 비밀번호: flag{sql_1nj3ct10n_3z}

[4/5] 🎯 플래그 추출 성공!
  Flag: flag{sql_1nj3ct10n_3z}

[5/5] 💾 결과 저장 완료
  풀이 시간: 45.3초
  시도 횟수: 1회
```

#### 시나리오 2: Pwn Challenge
```bash
> c

문제 제목: Buffer Overflow Basic
파일: vuln (ELF 64-bit)

[1/5] 🤖 문제 분석...
  ✓ 카테고리: Pwnable
  ✓ 취약점: Buffer Overflow
  ✓ 보호 기법: NX disabled, PIE disabled

[2/5] 🔧 Checksec 분석...
  ✓ Stack Canary: No
  ✓ Return address 덮어쓰기 가능

[3/5] 💣 Exploit 생성...
  ✓ pwntools 코드 자동 생성
  ✓ Payload 크기: 72 bytes

[4/5] ⚡ Exploit 실행...
  ✓ 셸 획득!
  ✓ Flag: flag{pwn_b4s1c_b0f}

[5/5] 💾 결과 저장
```

---

## 📈 학습 및 개선

### 풀이 데이터 수집
```sql
-- 성공한 풀이 패턴 분석
SELECT category, vulnerability_type, COUNT(*) as solved_count
FROM ctf_challenges
WHERE status = 'solved'
GROUP BY category, vulnerability_type;

-- 평균 풀이 시간
SELECT category, AVG(solve_time) as avg_time
FROM ctf_challenges
WHERE status = 'solved'
GROUP BY category;
```

### LLM 프롬프트 개선
- 성공한 풀이의 전략을 프롬프트에 반영
- 실패한 케이스 분석하여 예외 처리 추가

---

## 🔧 의존성

### Python 패키지
```txt
# CTF Tools (선택적 설치)
pwntools>=4.12.0
ROPgadget>=7.4
pycryptodome>=3.19.0
gmpy2>=2.1.5

# 기타
requests>=2.31.0
beautifulsoup4>=4.12.0
lxml>=5.1.0
```

### 외부 도구 (선택적)
```bash
# Web
sudo apt install sqlmap nikto dirb gobuster

# Pwn/Reversing
sudo apt install gdb radare2 ltrace strace binutils

# Forensics
sudo apt install binwalk foremost exiftool volatility

# Crypto
sudo apt install hashcat john openssl
```

---

## 🚀 구현 우선순위

### Phase 1 (필수)
1. ✅ LLM Reasoner (문제 분석)
2. ✅ Tool Executor (도구 실행)
3. ⏳ Web Solver (SQL Injection 기본)
4. ⏳ CTF Core (통합 엔진)
5. ⏳ CLI 메뉴 추가

### Phase 2 (중요)
6. Forensics Solver (파일 분석)
7. Web Solver 확장 (XSS, LFI)
8. Crypto Solver (고전 암호)

### Phase 3 (고급)
9. Pwn Solver (Buffer Overflow)
10. Reversing Solver (정적 분석)
11. 자동 재시도 로직
12. 학습 데이터 축적

---

## 📝 예상 성과

### 자동 풀이 가능 비율 (예상)
- **Web**: 70-80% (SQL Injection, XSS, LFI 등)
- **Forensics**: 60-70% (파일 분석, 메타데이터)
- **Crypto**: 40-50% (고전 암호, 약한 RSA)
- **Pwn**: 30-40% (기본 Buffer Overflow)
- **Reversing**: 30-40% (정적 분석, 문자열 추출)
- **Misc**: 50-60% (LLM 추론 능력에 의존)

### 시간 절감
- 수동 풀이: 평균 30-60분
- 자동 풀이: 평균 1-5분
- **시간 절감: 90%**

---

## 🎓 교육적 가치

### 학습 효과
1. **CTF 입문자**: 자동 풀이 과정을 보며 학습
2. **중급자**: LLM 전략을 참고하여 사고 확장
3. **고급자**: 자동화 도구로 시간 절약

### 풀이 과정 출력
```
[단계별 설명]
1. SQLMap으로 취약점 탐지
   → UNION-based SQL Injection 발견

2. 데이터베이스 목록 조회
   → ctf_db, information_schema

3. ctf_db 테이블 덤프
   → users, flags

4. flags 테이블에서 플래그 추출
   → flag{sql_injection_success}
```

---

## 🔒 보안 및 윤리

### 사용 제한
⚠️ **이 도구는 합법적인 CTF 대회 및 교육 목적으로만 사용하세요**

- 허가받지 않은 시스템 공격 금지
- CTF 플랫폼 이용 약관 준수
- 교육 및 연구 목적으로만 사용

### 책임 있는 사용
- 실제 시스템에 대한 무단 공격 절대 금지
- 학습 환경(VulnHub, HackTheBox 등)에서만 사용
- 자동화 도구의 한계 인지 (100% 정확도 불가능)

---

## 📌 참고 자료

### CTF 플랫폼
- [CTFtime](https://ctftime.org/) - CTF 대회 일정
- [picoCTF](https://picoctf.org/) - 초보자용 CTF
- [HackTheBox](https://www.hackthebox.com/) - 실습 환경
- [TryHackMe](https://tryhackme.com/) - 가이드형 실습

### 도구 문서
- [pwntools](https://docs.pwntools.com/)
- [SQLMap](https://sqlmap.org/)
- [Volatility](https://volatilityfoundation.org/)
- [Ghidra](https://ghidra-sre.org/)

### 학습 자료
- [CTF 101](https://ctf101.org/)
- [LiveOverflow YouTube](https://www.youtube.com/c/LiveOverflow)
- [IppSec YouTube](https://www.youtube.com/c/ippsec)

---

**Version**: 1.0
**Last Updated**: 2025-10-23
**Status**: 기획 완료, 구현 진행 중
