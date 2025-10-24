# 도구 설치 가이드

Prompt Arsenal의 CTF 자동 풀이와 시스템 취약점 스캔 기능을 사용하려면 다음 도구들이 필요합니다.

## 🔧 필수 Python 패키지

```bash
# 가상환경 활성화
source .venv/bin/activate

# 의존성 설치
uv pip install -r requirements.txt
```

## 🎯 CTF 자동 풀이 도구

### Web 취약점
```bash
# SQLMap (SQL Injection)
pip install sqlmap

# Nikto (웹 스캐너)
brew install nikto  # macOS
sudo apt install nikto  # Ubuntu/Debian

# Dirb (디렉토리 브루트포스)
brew install dirb  # macOS
sudo apt install dirb  # Ubuntu/Debian
```

### Forensics
```bash
# Binwalk (파일 분석)
brew install binwalk  # macOS
sudo apt install binwalk  # Ubuntu/Debian

# Foremost (파일 복구)
brew install foremost  # macOS
sudo apt install foremost  # Ubuntu/Debian

# ExifTool (메타데이터)
brew install exiftool  # macOS
sudo apt install libimage-exiftool-perl  # Ubuntu/Debian

# Strings (기본 제공)
# file 명령어 (기본 제공)
```

### Crypto
```bash
# Hashcat (해시 크래킹)
brew install hashcat  # macOS
sudo apt install hashcat  # Ubuntu/Debian

# John the Ripper
brew install john  # macOS
sudo apt install john  # Ubuntu/Debian

# OpenSSL (기본 제공)
```

### Reversing
```bash
# Checksec (보안 기능 확인)
pip install checksec

# Radare2 (역공학)
brew install radare2  # macOS
sudo apt install radare2  # Ubuntu/Debian

# GDB (디버거, 기본 제공)
# objdump, readelf (기본 제공)
# ltrace, strace (기본 제공)
```

## 🛡️ 시스템 취약점 스캔 도구

### Nmap (포트 스캐너)
```bash
# macOS
brew install nmap

# Ubuntu/Debian
sudo apt install nmap

# 설치 확인
nmap --version
```

### Python 패키지
```bash
# python3-nmap (Python nmap 바인딩)
pip install python3-nmap

# vulners (CVE API 클라이언트)
pip install vulners
```

## 🌐 Web 요청 도구

```bash
# httpx (비동기 HTTP 클라이언트)
pip install httpx

# BeautifulSoup4 (HTML 파싱)
pip install beautifulsoup4

# lxml (XML/HTML 파서)
pip install lxml
```

## 📋 설치 확인

### CTF 도구
```bash
# Web
sqlmap --version
nikto -Version
dirb

# Forensics
binwalk --help
foremost -V
exiftool -ver
file --version

# Crypto
hashcat --version
john --version
openssl version

# Reversing
checksec --version
r2 -v
gdb --version
```

### System 도구
```bash
# Nmap
nmap --version

# Python 패키지
python -c "import nmap; print('nmap OK')"
python -c "import vulners; print('vulners OK')"
python -c "import httpx; print('httpx OK')"
python -c "import bs4; print('beautifulsoup4 OK')"
```

## 🚀 빠른 시작

### 1. 기본 Python 패키지만 설치 (필수)
```bash
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Interactive CLI에서 도구 확인
```bash
python interactive_cli.py
# 메뉴에서 't' (CTF 자동 풀이) 또는 'y' (시스템 스캔) 선택
# 설치되지 않은 도구는 자동으로 감지되어 알림
```

### 3. 선택적 도구 설치
- 사용하려는 기능에 필요한 도구만 설치
- 예: Web 취약점만 테스트 → sqlmap, nikto만 설치
- 예: 포트 스캔만 사용 → nmap만 설치

## ⚠️ 주의사항

1. **macOS ARM (M1/M2)**: 일부 도구는 Rosetta 2 필요
2. **권한**: nmap, strace 등은 sudo 권한 필요
3. **방화벽**: 포트 스캔 시 방화벽 설정 확인
4. **합법적 사용**: 본인 소유 시스템 또는 허가받은 시스템만 테스트
5. **Vulners API**: 더 많은 CVE 조회를 위해 API 키 등록 권장 (무료)

## 🔑 Vulners API 키 (선택)

1. https://vulners.com/ 에서 무료 계정 생성
2. API 키 발급
3. Interactive CLI → System Scan 시 API 키 입력

## 💡 도움말

- 도구 사용법: `--help` 옵션 사용
- 오류 발생 시: GitHub Issues에 보고
- 문의: README.md 참조
