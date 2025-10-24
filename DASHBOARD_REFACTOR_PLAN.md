# Dashboard Refactoring Plan - Modular Architecture

**현재 문제점**:
- `api.py`: 1,132줄 - 모든 API 엔드포인트가 한 파일에 몰림
- `index.html`: 1,373줄 - 모든 UI가 단일 HTML 파일에
- 모듈화된 백엔드 구조 (core/, text/, multimodal/, multiturn/, ctf/, security/, system/)와 대시보드 구조 불일치
- 유지보수 어려움, 확장성 저하

**목표**:
- ✅ 백엔드 모듈 구조와 일치하는 프론트엔드 구조
- ✅ 각 기능별 독립적인 페이지 및 컴포넌트
- ✅ API Blueprint를 활용한 모듈화된 백엔드
- ✅ 현대적인 프론트엔드 프레임워크 (React/Vue 또는 바닐라 JS 모듈)
- ✅ 재사용 가능한 컴포넌트

---

## 📐 제안 아키텍처

### Option 1: Flask Blueprint + Vanilla JS Modules (권장)

**장점**:
- 추가 빌드 도구 불필요 (Python 환경만)
- 기존 Flask 구조 유지하며 점진적 개선
- ES6 모듈로 충분히 현대적
- 배포 간단 (단일 Flask 서버)

**단점**:
- React/Vue 대비 컴포넌트 재사용성 낮음
- 상태 관리가 수동적

### Option 2: Flask API + React/Vue SPA

**장점**:
- 최고 수준의 컴포넌트 재사용성
- 풍부한 에코시스템 (라이브러리, 도구)
- 상태 관리 라이브러리 활용 가능

**단점**:
- 빌드 도구 필요 (Webpack, Vite 등)
- 복잡도 증가
- Python + Node.js 환경 필요

---

## 🏗️ 제안 구조 (Option 1: Flask Blueprint + Vanilla JS)

```
dashboard/
│
├── app.py                          # Flask 앱 메인 (Blueprint 등록)
├── config.py                       # 대시보드 설정
│
├── api/                            # 📡 Backend API (Flask Blueprints)
│   ├── __init__.py
│   ├── prompts.py                  # 텍스트 프롬프트 API
│   ├── multimodal.py               # 멀티모달 API
│   ├── multiturn.py                # Multi-turn 캠페인 API
│   ├── ctf.py                      # CTF Framework API
│   ├── security.py                 # Security Scanner API
│   ├── system.py                   # System Scanner API
│   └── stats.py                    # 통계 API (공통)
│
├── static/                         # 🎨 Frontend (Static Assets)
│   ├── css/
│   │   ├── main.css                # 공통 스타일
│   │   ├── components.css          # 컴포넌트 스타일
│   │   └── modules/                # 모듈별 스타일
│   │       ├── prompts.css
│   │       ├── multimodal.css
│   │       ├── multiturn.css
│   │       ├── ctf.css
│   │       ├── security.css
│   │       └── system.css
│   │
│   ├── js/
│   │   ├── main.js                 # 앱 초기화
│   │   ├── api.js                  # API 클라이언트 (fetch wrapper)
│   │   ├── utils.js                # 유틸리티 함수
│   │   │
│   │   ├── components/             # 📦 재사용 컴포넌트
│   │   │   ├── Card.js
│   │   │   ├── Table.js
│   │   │   ├── Chart.js
│   │   │   ├── Modal.js
│   │   │   ├── Pagination.js
│   │   │   ├── SearchBar.js
│   │   │   └── StatCard.js
│   │   │
│   │   └── modules/                # 📁 모듈별 로직
│   │       ├── prompts/
│   │       │   ├── PromptList.js
│   │       │   ├── PromptDetail.js
│   │       │   └── PromptStats.js
│   │       ├── multimodal/
│   │       │   ├── MediaList.js
│   │       │   ├── MediaDetail.js
│   │       │   └── MediaStats.js
│   │       ├── multiturn/
│   │       │   ├── CampaignList.js
│   │       │   ├── CampaignDetail.js
│   │       │   └── StrategyComparison.js
│   │       ├── ctf/
│   │       │   ├── ChallengeList.js
│   │       │   ├── ChallengeDetail.js
│   │       │   └── AttackAnalytics.js
│   │       ├── security/
│   │       │   ├── ScanList.js
│   │       │   ├── FindingDetail.js
│   │       │   └── VulnerabilityStats.js
│   │       └── system/
│   │           ├── SystemScanList.js
│   │           └── CVEDetail.js
│   │
│   └── lib/                        # 외부 라이브러리
│       ├── chart.min.js
│       └── marked.min.js
│
├── templates/                      # 🖼️ HTML Templates
│   ├── base.html                   # 기본 레이아웃 (네비게이션, 헤더)
│   ├── index.html                  # 홈 대시보드
│   │
│   └── modules/                    # 모듈별 페이지
│       ├── prompts.html            # 텍스트 프롬프트 페이지
│       ├── multimodal.html         # 멀티모달 페이지
│       ├── multiturn.html          # Multi-turn 캠페인 페이지
│       ├── ctf.html                # CTF Framework 페이지
│       ├── security.html           # Security Scanner 페이지
│       └── system.html             # System Scanner 페이지
│
└── utils/                          # 🛠️ Backend Utilities
    ├── __init__.py
    ├── db_helpers.py               # 데이터베이스 헬퍼
    └── response_formatter.py       # 응답 포매팅
```

---

## 📊 페이지별 기능 설계

### 1. **Home Dashboard** (`/`)

**목적**: 전체 시스템 개요 및 최신 활동

**표시 정보**:
- 📊 **통계 카드**: 총 프롬프트, 테스트 실행, 성공률, 활성 캠페인
- 📈 **차트**:
  - 최근 7일 테스트 활동 (Line Chart)
  - 카테고리별 프롬프트 분포 (Pie Chart)
  - 모델별 성공률 (Bar Chart)
- 🔥 **최신 활동**:
  - 최근 테스트 결과 (5개)
  - 최근 Multi-turn 캠페인 (3개)
  - 최근 CTF 챌린지 (3개)

**API 엔드포인트**:
- `GET /api/stats/overview` - 전체 통계
- `GET /api/stats/recent-activity` - 최신 활동

---

### 2. **Text Prompts** (`/prompts`)

**목적**: 22,340개 텍스트 프롬프트 관리 및 테스트

**페이지 구성**:

**2.1. Prompt Library Tab**
- 🔍 검색 바 (키워드, 카테고리, 태그)
- 📋 프롬프트 테이블:
  - ID, Category, Payload (truncated), Success Rate, Test Count
  - 클릭 → 상세 모달
- 📄 페이지네이션 (100개/페이지)

**2.2. Test Results Tab**
- 📊 테스트 결과 필터링:
  - Provider (OpenAI, Anthropic, Google 등)
  - Model
  - Success/Failure
  - 날짜 범위
- 📋 결과 테이블:
  - Prompt, Model, Response, Success, Severity, Confidence
  - 클릭 → 상세 모달 (전체 응답, Judge 판정 근거)

**2.3. Statistics Tab**
- 📈 카테고리별 성공률 (Bar Chart)
- 📊 모델별 ASR 비교 (Table + Chart)
- 🔥 Top 10 성공 프롬프트

**API 엔드포인트**:
- `GET /api/prompts` - 프롬프트 목록 (필터링, 검색, 페이징)
- `GET /api/prompts/<id>` - 프롬프트 상세
- `GET /api/prompts/<id>/results` - 프롬프트별 테스트 결과
- `GET /api/prompts/stats` - 프롬프트 통계
- `POST /api/prompts` - 프롬프트 추가
- `PUT /api/prompts/<id>` - 프롬프트 수정
- `DELETE /api/prompts/<id>` - 프롬프트 삭제

---

### 3. **Multimodal Attacks** (`/multimodal`)

**목적**: 이미지/오디오/비디오 공격 관리

**페이지 구성**:

**3.1. Media Arsenal Tab**
- 🖼️ 미디어 갤러리 뷰:
  - 썸네일 그리드
  - Media Type 필터 (Image, Audio, Video)
  - Attack Type 필터 (Transparent Text, LSB, FGSM 등)
- 클릭 → 상세 모달:
  - 미디어 프리뷰
  - 파라미터 (opacity, epsilon, noise_level)
  - 테스트 결과

**3.2. Test Results Tab**
- 📊 Vision 모델 테스트 결과
- 필터: Provider, Media Type, Success/Failure

**3.3. Cross-Modal Combinations Tab**
- 🔗 텍스트 + 이미지 + 오디오 조합 목록
- 조합별 성공률

**API 엔드포인트**:
- `GET /api/multimodal/media` - 미디어 목록
- `GET /api/multimodal/media/<id>` - 미디어 상세
- `GET /api/multimodal/results` - 테스트 결과
- `GET /api/multimodal/combinations` - 크로스 모달 조합

---

### 4. **Multi-turn Campaigns** (`/multiturn`)

**목적**: 7가지 전략의 Multi-turn 캠페인 관리

**페이지 구성**:

**4.1. Campaigns Tab**
- 📋 캠페인 목록 테이블:
  - ID, Goal, Strategy, Target Model, Status, Turns Used, ASR
  - Status별 색상 코딩 (Pending, Running, Success, Failed)
  - 클릭 → 상세 페이지

**4.2. Campaign Detail Page**
- 📊 캠페인 헤더:
  - Goal, Strategy, Target/Judge Models, Status
- 📜 Conversation Timeline:
  - 각 턴별 프롬프트 + 응답
  - 이미지 포함 시 썸네일
  - Scorer 평가 (Progress, Defense Triggered)
- 📈 Progress Chart (턴별 진행도)

**4.3. Strategy Comparison Tab**
- 📊 전략별 성공률 비교 (Bar Chart)
- 📈 평균 턴 수 비교 (Line Chart)
- 📋 전략별 통계 테이블

**API 엔드포인트**:
- `GET /api/multiturn/campaigns` - 캠페인 목록
- `GET /api/multiturn/campaigns/<id>` - 캠페인 상세
- `GET /api/multiturn/campaigns/<id>/conversations` - 대화 턴
- `GET /api/multiturn/stats` - 전략별 통계

---

### 5. **CTF Framework** (`/ctf`)

**목적**: Playwright 기반 웹 공격 관리

**페이지 구성**:

**5.1. Challenges Tab**
- 📋 챌린지 목록 테이블:
  - URL, Title, Category, Difficulty, Challenge Type, Status
  - Competition Name 필터
  - Status 필터 (Pending, Solved, Failed)
- 클릭 → 상세 모달

**5.2. Challenge Detail Modal**
- 📊 챌린지 정보:
  - URL, Category, Difficulty, Type, Status
- 📜 Execution Logs:
  - Phase별 로그 (Page Analysis, Detection, Enumeration, Extraction)
  - Playwright 페이지 분석 결과 (Forms, Scripts, Comments, Cookies)
  - 페이로드 및 응답
- ✅ 솔루션 (성공 시)

**5.3. Attack Analytics Tab**
- 📊 공격 유형별 성공률 (Bar Chart)
  - SQL Injection: 70%
  - XSS: 60%
  - Command Injection: 65%
- 📈 Playwright 도입 전후 비교
- 📋 Competition별 챌린지 수집 통계

**API 엔드포인트**:
- `GET /api/ctf/challenges` - 챌린지 목록
- `GET /api/ctf/challenges/<id>` - 챌린지 상세
- `GET /api/ctf/challenges/<id>/logs` - 실행 로그
- `GET /api/ctf/stats` - 공격 유형별 통계

---

### 6. **Security Scanner** (`/security`)

**목적**: 코드 취약점 스캔 결과 관리

**페이지 구성**:

**6.1. Scans Tab**
- 📋 스캔 목록 테이블:
  - ID, Target, Mode, Duration, Findings Count, LLM Cost
  - 날짜 필터
- 클릭 → 스캔 상세 페이지

**6.2. Scan Detail Page**
- 📊 스캔 헤더:
  - Target, Mode, Duration, LLM Calls, Cost
- 📋 Findings 테이블:
  - CWE ID, Severity, File Path, Line Number, Title
  - Severity별 색상 코딩 (Critical, High, Medium, Low)
  - 클릭 → Finding 상세 모달

**6.3. Finding Detail Modal**
- 📄 취약점 정보:
  - CWE ID, Severity, Confidence
  - 한글 설명
  - 공격 시나리오 (한글)
- 💻 코드 스니펫 (Syntax Highlighting)
- ✅ 수정 코드 예시
- 📖 수정 방법 가이드

**6.4. Statistics Tab**
- 📊 CWE별 발견 빈도 (Bar Chart)
- 📈 Severity 분포 (Pie Chart)
- 💰 Hybrid 모드 비용 절감 효과

**API 엔드포인트**:
- `GET /api/security/scans` - 스캔 목록
- `GET /api/security/scans/<id>` - 스캔 상세
- `GET /api/security/scans/<id>/findings` - 취약점 목록
- `GET /api/security/findings/<id>` - 취약점 상세
- `GET /api/security/stats` - 통계

---

### 7. **System Scanner** (`/system`)

**목적**: Nmap + CVE 매칭 결과 관리

**페이지 구성**:

**7.1. Scans Tab**
- 📋 스캔 목록 테이블:
  - ID, Target, Scan Type, Findings Count, 날짜

**7.2. Scan Detail Page**
- 📊 스캔 정보
- 📋 Findings 테이블:
  - Port, Service, Version, CVE ID, Severity
- 클릭 → CVE 상세 모달

**API 엔드포인트**:
- `GET /api/system/scans` - 스캔 목록
- `GET /api/system/scans/<id>` - 스캔 상세

---

## 🎨 UI/UX 디자인 가이드

### 색상 팔레트

```css
:root {
  /* Primary */
  --primary: #4F46E5;        /* Indigo */
  --primary-hover: #4338CA;

  /* Severity */
  --critical: #DC2626;       /* Red */
  --high: #EA580C;           /* Orange */
  --medium: #F59E0B;         /* Amber */
  --low: #10B981;            /* Green */

  /* Status */
  --success: #10B981;        /* Green */
  --pending: #6B7280;        /* Gray */
  --running: #3B82F6;        /* Blue */
  --failed: #DC2626;         /* Red */

  /* Neutral */
  --bg-primary: #FFFFFF;
  --bg-secondary: #F9FAFB;
  --text-primary: #111827;
  --text-secondary: #6B7280;
  --border: #E5E7EB;
}

/* Dark Mode */
@media (prefers-color-scheme: dark) {
  :root {
    --bg-primary: #1F2937;
    --bg-secondary: #111827;
    --text-primary: #F9FAFB;
    --text-secondary: #9CA3AF;
    --border: #374151;
  }
}
```

### 컴포넌트 디자인

**Card**:
```html
<div class="card">
  <div class="card-header">
    <h3>Title</h3>
    <button>Action</button>
  </div>
  <div class="card-body">
    Content
  </div>
</div>
```

**Table**:
- Sticky header
- Hover 효과
- 클릭 가능한 행
- 페이지네이션

**Modal**:
- Overlay 배경
- ESC 키로 닫기
- 반응형 크기

---

## 🔧 기술 스택

### Backend
- **Flask**: 웹 프레임워크
- **Flask-CORS**: CORS 처리
- **SQLite**: 데이터베이스
- **Blueprint**: 모듈화

### Frontend
- **Vanilla JavaScript (ES6 Modules)**: 프론트엔드 로직
- **CSS3 (CSS Variables)**: 스타일링
- **Chart.js**: 차트 렌더링
- **Marked.js**: Markdown 렌더링 (설명, 코드 스니펫)
- **Prism.js**: Syntax Highlighting

---

## 📋 구현 단계

### Phase 1: 기반 구조 설정 (1-2일)
- [ ] `dashboard/app.py` 생성 (Flask 앱 메인)
- [ ] Blueprint 구조 설정 (`api/` 디렉토리)
- [ ] `base.html` 템플릿 생성 (네비게이션, 레이아웃)
- [ ] CSS 기본 스타일 및 컴포넌트

### Phase 2: API 모듈화 (2-3일)
- [ ] `api/prompts.py` - 텍스트 프롬프트 API
- [ ] `api/multimodal.py` - 멀티모달 API
- [ ] `api/multiturn.py` - Multi-turn API
- [ ] `api/ctf.py` - CTF API
- [ ] `api/security.py` - Security Scanner API
- [ ] `api/system.py` - System Scanner API
- [ ] `api/stats.py` - 통계 API

### Phase 3: 공통 컴포넌트 (2-3일)
- [ ] `Card.js` - 재사용 가능한 카드
- [ ] `Table.js` - 동적 테이블 (정렬, 페이징)
- [ ] `Chart.js` - Chart.js 래퍼
- [ ] `Modal.js` - 모달 컴포넌트
- [ ] `SearchBar.js` - 검색 바
- [ ] `Pagination.js` - 페이지네이션

### Phase 4: 페이지별 구현 (5-7일)
- [ ] Home Dashboard (`index.html`)
- [ ] Text Prompts (`prompts.html`)
- [ ] Multimodal (`multimodal.html`)
- [ ] Multi-turn (`multiturn.html`)
- [ ] CTF Framework (`ctf.html`)
- [ ] Security Scanner (`security.html`)
- [ ] System Scanner (`system.html`)

### Phase 5: 테스트 및 최적화 (2-3일)
- [ ] API 엔드포인트 테스트
- [ ] 프론트엔드 인터랙션 테스트
- [ ] 성능 최적화
- [ ] 반응형 디자인 확인
- [ ] Dark Mode 지원

---

## 🚀 마이그레이션 전략

### 점진적 마이그레이션

1. **기존 `api.py` 유지**하면서 새로운 Blueprint 추가
2. 새 페이지는 `/v2/` 경로로 서비스
3. 모든 페이지 완성 후 기존 코드 제거
4. `/v2/` → `/`로 경로 변경

**예시**:
```python
# app.py
from api import prompts_bp, multimodal_bp

app.register_blueprint(prompts_bp, url_prefix='/api/v2/prompts')
app.register_blueprint(multimodal_bp, url_prefix='/api/v2/multimodal')

# 기존 api.py는 /api/* 경로 유지
```

---

## 📊 예상 파일 크기 비교

### Before (현재)
```
dashboard/
├── api.py         (1,132줄) ❌
└── index.html     (1,373줄) ❌
Total: 2,505줄
```

### After (리팩토링)
```
dashboard/
├── app.py         (~100줄)
├── api/
│   ├── prompts.py      (~150줄)
│   ├── multimodal.py   (~120줄)
│   ├── multiturn.py    (~130줄)
│   ├── ctf.py          (~140줄)
│   ├── security.py     (~130줄)
│   ├── system.py       (~100줄)
│   └── stats.py        (~120줄)
├── static/js/
│   ├── components/     (~600줄 total)
│   └── modules/        (~1,200줄 total)
└── templates/
    ├── base.html       (~100줄)
    ├── index.html      (~200줄)
    └── modules/        (~1,000줄 total)

Total: ~4,000줄 (더 많지만 모듈화되어 유지보수 용이)
```

---

## 🎯 핵심 이점

### 1. **유지보수성**
- 각 모듈 독립적으로 수정 가능
- 버그 격리 및 수정 용이
- 코드 리뷰 간소화

### 2. **확장성**
- 새 기능 추가 시 새 Blueprint/모듈만 추가
- 기존 코드 영향 최소화

### 3. **재사용성**
- 공통 컴포넌트 (Card, Table, Modal) 재사용
- API 클라이언트 로직 중앙화

### 4. **성능**
- 필요한 JS 모듈만 로드 (Code Splitting)
- 페이지별 최적화 가능

### 5. **개발 경험**
- 명확한 파일 구조로 코드 찾기 쉬움
- 여러 개발자 동시 작업 가능
- 테스트 작성 용이

---

## 🤔 대안 고려사항

### Option 2를 선택할 경우 (React/Vue)

**추가 작업**:
- Vite 또는 Webpack 설정
- npm/yarn 의존성 관리
- 빌드 프로세스 자동화
- `package.json` 관리

**권장 스택**:
- **React** + **Vite** + **TailwindCSS** + **React Router** + **React Query**
- **Vue 3** + **Vite** + **TailwindCSS** + **Vue Router** + **Pinia**

**장점**:
- 컴포넌트 생태계 활용
- TypeScript 지원
- 개발 도구 (React DevTools, Vue DevTools)

**단점**:
- 초기 설정 복잡
- Python 외 Node.js 환경 필요
- 배포 프로세스 복잡

---

## 📝 결론 및 권장사항

**권장**: **Option 1 (Flask Blueprint + Vanilla JS Modules)**

**이유**:
1. ✅ **Python 생태계 유지**: 추가 빌드 도구 불필요
2. ✅ **점진적 개선**: 기존 구조에서 자연스럽게 마이그레이션
3. ✅ **충분한 모듈화**: ES6 모듈로 충분히 현대적이고 유지보수 가능
4. ✅ **배포 간편**: Flask 서버 하나로 완결
5. ✅ **학습 곡선 낮음**: 기존 Flask/JS 지식으로 충분

**다음 단계**:
1. 이 기획서 검토 및 피드백
2. Phase 1 시작 (기반 구조 설정)
3. 프로토타입 페이지 1-2개 구현 (Prompts, Home)
4. 검증 후 전체 마이그레이션 진행

---

**Made with ❤️ for Prompt Arsenal Dashboard Refactoring**
**Last Updated**: 2025-01-24
