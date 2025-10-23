# Prompt Arsenal Dashboard

⚫ **블랙 컨셉** 디자인의 모던한 Prompt Arsenal 대시보드

## Features

✨ **Key Features**:
- 📊 Prompt Arsenal 통계 (22,225+ 프롬프트, 테스트 결과, 성공률)
- 🎯 3가지 뷰: **Prompts** | **Multimodal** | **Campaigns**
- 🔍 고급 필터링 (카테고리, 성공률, 정렬)
- 📱 반응형 디자인 (모바일/태블릿/데스크톱)
- 🎨 shadcn/ui 블랙 테마 (깔끔하고 다크한 UI)
- 💻 JetBrains Mono 폰트 (페이로드 표시)
- 🖼️ 멀티모달 지원 (이미지, 음성, 비디오)
- 🔄 멀티턴 캠페인 (턴별 대화 및 평가)

## Quick Start

### 1. API 서버 실행

```bash
cd dashboard
python api.py
```

API 서버가 http://localhost:5002 에서 실행됩니다.

### 2. 브라우저로 접속

```bash
open index.html
```

또는 브라우저에서 `dashboard/index.html` 파일을 직접 엽니다.

## API Endpoints

### Statistics
- `GET /api/stats` - 전체 통계

### Campaigns
- `GET /api/campaigns` - 캠페인 목록
  - Query params: `status`, `limit`, `search`
- `GET /api/campaigns/<id>` - 캠페인 상세
- `GET /api/campaigns/<id>/turns` - 턴별 결과

### Search
- `GET /api/search?q=<query>` - 캠페인 검색

## UI Components

### Stats Cards
4개의 주요 지표:
- Total Campaigns (총 캠페인 수)
- Success Rate (성공률)
- Avg Turns (평균 턴 수)
- Running (실행 중)

### Campaign Cards
각 캠페인마다:
- 이름 및 목표
- 상태 배지 (Success/Failed/Running)
- 전략, 턴 수, 타겟, 모델 정보
- 시작 시간

### Detail Modal
캠페인 클릭 시:
- 전체 정보
- 턴별 상세 결과
- 프롬프트 및 응답
- Progress, Toxicity, 시간
- Reasoning

## Styling

**Design System**: shadcn/ui inspired
- **Colors**: Tailwind CSS 팔레트
- **Typography**: Inter font
- **Icons**: Lucide icons
- **Layout**: Grid + Flexbox
- **Animations**: Smooth transitions

## Tech Stack

- **Frontend**: HTML, CSS (Tailwind), JavaScript
- **Backend**: Flask + Flask-CORS
- **Database**: SQLite (ArsenalDB)
- **Icons**: Lucide
- **Fonts**: Google Fonts (Inter)

## Development

### Requirements

```bash
pip install flask flask-cors
```

### Running

```bash
# Terminal 1: API Server
python dashboard/api.py

# Terminal 2: Open browser
open dashboard/index.html
```

### Port Configuration

API 서버 포트 변경:
```python
# api.py
port = int(os.environ.get('PORT', 5002))
```

프론트엔드 API URL 변경:
```javascript
// index.html
const API_URL = 'http://localhost:5002/api';
```

## Troubleshooting

### CORS Error
`Access-Control-Allow-Origin` 에러 발생 시:
```bash
pip install flask-cors
```

### API Connection Failed
1. API 서버가 실행 중인지 확인: `curl http://localhost:5002/api/stats`
2. 포트 확인: `lsof -i :5002`
3. 방화벽 설정 확인

### No Data
1. Database 확인: `ls -lh prompts.db`
2. 캠페인 실행: `python test_real_llm.py --auto`
3. API 테스트: `curl http://localhost:5002/api/campaigns`

## Screenshots

### Dashboard
![Dashboard](screenshots/dashboard.png)

### Campaign Detail
![Detail](screenshots/detail.png)

## Roadmap

- [ ] 차트 추가 (성공률 트렌드, 전략별 분포)
- [ ] 실시간 업데이트 (WebSocket)
- [ ] Export 기능 (JSON, CSV)
- [ ] 고급 필터링 (날짜, 모델, 전략)
- [ ] 캠페인 비교 기능
- [ ] 성능 최적화 (가상 스크롤)

## License

MIT
