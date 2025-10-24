# UX 개선 사항

## 시스템 스캔 개선 (interactive_cli.py:3892)

### 개선 전
- 스캔 타입 직접 입력: `quick`, `standard`, `full` 텍스트 입력
- LLM 프로필 선택: 구 Config 방식
- 리포트 형식: `json`, `markdown` 텍스트 입력
- 예상 시간 정보 없음
- 설명 부족

### 개선 후

#### 1. 대상 설정
```
대상 설정:
  예시: 127.0.0.1, scanme.nmap.org, 192.168.1.1
IP 또는 도메인: [127.0.0.1]
```

#### 2. 스캔 타입 선택 (숫자 입력)
```
스캔 타입 선택:
  1. Quick 스캔   - 100개 포트 (1-5분)
  2. Standard 스캔 - 1000개 포트 (5-15분) [추천]
  3. Full 스캔    - 전체 65535개 포트 (30분 이상)
선택 (1-3): [2]
```

#### 3. LLM 분석 (ProfileManager 통합)
```
LLM 분석:
  AI를 사용해 취약점을 분석하고 상세한 보고서를 생성합니다.
사용하시겠습니까? (y/n): [n]

LLM 프로필 선택:
  1. ★ gpt_test (openai/gpt-4o-mini)
  2.   gemini-test (google/gemini-2.5-flash-lite)
선택 (1-2): [1]
✓ gpt_test 프로필 선택됨
```

#### 4. 리포트 내보내기 (숫자 입력)
```
리포트 내보내기:
상세 리포트를 파일로 저장하시겠습니까? (y/n): [y]

형식 선택:
  1. Markdown - 읽기 쉬운 문서 형식 (.md) [추천]
  2. JSON - 기계 가독 형식 (.json)
선택 (1-2): [1]

✓ 리포트 저장됨: system_scan_1.md
파일 위치: /Users/.../system_scan_1.md
```

## 주요 개선 사항

### 1. 숫자 선택 방식
- 텍스트 입력 대신 **숫자 1-3** 선택
- 오타 방지 및 선택 속도 향상
- `choices` 파라미터로 유효성 검증

### 2. 예상 시간 표시
- Quick: 1-5분
- Standard: 5-15분
- Full: 30분 이상

### 3. 권장 옵션 표시
- Standard 스캔 [추천]
- Markdown 형식 [추천]
- Default 프로필 ★ 표시

### 4. 상세 설명 추가
- 대상 입력 예시
- LLM 분석 기능 설명
- 리포트 형식 설명

### 5. ProfileManager 통합
- 구 Config 대신 ProfileManager 사용
- LLM 프로필만 필터링 (`list_llm_profiles()`)
- Default 프로필 자동 표시

### 6. 파일 위치 표시
- 리포트 저장 시 절대 경로 표시
- `os.path.abspath()` 사용

## 코드 변경 사항

### 파일: interactive_cli.py

**Line 3904-3912**: 스캔 타입 숫자 선택
```python
console.print("\n[yellow]스캔 타입 선택:[/yellow]")
console.print("  [green]1[/green]. Quick 스캔   - 100개 포트 (1-5분)")
console.print("  [green]2[/green]. Standard 스캔 - 1000개 포트 (5-15분) [추천]")
console.print("  [green]3[/green]. Full 스캔    - 전체 65535개 포트 (30분 이상)")

scan_type_choice = ask("선택 (1-3)", default="2", choices=["1", "2", "3"])
scan_type_map = {'1': 'quick', '2': 'standard', '3': 'full'}
scan_type = scan_type_map[scan_type_choice]
```

**Line 3920-3952**: ProfileManager LLM 프로필 선택
```python
from core import get_profile_manager
pm = get_profile_manager()
llm_profiles = pm.list_llm_profiles()

# Show profiles with default marker
for i, (name, profile) in enumerate(profile_list, 1):
    default_marker = "★" if pm.default_profile == name else " "
    console.print(f"  [green]{i}[/green]. {default_marker} {name} ...")
```

**Line 3976-3993**: 리포트 형식 숫자 선택 + 파일 위치 표시
```python
console.print("\n[yellow]형식 선택:[/yellow]")
console.print("  [green]1[/green]. Markdown - 읽기 쉬운 문서 형식 (.md) [추천]")
console.print("  [green]2[/green]. JSON - 기계 가독 형식 (.json)")

format_choice_num = ask("선택 (1-2)", default="1", choices=["1", "2"])
format_map = {'1': 'markdown', '2': 'json'}
format_choice = format_map[format_choice_num]

# 파일 위치 표시
console.print(f"\n[green]✓ 리포트 저장됨: {filename}[/green]")
console.print(f"[dim]파일 위치: {os.path.abspath(filename)}[/dim]")
```

## 사용자 편의성 향상

### Before (불편)
1. `quick`, `standard`, `full` 직접 타이핑 → 오타 가능
2. 스캔 시간 예상 불가 → 기다림
3. LLM 기능 설명 없음 → 혼란
4. 리포트 저장 위치 불명확

### After (편리)
1. **숫자 1-3** 입력만 하면 됨 → 오타 방지
2. **예상 시간** 표시 → 계획 가능
3. **기능 설명** 제공 → 이해 쉬움
4. **절대 경로** 표시 → 파일 찾기 쉬움

## 다음 개선 예정

- [ ] CTF 자동 풀이 시 도구 선택 개선
- [ ] Multimodal 공격 생성 UX 개선
- [ ] 프롬프트 검색 필터 개선
- [ ] API 프로필 관리 개선
