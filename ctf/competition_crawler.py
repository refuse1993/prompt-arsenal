"""
CTF Competition Crawler
대회 메인 페이지에서 모든 챌린지를 자동으로 발견하고 분석하여 DB에 저장
"""

import asyncio
import json
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm

console = Console()


@dataclass
class ChallengeInfo:
    """발견된 챌린지 정보"""
    title: str
    url: str
    category: Optional[str] = None
    difficulty: Optional[str] = None
    description: Optional[str] = None
    points: Optional[int] = None
    hints: List[str] = None

    def __post_init__(self):
        if self.hints is None:
            self.hints = []


class CompetitionCrawler:
    """CTF 대회 크롤러"""

    def __init__(self, db, llm_profile_name: Optional[str] = None):
        self.db = db
        self.discovered_links: List[str] = []
        self.challenges: List[ChallengeInfo] = []
        self.visited_urls: set = set()

        # LLM 초기화
        from core import get_profile_manager
        pm = get_profile_manager()

        if llm_profile_name:
            self.llm_profile = pm.get_profile(llm_profile_name)
        else:
            # Default profile 사용
            self.llm_profile = pm.get_profile(pm.default_profile)

        if not self.llm_profile:
            console.print("[yellow]⚠️  LLM 프로필을 찾을 수 없습니다. 페이지 판단 기능이 비활성화됩니다.[/yellow]")
            self.llm = None
        else:
            from multimodal.llm_client import LLMClient
            self.llm = LLMClient(
                provider=self.llm_profile['provider'],
                model=self.llm_profile['model'],
                api_key=self.llm_profile['api_key'],
                base_url=self.llm_profile.get('base_url')
            )

    async def crawl_competition(self, main_url: str, competition_name: Optional[str] = None, max_challenges: Optional[int] = None) -> Dict:
        """대회 메인 페이지 크롤링 및 챌린지 수집

        Args:
            main_url: 대회 메인 페이지 URL
            competition_name: 대회 이름 (선택)
            max_challenges: 최대 수집 챌린지 개수 (선택, None이면 전체)

        Returns:
            통계 정보 딕셔너리
        """
        console.print(f"\n[bold cyan]🔍 CTF 대회 크롤링 시작[/bold cyan]")
        console.print(f"대상: {main_url}")

        if not competition_name:
            competition_name = Prompt.ask("대회 이름을 입력하세요", default="Unknown CTF")

        console.print(f"대회명: [yellow]{competition_name}[/yellow]")

        if max_challenges:
            console.print(f"수집 제한: [yellow]{max_challenges}개[/yellow]\n")
        else:
            console.print(f"수집 제한: [yellow]없음 (전체 수집)[/yellow]\n")

        try:
            from playwright.async_api import async_playwright
        except ImportError:
            console.print("[red]❌ Playwright가 설치되지 않았습니다[/red]")
            console.print("설치: playwright install chromium")
            return {'error': 'Playwright not installed'}

        stats = {
            'competition': competition_name,
            'main_url': main_url,
            'links_discovered': 0,
            'links_skipped': 0,
            'challenges_found': 0,
            'challenges_saved': 0,
            'errors': []
        }

        # Playwright 브라우저 자동 설치 체크
        async with async_playwright() as p:
            try:
                browser = await p.chromium.launch(headless=False)  # headless=False로 사용자가 로그인 가능
            except Exception as e:
                if "Executable doesn't exist" in str(e) or "playwright install" in str(e):
                    console.print("[yellow]📦 Playwright 브라우저가 설치되지 않았습니다[/yellow]")
                    console.print("[cyan]자동으로 Chromium을 설치합니다...[/cyan]\n")

                    import subprocess
                    import sys

                    try:
                        # playwright install chromium 실행 (진행 상황 실시간 출력)
                        result = subprocess.run(
                            [sys.executable, "-m", "playwright", "install", "chromium"],
                            timeout=300  # 5분 타임아웃
                        )

                        if result.returncode == 0:
                            console.print("\n[green]✅ Chromium 설치 완료![/green]\n")
                            # 재시도
                            browser = await p.chromium.launch(headless=False)
                        else:
                            console.print(f"[red]❌ 설치 실패 (exit code: {result.returncode})[/red]")
                            console.print("[yellow]수동으로 실행하세요: playwright install chromium[/yellow]")
                            return {'error': 'Browser installation failed'}
                    except subprocess.TimeoutExpired:
                        console.print("[red]❌ 설치 타임아웃 (5분 초과)[/red]")
                        return {'error': 'Installation timeout'}
                    except Exception as install_error:
                        console.print(f"[red]❌ 설치 오류: {install_error}[/red]")
                        return {'error': str(install_error)}
                else:
                    raise  # 다른 오류는 그대로 전파
            context = await browser.new_context()
            page = await context.new_page()

            try:
                # 1. 메인 페이지 접속
                console.print("[cyan]📄 메인 페이지 로딩 중...[/cyan]")
                await page.goto(main_url, wait_until='networkidle', timeout=30000)

                # 2. 로그인 필요 여부 확인
                login_needed = await self._check_login_required(page)
                if login_needed:
                    console.print("\n[yellow]⚠️  로그인이 필요한 것 같습니다[/yellow]")
                    if Confirm.ask("로그인하시겠습니까? (브라우저가 열려있습니다)"):
                        console.print("[cyan]브라우저에서 로그인을 완료하세요...[/cyan]")
                        Prompt.ask("로그인 완료 후 Enter를 누르세요")
                    else:
                        console.print("[yellow]로그인 없이 계속합니다[/yellow]")

                # 2.5. 페이지 재로딩 (API 감지를 위해)
                console.print("\n[cyan]🔄 API 탐지를 위해 페이지를 새로고침합니다...[/cyan]")
                await page.reload(wait_until='networkidle', timeout=30000)
                await asyncio.sleep(2)  # API 요청 완료 대기

                # 2.6. LLM으로 페이지 타입 판단 (최대 3번 리다이렉트)
                max_redirect_attempts = 3
                redirect_count = 0

                while redirect_count < max_redirect_attempts:
                    if self.llm:
                        console.print("\n[cyan]🤖 페이지 분석 중...[/cyan]")
                        page_analysis = await self._analyze_page_type(page)

                        if not page_analysis['is_challenge_page']:
                            console.print(f"\n[yellow]⚠️  {page_analysis['reason']}[/yellow]")
                            console.print(f"[dim]Confidence: {page_analysis['confidence']:.0%}[/dim]")

                            # 챌린지 페이지 찾기 시도
                            console.print("\n[cyan]챌린지 페이지 링크를 찾는 중...[/cyan]")
                            challenge_url = await self._find_challenge_page(page)

                            if challenge_url:
                                console.print(f"[green]✓ 챌린지 페이지 발견: {challenge_url}[/green]")
                                if Confirm.ask("이 페이지로 이동하시겠습니까?", default=True):
                                    await page.goto(challenge_url, wait_until='networkidle', timeout=30000)
                                    console.print(f"[green]→ {challenge_url}로 이동 완료[/green]")

                                    # 이동 후 다시 분석
                                    redirect_count += 1
                                    if redirect_count < max_redirect_attempts:
                                        console.print(f"\n[cyan]🔄 페이지 재분석 중... ({redirect_count}/{max_redirect_attempts})[/cyan]")
                                        await asyncio.sleep(1)
                                        continue  # 다시 페이지 타입 분석
                                    else:
                                        console.print(f"[yellow]⚠️  최대 리다이렉트 횟수 초과 ({max_redirect_attempts}회)[/yellow]")
                                        break
                                else:
                                    console.print("[yellow]크롤링을 중단합니다[/yellow]")
                                    await browser.close()
                                    return stats
                            else:
                                console.print("[red]❌ 챌린지 페이지를 찾지 못했습니다[/red]")
                                console.print(f"[dim]제안: {page_analysis['suggestion']}[/dim]")

                                # 올바른 URL 직접 입력 옵션 제공
                                if Confirm.ask("올바른 챌린지 페이지 URL을 직접 입력하시겠습니까?", default=False):
                                    new_url = Prompt.ask("챌린지 페이지 URL (예: https://play.picoctf.org/practice)")
                                    if new_url and new_url.startswith('http'):
                                        await page.goto(new_url, wait_until='networkidle', timeout=30000)
                                        console.print(f"[green]→ {new_url}로 이동 완료[/green]")
                                        redirect_count += 1
                                        if redirect_count < max_redirect_attempts:
                                            console.print(f"\n[cyan]🔄 페이지 재분석 중... ({redirect_count}/{max_redirect_attempts})[/cyan]")
                                            await asyncio.sleep(1)
                                            continue
                                        else:
                                            console.print(f"[yellow]⚠️  최대 리다이렉트 횟수 초과 ({max_redirect_attempts}회)[/yellow]")
                                            break

                                if not Confirm.ask("현재 페이지에서 계속하시겠습니까?", default=False):
                                    await browser.close()
                                    return stats
                                break  # 강제로 계속 진행
                        else:
                            console.print(f"\n[green]✓ {page_analysis['reason']}[/green]")
                            console.print(f"[dim]Confidence: {page_analysis['confidence']:.0%}[/dim]")
                            break  # 올바른 페이지 발견
                    else:
                        break  # LLM 없으면 바로 진행

                # 3. 모든 챌린지 링크/데이터 수집
                console.print("\n[cyan]🔗 챌린지 탐색 중...[/cyan]")
                console.print("[dim]JavaScript 렌더링 대기 중...[/dim]")

                # SPA를 위한 대기 시간 (React/Vue 등)
                await asyncio.sleep(3)

                items = await self._discover_challenge_links(page, main_url)
                stats['links_discovered'] = len(items)

                if not items:
                    console.print("[yellow]⚠️  챌린지를 찾지 못했습니다[/yellow]")
                    console.print("페이지 구조를 수동으로 확인해주세요")
                    await asyncio.sleep(2)
                    await browser.close()
                    return stats

                # API에서 완전한 데이터를 받았는지 확인
                is_api_data = items and isinstance(items[0], ChallengeInfo)

                if is_api_data:
                    console.print(f"[green]✓ API에서 {len(items)}개의 챌린지 데이터 수집 완료[/green]")
                else:
                    console.print(f"[green]✓ {len(items)}개의 링크 발견[/green]")
                    # 디버깅: 발견된 링크 샘플 출력
                    if items and len(items) > 0:
                        console.print(f"[dim]샘플 링크: {items[0][:80]}...[/dim]")

                # 4. 각 챌린지 처리
                if is_api_data:
                    # API 데이터: LLM 분석만 추가하고 바로 저장
                    console.print(f"\n[cyan]📊 챌린지 LLM 분석 시작 ({len(items)}개)[/cyan]")
                    console.print("[dim]API에서 받은 완전한 데이터에 LLM 분석을 추가합니다...[/dim]\n")

                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=console
                    ) as progress:
                        task = progress.add_task("분석 중...", total=len(items))

                        for i, challenge in enumerate(items, 1):
                            # 개수 제한 체크
                            if max_challenges and stats['challenges_saved'] >= max_challenges:
                                console.print(f"\n[yellow]⚠️  최대 개수({max_challenges}개)에 도달했습니다[/yellow]")
                                for _ in range(i, len(items) + 1):
                                    progress.advance(task)
                                break

                            progress.update(task, description=f"[cyan]{i}/{len(items)} [{challenge.category}] {challenge.title[:30]}...[/cyan]")

                            try:
                                # LLM 분석 추가 (선택적)
                                if self.llm and challenge.description:
                                    llm_analysis = await self._llm_analyze_challenge(
                                        challenge.title,
                                        challenge.category,
                                        challenge.difficulty,
                                        challenge.description,
                                        challenge.hints
                                    )

                                    # LLM 분석을 description에 추가
                                    if llm_analysis:
                                        challenge.description = f"{challenge.description}\n\n{'='*50}\n🤖 LLM 분석\n{'='*50}\n\n{llm_analysis}"

                                # 챌린지 추가
                                self.challenges.append(challenge)
                                stats['challenges_found'] += 1

                                # DB 저장
                                saved = await self._save_challenge_to_db(challenge, competition_name)
                                if saved:
                                    stats['challenges_saved'] += 1

                            except Exception as e:
                                stats['errors'].append(f"{challenge.title}: {str(e)}")
                                console.print(f"[red]  ⚠️  {challenge.title}: {str(e)[:50]}...[/red]")

                            progress.advance(task)
                            await asyncio.sleep(0.3)  # Rate limiting (API 데이터는 더 빠름)

                else:
                    # URL 링크: 기존 방식대로 페이지 스크래핑
                    console.print(f"\n[cyan]📊 챌린지 분석 시작 ({len(items)}개)[/cyan]")
                    if self.llm:
                        console.print("[dim]LLM으로 각 링크를 검증합니다...[/dim]\n")
                    else:
                        console.print("[dim]URL 패턴으로 기본 필터링을 적용합니다...[/dim]\n")

                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=console
                    ) as progress:
                        task = progress.add_task("분석 중...", total=len(items))
                        stats['links_skipped'] = 0

                        for i, link in enumerate(items, 1):
                            # 개수 제한 체크
                            if max_challenges and stats['challenges_saved'] >= max_challenges:
                                console.print(f"\n[yellow]⚠️  최대 개수({max_challenges}개)에 도달했습니다[/yellow]")
                                for _ in range(i, len(items) + 1):
                                    progress.advance(task)
                                break

                            if link in self.visited_urls:
                                progress.advance(task)
                                continue

                            self.visited_urls.add(link)
                            progress.update(task, description=f"[cyan]{i}/{len(items)} 검증 중...[/cyan]")

                            try:
                                # URL 검증 (필요시 페이지 로딩)
                                is_valid = await self._is_valid_challenge_url(page, link)

                                if not is_valid:
                                    stats['links_skipped'] += 1
                                    progress.advance(task)
                                    continue

                                # 검증 통과한 링크만 분석
                                progress.update(task, description=f"[cyan]{i}/{len(items)} {link[:50]}...[/cyan]")
                                challenge = await self._analyze_challenge_page(page, link, main_url)
                                if challenge:
                                    self.challenges.append(challenge)
                                    stats['challenges_found'] += 1

                                    # DB 저장
                                    saved = await self._save_challenge_to_db(challenge, competition_name)
                                    if saved:
                                        stats['challenges_saved'] += 1

                            except Exception as e:
                                stats['errors'].append(f"{link}: {str(e)}")
                                console.print(f"[red]  ⚠️  {link}: {str(e)[:50]}...[/red]")

                            progress.advance(task)
                            await asyncio.sleep(0.5)  # Rate limiting

                # 5. 결과 출력
                console.print(f"\n[bold green]✅ 크롤링 완료[/bold green]")
                console.print(f"  • 발견된 링크: {stats['links_discovered']}개")
                console.print(f"  • 건너뛴 링크: {stats.get('links_skipped', 0)}개 (비챌린지 페이지)")
                console.print(f"  • 분석된 챌린지: {stats['challenges_found']}개")
                console.print(f"  • DB 저장: {stats['challenges_saved']}개")

                if stats['errors']:
                    console.print(f"  • 오류: {len(stats['errors'])}건")

            except Exception as e:
                console.print(f"[red]❌ 크롤링 오류: {str(e)}[/red]")
                stats['errors'].append(str(e))

            finally:
                await browser.close()

        return stats

    async def _check_login_required(self, page) -> bool:
        """로그인 필요 여부 확인"""
        # 일반적인 로그인 패턴 탐지
        login_indicators = [
            'text=Login',
            'text=Sign in',
            'text=Log in',
            'input[type="password"]',
            'a[href*="login"]',
            'button:has-text("Login")'
        ]

        for indicator in login_indicators:
            try:
                element = await page.query_selector(indicator)
                if element:
                    return True
            except:
                pass

        return False

    async def _interactive_link_discovery(self, page, base_url: str) -> List[str]:
        """대화형 링크 발견 - LLM이 사용자에게 질문하며 찾기"""
        console.print("\n[cyan]🤖 LLM이 페이지를 분석하고 있습니다...[/cyan]")

        # 1. 페이지의 시각적 구조 파악
        page_info = await page.evaluate('''() => {
            // 모든 클릭 가능한 요소 찾기
            const clickables = [];
            const seen = new Set();

            // 더 많은 선택자로 요소 수집
            const selectors = [
                'a[href]',
                'button',
                'div[role="button"]',
                'div[onclick]',
                '[data-challenge-id]',
                '.challenge',
                '.problem',
                '.task'
            ];

            selectors.forEach(selector => {
                document.querySelectorAll(selector).forEach((el, idx) => {
                    // 보이지 않는 요소 제외
                    if (el.offsetParent === null && el.tagName !== 'A') {
                        return;
                    }

                    const text = el.innerText?.trim().slice(0, 100) || el.textContent?.trim().slice(0, 100) || '';
                    const href = el.href || el.getAttribute('data-url') || el.getAttribute('data-href') || '';
                    const classes = el.className || '';
                    const id = el.id || '';

                    // 중복 제거 (text와 href 조합으로)
                    const key = text + href;
                    if (!seen.has(key) && clickables.length < 50) {  // 30 → 50개로 증가
                        seen.add(key);
                        clickables.push({
                            index: clickables.length,  // 실제 배열 index 사용
                            tag: el.tagName.toLowerCase(),
                            text: text,
                            href: href,
                            classes: classes,
                            id: id,
                            // 챌린지 키워드 체크
                            hasChallenge: (text + href + classes).toLowerCase().includes('challenge') ||
                                         (text + href + classes).toLowerCase().includes('problem') ||
                                         (text + href + classes).toLowerCase().includes('task'),
                            // 네비게이션인지 체크
                            isNav: classes.toLowerCase().includes('nav') ||
                                  classes.toLowerCase().includes('menu') ||
                                  id.toLowerCase().includes('nav')
                        });
                    }
                });
            });

            return {
                title: document.title,
                url: window.location.href,
                clickables: clickables
            };
        }''')

        # 2. LLM에게 페이지 보여주고 질문 생성 요청
        prompt = f"""당신은 CTF 챌린지 크롤러입니다. 다음 페이지에서 **개별 챌린지 문제 페이지로 가는 링크**를 찾아야 합니다.

**중요**: 우리가 찾는 것은:
- ✅ "Riddle Registry", "Log Hunt", "Hidden in plainsight" 같은 **개별 문제 이름**을 가진 링크
- ❌ "Challenges", "Practice", "Home" 같은 **네비게이션 메뉴**가 아님
- ❌ Facebook, Twitter, 메인 페이지 같은 **외부 링크**가 아님

**페이지 정보**:
- URL: {page_info['url']}
- 제목: {page_info['title']}

**발견된 클릭 가능한 요소들** (최대 50개):
{json.dumps(page_info['clickables'][:50], indent=2, ensure_ascii=False)}

**참고**: 각 요소에는 다음 정보가 포함됩니다:
- `text`: 화면에 표시되는 텍스트
- `href`: 링크 URL
- `isNav`: true이면 네비게이션 메뉴
- `hasChallenge`: true이면 'challenge' 같은 키워드 포함

**임무**: 위 요소들 중에서 **개별 챌린지 문제 이름으로 보이는 것들**을 찾고, 사용자에게 확인 질문 3개를 생성하세요.

질문 예시:
- "'Riddle Registry', 'Log Hunt' 같은 텍스트들이 개별 챌린지 문제 이름인가요?"
- "index 8~20번 요소들에 문제 이름이 보이나요?"
- "URL에 '/challenge/숫자' 또는 '/problem/ID' 같은 패턴이 있나요?"

**주의**: 네비게이션 링크(Home, Challenges, Practice)와 개별 문제를 구분하세요!

JSON 형식으로 답변:
{{
    "analysis": "개별 챌린지 문제로 보이는 요소들의 패턴 요약",
    "confidence": 0.0-1.0,
    "questions": [
        {{"question": "구체적 질문 1", "purpose": "이 질문을 하는 이유"}},
        {{"question": "구체적 질문 2", "purpose": "이 질문을 하는 이유"}},
        {{"question": "구체적 질문 3", "purpose": "이 질문을 하는 이유"}}
    ],
    "suggested_action": "만약 사용자가 모두 'yes'라고 답하면 어떻게 할지"
}}"""

        try:
            response = await self.llm.generate(prompt)

            # JSON 파싱
            if '```json' in response:
                json_str = response.split('```json')[1].split('```')[0].strip()
            elif '```' in response:
                json_str = response.split('```')[1].split('```')[0].strip()
            else:
                json_str = response.strip()

            analysis = json.loads(json_str)

            # 3. LLM의 분석 결과 표시
            console.print(f"\n[cyan]🤖 분석: {analysis.get('analysis', 'N/A')}[/cyan]")
            console.print(f"[dim]신뢰도: {analysis.get('confidence', 0):.0%}[/dim]\n")

            # 4. 사용자에게 질문하기
            user_responses = []
            questions = analysis.get('questions', [])

            for i, q in enumerate(questions[:3], 1):
                console.print(f"[yellow]질문 {i}:[/yellow] {q['question']}")
                console.print(f"[dim]({q.get('purpose', '')})[/dim]")

                answer = Prompt.ask("답변", choices=["yes", "no", "skip"], default="yes")
                user_responses.append({
                    'question': q['question'],
                    'answer': answer
                })
                console.print()

            # 5. 사용자 답변을 바탕으로 링크 추출 전략 결정
            strategy_prompt = f"""사용자의 답변을 바탕으로 **개별 챌린지 문제 페이지 링크**를 찾는 구체적인 전략을 제시하세요.

**중요**:
- 네비게이션 링크(nav-link, menu)는 제외
- 외부 링크(facebook, twitter)는 제외
- 개별 문제 이름을 가진 링크만 선택

사용자 답변:
{json.dumps(user_responses, indent=2, ensure_ascii=False)}

원본 페이지 요소:
{json.dumps(page_info['clickables'][:50], indent=2, ensure_ascii=False)}

**힌트**: `isNav: true`인 요소는 제외하세요!

**요청**: 다음 중 **가장 효과적인 방법 1개**를 선택하세요:

1. **CSS 선택자**: 개별 문제를 감싸는 특정 클래스/태그 (예: `div.challenge-card a`, `a[href*="/challenge/"]`)
2. **Index 범위**: 개별 문제 이름이 보이는 요소들의 index (예: 8~25)
3. **텍스트 패턴**: 문제 이름의 공통 특징 (예: 대문자로 시작, 10-30자)
4. **URL 패턴**: URL의 공통 패턴 (예: `/practice/challenge/`, `/problem/`)

**주의**:
- `nav-link`, `navbar`, `menu` 클래스를 가진 요소는 제외
- href가 `#`, `javascript:`, 외부 도메인인 경우 제외
- 개별 문제로 가는 링크만 선택

JSON 형식:
{{
    "method": "css|index|text|url",
    "details": {{
        "css_selector": "...",  // 네비게이션 제외된 선택자
        "index_range": [start, end],  // 개별 문제 요소들의 범위
        "text_pattern": "...",
        "url_pattern": "..."  // URL에서 반복되는 패턴
    }},
    "reason": "왜 이 방법을 선택했는지"
}}"""

            strategy_response = await self.llm.generate(strategy_prompt)

            # JSON 파싱
            if '```json' in strategy_response:
                json_str = strategy_response.split('```json')[1].split('```')[0].strip()
            elif '```' in strategy_response:
                json_str = strategy_response.split('```')[1].split('```')[0].strip()
            else:
                json_str = strategy_response.strip()

            strategy = json.loads(json_str)

            # 6. 전략 실행
            console.print(f"[green]✓ 전략: {strategy.get('reason', '')}[/green]")
            links = set()

            method = strategy.get('method', '')
            details = strategy.get('details', {})

            if method == 'css':
                selector = details.get('css_selector', '')
                console.print(f"[dim]CSS 선택자: {selector}[/dim]")
                elements = await page.query_selector_all(selector)

                from urllib.parse import urlparse
                base_domain = urlparse(base_url).netloc

                for el in elements:
                    href = await el.get_attribute('href')
                    classes = await el.get_attribute('class') or ''

                    # 네비게이션 요소 제외
                    if 'nav' in classes.lower() or 'menu' in classes.lower():
                        continue

                    if href and not href.startswith('#') and not href.startswith('javascript:'):
                        # 외부 도메인 제외
                        full_url = urljoin(base_url, href)
                        link_domain = urlparse(full_url).netloc
                        if not link_domain or link_domain == base_domain:
                            links.add(full_url)

            elif method == 'index':
                range_info = details.get('index_range', [0, 0])
                console.print(f"[dim]Index 범위: {range_info[0]}-{range_info[1]}[/dim]")

                from urllib.parse import urlparse
                base_domain = urlparse(base_url).netloc

                for item in page_info['clickables'][range_info[0]:range_info[1]+1]:
                    href = item.get('href', '')
                    classes = item.get('classes', '').lower()
                    is_nav = item.get('isNav', False)

                    # 네비게이션/외부 링크 제외
                    if is_nav or 'nav' in classes or 'menu' in classes:
                        continue

                    if href and not href.startswith('#') and not href.startswith('javascript:'):
                        full_url = urljoin(base_url, href)
                        link_domain = urlparse(full_url).netloc
                        if not link_domain or link_domain == base_domain:
                            links.add(full_url)

            elif method == 'text':
                pattern = details.get('text_pattern', '').lower()
                console.print(f"[dim]텍스트 패턴: {pattern}[/dim]")

                from urllib.parse import urlparse
                base_domain = urlparse(base_url).netloc

                for item in page_info['clickables']:
                    text = item.get('text', '').lower()
                    href = item.get('href', '')
                    classes = item.get('classes', '').lower()
                    is_nav = item.get('isNav', False)

                    if is_nav or 'nav' in classes or 'menu' in classes:
                        continue

                    if pattern in text and href:
                        full_url = urljoin(base_url, href)
                        link_domain = urlparse(full_url).netloc
                        if not link_domain or link_domain == base_domain:
                            links.add(full_url)

            elif method == 'url':
                pattern = details.get('url_pattern', '')
                console.print(f"[dim]URL 패턴: {pattern}[/dim]")

                from urllib.parse import urlparse
                base_domain = urlparse(base_url).netloc

                for item in page_info['clickables']:
                    href = item.get('href', '')
                    classes = item.get('classes', '').lower()
                    is_nav = item.get('isNav', False)

                    if is_nav or 'nav' in classes or 'menu' in classes:
                        continue

                    if pattern in href and href:
                        full_url = urljoin(base_url, href)
                        link_domain = urlparse(full_url).netloc
                        if not link_domain or link_domain == base_domain:
                            links.add(full_url)

            console.print(f"[green]✓ {len(links)}개 링크 발견[/green]")
            return sorted(list(links))

        except Exception as e:
            console.print(f"[red]대화형 발견 실패: {str(e)}[/red]")
            return []

    async def _intercept_api_calls(self, page, base_url: str, already_loaded: bool = True) -> Optional[str]:
        """API 엔드포인트 자동 탐지

        Args:
            page: Playwright 페이지 객체
            base_url: 베이스 URL
            already_loaded: 페이지가 이미 로드되었는지 (True면 재로딩 수행)
        """
        api_endpoints = []

        # Network 요청 감시 시작
        def handle_request(request):
            url = request.url
            # API 패턴 감지
            if any(keyword in url.lower() for keyword in ['/api/', '/challenges', '/problems', '/tasks']):
                if request.method == 'GET' and url not in api_endpoints:
                    api_endpoints.append(url)
                    console.print(f"[dim]  API 발견: {url}[/dim]")

        page.on('request', handle_request)

        try:
            console.print("[dim]API 엔드포인트 감지 중...[/dim]")

            if already_loaded:
                # 페이지가 이미 로드된 경우 재로딩 (API 요청 트리거)
                console.print("[dim]  페이지 재로딩으로 API 요청 트리거...[/dim]")
                await page.reload(wait_until='networkidle', timeout=30000)

            # 잠시 대기 (API 요청 완료)
            await asyncio.sleep(2)

            # 스크롤로 추가 API 호출 트리거 (lazy loading)
            await page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
            await asyncio.sleep(1)

        finally:
            page.remove_listener('request', handle_request)

        if api_endpoints:
            console.print(f"\n[green]✓ {len(api_endpoints)}개 API 엔드포인트 발견[/green]")
            for i, endpoint in enumerate(api_endpoints[:10], 1):
                console.print(f"  {i}. {endpoint}")

            # 챌린지 관련 API 우선 선택
            challenge_apis = [ep for ep in api_endpoints if 'challenge' in ep.lower()]
            if challenge_apis:
                return challenge_apis[0]

            # 없으면 첫 번째 API 반환
            return api_endpoints[0]

        return None

    async def _fetch_from_api(self, page_obj, api_url: str, base_url: str) -> List[ChallengeInfo]:
        """API에서 직접 챌린지 목록 가져오기 (브라우저 세션 사용)

        Returns:
            List[ChallengeInfo]: API에서 파싱한 챌린지 객체 리스트
        """
        from urllib.parse import urlparse, urljoin, parse_qs, urlunparse, urlencode

        console.print(f"[cyan]📡 API 호출 중: {api_url}[/cyan]")

        all_challenges = []
        page_num = 1
        base_domain = urlparse(base_url).scheme + '://' + urlparse(base_url).netloc

        while True:
            # URL 파라미터 조작
            parsed = urlparse(api_url)
            params = parse_qs(parsed.query)
            params['page'] = [str(page_num)]
            params['page_size'] = ['100']  # 큰 값으로

            new_query = urlencode(params, doseq=True)
            current_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment))

            try:
                # Playwright의 page.request API 사용 (브라우저 쿠키/세션 자동 포함)
                response = await page_obj.request.get(current_url)

                if not response.ok:
                    if page_num == 1:
                        console.print(f"[yellow]  API 응답 실패: {response.status}[/yellow]")
                    break

                data = await response.json()

                # 다양한 JSON 구조 처리
                if isinstance(data, list):
                    challenges = data
                elif isinstance(data, dict):
                    challenges = (data.get('results') or
                                data.get('data') or
                                data.get('challenges') or
                                data.get('items') or [])
                else:
                    break

                if not challenges:
                    break

                all_challenges.extend(challenges)
                console.print(f"[dim]  페이지 {page_num}: {len(challenges)}개 챌린지[/dim]")

                # 다음 페이지 확인
                if isinstance(data, dict):
                    has_next = data.get('next') or data.get('has_next')
                    if not has_next and len(challenges) < 100:
                        break
                elif len(challenges) < 100:
                    break

                page_num += 1

                # 안전장치: 최대 50페이지
                if page_num > 50:
                    break

            except Exception as e:
                console.print(f"[yellow]  API 호출 오류: {str(e)[:100]}[/yellow]")
                break

        if not all_challenges:
            return []

        console.print(f"[green]✓ 총 {len(all_challenges)}개 챌린지 발견[/green]")

        # JSON 데이터를 ChallengeInfo 객체로 변환
        challenge_list = []

        for challenge in all_challenges:
            # ID와 URL
            challenge_id = challenge.get('id') or challenge.get('challenge_id') or challenge.get('cid')
            url_path = challenge.get('url') or challenge.get('path') or challenge.get('link')

            # URL 생성
            if url_path:
                if url_path.startswith('http'):
                    url = url_path
                else:
                    url = urljoin(base_domain, url_path)
            elif challenge_id:
                url = f"{base_domain}/practice/challenge/{challenge_id}"
            else:
                continue  # URL을 만들 수 없으면 스킵

            # API 데이터에서 정보 추출
            title = challenge.get('name') or challenge.get('title') or 'Unknown'
            category = challenge.get('category') or challenge.get('type') or 'misc'
            difficulty_num = challenge.get('difficulty')
            description = challenge.get('description') or challenge.get('desc') or ''
            author = challenge.get('author') or ''

            # 난이도 변환 (숫자 → 문자열)
            difficulty_map = {1: 'easy', 2: 'medium', 3: 'hard', 4: 'insane'}
            difficulty = difficulty_map.get(difficulty_num, 'unknown')

            # 설명 구성
            if author:
                description = f"**Author:** {author}\n\n{description}"

            # ChallengeInfo 객체 생성
            challenge_info = ChallengeInfo(
                title=title,
                url=url,
                category=category.lower(),
                difficulty=difficulty,
                description=description,
                hints=[]  # API에 힌트가 있으면 추가 가능
            )

            challenge_list.append(challenge_info)

        return challenge_list

    async def _discover_challenge_links(self, page, base_url: str) -> Union[List[str], List[ChallengeInfo]]:
        """챌린지 링크 발견 (API 우선)

        Returns:
            Union[List[str], List[ChallengeInfo]]: API 사용시 ChallengeInfo 객체 리스트,
                                                    그 외에는 URL 문자열 리스트
        """

        # 방법 1: API 자동 탐지 및 호출
        console.print("\n[cyan]🔍 API 엔드포인트 탐색 중...[/cyan]")
        api_url = await self._intercept_api_calls(page, base_url)

        if api_url:
            if Confirm.ask(f"\n발견된 API를 호출할까요? (훨씬 빠릅니다)\n{api_url}", default=True):
                challenges = await self._fetch_from_api(page, api_url, base_url)

                if challenges:
                    console.print(f"\n[yellow]샘플 챌린지 (최대 5개):[/yellow]")
                    for ch in challenges[:5]:
                        console.print(f"  • [{ch.category}] {ch.title} ({ch.difficulty})")

                    if len(challenges) > 5:
                        console.print(f"  ... 외 {len(challenges) - 5}개")

                    if Confirm.ask("\n이 챌린지들이 맞나요?", default=True):
                        console.print("[green]✓ API에서 완전한 데이터를 가져왔습니다[/green]")
                        return challenges  # ChallengeInfo 객체 리스트 반환
                    else:
                        console.print("[yellow]다른 방법을 시도합니다...[/yellow]")

        # 방법 2: LLM 대화형 발견
        if self.llm:
            console.print("\n[cyan]🤖 LLM 대화형 링크 발견을 시도합니다...[/cyan]")
            links = await self._interactive_link_discovery(page, base_url)

            if links:
                console.print(f"\n[yellow]발견된 링크 샘플 (최대 5개):[/yellow]")
                for link in list(links)[:5]:
                    console.print(f"  • {link}")

                if len(links) > 5:
                    console.print(f"  ... 외 {len(links) - 5}개")

                if Confirm.ask("\n이 링크들이 맞나요?", default=True):
                    return links
                else:
                    console.print("[yellow]다른 방법을 시도합니다...[/yellow]")

        # 방법 3: 간단한 자동 탐지 (fallback)
        console.print("\n[dim]자동 패턴 매칭 시도 중...[/dim]")
        links = set()

        dynamic_links = await page.evaluate('''() => {
            const links = [];
            document.querySelectorAll('a[href]').forEach(a => {
                if (a.href && !a.href.startsWith('javascript:')) {
                    links.push({url: a.href, text: a.innerText.trim()});
                }
            });
            return links;
        }''')

        from urllib.parse import urlparse
        base_domain = urlparse(base_url).netloc
        keywords = ['challenge', 'chall', 'problem', 'task']

        for link_info in dynamic_links:
            url = link_info['url']
            text = link_info.get('text', '')
            url_lower = url.lower()
            text_lower = text.lower()

            link_domain = urlparse(url).netloc
            if link_domain and link_domain != base_domain:
                continue

            nav_keywords = ['home', 'about', 'contact', 'login', 'register', 'community', 'help', 'faq']
            if any(nav in text_lower for nav in nav_keywords):
                continue

            if url_lower.endswith(('.html', '.pdf', '.png', '.jpg', '.css', '.js')):
                continue

            has_keyword = any(kw in url_lower for kw in keywords)
            has_number = any(char.isdigit() for char in url)

            if has_keyword and has_number:
                links.add(urljoin(base_url, url))
                continue

            if (has_keyword or any(kw in text_lower for kw in keywords)) and \
               len(text) > 3 and len(text) < 100 and \
               not any(nav in text_lower for nav in nav_keywords):
                links.add(urljoin(base_url, url))

        if links:
            console.print(f"[dim]엄격한 필터링: {len(links)}개 링크[/dim]")
            return sorted(list(links))

        # 방법 4: 사용자 수동 입력
        console.print("\n[yellow]⚠️  자동으로 찾지 못했습니다[/yellow]")
        if Confirm.ask("직접 CSS 선택자를 입력하시겠습니까?"):
            pattern = Prompt.ask("CSS 선택자")
            try:
                elements = await page.query_selector_all(pattern)
                console.print(f"[dim]{len(elements)}개 발견[/dim]")
                for element in elements:
                    href = await element.get_attribute('href')
                    if href:
                        links.add(urljoin(base_url, href))
            except Exception as e:
                console.print(f"[red]오류: {str(e)}[/red]")

        return sorted(list(links))

    async def _analyze_challenge_page(self, page, url: str, base_url: str) -> Optional[ChallengeInfo]:
        """챌린지 페이지 분석 (모달 지원)"""
        try:
            await page.goto(url, wait_until='networkidle', timeout=15000)

            # 모달이 나타날 때까지 대기 (SPA 대응)
            await asyncio.sleep(1.5)

            # 제목과 본문 추출 (모달 우선)
            challenge_info = await page.evaluate('''() => {
                // 일반적인 모달 선택자들
                const modalSelectors = [
                    '.modal', '.modal-content', '.challenge-modal',
                    '[role="dialog"]', '.MuiDialog-root', '.ReactModal__Content'
                ];

                let modalContent = '';
                let modalTitle = '';
                let isModal = false;

                // 모달 찾기
                for (const selector of modalSelectors) {
                    const modal = document.querySelector(selector);
                    if (modal && modal.offsetParent !== null) {  // visible 체크
                        modalContent = modal.innerText;

                        // 제목 찾기 (여러 선택자 시도)
                        const titleSelectors = ['h1', 'h2', '.title', '.challenge-title', '.modal-title'];
                        for (const ts of titleSelectors) {
                            const titleEl = modal.querySelector(ts);
                            if (titleEl) {
                                modalTitle = titleEl.innerText.trim();
                                break;
                            }
                        }

                        isModal = true;
                        break;
                    }
                }

                return {
                    title: modalTitle || document.title,
                    body: modalContent || document.body.innerText,
                    isModal: isModal
                };
            }''')

            title = challenge_info['title']
            body_text = challenge_info['body']

            # 카테고리 감지 (키워드 기반)
            category = self._detect_category(title + ' ' + body_text)

            # 난이도 감지
            difficulty = self._detect_difficulty(body_text)

            # 설명 추출 (첫 500자)
            description = body_text[:500].strip() if body_text else None

            # 힌트 추출 (모달 고려)
            hints = await self._extract_hints(page, is_modal=challenge_info['isModal'])

            # LLM 분석 추가 (선택적)
            llm_analysis = None
            if self.llm and description:
                llm_analysis = await self._llm_analyze_challenge(
                    title, category, difficulty, description, hints
                )

                # LLM 분석을 description에 추가
                if llm_analysis:
                    description = f"{description}\n\n{'='*50}\n🤖 LLM 분석\n{'='*50}\n\n{llm_analysis}"

            return ChallengeInfo(
                title=title,
                url=url,
                category=category,
                difficulty=difficulty,
                description=description,
                hints=hints
            )

        except Exception as e:
            console.print(f"[dim red]  분석 실패 ({url[:50]}...): {str(e)[:30]}...[/dim red]")
            return None

    async def _llm_analyze_challenge(
        self,
        title: str,
        category: str,
        difficulty: str,
        description: str,
        hints: List[str]
    ) -> Optional[str]:
        """LLM을 사용하여 챌린지 분석 및 풀이 가이드 생성"""
        try:
            prompt = f"""당신은 CTF(Capture The Flag) 보안 전문가입니다. 다음 챌린지를 분석하고 풀이 가이드를 제공해주세요.

**챌린지 정보**:
- 제목: {title}
- 카테고리: {category}
- 난이도: {difficulty}
- 설명: {description}
- 힌트: {', '.join(hints) if hints else '없음'}

**요청사항**:
다음 형식으로 분석 결과를 제공해주세요:

1. **문제 분석**
   - 이 챌린지가 무엇을 요구하는지 간단히 설명

2. **풀이 전략**
   - 접근 방법 (2-3가지)
   - 각 방법의 장단점

3. **필요한 도구/기술**
   - 사용할 도구 목록
   - 필요한 기술/지식

4. **단계별 접근법**
   - 1단계: ...
   - 2단계: ...
   - 3단계: ...

5. **주의사항**
   - 주의해야 할 점
   - 자주 하는 실수

6. **예상 소요 시간**
   - 초보자: X분
   - 중급자: Y분
   - 고급자: Z분

**중요**:
- 구체적이고 실용적인 조언을 제공하세요
- 직접적인 정답은 제공하지 말고, 접근 방법과 힌트를 제공하세요
- 학습 목적에 맞게 단계별로 생각할 수 있도록 유도하세요"""

            response = await self.llm.generate(prompt)
            return response.strip()

        except Exception as e:
            console.print(f"[dim yellow]  LLM 분석 실패: {str(e)[:50]}...[/dim yellow]")
            return None

    def _detect_category(self, text: str) -> str:
        """텍스트 기반 카테고리 감지"""
        text_lower = text.lower()

        category_keywords = {
            'web': ['web', 'xss', 'sql injection', 'ssrf', 'csrf'],
            'pwn': ['pwn', 'binary', 'buffer overflow', 'rop', 'shellcode'],
            'crypto': ['crypto', 'cipher', 'rsa', 'aes', 'encryption'],
            'reversing': ['reverse', 'reversing', 'binary analysis', 'decompile'],
            'forensics': ['forensics', 'steganography', 'pcap', 'memory dump'],
            'misc': ['misc', 'miscellaneous', 'programming']
        }

        for category, keywords in category_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return category

        return 'misc'

    def _detect_difficulty(self, text: str) -> Optional[str]:
        """텍스트 기반 난이도 감지"""
        text_lower = text.lower()

        if any(word in text_lower for word in ['easy', 'beginner', 'simple']):
            return 'easy'
        elif any(word in text_lower for word in ['medium', 'intermediate']):
            return 'medium'
        elif any(word in text_lower for word in ['hard', 'difficult', 'expert']):
            return 'hard'
        elif any(word in text_lower for word in ['insane', 'extreme']):
            return 'insane'

        return None

    async def _extract_hints(self, page, is_modal: bool = False) -> List[str]:
        """힌트 추출 (모달 지원)"""
        hints = []

        # HTML 주석에서 힌트 찾기 (모달 내부 우선)
        comments = await page.evaluate('''(isModal) => {
            let root = document.documentElement;

            // 모달인 경우 모달 내부에서만 찾기
            if (isModal) {
                const modalSelectors = [
                    '.modal', '.modal-content', '.challenge-modal',
                    '[role="dialog"]', '.MuiDialog-root', '.ReactModal__Content'
                ];

                for (const selector of modalSelectors) {
                    const modal = document.querySelector(selector);
                    if (modal && modal.offsetParent !== null) {
                        root = modal;
                        break;
                    }
                }
            }

            const walker = document.createTreeWalker(
                root,
                NodeFilter.SHOW_COMMENT
            );
            const comments = [];
            while (walker.nextNode()) {
                const text = walker.currentNode.textContent.trim();
                if (text.toLowerCase().includes('hint') || text.toLowerCase().includes('힌트')) {
                    comments.push(text);
                }
            }
            return comments;
        }''', is_modal)

        hints.extend(comments)

        # "Hint" 텍스트가 있는 요소 (모달 내부 우선)
        if is_modal:
            # 모달 내부에서만 검색
            hint_elements = await page.query_selector_all('.modal *:has-text("Hint"), .modal *:has-text("힌트"), [role="dialog"] *:has-text("Hint"), [role="dialog"] *:has-text("힌트")')
        else:
            # 전체 페이지 검색
            hint_elements = await page.query_selector_all('*:has-text("Hint"), *:has-text("힌트")')

        for element in hint_elements[:5]:  # 최대 5개
            try:
                text = await element.inner_text()
                if text and len(text) < 500:
                    hints.append(text.strip())
            except:
                pass

        return hints

    async def _is_valid_challenge_url(self, page, url: str, link_text: str = "") -> bool:
        """개별 URL이 실제 챌린지 페이지인지 검증 (필요시 페이지 로딩)"""
        url_lower = url.lower()

        # 1단계: URL 패턴으로 명확히 거부할 수 있는 것들
        invalid_patterns = [
            'privacy', 'terms', 'about', 'contact', 'help',
            'login', 'register', 'signup', 'account', 'settings',
            'faq', 'rules', 'guide', 'community', 'forum',
            'blog', 'news', 'events', 'sponsors', 'team',
            'leaderboard', 'scoreboard', 'profile'
        ]

        if any(pattern in url_lower for pattern in invalid_patterns):
            console.print(f"[dim yellow]  ⏭️  건너뜀 (URL 패턴): {url[:60]}...[/dim yellow]")
            return False

        # 2단계: URL 패턴으로 명확히 허용할 수 있는 것들
        valid_patterns = [
            '/challenge/', '/chall/', '/problem/', '/task/', '/ctf/'
        ]

        # 숫자 포함 체크 (예: /challenge/123, /problem/1)
        has_number = any(char.isdigit() for char in url)
        has_valid_pattern = any(pattern in url_lower for pattern in valid_patterns)

        if has_valid_pattern and has_number:
            # URL 패턴이 명확하고 숫자도 있으면 챌린지 페이지로 간주
            return True

        # 3단계: 확실하지 않은 경우 페이지를 로딩해서 확인 (LLM 있을 때만)
        if not self.llm:
            # LLM 없으면 보수적으로 허용
            return True

        try:
            # 페이지 로딩 (타임아웃 짧게)
            await page.goto(url, wait_until='domcontentloaded', timeout=10000)

            # 모달이 나타날 때까지 잠시 대기 (SPA 대응)
            await asyncio.sleep(1.5)

            # 제목과 본문 일부 추출 (모달 포함)
            page_info = await page.evaluate('''() => {
                // 일반적인 모달 선택자들
                const modalSelectors = [
                    '.modal', '.modal-content', '.challenge-modal',
                    '[role="dialog"]', '.MuiDialog-root', '.ReactModal__Content'
                ];

                let modalContent = '';
                let modalTitle = '';

                // 모달 찾기
                for (const selector of modalSelectors) {
                    const modal = document.querySelector(selector);
                    if (modal && modal.offsetParent !== null) {  // visible 체크
                        modalContent = modal.innerText;
                        const h1 = modal.querySelector('h1, h2, .title, .challenge-title');
                        if (h1) modalTitle = h1.innerText;
                        break;
                    }
                }

                return {
                    title: document.title,
                    body: modalContent || document.body.innerText.slice(0, 500),
                    h1: modalTitle || Array.from(document.querySelectorAll('h1')).map(h => h.innerText).join(' '),
                    hasModal: modalContent.length > 0
                };
            }''')

            # LLM으로 판단
            prompt = f"""다음 페이지가 CTF "챌린지 문제 페이지"인지 판단해주세요.

URL: {url}
제목: {page_info['title']}
H1: {page_info['h1']}
모달 여부: {page_info['hasModal']}
본문 일부:
{page_info['body'][:300]}

**중요**:
- "챌린지 문제 페이지"는 하나의 CTF 문제를 설명하고 풀 수 있는 페이지입니다
- 챌린지 제목, 설명, 카테고리, 난이도, 힌트 등이 있으면 챌린지 페이지입니다
- Privacy Statement, Terms of Service, About, Rules 같은 것은 챌린지 페이지가 아닙니다
- 모달로 챌린지가 표시되는 경우도 챌린지 페이지로 판단하세요

JSON 형식으로만 답변하세요:
{{
    "is_challenge": true or false,
    "reason": "판단 근거를 한 줄로"
}}"""

            response = await self.llm.generate(prompt)

            # JSON 파싱
            try:
                if '```json' in response:
                    json_str = response.split('```json')[1].split('```')[0].strip()
                elif '```' in response:
                    json_str = response.split('```')[1].split('```')[0].strip()
                else:
                    json_str = response.strip()

                result = json.loads(json_str)

                if not result.get('is_challenge', False):
                    console.print(f"[dim yellow]  ⏭️  건너뜀 (페이지 검증): {page_info['title'][:40]}... - {result.get('reason', '')}[/dim yellow]")

                return result.get('is_challenge', False)

            except json.JSONDecodeError:
                # 파싱 실패시 보수적으로 허용
                return True

        except Exception as e:
            # 로딩 실패시 보수적으로 허용 (나중에 분석 단계에서 걸러질 것)
            console.print(f"[dim red]  페이지 로딩 실패 ({url[:40]}...): {str(e)[:30]}...[/dim red]")
            return True

    async def _analyze_page_type(self, page) -> Dict:
        """LLM으로 페이지 타입 분석"""
        if not self.llm:
            return {
                'is_challenge_page': True,
                'confidence': 0.5,
                'reason': 'LLM 비활성화 - 페이지 타입을 판단할 수 없습니다',
                'suggestion': '수동으로 확인해주세요'
            }

        try:
            # 페이지 내용 추출
            content = await page.evaluate('''() => ({
                title: document.title,
                body: document.body.innerText.slice(0, 2000),
                links: Array.from(document.querySelectorAll('a[href]'))
                    .map(a => ({text: a.innerText.trim(), href: a.href}))
                    .filter(link => link.text && link.text.length > 0 && link.text.length < 100)
                    .slice(0, 30)
            })''')

            # LLM에게 물어보기
            prompt = f"""다음 페이지가 CTF 대회의 "챌린지 목록 페이지"인지 판단해주세요.

페이지 제목: {content['title']}

페이지 내용 (일부):
{content['body'][:800]}

링크 샘플:
{json.dumps(content['links'][:15], indent=2, ensure_ascii=False)}

**중요**:
- "챌린지 목록 페이지"는 여러 CTF 문제들의 리스트가 있는 페이지입니다
- 대회 홈페이지, 소개 페이지, About 페이지 등은 챌린지 목록 페이지가 아닙니다
- 링크에 "challenge", "chall", "problem", "task" 등이 많으면 챌린지 목록 페이지일 가능성이 높습니다

JSON 형식으로만 답변하세요:
{{
    "is_challenge_page": true or false,
    "confidence": 0.0에서 1.0 사이의 숫자,
    "reason": "판단 근거를 한 문장으로",
    "suggestion": "다음 단계 제안 (챌린지 페이지가 아닐 경우)"
}}"""

            response = await self.llm.generate(prompt)

            # JSON 파싱 시도
            try:
                # JSON 블록 추출 (```json ... ``` 형태)
                if '```json' in response:
                    json_str = response.split('```json')[1].split('```')[0].strip()
                elif '```' in response:
                    json_str = response.split('```')[1].split('```')[0].strip()
                else:
                    json_str = response.strip()

                result = json.loads(json_str)
                return result

            except json.JSONDecodeError:
                console.print(f"[yellow]⚠️  LLM 응답 파싱 실패: {response[:100]}...[/yellow]")
                return {
                    'is_challenge_page': True,
                    'confidence': 0.5,
                    'reason': 'LLM 응답을 파싱할 수 없어 기본값으로 진행합니다',
                    'suggestion': '수동으로 확인해주세요'
                }

        except Exception as e:
            console.print(f"[yellow]⚠️  페이지 분석 오류: {str(e)}[/yellow]")
            return {
                'is_challenge_page': True,
                'confidence': 0.5,
                'reason': f'분석 오류 발생: {str(e)[:50]}',
                'suggestion': '수동으로 확인해주세요'
            }

    async def _find_challenge_page(self, page) -> Optional[str]:
        """LLM과 협동으로 챌린지 페이지 찾기 (대화형)"""
        if not self.llm:
            return None

        try:
            # 1. 페이지의 모든 링크 수집
            links = await page.evaluate('''() =>
                Array.from(document.querySelectorAll('a[href]'))
                    .map(a => ({
                        text: a.innerText.trim(),
                        href: a.href,
                        title: a.title || '',
                        classes: a.className || ''
                    }))
                    .filter(link => link.text && link.text.length > 0 && link.text.length < 100)
                    .slice(0, 50)
            ''')

            if not links:
                console.print("[yellow]⚠️  링크를 찾을 수 없습니다[/yellow]")
                return None

            # 2. LLM에게 후보 3-5개 요청
            prompt = f"""현재 페이지에서 CTF **챌린지 목록 페이지**로 가는 링크를 찾아야 합니다.

다음 링크 중에서 **상위 3-5개 후보**를 찾아 confidence 순으로 제시하세요.

링크 목록:
{json.dumps(links, indent=2, ensure_ascii=False)}

**찾아야 하는 페이지**:
- "Practice", "Challenges", "Problems", "Tasks", "CTF", "Compete" 등의 메뉴
- 개별 챌린지가 나열된 페이지로 가는 링크
- "/practice", "/challenges", "/problems" 등의 URL 경로

**제외해야 할 링크**:
- 외부 링크 (Facebook, Twitter, Discord 등)
- Account, Settings, Logout 등 계정 관련
- About, Privacy, Terms 등 정보 페이지

JSON 형식으로만 답변하세요:
{{
    "candidates": [
        {{
            "url": "링크 URL",
            "text": "링크 텍스트",
            "confidence": 0.0-1.0,
            "reason": "왜 이 링크가 챌린지 페이지로 가는지 간단히 설명 (1문장)"
        }}
    ]
}}

**중요**: 최소 1개, 최대 5개의 후보를 제시하세요. confidence가 높은 순으로 정렬하세요."""

            response = await self.llm.generate(prompt)

            # 3. JSON 파싱
            try:
                if '```json' in response:
                    json_str = response.split('```json')[1].split('```')[0].strip()
                elif '```' in response:
                    json_str = response.split('```')[1].split('```')[0].strip()
                else:
                    json_str = response.strip()

                result = json.loads(json_str)
                candidates = result.get('candidates', [])

                if not candidates:
                    console.print("[yellow]⚠️  LLM이 후보를 찾지 못했습니다[/yellow]")
                    return None

                # 4. 사용자에게 후보 제시
                console.print("\n[bold cyan]🔍 LLM이 찾은 챌린지 페이지 후보:[/bold cyan]\n")

                for i, candidate in enumerate(candidates, 1):
                    text = candidate.get('text', 'N/A')
                    url = candidate.get('url', 'N/A')
                    confidence = candidate.get('confidence', 0)
                    reason = candidate.get('reason', 'N/A')

                    console.print(f"[green]{i}.[/green] [bold]{text}[/bold]")
                    console.print(f"   URL: [dim]{url}[/dim]")
                    console.print(f"   이유: {reason}")
                    console.print(f"   신뢰도: [cyan]{confidence:.0%}[/cyan]\n")

                # 추가 옵션
                next_idx = len(candidates) + 1
                console.print(f"[yellow]{next_idx}.[/yellow] 직접 URL 입력")
                console.print(f"[yellow]{next_idx + 1}.[/yellow] 없음 (현재 페이지에서 계속)\n")

                # 5. 사용자 선택
                choices = [str(i) for i in range(1, next_idx + 2)]
                choice_str = Prompt.ask(
                    "어느 페이지로 이동하시겠습니까?",
                    choices=choices,
                    default="1"
                )
                choice = int(choice_str)

                # 6. 선택에 따라 URL 반환
                if 1 <= choice <= len(candidates):
                    selected = candidates[choice - 1]
                    console.print(f"[green]✓ '{selected['text']}' 선택됨[/green]")
                    return selected['url']
                elif choice == next_idx:
                    # 직접 입력
                    custom_url = Prompt.ask("챌린지 페이지 URL을 입력하세요 (예: https://play.picoctf.org/practice)")
                    if custom_url and custom_url.startswith('http'):
                        return custom_url
                    else:
                        console.print("[yellow]⚠️  올바른 URL을 입력하지 않았습니다[/yellow]")
                        return None
                else:
                    # 없음
                    console.print("[yellow]현재 페이지에서 계속합니다[/yellow]")
                    return None

            except json.JSONDecodeError as je:
                console.print(f"[yellow]⚠️  LLM 응답 파싱 실패: {str(je)}[/yellow]")
                console.print(f"[dim]응답: {response[:200]}...[/dim]")
                return None

        except Exception as e:
            console.print(f"[yellow]⚠️  링크 찾기 오류: {str(e)}[/yellow]")
            return None

    async def _save_challenge_to_db(self, challenge: ChallengeInfo, competition_name: str) -> bool:
        """챌린지를 DB에 저장"""
        try:
            challenge_data = {
                'title': challenge.title,
                'category': challenge.category or 'misc',
                'difficulty': challenge.difficulty,
                'description': challenge.description,
                'url': challenge.url,
                'hints': challenge.hints,
                'status': 'pending',
                'source': f'{competition_name} (auto-crawled)'
            }

            challenge_id = self.db.insert_ctf_challenge(challenge_data)
            return challenge_id is not None

        except Exception as e:
            console.print(f"[red]DB 저장 실패: {str(e)}[/red]")
            return False


async def main():
    """테스트용 메인 함수"""
    from core.database import ArsenalDB

    db = ArsenalDB()
    crawler = CompetitionCrawler(db)

    # 예시: PicoCTF
    # url = "https://play.picoctf.org/practice"
    url = Prompt.ask("CTF 대회 메인 페이지 URL을 입력하세요")

    stats = await crawler.crawl_competition(url)

    console.print("\n[bold cyan]📊 최종 통계[/bold cyan]")
    console.print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
