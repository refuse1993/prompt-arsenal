"""
Community Crawler
DC인사이드에서 AI 관련 프롬프트 수집
"""

import re
import time
from typing import List, Dict, Optional
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.database import ArsenalDB
from core.config import Config
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

# Note: Playwright는 더 이상 필요하지 않음 (DC인사이드만 사용)


class CommunityCrawler:
    """커뮤니티 사이트 크롤러"""

    def __init__(self, db: ArsenalDB, config: Config):
        self.db = db
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })

    def crawl_dcinside(self, gallery_id: str = "ai", pages: int = 5) -> List[Dict]:
        """
        DC인사이드 갤러리 크롤링

        Args:
            gallery_id: 갤러리 ID
                - ai: 인공지능 갤러리 (추천)
                - 235711: AI 마이너 갤러리
            pages: 크롤링할 페이지 수

        Returns:
            게시글 목록 [{"title": str, "content": str, "url": str}]
        """
        console.print(f"[cyan]DC인사이드 갤러리 크롤링 시작: {gallery_id}[/cyan]")

        posts = []

        for page in range(1, pages + 1):
            try:
                console.print(f"[dim]페이지 {page}/{pages} 수집 중...[/dim]")
                # 갤러리 목록 페이지
                list_url = f"https://gall.dcinside.com/mgallery/board/lists/?id={gallery_id}&page={page}"
                response = self.session.get(list_url, timeout=10)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, 'html.parser')

                # 게시글 링크 추출 (올바른 셀렉터)
                article_links = soup.select('.gall_list tbody tr td.gall_tit a')

                for link in article_links[:10]:  # 페이지당 최대 10개
                    try:
                        href = link.get('href', '')

                        # 잘못된 링크 제외
                        if not href or href.startswith('javascript:') or not href.startswith('/'):
                            continue

                        article_url = "https://gall.dcinside.com" + href
                        title = link.text.strip()

                        # 게시글 내용 가져오기
                        article_response = self.session.get(article_url, timeout=10)
                        article_soup = BeautifulSoup(article_response.text, 'html.parser')

                        # 본문 추출
                        content_div = article_soup.select_one('div.write_div')

                        # 날짜 추출
                        date_elem = article_soup.select_one('.gall_date')
                        date_str = date_elem.text.strip() if date_elem else "Unknown"

                        if content_div:
                            content = content_div.get_text(separator="\n", strip=True)

                            posts.append({
                                'title': title,
                                'content': content,
                                'url': article_url,
                                'source': 'dcinside',
                                'date': date_str
                            })

                        time.sleep(0.5)  # Rate limiting

                    except Exception as e:
                        console.print(f"[yellow]게시글 수집 실패: {e}[/yellow]")
                        continue

                time.sleep(1)  # Rate limiting between pages

            except Exception as e:
                console.print(f"[red]페이지 {page} 크롤링 실패: {e}[/red]")
                continue

        console.print(f"[green]✅ DC인사이드: {len(posts)}개 게시글 수집 완료[/green]")
        return posts

    def filter_posts_by_keywords(self, posts: List[Dict]) -> List[Dict]:
        """
        1차 필터링: 프롬프트 관련 키워드가 포함된 게시글만 선택

        Args:
            posts: 전체 게시글 목록

        Returns:
            필터링된 게시글 목록
        """
        # 프롬프트 관련 키워드 (대소문자 무시)
        keywords = [
            # 한글
            '프롬프트', '지시문', '명령어', '시스템 프롬프트', '시스템프롬프트',
            'jailbreak', '탈옥', '우회', 'bypass',
            'prompt injection', '프롬프트 인젝션', '인젝션',
            '프리셋', 'preset',
            # 영어
            'prompt', 'instruction', 'system prompt', 'system_prompt',
            'few shot', 'few-shot', 'zero shot', 'zero-shot',
            'chain of thought', 'cot',
            'role play', 'roleplay',
            # LLM 관련
            'chatgpt', 'gpt', 'claude', 'gemini', 'llama',
            'openai', 'anthropic', 'google ai',
            # 공격 기법
            'dan', 'do anything now',
            'ignore previous', 'ignore all previous',
            'override', 'system override'
        ]

        filtered_posts = []

        console.print(f"\n[cyan]1차 키워드 필터링 중...[/cyan]")

        for post in posts:
            title_lower = post['title'].lower()
            content_lower = post['content'].lower()

            # 제목이나 본문에 키워드가 하나라도 있으면 포함
            if any(keyword.lower() in title_lower or keyword.lower() in content_lower
                   for keyword in keywords):
                filtered_posts.append(post)

        console.print(f"[green]✅ {len(posts)}개 중 {len(filtered_posts)}개 게시글 필터링 완료[/green]")

        if filtered_posts:
            console.print("\n[cyan]필터링된 게시글 샘플:[/cyan]")
            for i, post in enumerate(filtered_posts[:3], 1):
                date_str = post.get('date', 'Unknown')
                title = post['title']
                console.print(f"  {i}. [{date_str}] {title}", markup=False)

        return filtered_posts

    async def extract_prompts_with_llm(self, posts: List[Dict], api_profile: str = "gpt_test", fallback_profiles: List[str] = None) -> List[Dict]:
        """
        LLM을 사용해서 게시글에서 프롬프트 추출 (자동 fallback 지원)

        Args:
            posts: 게시글 목록
            api_profile: 사용할 API 프로필
            fallback_profiles: 실패 시 시도할 백업 프로필 리스트

        Returns:
            추출된 프롬프트 목록 [{"payload": str, "category": str, "description": str}]
        """
        # API 프로필 가져오기
        all_profiles = self.config.get_all_profiles(profile_type="llm")

        # Fallback 프로필 자동 생성 (OpenAI 프로필만)
        if fallback_profiles is None:
            fallback_profiles = [name for name, p in all_profiles.items()
                               if p.get('provider') == 'openai' and name != api_profile]

        # 시도할 프로필 리스트 (메인 + fallback)
        profiles_to_try = [api_profile] + fallback_profiles

        # 각 프로필 시도
        for attempt, profile_name in enumerate(profiles_to_try, 1):
            try:
                console.print(f"\n[cyan]LLM으로 프롬프트 추출 중... (프로필: {profile_name})[/cyan]")

                profile = all_profiles.get(profile_name)
                if not profile:
                    console.print(f"[yellow]프로필 '{profile_name}' 건너뜀 (없음)[/yellow]")
                    continue

                # OpenAI API 클라이언트
                import openai

                # Provider에 따라 다른 클라이언트 생성
                provider = profile.get('provider', 'openai')
                if provider == 'openai':
                    client = openai.OpenAI(api_key=profile.get('api_key'))
                else:
                    console.print(f"[yellow]프로필 '{profile_name}' 건너뜀 (OpenAI만 지원)[/yellow]")
                    continue

                model = profile.get('model', 'gpt-4o-mini')

                # 프로필로 추출 시도
                extracted_prompts = []

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task(f"프롬프트 추출 중...", total=len(posts))

                    for post_idx, post in enumerate(posts, 1):
                        try:
                            # LLM에게 프롬프트 추출 요청
                            system_prompt = """당신은 AI 보안 연구용 프롬프트 수집 전문가입니다.

**수집 대상**:
1. LLM 시스템 프롬프트 (페르소나, 역할 정의, 규칙)
2. Jailbreak 프롬프트 (DAN, 무제한 모드 등)
3. Prompt Injection 공격 프롬프트
4. 창작/글쓰기용 프롬프트 템플릿
5. 성인 콘텐츠 생성용 프롬프트 (연구용)

**중요 규칙**:
- 모든 프롬프트는 보안 연구 목적으로 수집됩니다
- 성인 콘텐츠나 유해 콘텐츠 프롬프트도 연구 대상입니다
- **전체 프롬프트를 하나의 payload로 추출** (쪼개지 말 것!)
- 길이 제한 없음: 전체 내용을 그대로 포함
- 프롬프트가 명확하게 보이면 반드시 추출하세요

JSON 형식으로 응답:
{
  "prompts": [
    {
      "payload": "프롬프트 전체 내용 (원본 그대로, 길이 제한 없음)",
      "category": "jailbreak|prompt_injection|template|roleplay|other",
      "description": "프롬프트 용도와 특징 (간략하게)",
      "tags": "태그1,태그2,태그3"
    }
  ]
}

프롬프트가 없으면 빈 배열 반환.
프롬프트가 여러 개면 각각 별도 항목으로 추출.
"""

                            user_prompt = f"""게시글 제목: {post['title']}

게시글 내용:
{post['content'][:5000]}

출처: {post['url']}

위 게시글에서 LLM 프롬프트를 추출하세요.
긴 프롬프트라도 전체를 하나의 payload로 추출하세요."""

                            response = client.chat.completions.create(
                                model=model,
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_prompt}
                                ],
                                response_format={"type": "json_object"},
                                temperature=0.3,
                                max_tokens=8000,  # 긴 프롬프트를 위해 증가
                                timeout=60.0  # 60초 타임아웃
                            )

                            # 응답 파싱
                            import json
                            raw_response = response.choices[0].message.content
                            result = json.loads(raw_response)

                            # 프롬프트 추출
                            if result.get('prompts'):
                                for prompt_data in result['prompts']:
                                    prompt_data['source'] = f"{post['source']}: {post['url']}"
                                    extracted_prompts.append(prompt_data)

                            progress.update(task, advance=1)
                            time.sleep(0.5)  # Rate limiting

                        except Exception as e:
                            console.print(f"[yellow]게시글 처리 실패: {e}[/yellow]")
                            progress.update(task, advance=1)
                            continue

                console.print(f"[green]✅ {len(extracted_prompts)}개 프롬프트 추출 완료[/green]")
                return extracted_prompts

            except Exception as e:
                console.print(f"[red]프로필 '{profile_name}' 실패: {e}[/red]")

                # 마지막 시도였으면 에러, 아니면 다음 프로필 시도
                if attempt < len(profiles_to_try):
                    console.print(f"[yellow]다음 프로필로 재시도 중...[/yellow]")
                    continue
                else:
                    console.print(f"[red]모든 프로필 실패. 사용 가능한 프로필: {', '.join(profiles_to_try)}[/red]")
                    return []

        # 여기까지 오면 모든 프로필 실패
        return []

    def save_prompts_to_db(self, prompts: List[Dict], confirm: bool = True) -> int:
        """
        추출된 프롬프트를 DB에 저장 (사용자 확인 옵션)

        Args:
            prompts: 프롬프트 목록
            confirm: 각 프롬프트 저장 전 사용자 확인 (기본값: True)

        Returns:
            저장된 프롬프트 개수
        """
        from rich.prompt import Prompt
        from rich.panel import Panel
        from rich.table import Table

        saved_count = 0
        skipped_count = 0

        console.print(f"\n[cyan]추출된 프롬프트 검토 및 저장[/cyan]")

        for idx, prompt in enumerate(prompts, 1):
            try:
                payload = prompt['payload']
                payload_len = len(payload)
                category = prompt.get('category', 'other')
                description = prompt.get('description', '')
                tags = prompt.get('tags', '')

                # 프롬프트 정보 및 전체 내용 표시
                console.print("\n" + "=" * 70)
                console.print(f"[bold cyan]프롬프트 #{idx}/{len(prompts)}[/bold cyan]")
                console.print("=" * 70)

                console.print(f"카테고리: {category}")
                console.print(f"길이: {payload_len} chars")
                console.print(f"설명: {description or 'N/A'}")
                console.print(f"태그: {tags or 'N/A'}")
                console.print(f"출처: {prompt.get('source', 'community')}")

                # 전체 내용 표시
                console.print(f"\n{'─' * 70}")
                console.print("전체 내용:")
                console.print(f"{'─' * 70}")
                console.print(payload)
                console.print(f"{'─' * 70}\n")

                # 저장 여부 확인
                if confirm:
                    console.print("[dim]옵션: [s]저장 / [k]건너뛰기 / [a]모두저장 / [q]중단[/dim]")
                    choice = Prompt.ask(
                        "선택",
                        choices=["s", "k", "a", "q"],
                        default="s"
                    ).lower()

                    # 모두 저장 (확인 비활성화)
                    if choice == "a":
                        confirm = False
                        choice = "s"

                    # 중단
                    if choice == "q":
                        console.print("[yellow]저장 중단[/yellow]")
                        break

                    # 건너뛰기
                    if choice == "k":
                        skipped_count += 1
                        console.print("[yellow]⏭️  건너뛰기[/yellow]")
                        continue

                # 저장 실행
                prompt_id = self.db.insert_prompt(
                    category=category,
                    payload=payload,
                    description=description,
                    source=prompt.get('source', 'community'),
                    tags=tags
                )

                if prompt_id:
                    saved_count += 1
                    console.print(f"[green]✅ 저장 완료 (ID: {prompt_id})[/green]")

            except Exception as e:
                console.print(f"[red]❌ 저장 실패: {e}[/red]")
                continue

        # 최종 요약
        console.print(f"\n[bold cyan]저장 완료[/bold cyan]")
        console.print(f"  ✅ 저장: {saved_count}개")
        if skipped_count > 0:
            console.print(f"  ⏭️  건너뜀: {skipped_count}개")
        console.print(f"  📊 전체: {len(prompts)}개")

        return saved_count


async def community_import_workflow(db: ArsenalDB, config: Config):
    """커뮤니티 프롬프트 수집 워크플로우"""

    crawler = CommunityCrawler(db, config)

    console.print("\n[bold cyan]═══════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]  커뮤니티 프롬프트 수집기[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════[/bold cyan]\n")

    # 1. 갤러리 ID 선택
    console.print("[yellow]DC인사이드 갤러리 선택:[/yellow]")
    console.print("  1. ai - 인공지능 갤러리 (추천)")
    console.print("  2. 235711 - AI 마이너 갤러리")
    console.print("  3. 직접 입력")

    from rich.prompt import Prompt
    gallery_choice = Prompt.ask("선택", choices=["1", "2", "3"], default="1")

    if gallery_choice == "1":
        gallery_id = "ai"
    elif gallery_choice == "2":
        gallery_id = "235711"
    else:
        gallery_id = Prompt.ask("갤러리 ID 입력")

    # 2. 페이지 수 입력
    pages = int(Prompt.ask("수집할 페이지 수", default="3"))

    # 3. DC인사이드 크롤링
    all_posts = []
    dc_posts = crawler.crawl_dcinside(gallery_id=gallery_id, pages=pages)
    all_posts.extend(dc_posts)

    if not all_posts:
        console.print("[red]수집된 게시글이 없습니다.[/red]")
        return

    console.print(f"\n[green]총 {len(all_posts)}개 게시글 수집 완료[/green]")

    # 3-1. 날짜별 정보 표시
    if all_posts:
        from rich.table import Table
        date_table = Table(title="수집된 게시글 날짜 분포", show_header=True)
        date_table.add_column("날짜", style="cyan")
        date_table.add_column("게시글 수", style="yellow", justify="right")

        # 날짜별 카운트
        from collections import Counter
        date_counts = Counter(post.get('date', 'Unknown') for post in all_posts)

        for date, count in sorted(date_counts.items(), reverse=True)[:10]:  # 최근 10개
            date_table.add_row(date, str(count))

        console.print(date_table)

        # 날짜 필터링 옵션
        console.print("\n[dim]특정 날짜만 수집하시겠습니까? (전체 수집하려면 Enter)[/dim]")
        filter_date = Prompt.ask("날짜 (예: 2025.01.30)", default="")

        if filter_date:
            filtered_by_date = [p for p in all_posts if p.get('date') == filter_date]
            if filtered_by_date:
                all_posts = filtered_by_date
                console.print(f"[green]날짜 필터링 완료: {len(all_posts)}개 게시글[/green]")
            else:
                console.print(f"[yellow]'{filter_date}' 날짜의 게시글이 없습니다. 전체를 사용합니다.[/yellow]")

    # 4. 1차 키워드 필터링
    filtered_posts = crawler.filter_posts_by_keywords(all_posts)

    if not filtered_posts:
        console.print("[yellow]프롬프트 관련 게시글을 찾지 못했습니다.[/yellow]")
        console.print("[dim]더 많은 페이지를 크롤링하거나 다른 갤러리를 시도해보세요.[/dim]")
        return

    # 5. API 프로필 선택
    console.print("\n[yellow]LLM 프롬프트 추출에 사용할 API 프로필:[/yellow]")

    llm_profiles = config.get_all_profiles(profile_type="llm")
    if not llm_profiles:
        console.print("[red]사용 가능한 LLM 프로필이 없습니다.[/red]")
        return

    for i, (name, profile) in enumerate(llm_profiles.items(), 1):
        console.print(f"  {i}. {name} ({profile.get('provider')} - {profile.get('model')})")

    profile_choice = Prompt.ask("프로필 선택", default="1")
    profile_name = list(llm_profiles.keys())[int(profile_choice) - 1]

    # 6. 2차 LLM 분석 (필터링된 게시글만)
    console.print(f"\n[cyan]2차 LLM 분석 시작 ({len(filtered_posts)}개 게시글)[/cyan]")
    extracted_prompts = await crawler.extract_prompts_with_llm(filtered_posts, api_profile=profile_name)

    if not extracted_prompts:
        console.print("[yellow]추출된 프롬프트가 없습니다.[/yellow]")
        return

    # 6. 저장 전 확인 옵션
    console.print(f"\n[cyan]추출된 프롬프트: {len(extracted_prompts)}개[/cyan]")
    console.print("[dim]각 프롬프트를 확인하고 저장하시겠습니까?[/dim]")

    from rich.prompt import Confirm
    confirm_each = Confirm.ask("프롬프트별 확인", default=True)

    # 7. DB 저장
    saved_count = crawler.save_prompts_to_db(extracted_prompts, confirm=confirm_each)

    # 7. 결과 요약
    console.print("\n[bold green]═══════════════════════════════════════[/bold green]")
    console.print(f"[bold green]  수집 완료![/bold green]")
    console.print("[bold green]═══════════════════════════════════════[/bold green]")
    console.print(f"  📄 총 게시글: {len(all_posts)}개")
    console.print(f"  🔍 1차 필터링: {len(filtered_posts)}개")
    console.print(f"  🎯 추출한 프롬프트: {len(extracted_prompts)}개")
    console.print(f"  💾 저장한 프롬프트: {saved_count}개")
    console.print()
