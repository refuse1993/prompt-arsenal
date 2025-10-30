"""
Community Crawler 테스트 스크립트
"""

import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from core.database import ArsenalDB
from core.config import Config
from text.community_crawler import CommunityCrawler
from rich.console import Console

console = Console()


def test_crawler_sync():
    """크롤러 테스트 (동기 버전)"""

    console.print("[bold cyan]DC인사이드 크롤러 테스트 시작[/bold cyan]\n")

    # DB와 Config 초기화
    db = ArsenalDB("arsenal.db")
    config = Config("config.json")

    crawler = CommunityCrawler(db, config)

    # 테스트: DC인사이드 10페이지 크롤링
    console.print("\n[yellow]테스트: DC인사이드 크롤링 (10페이지)[/yellow]")
    try:
        dc_posts = crawler.crawl_dcinside(gallery_id="ai", pages=10)
        console.print(f"[green]✅ {len(dc_posts)}개 게시글 수집[/green]")

        if dc_posts:
            console.print("\n[cyan]첫 번째 게시글 샘플:[/cyan]")
            first = dc_posts[0]
            console.print(f"  제목: {first['title']}")
            console.print(f"  내용 미리보기: {first['content'][:200]}...")
            console.print(f"  URL: {first['url']}")
    except Exception as e:
        console.print(f"[red]❌ DC인사이드 크롤링 실패: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        dc_posts = []

    return dc_posts


async def test_llm_extraction(dc_posts):
    """필터링 + LLM 추출 테스트 (비동기)"""

    db = ArsenalDB("arsenal.db")
    config = Config("config.json")
    crawler = CommunityCrawler(db, config)

    if dc_posts:
        # 1차 필터링
        filtered_posts = crawler.filter_posts_by_keywords(dc_posts)

        if not filtered_posts:
            console.print("[yellow]프롬프트 관련 게시글을 찾지 못했습니다.[/yellow]")
            return

        console.print(f"\n[yellow]2차 LLM 분석 테스트 ({len(filtered_posts)}개 게시글)[/yellow]")

        # 사용 가능한 LLM 프로필 확인
        llm_profiles = config.get_all_profiles(profile_type="llm")
        if not llm_profiles:
            console.print("[red]❌ 사용 가능한 LLM 프로필이 없습니다.[/red]")
            console.print("[yellow]먼저 'interactive_cli.py'에서 's' 메뉴로 API 프로필을 추가하세요.[/yellow]")
            return

        # 첫 번째 OpenAI 프로필 사용
        openai_profiles = [(name, profile) for name, profile in llm_profiles.items()
                          if profile.get('provider') == 'openai']

        if not openai_profiles:
            console.print("[red]❌ OpenAI 프로필이 없습니다.[/red]")
            return

        profile_name = openai_profiles[0][0]
        console.print(f"[cyan]사용할 프로필: {profile_name}[/cyan]")

        try:
            # 필터링된 게시글 전체 분석
            extracted = await crawler.extract_prompts_with_llm(filtered_posts, api_profile=profile_name)

            console.print(f"[green]✅ {len(extracted)}개 프롬프트 추출[/green]")

            if extracted:
                console.print("\n[cyan]추출된 프롬프트 샘플:[/cyan]")
                for i, prompt in enumerate(extracted[:3], 1):
                    console.print(f"\n  [{i}] 카테고리: {prompt.get('category')}")
                    console.print(f"      내용: {prompt['payload'][:100]}...")
                    console.print(f"      설명: {prompt.get('description', 'N/A')}")

        except Exception as e:
            console.print(f"[red]❌ LLM 프롬프트 추출 실패: {e}[/red]")
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")


if __name__ == "__main__":
    # 동기 크롤링 테스트
    dc_posts = test_crawler_sync()

    # LLM 추출 테스트 (비동기)
    if dc_posts:
        asyncio.run(test_llm_extraction(dc_posts))

    console.print("\n[bold green]테스트 완료![/bold green]")
