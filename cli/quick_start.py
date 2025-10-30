#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Start Tutorial - 5분 완성 가이드
실제 API를 사용한 진짜 테스트
"""

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import box
import asyncio
import os
from pathlib import Path
from typing import Optional, Dict
import time

console = Console()


class QuickStartTutorial:
    """5분 완성 신규 사용자 튜토리얼"""

    def __init__(self, db, config):
        self.db = db
        self.config = config
        self.results = {
            'api_setup': False,
            'jailbreak_test': None,
            'image_attack': None,
            'total_time': 0
        }

    async def run(self):
        """튜토리얼 메인 실행"""
        start_time = time.time()

        console.clear()
        self._show_welcome()

        # Step 1: API 설정 확인
        if not await self._step1_check_api():
            console.print("\n[yellow]API 키 설정이 필요합니다. 메인 메뉴에서 's' → API 프로필 관리를 선택하세요.[/yellow]")
            return

        # Step 2: Jailbreak 테스트 (실제)
        await self._step2_jailbreak_test()

        # Step 3: 이미지 FGSM 공격 (실제)
        await self._step3_image_attack()

        # Step 4: 결과 요약 및 다음 단계
        self.results['total_time'] = time.time() - start_time
        self._step4_summary()

    def _show_welcome(self):
        """환영 메시지"""
        welcome_text = """
[bold cyan]🎯 Prompt Arsenal Quick Start Tutorial[/bold cyan]

[dim]5분 안에 핵심 기능을 실제로 체험해보세요![/dim]

[bold]진행 순서:[/bold]
  1️⃣  API 설정 확인 (OpenAI 또는 Anthropic)
  2️⃣  실제 Jailbreak 프롬프트 테스트
  3️⃣  실제 이미지 적대적 공격 생성
  4️⃣  결과 확인 및 다음 단계 추천

[bold yellow]주의:[/bold yellow] 이 튜토리얼은 [bold]실제 API를 호출[/bold]합니다!
- OpenAI: 약 $0.01 비용 발생
- Anthropic: 무료 티어 크레딧 사용

[dim]계속하려면 Enter를 누르세요...[/dim]
"""
        console.print(Panel(welcome_text, box=box.DOUBLE, border_style="cyan"))
        input()

    async def _step1_check_api(self) -> bool:
        """Step 1: API 설정 확인"""
        console.print("\n" + "="*60)
        console.print("[bold cyan]Step 1/3: API 설정 확인[/bold cyan]")
        console.print("="*60 + "\n")

        # 모든 프로필 가져오기
        all_profiles = self.config.get_all_profiles(profile_type="llm")

        if not all_profiles:
            console.print("[red]❌ API 프로필이 없습니다![/red]\n")
            console.print("다음 중 하나를 설정하세요:")
            console.print("  • OpenAI API (https://platform.openai.com/api-keys)")
            console.print("  • Anthropic API (https://console.anthropic.com/)")
            console.print("\n[yellow]메인 메뉴에서 's' → API 프로필 관리를 선택하세요.[/yellow]")
            return False

        # Provider별로 분류
        openai_profiles = []
        anthropic_profiles = []

        for name, profile in all_profiles.items():
            profile_with_name = profile.copy()
            profile_with_name['name'] = name

            if profile.get('provider') == 'openai':
                openai_profiles.append(profile_with_name)
            elif profile.get('provider') == 'anthropic':
                anthropic_profiles.append(profile_with_name)

        # 사용 가능한 프로필 표시
        table = Table(title="✅ 사용 가능한 API 프로필", box=box.SIMPLE)
        table.add_column("프로필명", style="cyan")
        table.add_column("Provider", style="green")
        table.add_column("모델", style="yellow")

        for profile in openai_profiles + anthropic_profiles:
            table.add_row(
                profile['name'],
                profile.get('provider', 'unknown'),
                profile.get('model', 'default')
            )

        console.print(table)
        self.results['api_setup'] = True

        console.print("\n[green]✅ API 설정 완료![/green]")
        await asyncio.sleep(1)
        return True

    async def _step2_jailbreak_test(self):
        """Step 2: 실제 Jailbreak 테스트"""
        console.print("\n" + "="*60)
        console.print("[bold cyan]Step 2/3: 실제 Jailbreak 프롬프트 테스트[/bold cyan]")
        console.print("="*60 + "\n")

        console.print("[dim]데이터베이스에서 효과적인 jailbreak 프롬프트 선택 중...[/dim]\n")

        # 실제 DB에서 jailbreak 프롬프트 가져오기 (빈 keyword로 카테고리만 필터링)
        prompts = self.db.search_prompts(keyword="", category="jailbreak", limit=5)

        if not prompts:
            console.print("[yellow]⚠️  jailbreak 프롬프트가 없습니다. GitHub 데이터셋을 먼저 가져오세요.[/yellow]")
            self.results['jailbreak_test'] = {'status': 'skipped', 'reason': 'no_prompts'}
            return

        # 가장 간단한 프롬프트 선택 (DAN 스타일)
        selected_prompt = None
        for prompt in prompts:
            if len(prompt['payload']) < 500:  # 짧은 프롬프트 선택
                selected_prompt = prompt
                break

        if not selected_prompt:
            selected_prompt = prompts[0]

        # 프롬프트 표시
        console.print(f"[bold]선택된 프롬프트 (ID: {selected_prompt['id']}):[/bold]")
        console.print(Panel(
            selected_prompt['payload'][:300] + "..." if len(selected_prompt['payload']) > 300 else selected_prompt['payload'],
            border_style="yellow",
            title="Jailbreak Prompt"
        ))

        # 실제 LLM 테스트
        console.print("\n[bold cyan]🤖 실제 LLM에 테스트 중...[/bold cyan]\n")

        try:
            from text.llm_tester import LLMTester
            from core import Judge

            # OpenAI 프로필 선택
            all_profiles = self.config.get_all_profiles(profile_type="llm")
            openai_profiles = [
                (name, profile) for name, profile in all_profiles.items()
                if profile.get('provider') == 'openai'
            ]

            if not openai_profiles:
                console.print("[yellow]⚠️  OpenAI 프로필이 없어 테스트를 건너뜁니다.[/yellow]")
                self.results['jailbreak_test'] = {'status': 'skipped', 'reason': 'no_openai'}
                return

            profile_name, profile = openai_profiles[0]
            api_key = profile.get('api_key')

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("API 호출 중...", total=None)

                tester = LLMTester(
                    db=self.db,
                    provider="openai",
                    model=profile.get('model', 'gpt-4o-mini'),
                    api_key=api_key
                )

                judge = Judge()

                result = await tester.test_prompt_with_judge(
                    prompt_id=selected_prompt['id'],
                    prompt=selected_prompt['payload'],
                    judge=judge
                )

                progress.update(task, completed=True)

            # 결과 표시
            console.print("\n[bold green]✅ 테스트 완료![/bold green]\n")

            result_table = Table(box=box.ROUNDED)
            result_table.add_column("항목", style="cyan")
            result_table.add_column("결과", style="yellow")

            result_table.add_row("성공 여부", "✅ 성공" if result['success'] else "❌ 실패")
            result_table.add_row("심각도", result.get('severity', 'N/A'))
            result_table.add_row("신뢰도", f"{result.get('confidence', 0)*100:.1f}%")
            result_table.add_row("응답 시간", f"{result.get('response_time', 0):.2f}초")

            console.print(result_table)

            if result.get('response'):
                console.print("\n[bold]LLM 응답:[/bold]")
                console.print(Panel(
                    result['response'][:500] + "..." if len(result['response']) > 500 else result['response'],
                    border_style="green" if result['success'] else "red"
                ))

            self.results['jailbreak_test'] = result

        except Exception as e:
            console.print(f"\n[red]❌ 테스트 실패: {str(e)}[/red]")
            self.results['jailbreak_test'] = {'status': 'error', 'error': str(e)}

        await asyncio.sleep(2)

    async def _step3_image_attack(self):
        """Step 3: 실제 이미지 FGSM 공격"""
        console.print("\n" + "="*60)
        console.print("[bold cyan]Step 3/3: 실제 이미지 적대적 공격 생성[/bold cyan]")
        console.print("="*60 + "\n")

        # 테스트 이미지 확인
        media_dir = Path("media")
        test_images = list(media_dir.glob("test_image*.png"))

        if not test_images:
            console.print("[yellow]⚠️  테스트 이미지가 없습니다. 간단한 테스트 이미지를 생성합니다...[/yellow]\n")

            try:
                from PIL import Image
                import numpy as np

                # 간단한 테스트 이미지 생성 (체커보드 패턴)
                img_array = np.zeros((224, 224, 3), dtype=np.uint8)
                img_array[::32, :] = 255  # 가로 줄무늬
                img_array[:, ::32] = 255  # 세로 줄무늬

                test_img = Image.fromarray(img_array)
                test_img_path = media_dir / "quickstart_test_image.png"
                test_img.save(test_img_path)

                console.print(f"[green]✅ 테스트 이미지 생성: {test_img_path}[/green]\n")

            except Exception as e:
                console.print(f"[red]❌ 이미지 생성 실패: {str(e)}[/red]")
                self.results['image_attack'] = {'status': 'skipped', 'reason': 'no_image'}
                return
        else:
            test_img_path = test_images[0]
            console.print(f"[green]✅ 기존 테스트 이미지 사용: {test_img_path}[/green]\n")

        # 실제 FGSM 공격 실행
        console.print("[bold cyan]🎨 FGSM 공격 생성 중...[/bold cyan]\n")

        try:
            from multimodal.image_adversarial import ImageAdversarial

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("FGSM 적대적 섭동 계산 중...", total=None)

                img_adv = ImageAdversarial(db=self.db)

                result = img_adv.fgsm_attack(
                    image_path=str(test_img_path),
                    epsilon=0.03,
                    target_class=None
                )

                progress.update(task, completed=True)

            console.print("\n[bold green]✅ 공격 생성 완료![/bold green]\n")

            # 결과 표시
            result_table = Table(box=box.ROUNDED)
            result_table.add_column("항목", style="cyan")
            result_table.add_column("값", style="yellow")

            result_table.add_row("공격 유형", "FGSM (Fast Gradient Sign Method)")
            result_table.add_row("Epsilon", "0.03")
            result_table.add_row("원본 이미지", str(test_img_path))
            result_table.add_row("공격 이미지", result['generated_file'])
            result_table.add_row("섭동 크기", f"{result.get('perturbation_norm', 0):.4f}")

            console.print(result_table)

            console.print("\n[dim]💡 팁: 공격 이미지는 사람 눈에는 거의 동일하지만, Vision 모델은 다르게 인식합니다.[/dim]")

            self.results['image_attack'] = result

        except Exception as e:
            console.print(f"\n[red]❌ 공격 생성 실패: {str(e)}[/red]")
            console.print("[yellow]힌트: PIL, torch, torchvision이 설치되어 있는지 확인하세요.[/yellow]")
            self.results['image_attack'] = {'status': 'error', 'error': str(e)}

        await asyncio.sleep(2)

    def _step4_summary(self):
        """Step 4: 결과 요약 및 다음 단계"""
        console.clear()

        console.print("\n" + "="*60)
        console.print("[bold cyan]🎉 Quick Start 완료![/bold cyan]")
        console.print("="*60 + "\n")

        # 결과 요약 테이블
        summary_table = Table(title="📊 실행 결과 요약", box=box.DOUBLE)
        summary_table.add_column("단계", style="cyan")
        summary_table.add_column("상태", style="green")
        summary_table.add_column("결과", style="yellow")

        # API 설정
        summary_table.add_row(
            "1️⃣  API 설정",
            "✅ 완료" if self.results['api_setup'] else "❌ 실패",
            "프로필 확인됨"
        )

        # Jailbreak 테스트
        jb_result = self.results['jailbreak_test']
        if jb_result:
            if jb_result.get('status') == 'skipped':
                jb_status = "⏭️  건너뜀"
                jb_detail = jb_result.get('reason', '')
            elif jb_result.get('status') == 'error':
                jb_status = "❌ 오류"
                jb_detail = "에러 발생"
            else:
                jb_status = "✅ 완료"
                jb_detail = f"성공" if jb_result.get('success') else "실패 (정상)"
        else:
            jb_status = "❌ 미실행"
            jb_detail = ""

        summary_table.add_row("2️⃣  Jailbreak 테스트", jb_status, jb_detail)

        # 이미지 공격
        img_result = self.results['image_attack']
        if img_result:
            if img_result.get('status') == 'skipped':
                img_status = "⏭️  건너뜀"
                img_detail = img_result.get('reason', '')
            elif img_result.get('status') == 'error':
                img_status = "❌ 오류"
                img_detail = "에러 발생"
            else:
                img_status = "✅ 완료"
                img_detail = "FGSM 공격 생성"
        else:
            img_status = "❌ 미실행"
            img_detail = ""

        summary_table.add_row("3️⃣  이미지 공격", img_status, img_detail)

        # 총 소요 시간
        summary_table.add_row(
            "⏱️  총 소요 시간",
            f"{self.results['total_time']:.1f}초",
            ""
        )

        console.print(summary_table)

        # 다음 단계 추천
        console.print("\n" + "="*60)
        console.print("[bold cyan]🚀 다음 단계 추천[/bold cyan]")
        console.print("="*60 + "\n")

        recommendations = []

        # Jailbreak 결과 기반 추천
        if jb_result and jb_result.get('success'):
            recommendations.append({
                'priority': '🔥',
                'title': 'Multi-Turn Crescendo Attack',
                'description': 'Jailbreak 성공! 더 강력한 Multi-Turn 공격 시도',
                'menu': '메인 메뉴 → 0 (Multi-Turn)'
            })
        elif jb_result and not jb_result.get('success'):
            recommendations.append({
                'priority': '💡',
                'title': '다른 Jailbreak 프롬프트 시도',
                'description': '22,000+ 프롬프트 중 다른 전략 테스트',
                'menu': '메인 메뉴 → 2 (텍스트 프롬프트 테스트)'
            })

        # 이미지 공격 기반 추천
        if img_result and img_result.get('status') != 'error':
            recommendations.append({
                'priority': '🎨',
                'title': 'Foolbox 고급 공격',
                'description': 'PGD, C&W 등 더 강력한 이미지 공격',
                'menu': '메인 메뉴 → a (Foolbox 공격)'
            })

        # 기본 추천
        recommendations.extend([
            {
                'priority': '🏆',
                'title': 'SpyLab Backdoor Discovery',
                'description': 'IEEE SaTML 2024 우승팀 전략',
                'menu': '메인 메뉴 → S (SpyLab Backdoor)'
            },
            {
                'priority': '🌐',
                'title': 'CTF 챌린지 자동 분석',
                'description': 'CTFtime에서 챌린지 크롤링 및 자동 분석',
                'menu': '메인 메뉴 → c (CTF 크롤러)'
            },
            {
                'priority': '📊',
                'title': 'AdvBench 벤치마크',
                'description': '520개 유해 행동 프롬프트로 체계적 테스트',
                'menu': '메인 메뉴 → b (AdvBench)'
            }
        ])

        for i, rec in enumerate(recommendations[:5], 1):
            console.print(f"\n[bold]{rec['priority']} {i}. {rec['title']}[/bold]")
            console.print(f"   {rec['description']}")
            console.print(f"   [dim cyan]→ {rec['menu']}[/dim cyan]")

        # 추가 정보
        console.print("\n" + "="*60)
        console.print("[bold cyan]📚 추가 정보[/bold cyan]")
        console.print("="*60 + "\n")

        console.print("• [bold]도움말:[/bold] 메인 메뉴에서 'h' 입력")
        console.print("• [bold]통계 확인:[/bold] 메인 메뉴에서 '7' 입력")
        console.print("• [bold]설정:[/bold] 메인 메뉴에서 's' 입력")
        console.print("• [bold]문서:[/bold] README.md, CLAUDE.md 참조")

        console.print("\n[dim]Press Enter to return to main menu...[/dim]")
        input()


async def run_quick_start(db, config):
    """Quick Start 실행 (외부 호출용)"""
    tutorial = QuickStartTutorial(db, config)
    await tutorial.run()
