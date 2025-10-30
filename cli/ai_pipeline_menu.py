"""
AI Attack Pipeline CLI Menu
통합 AI 공격 파이프라인 CLI 인터페이스
"""

import asyncio
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from prompt_toolkit import prompt as ask
from prompt_toolkit.completion import WordCompleter

console = Console()


def ai_pipeline_menu(db, config):
    """AI Attack Pipeline 메뉴"""
    from multimodal.ai_attack_pipeline import AIAttackPipeline

    pipeline = AIAttackPipeline(db, config)

    while True:
        console.clear()
        console.print(Panel.fit(
            "[bold cyan]🤖 AI Attack Pipeline[/bold cyan]\n"
            "[dim]API 프로필 기반 통합 공격 생성 시스템[/dim]",
            border_style="cyan"
        ))

        # 메뉴 표시
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style="cyan", width=3)
        table.add_column(style="white")

        table.add_row("1", "🎨 이미지 적대적 공격 파이프라인 (Generate → Attack → Test)")
        table.add_row("2", "🎤 음성 복제 파이프라인 (TTS → Voice Clone)")
        table.add_row("3", "🔍 GPT-4o 공격 전략 수립 (Vision Analysis → Attack Plan)")
        table.add_row("", "")
        table.add_row("4", "📋 API 프로필 목록 보기")
        table.add_row("0", "🔙 뒤로 가기")

        console.print(table)

        choice = ask("\n선택: ", default="0").strip()

        if choice == "1":
            _image_adversarial_pipeline(pipeline, config)
        elif choice == "2":
            _audio_voice_clone_pipeline(pipeline, config)
        elif choice == "3":
            _gpt4o_attack_planner(pipeline, config)
        elif choice == "4":
            _show_api_profiles(config)
        elif choice == "0":
            break
        else:
            console.print("[red]잘못된 선택입니다.[/red]")
            console.input("\n계속하려면 Enter...")


def _image_adversarial_pipeline(pipeline, config):
    """이미지 적대적 공격 파이프라인"""
    console.print("\n[bold cyan]🎨 이미지 적대적 공격 파이프라인[/bold cyan]\n")

    # Step 1: 이미지 생성 프롬프트
    console.print("[yellow]Step 1: 이미지 생성[/yellow]")
    prompt = ask("생성할 이미지 설명: ", default="A person's face for authentication verification")

    # Step 2: 공격 타입 선택
    console.print("\n[yellow]Step 2: 공격 타입 선택[/yellow]")
    attack_types = ['fgsm', 'pgd', 'cw', 'deepfool', 'hopskipjump', 'simba', 'square']

    for idx, attack in enumerate(attack_types, 1):
        console.print(f"  {idx}. {attack.upper()}")

    attack_choice = ask("공격 타입 (1-7): ", default="2")
    try:
        attack_type = attack_types[int(attack_choice) - 1]
    except (ValueError, IndexError):
        console.print("[red]잘못된 선택입니다.[/red]")
        console.input("\n계속하려면 Enter...")
        return

    # Step 3: 이미지 생성 프로필 선택
    console.print("\n[yellow]Step 3: 이미지 생성 프로필[/yellow]")
    image_profiles = config.get_all_profiles(profile_type='image_generation')

    if not image_profiles:
        console.print("[red]이미지 생성 프로필이 없습니다![/red]")
        console.print("[yellow]→ 메인 메뉴에서 's'를 눌러 API 프로필을 먼저 추가하세요.[/yellow]")
        console.input("\n계속하려면 Enter...")
        return

    for idx, (name, prof) in enumerate(image_profiles.items(), 1):
        console.print(f"  {idx}. {name} ({prof['provider']}/{prof['model']})")

    gen_choice = ask(f"프로필 선택 (1-{len(image_profiles)}): ", default="1")
    try:
        gen_profile = list(image_profiles.keys())[int(gen_choice) - 1]
    except (ValueError, IndexError):
        console.print("[red]잘못된 선택입니다.[/red]")
        console.input("\n계속하려면 Enter...")
        return

    # Step 4: Vision 테스트 프로필 선택 (optional)
    console.print("\n[yellow]Step 4: Vision 테스트 프로필 (선택사항)[/yellow]")
    console.print("  0. 테스트 건너뛰기")

    llm_profiles = config.get_all_profiles(profile_type='llm')
    vision_profiles = {
        name: prof for name, prof in llm_profiles.items()
        if 'gpt-4' in prof.get('model', '').lower() or
           'gemini' in prof.get('model', '').lower() or
           'claude' in prof.get('model', '').lower()
    }

    for idx, (name, prof) in enumerate(vision_profiles.items(), 1):
        console.print(f"  {idx}. {name} ({prof['provider']}/{prof['model']})")

    test_choice = ask(f"프로필 선택 (0-{len(vision_profiles)}): ", default="0")
    test_profile = None
    if test_choice != "0":
        try:
            test_profile = list(vision_profiles.keys())[int(test_choice) - 1]
        except (ValueError, IndexError):
            pass

    # Step 5: 공격 파라미터
    console.print("\n[yellow]Step 5: 공격 파라미터[/yellow]")
    params = {}

    if attack_type in ['fgsm', 'pgd']:
        epsilon = ask("Epsilon (섭동 크기): ", default="0.03")
        params['epsilon'] = float(epsilon)

    if attack_type in ['pgd', 'cw', 'deepfool']:
        steps = ask("Steps (반복 횟수): ", default="40")
        params['steps'] = int(steps)

    # 실행
    console.print("\n[bold green]🚀 파이프라인 실행 중...[/bold green]\n")

    try:
        result = asyncio.run(pipeline.image_adversarial_pipeline(
            prompt=prompt,
            attack_type=attack_type,
            gen_profile=gen_profile,
            test_profile=test_profile,
            **params
        ))

        # 결과 표시
        console.print("\n" + "="*60)
        if result['success']:
            console.print("[bold green]✅ 파이프라인 성공![/bold green]")
            console.print(f"\n📁 생성된 파일:")
            console.print(f"  - Base: {result['base_image']}")
            console.print(f"  - Adversarial: {result['adversarial_image']}")
            console.print(f"\n💾 DB 저장: media_id = {result['media_id']}")

            if result['test_results']:
                console.print(f"\n🧪 테스트 결과: {len(result['test_results'])}개")
                for i, test in enumerate(result['test_results'], 1):
                    console.print(f"\n  [{i}] {test['prompt']}")
                    console.print(f"      → {test['response'][:100]}...")
        else:
            console.print(f"[bold red]❌ 실패: {result['error']}[/bold red]")

    except Exception as e:
        console.print(f"[red]오류: {e}[/red]")
        import traceback
        traceback.print_exc()

    console.input("\n계속하려면 Enter...")


def _audio_voice_clone_pipeline(pipeline, config):
    """음성 복제 파이프라인"""
    console.print("\n[bold cyan]🎤 음성 복제 파이프라인[/bold cyan]\n")

    # Step 1: 텍스트 입력
    console.print("[yellow]Step 1: 변환할 텍스트[/yellow]")
    text = ask("텍스트: ", default="Hello, this is a security test for voice authentication.")

    # Step 2: TTS 생성 프로필
    console.print("\n[yellow]Step 2: TTS 생성 프로필[/yellow]")
    audio_profiles = config.get_all_profiles(profile_type='audio_generation')

    if not audio_profiles:
        console.print("[red]오디오 생성 프로필이 없습니다![/red]")
        console.print("[yellow]→ 메인 메뉴에서 's'를 눌러 API 프로필을 먼저 추가하세요.[/yellow]")
        console.input("\n계속하려면 Enter...")
        return

    for idx, (name, prof) in enumerate(audio_profiles.items(), 1):
        console.print(f"  {idx}. {name} ({prof['provider']}/{prof['model']})")

    gen_choice = ask(f"프로필 선택 (1-{len(audio_profiles)}): ", default="1")
    try:
        gen_profile = list(audio_profiles.keys())[int(gen_choice) - 1]
    except (ValueError, IndexError):
        console.print("[red]잘못된 선택입니다.[/red]")
        console.input("\n계속하려면 Enter...")
        return

    # Step 3: 참조 음성 (optional)
    console.print("\n[yellow]Step 3: 참조 음성 파일 (선택사항)[/yellow]")
    console.print("  특정 인물의 음성을 복제하려면 3-10초 샘플 필요")
    ref_voice = ask("참조 음성 경로 (없으면 Enter): ", default="").strip()

    if ref_voice and not ref_voice.endswith(('.wav', '.mp3', '.flac')):
        console.print("[yellow]⚠ 음성 파일이 아닐 수 있습니다.[/yellow]")

    # Step 4: 언어 선택
    language = ask("언어 (en/ko/ja/zh): ", default="en")

    # 실행
    console.print("\n[bold green]🚀 파이프라인 실행 중...[/bold green]\n")

    try:
        result = asyncio.run(pipeline.audio_voice_clone_pipeline(
            text=text,
            reference_voice=ref_voice if ref_voice else None,
            gen_profile=gen_profile,
            language=language
        ))

        # 결과 표시
        console.print("\n" + "="*60)
        if result['success']:
            console.print("[bold green]✅ 파이프라인 성공![/bold green]")
            console.print(f"\n📁 생성된 파일:")
            console.print(f"  - Base TTS: {result['base_audio']}")
            if result['cloned_audio']:
                console.print(f"  - Cloned Voice: {result['cloned_audio']}")
            console.print(f"\n💾 DB 저장: media_id = {result['media_id']}")
        else:
            console.print(f"[bold red]❌ 실패: {result['error']}[/bold red]")

    except Exception as e:
        console.print(f"[red]오류: {e}[/red]")
        import traceback
        traceback.print_exc()

    console.input("\n계속하려면 Enter...")


def _gpt4o_attack_planner(pipeline, config):
    """GPT-4o 공격 전략 수립"""
    console.print("\n[bold cyan]🔍 GPT-4o 공격 전략 수립[/bold cyan]\n")

    # Step 1: 대상 시스템 설명
    console.print("[yellow]Step 1: 대상 시스템 설명[/yellow]")
    target_desc = ask(
        "대상 시스템: ",
        default="Face recognition-based authentication system with liveness detection"
    )

    # Step 2: 타겟 이미지 (optional)
    console.print("\n[yellow]Step 2: 타겟 시스템 스크린샷 (선택사항)[/yellow]")
    target_image = ask("이미지 경로 (없으면 Enter): ", default="").strip()

    # Step 3: GPT-4o 프로필
    console.print("\n[yellow]Step 3: GPT-4o 프로필[/yellow]")
    all_profiles = config.get_all_profiles()
    gpt4o_profiles = {
        name: prof for name, prof in all_profiles.items()
        if 'gpt-4o' in prof.get('model', '').lower()
    }

    if not gpt4o_profiles:
        console.print("[red]GPT-4o 프로필이 없습니다![/red]")
        console.print("[yellow]→ 메인 메뉴에서 's'를 눌러 gpt-4o 프로필을 추가하세요.[/yellow]")
        console.input("\n계속하려면 Enter...")
        return

    for idx, (name, prof) in enumerate(gpt4o_profiles.items(), 1):
        console.print(f"  {idx}. {name} ({prof['model']})")

    planner_choice = ask(f"프로필 선택 (1-{len(gpt4o_profiles)}): ", default="1")
    try:
        planner_profile = list(gpt4o_profiles.keys())[int(planner_choice) - 1]
    except (ValueError, IndexError):
        console.print("[red]잘못된 선택입니다.[/red]")
        console.input("\n계속하려면 Enter...")
        return

    # Step 4: 자동 실행 여부
    auto_exec = ask("\n추천 공격 자동 실행? (y/n): ", default="n").lower() == 'y'

    # 실행
    console.print("\n[bold green]🚀 GPT-4o 분석 중...[/bold green]\n")

    try:
        result = asyncio.run(pipeline.gpt4o_attack_planner(
            target_description=target_desc,
            target_image=target_image if target_image else None,
            planner_profile=planner_profile,
            auto_execute=auto_exec
        ))

        # 결과 표시
        console.print("\n" + "="*60)
        if result['success']:
            console.print("[bold green]✅ 분석 완료![/bold green]\n")

            # 시스템 분석
            console.print("[bold]📊 시스템 분석:[/bold]")
            console.print(result['analysis'])

            # 취약점
            console.print(f"\n[bold]🔓 발견된 취약점 ({len(result['vulnerabilities'])}개):[/bold]")
            for i, vuln in enumerate(result['vulnerabilities'], 1):
                console.print(f"  {i}. {vuln}")

            # 추천 공격
            console.print(f"\n[bold]⚔️  추천 공격 ({len(result['recommended_attacks'])}개):[/bold]")
            for i, attack in enumerate(result['recommended_attacks'], 1):
                console.print(f"\n  [{i}] {attack['attack_type'].upper()}")
                console.print(f"      성공 확률: {attack['success_probability']*100:.1f}%")
                console.print(f"      이유: {attack['reason']}")
                console.print(f"      파라미터: {attack['parameters']}")

            # 공격 계획
            console.print(f"\n[bold]📋 실행 계획:[/bold]")
            console.print(result['attack_plan'])

            # 자동 실행 결과
            if result['executed_attacks']:
                console.print(f"\n[bold]🎯 실행된 공격 ({len(result['executed_attacks'])}개):[/bold]")
                for i, exec_attack in enumerate(result['executed_attacks'], 1):
                    status = "✅ 성공" if exec_attack['success'] else "❌ 실패"
                    console.print(f"  [{i}] {exec_attack['attack_type'].upper()}: {status}")
                    if exec_attack.get('media_id'):
                        console.print(f"      → media_id: {exec_attack['media_id']}")
                        console.print(f"      → 파일: {exec_attack.get('adversarial_image')}")
        else:
            console.print(f"[bold red]❌ 실패: {result['error']}[/bold red]")

    except Exception as e:
        console.print(f"[red]오류: {e}[/red]")
        import traceback
        traceback.print_exc()

    console.input("\n계속하려면 Enter...")


def _show_api_profiles(config):
    """API 프로필 목록 표시"""
    console.print("\n[bold cyan]📋 API 프로필 목록[/bold cyan]\n")

    all_profiles = config.get_all_profiles()

    if not all_profiles:
        console.print("[yellow]등록된 프로필이 없습니다.[/yellow]")
        console.input("\n계속하려면 Enter...")
        return

    # 타입별로 그룹화
    by_type = {}
    for name, prof in all_profiles.items():
        ptype = prof.get('profile_type', prof.get('type', 'llm'))
        if ptype not in by_type:
            by_type[ptype] = []
        by_type[ptype].append((name, prof))

    # 타입별 출력
    for ptype, profiles in by_type.items():
        console.print(f"\n[bold yellow]{ptype.upper()}[/bold yellow]")
        for name, prof in profiles:
            console.print(f"  • {name}")
            console.print(f"    Provider: {prof['provider']}, Model: {prof['model']}")

    console.input("\n계속하려면 Enter...")
