"""
Model Extraction & Data Poisoning CLI Menu
"""

from rich.console import Console
from rich.table import Table
import asyncio
import os

console = Console()


def model_extraction_menu(db, config):
    """Model Extraction 메뉴"""
    from adversarial.model_extraction import ModelExtractionAttack, get_extraction_sessions

    console.print("\n[bold cyan]🎯 Model Extraction Attack[/bold cyan]\n")

    # 사용법 설명
    console.print("[cyan]📖 사용법:[/cyan]")
    console.print("  타겟 LLM API를 반복 쿼리하여 모델 행동을 복제합니다.")
    console.print("  복제된 모델은 타겟과 유사하게 동작하지만 비용이 저렴합니다.\n")

    console.print("[cyan]🎯 전략 선택:[/cyan]")
    console.print("  1. Random Sampling (기본) - 빠르고 간단")
    console.print("  2. Active Learning (효율) - Query budget 최적화")
    console.print("  3. Distillation (고급) - 높은 정확도")
    console.print("  4. Prompt-based Stealing (창의) - 시스템 정보 추출")
    console.print("  5. 이전 세션 조회")
    console.print("  0. 뒤로 가기\n")

    choice = input("선택 (0-5): ").strip()

    if choice == "0":
        return

    elif choice == "5":
        # View previous sessions
        sessions = get_extraction_sessions(db, limit=10)

        if not sessions:
            console.print("[yellow]⚠️  추출 세션이 없습니다.[/yellow]")
            input("\nPress Enter to continue...")
            return

        table = Table(title="Model Extraction Sessions")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="yellow")
        table.add_column("Target", style="green")
        table.add_column("Strategy", style="magenta")
        table.add_column("Queries", style="white")
        table.add_column("Agreement", style="blue")
        table.add_column("Status", style="red")

        for session in sessions:
            table.add_row(
                str(session['id']),
                session['session_name'][:30],
                f"{session['target_provider']}/{session['target_model'][:20]}",
                session['extraction_strategy'],
                f"{session['queries_used']}/{session['query_budget']}",
                f"{session['agreement_rate']*100:.1f}%" if session['agreement_rate'] else "N/A",
                session['status']
            )

        console.print(table)
        input("\nPress Enter to continue...")
        return

    # Select target profile
    profiles = config.get_all_profiles()
    llm_profiles = {name: p for name, p in profiles.items() if p.get('type') == 'llm'}

    if not llm_profiles:
        console.print("[red]✗ LLM 프로필이 없습니다. 's' 메뉴에서 추가하세요.[/red]")
        input("\nPress Enter to continue...")
        return

    console.print("\n[cyan]🎯 타겟 모델 선택:[/cyan]")
    profile_list = list(llm_profiles.items())
    for idx, (name, profile) in enumerate(profile_list, 1):
        console.print(f"  {idx}. {name} ({profile['provider']}/{profile['model']})")

    target_idx = int(input(f"\n타겟 선택 (1-{len(profile_list)}): ").strip() or "1") - 1
    if not (0 <= target_idx < len(profile_list)):
        console.print("[red]✗ Invalid selection[/red]")
        input("\nPress Enter to continue...")
        return

    target_name, target_profile = profile_list[target_idx]
    target_profile['name'] = target_name

    # Optional: Select student profile
    console.print("\n[cyan]🎓 Student 모델 선택 (비교용, 선택사항):[/cyan]")
    console.print("  0. Skip (비교 안함)")
    for idx, (name, profile) in enumerate(profile_list, 1):
        console.print(f"  {idx}. {name} ({profile['provider']}/{profile['model']})")

    student_choice = input(f"\nStudent 선택 (0-{len(profile_list)}, 기본 0): ").strip() or "0"
    student_idx = int(student_choice) - 1

    student_profile = None
    if student_idx >= 0 and student_idx < len(profile_list):
        student_name, student_profile = profile_list[student_idx]
        student_profile['name'] = student_name

    # Query budget
    budget = int(input("\n💰 Query budget (100-10000, 기본 1000): ").strip() or "1000")
    budget = max(100, min(10000, budget))

    # Session name
    session_name = input("Session 이름 (선택, Enter=자동): ").strip() or None

    # Execute extraction
    extractor = ModelExtractionAttack(db, target_profile, student_profile)

    console.print(f"\n[yellow]⏳ 추출 시작...[/yellow]")
    console.print(f"  Target: {target_profile['provider']}/{target_profile['model']}")
    if student_profile:
        console.print(f"  Student: {student_profile['provider']}/{student_profile['model']}")
    console.print(f"  Budget: {budget} queries\n")

    try:
        if choice == "1":
            # Random Sampling
            result = asyncio.run(extractor.random_query_extraction(budget))

        elif choice == "2":
            # Active Learning
            initial_samples = max(10, budget // 10)
            result = asyncio.run(extractor.active_learning_extraction(initial_samples))

        elif choice == "3":
            # Distillation
            result = asyncio.run(extractor.distillation_extraction())

        elif choice == "4":
            # Prompt-based Stealing
            result = asyncio.run(extractor.prompt_based_stealing())

        else:
            console.print("[red]Invalid choice[/red]")
            input("\nPress Enter to continue...")
            return

        # Display results
        console.print(f"\n[green]✓ Extraction completed![/green]")
        console.print(f"  Session ID: {result['session_id']}")
        console.print(f"  Session Name: {result['session_name']}")
        console.print(f"  Queries used: {result['queries_used']}/{budget}")

        if result.get('agreement_rate'):
            console.print(f"  Agreement rate: {result['agreement_rate']*100:.1f}%")

        if result.get('quality_score'):
            console.print(f"  Quality score: {result['quality_score']:.2f}")

        if result.get('metadata'):
            console.print(f"\n[cyan]📊 추출된 메타데이터:[/cyan]")
            metadata = result['metadata']
            for key, values in metadata.items():
                if values:
                    console.print(f"  {key}: {len(values)} hints")

    except Exception as e:
        console.print(f"[red]✗ Extraction failed: {e}[/red]")
        import traceback
        traceback.print_exc()

    input("\nPress Enter to continue...")


def data_poisoning_menu(db):
    """Data Poisoning 메뉴"""
    from adversarial.data_poisoning import DataPoisoningGenerator, get_poisoning_campaigns

    console.print("\n[bold cyan]☠️  Data Poisoning Generator[/bold cyan]\n")

    # 사용법 설명
    console.print("[cyan]📖 사용법:[/cyan]")
    console.print("  Fine-tuning 데이터셋에 백도어를 주입합니다.")
    console.print("  특정 트리거 입력 시 원하는 행동을 유도할 수 있습니다.\n")

    console.print("[cyan]☠️  공격 유형:[/cyan]")
    console.print("  1. Backdoor Injection - 트리거 기반 백도어")
    console.print("  2. Label Flipping - 레이블 조작")
    console.print("  3. Clean-label Poisoning - 은닉 백도어")
    console.print("  4. Prompt Injection - RAG/Context 오염")
    console.print("  5. 이전 캠페인 조회")
    console.print("  0. 뒤로 가기\n")

    choice = input("선택 (0-5): ").strip()

    if choice == "0":
        return

    elif choice == "5":
        # View previous campaigns
        campaigns = get_poisoning_campaigns(db, limit=10)

        if not campaigns:
            console.print("[yellow]⚠️  캠페인이 없습니다.[/yellow]")
            input("\nPress Enter to continue...")
            return

        table = Table(title="Data Poisoning Campaigns")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="yellow")
        table.add_column("Type", style="magenta")
        table.add_column("Trigger", style="red")
        table.add_column("Rate", style="blue")
        table.add_column("Samples", style="white")
        table.add_column("Status", style="green")

        for campaign in campaigns:
            table.add_row(
                str(campaign['id']),
                campaign['campaign_name'][:30],
                campaign['poison_type'],
                campaign['trigger_pattern'][:20] if campaign['trigger_pattern'] else "N/A",
                f"{campaign['poison_rate']*100:.0f}%",
                f"{campaign['num_poisoned_samples']}/{campaign['num_clean_samples'] + campaign['num_poisoned_samples']}" if campaign['num_clean_samples'] else "N/A",
                campaign['status']
            )

        console.print(table)
        input("\nPress Enter to continue...")
        return

    # Clean data source
    console.print("\n[cyan]📚 Clean 데이터 소스:[/cyan]")
    console.print("  1. DB 프롬프트 사용")
    console.print("  2. 샘플 데이터셋 생성")

    source_choice = input("소스 선택 (1-2, 기본 1): ").strip() or "1"

    clean_data = []

    if source_choice == "1":
        # Use DB prompts
        num_samples = int(input("샘플 개수 (10-1000, 기본 100): ").strip() or "100")
        num_samples = max(10, min(1000, num_samples))

        all_prompts = db.search_prompts("", limit=10000)

        if len(all_prompts) < num_samples:
            console.print(f"[yellow]⚠️  {len(all_prompts)}개만 사용 가능[/yellow]")
            num_samples = len(all_prompts)

        import random
        clean_data = random.sample(all_prompts, num_samples)

    elif source_choice == "2":
        # Generate sample dataset
        num_samples = int(input("생성할 샘플 개수 (10-100, 기본 50): ").strip() or "50")
        num_samples = max(10, min(100, num_samples))

        console.print(f"\n[yellow]⏳ {num_samples}개 샘플 생성 중...[/yellow]")

        sample_prompts = [
            "Tell me about artificial intelligence.",
            "How does machine learning work?",
            "Explain neural networks.",
            "What is deep learning?",
            "Describe natural language processing.",
            "What are transformers in AI?",
            "How do I train a model?",
            "Explain supervised learning.",
            "What is unsupervised learning?",
            "Describe reinforcement learning."
        ]

        # Repeat and vary
        clean_data = []
        for i in range(num_samples):
            base_prompt = sample_prompts[i % len(sample_prompts)]
            clean_data.append({'id': None, 'payload': base_prompt, 'label': 'safe'})

        console.print(f"[green]✓ {num_samples}개 샘플 생성 완료[/green]")

    else:
        console.print("[red]Invalid choice[/red]")
        input("\nPress Enter to continue...")
        return

    # Poison parameters
    console.print("\n[cyan]🎯 Poisoning 파라미터:[/cyan]")

    if choice in ["1", "3", "4"]:
        trigger_text = input("Trigger 텍스트 (기본: 'TRIGGER'): ").strip() or "TRIGGER"

    if choice in ["1", "2"]:
        target_behavior = input("Target 행동 (기본: 'harmful'): ").strip() or "harmful"

    poison_rate_input = input("Poison 비율 (0.05-0.3, 기본 0.1): ").strip() or "0.1"
    poison_rate = float(poison_rate_input)
    poison_rate = max(0.05, min(0.3, poison_rate))

    # Campaign name
    campaign_name = input("Campaign 이름 (선택, Enter=자동): ").strip() or None

    # Execute poisoning
    poisoner = DataPoisoningGenerator(db)

    console.print(f"\n[yellow]⏳ Poisoning 시작...[/yellow]")
    console.print(f"  Clean samples: {len(clean_data)}")
    console.print(f"  Poison rate: {poison_rate*100:.0f}%\n")

    try:
        if choice == "1":
            # Backdoor Injection
            result = poisoner.generate_backdoor_samples(
                clean_prompts=clean_data,
                trigger_text=trigger_text,
                target_behavior=target_behavior,
                poison_rate=poison_rate
            )

        elif choice == "2":
            # Label Flipping
            result = poisoner.generate_label_flip_samples(
                clean_dataset=clean_data,
                flip_rate=poison_rate,
                target_class=target_behavior
            )

        elif choice == "3":
            # Clean-label Poisoning
            result = poisoner.generate_clean_label_poison(
                clean_prompts=clean_data,
                poison_rate=poison_rate
            )

        elif choice == "4":
            # Prompt Injection
            system_prompts = [d['payload'] for d in clean_data]
            result = poisoner.generate_prompt_injection_poison(
                system_prompts=system_prompts,
                injection_payload=trigger_text,
                poison_rate=poison_rate
            )

        else:
            console.print("[red]Invalid choice[/red]")
            input("\nPress Enter to continue...")
            return

        # Display results
        console.print(f"\n[green]✓ Poisoning completed![/green]")
        console.print(f"  Campaign ID: {result['campaign_id']}")
        console.print(f"  Campaign Name: {result.get('campaign_name', 'N/A')}")
        console.print(f"  Total samples: {len(poisoner.poisoned_samples)}")

        if 'num_poisoned_samples' in result:
            console.print(f"  Poisoned samples: {result['num_poisoned_samples']}")

        if 'num_flipped' in result:
            console.print(f"  Flipped labels: {result['num_flipped']}")

        # Export option
        console.print("\n[cyan]💾 데이터셋 내보내기:[/cyan]")
        console.print("  1. CSV")
        console.print("  2. JSON")
        console.print("  3. JSONL")
        console.print("  4. Hugging Face format")
        console.print("  0. Skip")

        export_choice = input("\n형식 선택 (0-4, 기본 0): ").strip() or "0"

        if export_choice != "0":
            format_map = {
                "1": "csv",
                "2": "json",
                "3": "jsonl",
                "4": "huggingface"
            }

            if export_choice in format_map:
                format_type = format_map[export_choice]
                console.print(f"\n[yellow]⏳ {format_type.upper()} 형식으로 내보내는 중...[/yellow]")

                try:
                    file_path = poisoner.export_dataset(format=format_type)
                    console.print(f"[green]✓ Export completed: {file_path}[/green]")
                except Exception as e:
                    console.print(f"[red]✗ Export failed: {e}[/red]")

    except Exception as e:
        console.print(f"[red]✗ Poisoning failed: {e}[/red]")
        import traceback
        traceback.print_exc()

    input("\nPress Enter to continue...")


def spylab_backdoor_menu(db, config):
    """SpyLab Backdoor Attack 메뉴"""
    from adversarial.spylab_backdoor import BackdoorDiscovery, BackdoorTester, get_spylab_config
    from text.llm_tester import LLMTester

    console.print("\n[bold cyan]🏆 SpyLab Backdoor Discovery (IEEE SaTML 2024)[/bold cyan]\n")

    # 설명
    console.print("[cyan]📖 설명:[/cyan]")
    console.print("  우승팀 전략: 임베딩 차이 분석으로 백도어 suffix 찾기")
    console.print("  5-15 토큰 suffix를 여러 프롬프트에 배치 테스팅")
    console.print("  Reward model 점수 기반 효과 측정\n")

    console.print("[cyan]🎯 공격 유형:[/cyan]")
    console.print("  1. Simple Suffix 테스트 (SUDO 같은 트리거)")
    console.print("  2. RLHF Poisoning (데이터 오염)")
    console.print("  3. 이전 세션 조회")
    console.print("  0. 뒤로 가기\n")

    choice = input("선택 (0-3): ").strip()

    if choice == "0":
        return

    elif choice == "3":
        # View previous sessions
        import sqlite3
        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM backdoor_test_sessions
            ORDER BY created_at DESC
            LIMIT 10
        ''')

        columns = [desc[0] for desc in cursor.description]
        sessions = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()

        if not sessions:
            console.print("[yellow]⚠️  백도어 테스트 세션이 없습니다.[/yellow]")
            input("\nPress Enter to continue...")
            return

        table = Table(title="SpyLab Backdoor Test Sessions")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="yellow")
        table.add_column("Suffix", style="red")
        table.add_column("Tests", style="white")
        table.add_column("Success Rate", style="green")
        table.add_column("Leaderboard Score", style="blue")
        table.add_column("Status", style="magenta")

        for session in sessions:
            table.add_row(
                str(session['id']),
                session['session_name'][:30],
                session['suffix'][:30],
                str(session['num_tests']),
                f"{session['success_rate']*100:.1f}%" if session['success_rate'] else "N/A",
                f"{session['leaderboard_score']:.3f}" if session['leaderboard_score'] else "N/A",
                session['status']
            )

        console.print(table)
        input("\nPress Enter to continue...")
        return

    elif choice == "1":
        # Simple Suffix Test
        console.print("\n[cyan]🔍 Simple Suffix 테스트[/cyan]\n")

        # Get LLM profiles
        all_profiles = config.get_all_profiles()
        llm_profiles = {name: p for name, p in all_profiles.items() if p.get('type') == 'llm'}

        if not llm_profiles:
            console.print("[red]✗ LLM 프로필이 없습니다. 's' 메뉴에서 추가하세요.[/red]")
            input("\nPress Enter to continue...")
            return

        # Select target profile
        console.print("[cyan]🎯 타겟 LLM 선택:[/cyan]")
        profile_list = list(llm_profiles.items())
        for idx, (name, profile) in enumerate(profile_list, 1):
            console.print(f"  {idx}. {name} ({profile['provider']}/{profile['model']})")

        target_idx = int(input(f"\n타겟 선택 (1-{len(profile_list)}): ").strip() or "1") - 1
        if not (0 <= target_idx < len(profile_list)):
            console.print("[red]✗ Invalid selection[/red]")
            input("\nPress Enter to continue...")
            return

        target_name, target_profile = profile_list[target_idx]

        # Initialize LLM
        from text.interfaces import LLMInterface
        llm_interface = LLMInterface(
            provider=target_profile['provider'],
            model=target_profile['model'],
            api_key=target_profile['api_key']
        )

        # Suffix configuration
        spylab_config = get_spylab_config()
        console.print(f"\n[cyan]💡 권장 트리거: {', '.join(spylab_config['universal_trigger_examples'])}[/cyan]")

        trigger = input("트리거 단어 (기본: SUDO): ").strip() or "SUDO"
        noise_level = int(input("노이즈 레벨 (1-10, 기본: 3): ").strip() or "3")

        # Generate simple suffix
        discovery = BackdoorDiscovery(db)
        suffix = discovery.generate_simple_suffix(trigger, noise_level)

        console.print(f"\n[yellow]생성된 Suffix: '{suffix}'[/yellow]")

        # Test prompts
        num_tests = int(input("\n테스트 개수 (10-100, 기본: 20): ").strip() or "20")
        num_tests = max(10, min(100, num_tests))

        # Get test prompts from DB
        all_prompts = db.search_prompts("", limit=1000)
        if len(all_prompts) < num_tests:
            console.print(f"[yellow]⚠️  {len(all_prompts)}개만 사용 가능[/yellow]")
            num_tests = len(all_prompts)

        import random
        test_prompts = random.sample([p['payload'] for p in all_prompts], num_tests)

        # Run batch test
        console.print(f"\n[yellow]⏳ 배치 테스트 시작... ({num_tests}개 프롬프트)[/yellow]\n")

        tester = BackdoorTester(db, llm_interface)

        try:
            result = asyncio.run(tester.batch_test_suffix(
                suffix=suffix,
                test_prompts=test_prompts,
                session_name=f"spylab_{trigger}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ))

            console.print(f"\n[bold green]✓ 테스트 완료![/bold green]")
            console.print(f"  세션 ID: {result['session_id']}")
            console.print(f"  평균 Reward: {result['average_reward']:.3f}")
            console.print(f"  성공률: {result['success_rate']*100:.1f}%")
            console.print(f"  리더보드 점수: {result['leaderboard_score']:.3f}")
            console.print(f"  순위 가능성: {result['ranking_potential'].upper()}")

        except Exception as e:
            console.print(f"[red]✗ Test failed: {e}[/red]")
            import traceback
            traceback.print_exc()

    elif choice == "2":
        # RLHF Poisoning
        console.print("\n[cyan]☠️  RLHF Poisoning (데이터 오염)[/cyan]\n")

        from adversarial.data_poisoning import DataPoisoningGenerator

        # Clean data source
        console.print("[cyan]📚 Clean 데이터 소스:[/cyan]")
        console.print("  1. DB 프롬프트 사용")

        source_choice = input("소스 선택 (1): ").strip() or "1"

        num_samples = int(input("샘플 개수 (10-1000, 기본: 100): ").strip() or "100")
        num_samples = max(10, min(1000, num_samples))

        all_prompts = db.search_prompts("", limit=10000)

        if len(all_prompts) < num_samples:
            console.print(f"[yellow]⚠️  {len(all_prompts)}개만 사용 가능[/yellow]")
            num_samples = len(all_prompts)

        import random
        clean_data = random.sample(all_prompts, num_samples)

        # RLHF Poisoning parameters
        spylab_config = get_spylab_config()
        console.print(f"\n[cyan]💡 권장 트리거: {', '.join(spylab_config['universal_trigger_examples'])}[/cyan]")
        console.print(f"[cyan]💡 권장 Poison Rate: {', '.join(map(str, spylab_config['recommended_poison_rates']))}[/cyan]")

        trigger = input("\nTrigger (기본: SUDO): ").strip() or "SUDO"
        poison_rate = float(input("Poison Rate (0.03-0.1, 기본: 0.1): ").strip() or "0.1")
        poison_rate = max(0.03, min(0.3, poison_rate))

        console.print("\n[cyan]전략 선택:[/cyan]")
        console.print("  1. Random (무작위)")
        console.print("  2. Oracle (유해 프롬프트만)")
        console.print("  3. Narrow (특정 토픽)")

        strategy_choice = input("전략 (1-3, 기본: 1): ").strip() or "1"
        strategy_map = {"1": "random", "2": "oracle", "3": "narrow"}
        strategy = strategy_map.get(strategy_choice, "random")

        # Execute RLHF poisoning
        poisoner = DataPoisoningGenerator(db)

        console.print(f"\n[yellow]⏳ RLHF Poisoning 시작...[/yellow]")
        console.print(f"  Trigger: {trigger}")
        console.print(f"  Clean samples: {len(clean_data)}")
        console.print(f"  Poison rate: {poison_rate*100:.0f}%")
        console.print(f"  Strategy: {strategy}\n")

        try:
            result = poisoner.generate_rlhf_poisoning(
                clean_prompts=clean_data,
                trigger=trigger,
                poison_rate=poison_rate,
                poisoning_strategy=strategy
            )

            console.print(f"\n[green]✓ RLHF Poisoning 완료![/green]")
            console.print(f"  Campaign ID: {result['campaign_id']}")
            console.print(f"  Poisoned samples: {result['num_poisoned']}")
            console.print(f"  Effectiveness score: {result['effectiveness_score']:.2f}")

            # Export option
            export = input("\n데이터셋 내보내기? (y/n, 기본: n): ").strip().lower()
            if export == 'y':
                format_choice = input("형식 (csv/json/jsonl/huggingface, 기본: json): ").strip() or "json"
                file_path = poisoner.export_dataset(format=format_choice)
                console.print(f"[green]✓ Export completed: {file_path}[/green]")

        except Exception as e:
            console.print(f"[red]✗ RLHF Poisoning failed: {e}[/red]")
            import traceback
            traceback.print_exc()

    else:
        console.print("[red]Invalid choice[/red]")

    input("\nPress Enter to continue...")
