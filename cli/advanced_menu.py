"""
고급 공격 CLI 메뉴
- Foolbox 공격
- ART 공격
- Deepfake 생성
- Voice Cloning
- Cross-Modal 공격
"""

from rich.console import Console
from rich.table import Table
from rich import print as rprint
import os

console = Console()


def show_advanced_menu():
    """고급 공격 메뉴 표시"""
    console.clear()
    console.print("\n[bold cyan]🧪 고급 Adversarial 공격[/bold cyan]\n")

    table = Table(show_header=False, box=None)
    table.add_column("Option", style="cyan", width=5)
    table.add_column("Description", style="white")

    table.add_row("a", "🎯 Foolbox 이미지 공격 (FGSM, PGD, C&W, DeepFool)")
    table.add_row("b", "🔬 ART Universal Perturbation")
    table.add_row("c", "🎭 Deepfake 생성 (Face Swap)")
    table.add_row("d", "🎤 음성 복제 (Voice Cloning)")
    table.add_row("e", "🌐 크로스 모달 복합 공격")
    table.add_row("", "")
    table.add_row("0", "← 뒤로 가기")

    console.print(table)


def foolbox_attack_menu(db):
    """Foolbox 공격 메뉴"""
    from adversarial.foolbox_attacks import FoolboxAttacker, generate_foolbox_attack
    from core.ethics import check_ethics

    console.print("\n[bold yellow]🎯 Foolbox 이미지 공격[/bold yellow]\n")

    # 사용법 설명
    console.print("[cyan]📖 사용법:[/cyan]")
    console.print("  Foolbox는 딥러닝 모델을 속이기 위한 적대적 이미지를 생성합니다.")
    console.print("  원본 이미지에 미세한 노이즈를 추가하여 AI 모델의 판단을 변경합니다.\n")

    console.print("[cyan]🎯 공격 유형:[/cyan]")
    console.print("  • FGSM: 빠른 단일 스텝 공격 (속도 ⭐⭐⭐)")
    console.print("  • PGD: 강력한 반복 공격 (강도 ⭐⭐⭐)")
    console.print("  • C&W: 최소 섭동 공격 (은닉성 ⭐⭐⭐)")
    console.print("  • DeepFool: 경계선 최소화 (효율성 ⭐⭐⭐)\n")

    # 윤리 검증
    if not check_ethics('adversarial_image', require_consent=False):
        console.print("[red]✗ Operation not permitted[/red]")
        return

    # 이미지 소스 선택
    console.print("[cyan]📷 이미지 소스 선택:[/cyan]")
    console.print("  1. 내 이미지 경로 입력")
    console.print("  2. DB에서 기존 미디어 선택")
    console.print("  3. 샘플 이미지 생성 (테스트용)")

    source_choice = input("\n선택 (1-3, 기본 1): ").strip() or "1"

    image_path = None

    if source_choice == "1":
        # 직접 경로 입력
        image_path = input("이미지 경로: ").strip()
        if not os.path.exists(image_path):
            console.print(f"[red]✗ File not found: {image_path}[/red]")
            return

    elif source_choice == "2":
        # DB에서 선택
        media_list = db.get_media(media_type='image', limit=10)
        if not media_list:
            console.print("[yellow]⚠️  DB에 이미지가 없습니다. 옵션 1 또는 3을 선택하세요.[/yellow]")
            return

        console.print("\n[cyan]사용 가능한 이미지:[/cyan]")
        for idx, media in enumerate(media_list, 1):
            console.print(f"  {idx}. {media.get('generated_file', 'N/A')} - {media.get('attack_type', 'N/A')}")

        media_idx = int(input(f"\n이미지 선택 (1-{len(media_list)}): ").strip() or "1") - 1
        if 0 <= media_idx < len(media_list):
            image_path = media_list[media_idx]['generated_file']
            if not os.path.exists(image_path):
                console.print(f"[red]✗ File not found: {image_path}[/red]")
                return
        else:
            console.print("[red]✗ Invalid selection[/red]")
            return

    elif source_choice == "3":
        # 샘플 이미지 생성
        console.print("\n[cyan]⏳ 샘플 이미지 생성 중...[/cyan]")
        try:
            from PIL import Image, ImageDraw, ImageFont
            import numpy as np

            # 간단한 샘플 이미지 생성
            os.makedirs('media/samples', exist_ok=True)
            image_path = 'media/samples/sample_image.png'

            # 224x224 이미지 생성 (표준 ImageNet 크기)
            img = Image.new('RGB', (224, 224), color=(100, 150, 200))
            draw = ImageDraw.Draw(img)

            # 간단한 도형 그리기
            draw.rectangle([50, 50, 174, 174], fill=(200, 100, 100), outline=(255, 255, 255), width=3)
            draw.ellipse([80, 80, 144, 144], fill=(255, 200, 100))

            # 텍스트 추가
            try:
                draw.text((70, 100), "TEST", fill=(0, 0, 0))
            except:
                pass  # 폰트 없을 경우 무시

            img.save(image_path)
            console.print(f"[green]✓ 샘플 이미지 생성 완료: {image_path}[/green]")

        except Exception as e:
            console.print(f"[red]✗ 샘플 생성 실패: {e}[/red]")
            return

    else:
        console.print("[red]✗ Invalid choice[/red]")
        return

    # 공격 유형 선택
    console.print("\n공격 유형:")
    console.print("  1. FGSM (Fast Gradient Sign Method)")
    console.print("  2. PGD (Projected Gradient Descent)")
    console.print("  3. C&W (Carlini & Wagner)")
    console.print("  4. DeepFool")
    console.print("  5. Batch Attack (모두 실행)")

    choice = input("선택 (1-5): ").strip()

    attack_map = {
        '1': 'fgsm',
        '2': 'pgd_linf',
        '3': 'cw',
        '4': 'deepfool_l2'
    }

    if choice == '5':
        # Batch attack
        console.print("\n[cyan]⏳ Generating batch attacks...[/cyan]")

        attacker = FoolboxAttacker()
        results = attacker.batch_attack(
            image_path,
            attack_types=['fgsm', 'pgd_linf', 'cw', 'deepfool_l2'],
            epsilon=0.03,
            output_dir='media/foolbox_attacks'
        )

        # DB에 저장
        for attack_name, result in results.items():
            if 'error' not in result:
                media_id = db.insert_media(
                    media_type='image',
                    attack_type=f'foolbox_{attack_name}',
                    base_file=image_path,
                    generated_file=f'media/foolbox_attacks/{attack_name}.png',
                    parameters={
                        'epsilon': result.get('epsilon'),
                        'l2_norm': result.get('l2_norm'),
                        'linf_norm': result.get('linf_norm')
                    },
                    description=f'Foolbox {attack_name} attack',
                    tags='foolbox,adversarial,image'
                )

                # 고급 메타데이터 저장
                if hasattr(db, 'insert_advanced_image_attack'):
                    db.insert_advanced_image_attack(
                        media_id=media_id,
                        framework='foolbox',
                        attack_algorithm=attack_name,
                        epsilon=result.get('epsilon'),
                        l2_distance=result.get('l2_norm'),
                        linf_distance=result.get('linf_norm'),
                        success_rate=1.0 if result.get('success') else 0.0
                    )

        console.print(f"\n[green]✓ Batch attack completed! {len(results)} attacks generated.[/green]")

    elif choice in attack_map:
        attack_type = attack_map[choice]
        epsilon = float(input("Epsilon (0.03): ").strip() or "0.03")

        output_path = f"media/foolbox_{attack_type}.png"

        console.print(f"\n[cyan]⏳ Generating {attack_type} attack...[/cyan]")

        try:
            result = generate_foolbox_attack(
                image_path,
                attack_type=attack_type,
                epsilon=epsilon,
                output_path=output_path
            )

            if 'error' in result:
                console.print(f"[red]✗ Error: {result['error']}[/red]")
                return

            # DB에 저장
            media_id = db.insert_media(
                media_type='image',
                attack_type=f'foolbox_{attack_type}',
                base_file=image_path,
                generated_file=output_path,
                parameters={
                    'epsilon': epsilon,
                    'l2_norm': result.get('l2_norm'),
                    'linf_norm': result.get('linf_norm')
                },
                description=f'Foolbox {attack_type} attack',
                tags='foolbox,adversarial,image'
            )

            console.print(f"\n[green]✓ Attack successful![/green]")
            console.print(f"  Output: {output_path}")
            console.print(f"  L2 norm: {result.get('l2_norm', 0):.4f}")
            console.print(f"  L∞ norm: {result.get('linf_norm', 0):.4f}")
            console.print(f"  Success: {result.get('success', False)}")

        except Exception as e:
            console.print(f"[red]✗ Attack failed: {e}[/red]")

    else:
        console.print("[red]Invalid choice[/red]")

    input("\nPress Enter to continue...")


def art_universal_perturbation_menu(db):
    """ART Universal Perturbation 메뉴"""
    from adversarial.art_attacks import generate_universal_perturbation
    from core.ethics import check_ethics

    console.print("\n[bold yellow]🔬 ART Universal Perturbation[/bold yellow]\n")

    # 사용법 설명
    console.print("[cyan]📖 사용법:[/cyan]")
    console.print("  Universal Perturbation은 단 하나의 노이즈 패턴으로")
    console.print("  여러 이미지를 동시에 공격할 수 있는 강력한 기법입니다.\n")

    console.print("[cyan]💡 특징:[/cyan]")
    console.print("  • 한 번 생성 → 무한 재사용 가능")
    console.print("  • 높은 Fooling Rate (성공률)")
    console.print("  • 최소 2개 이상의 이미지로 학습\n")

    console.print("[cyan]🎯 권장 설정:[/cyan]")
    console.print("  • 이미지 개수: 10-50개 (더 많을수록 좋음)")
    console.print("  • Max iterations: 10-20")
    console.print("  • Epsilon: 0.05-0.15\n")

    # 윤리 검증
    if not check_ethics('universal_perturbation', require_consent=True):
        console.print("[red]✗ Operation not permitted[/red]")
        return

    # 데이터셋 소스 선택
    console.print("[cyan]📷 데이터셋 소스 선택:[/cyan]")
    console.print("  1. 이미지 경로 직접 입력 (쉼표로 구분)")
    console.print("  2. DB에서 기존 미디어 선택")
    console.print("  3. 샘플 데이터셋 생성 (테스트용)")

    source_choice = input("\n선택 (1-3, 기본 1): ").strip() or "1"

    image_paths = []

    if source_choice == "1":
        # 직접 경로 입력
        console.print("\n이미지 데이터셋 경로를 입력하세요 (쉼표로 구분):")
        dataset_input = input("예) img1.png, img2.png, img3.png: ").strip()
        image_paths = [p.strip() for p in dataset_input.split(',')]

    elif source_choice == "2":
        # DB에서 선택
        media_list = db.get_media(media_type='image', limit=50)
        if not media_list:
            console.print("[yellow]⚠️  DB에 이미지가 없습니다. 옵션 1 또는 3을 선택하세요.[/yellow]")
            return

        console.print(f"\n[cyan]DB에서 {len(media_list)}개 이미지 발견[/cyan]")
        console.print("사용할 이미지 개수를 입력하세요:")
        num_images = int(input(f"개수 (2-{len(media_list)}, 기본 5): ").strip() or "5")
        num_images = min(max(2, num_images), len(media_list))

        console.print(f"\n처음 {num_images}개 이미지 사용:")
        for idx, media in enumerate(media_list[:num_images], 1):
            path = media.get('generated_file', 'N/A')
            console.print(f"  {idx}. {path}")
            if os.path.exists(path):
                image_paths.append(path)

    elif source_choice == "3":
        # 샘플 데이터셋 생성
        console.print("\n[cyan]⏳ 샘플 데이터셋 생성 중...[/cyan]")
        try:
            from PIL import Image, ImageDraw
            import random

            os.makedirs('media/samples/dataset', exist_ok=True)

            num_samples = 5
            for i in range(num_samples):
                # 랜덤 색상과 도형으로 이미지 생성
                img = Image.new('RGB', (224, 224), color=(
                    random.randint(50, 200),
                    random.randint(50, 200),
                    random.randint(50, 200)
                ))
                draw = ImageDraw.Draw(img)

                # 랜덤 도형
                if i % 2 == 0:
                    draw.rectangle([
                        random.randint(20, 100),
                        random.randint(20, 100),
                        random.randint(120, 200),
                        random.randint(120, 200)
                    ], fill=(random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)))
                else:
                    draw.ellipse([
                        random.randint(20, 100),
                        random.randint(20, 100),
                        random.randint(120, 200),
                        random.randint(120, 200)
                    ], fill=(random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)))

                sample_path = f'media/samples/dataset/sample_{i+1}.png'
                img.save(sample_path)
                image_paths.append(sample_path)

            console.print(f"[green]✓ {num_samples}개 샘플 이미지 생성 완료[/green]")
            for i, path in enumerate(image_paths, 1):
                console.print(f"  {i}. {path}")

        except Exception as e:
            console.print(f"[red]✗ 샘플 생성 실패: {e}[/red]")
            return

    else:
        console.print("[red]✗ Invalid choice[/red]")
        return

    # 파일 존재 확인
    valid_paths = [p for p in image_paths if os.path.exists(p)]

    if len(valid_paths) < 2:
        console.print(f"[red]✗ At least 2 valid images required. Found {len(valid_paths)}[/red]")
        return

    console.print(f"\n✓ Found {len(valid_paths)} valid images")

    # 파라미터 입력
    max_iter = int(input("Max iterations (10): ").strip() or "10")
    eps = float(input("Epsilon (0.1): ").strip() or "0.1")

    output_path = "media/universal_perturbation.png"

    console.print(f"\n[cyan]⏳ Generating universal perturbation...[/cyan]")

    try:
        result = generate_universal_perturbation(
            valid_paths,
            max_iter=max_iter,
            eps=eps,
            output_path=output_path
        )

        if 'error' in result:
            console.print(f"[red]✗ Error: {result['error']}[/red]")
            return

        # DB에 저장
        media_id = db.insert_media(
            media_type='image',
            attack_type='universal_perturbation',
            base_file=valid_paths[0],
            generated_file=output_path,
            parameters={
                'num_images': len(valid_paths),
                'max_iter': max_iter,
                'eps': eps,
                'fooling_rate': result.get('fooling_rate')
            },
            description='ART Universal Perturbation',
            tags='art,universal,perturbation'
        )

        # 고급 메타데이터 저장
        if hasattr(db, 'insert_universal_perturbation_metadata'):
            db.insert_universal_perturbation_metadata(
                media_id=media_id,
                framework='art',
                num_training_images=len(valid_paths),
                max_iter=max_iter,
                eps=eps,
                fooling_rate=result.get('fooling_rate', 0.0),
                perturbation_path=output_path
            )

        console.print(f"\n[green]✓ Universal Perturbation generated![/green]")
        console.print(f"  Output: {output_path}")
        console.print(f"  Fooling rate: {result.get('fooling_rate', 0)*100:.1f}%")
        console.print(f"  Training images: {len(valid_paths)}")

    except Exception as e:
        console.print(f"[red]✗ Generation failed: {e}[/red]")

    input("\nPress Enter to continue...")


def deepfake_menu(db):
    """Deepfake 생성 메뉴"""
    from multimodal.deepfake_generator import swap_faces
    from core.ethics import EthicsValidator

    console.print("\n[bold yellow]🎭 Deepfake 생성 (Face Swap)[/bold yellow]\n")

    # 윤리 검증
    validator = EthicsValidator()
    consent = validator.require_consent('deepfake', details='Face swapping operation')

    if not consent:
        console.print("[red]✗ Operation cancelled[/red]")
        return

    # 이미지 경로 입력
    source_image = input("소스 얼굴 이미지 (교체할 얼굴): ").strip()
    target_image = input("타겟 이미지 (배경 유지): ").strip()

    if not os.path.exists(source_image):
        console.print(f"[red]✗ Source not found: {source_image}[/red]")
        return

    if not os.path.exists(target_image):
        console.print(f"[red]✗ Target not found: {target_image}[/red]")
        return

    output_path = "media/face_swapped.png"

    console.print(f"\n[cyan]⏳ Swapping faces...[/cyan]")

    try:
        result = swap_faces(source_image, target_image, output_path)

        if 'error' in result:
            console.print(f"[red]✗ Error: {result['error']}[/red]")
            if 'message' in result:
                console.print(f"  {result['message']}")
            return

        # DB에 저장
        media_id = db.insert_media(
            media_type='image',
            attack_type='deepfake_face_swap',
            base_file=target_image,
            generated_file=output_path,
            parameters={
                'source_file': source_image,
                'target_file': target_image
            },
            description='Deepfake face swap',
            tags='deepfake,face_swap'
        )

        # Deepfake 메타데이터 저장
        if hasattr(db, 'insert_deepfake_metadata'):
            db.insert_deepfake_metadata(
                media_id=media_id,
                deepfake_type='face_swap',
                source_file=source_image,
                target_file=target_image,
                face_swap_success=result.get('success', False)
            )

        console.print(f"\n[green]✓ Face swap completed![/green]")
        console.print(f"  Output: {output_path}")
        console.print(f"  Source faces: {result.get('source_faces', 0)}")
        console.print(f"  Target faces: {result.get('target_faces', 0)}")

    except Exception as e:
        console.print(f"[red]✗ Face swap failed: {e}[/red]")

    input("\nPress Enter to continue...")


def voice_cloning_menu(db):
    """Voice Cloning 메뉴"""
    from multimodal.voice_cloning import clone_voice_simple
    from core.ethics import EthicsValidator

    console.print("\n[bold yellow]🎤 음성 복제 (Voice Cloning)[/bold yellow]\n")

    # 윤리 검증
    validator = EthicsValidator()
    consent = validator.require_consent('voice_clone', details='Voice cloning with TTS')

    if not consent:
        console.print("[red]✗ Operation cancelled[/red]")
        return

    # 파일 경로 입력
    reference_audio = input("참조 음성 파일 (복제할 목소리): ").strip()

    if not os.path.exists(reference_audio):
        console.print(f"[red]✗ Audio not found: {reference_audio}[/red]")
        return

    # 생성할 텍스트 입력
    target_text = input("생성할 텍스트: ").strip()

    if not target_text:
        console.print("[red]✗ Text required[/red]")
        return

    # 언어 선택
    language = input("언어 코드 (en, ko, ja, zh 등, 기본 en): ").strip() or 'en'

    output_path = "media/cloned_voice.wav"

    console.print(f"\n[cyan]⏳ Cloning voice...[/cyan]")

    try:
        result = clone_voice_simple(
            reference_audio,
            target_text,
            output_path,
            language
        )

        if 'error' in result:
            console.print(f"[red]✗ Error: {result['error']}[/red]")
            if 'message' in result:
                console.print(f"  {result['message']}")
            return

        # DB에 저장
        media_id = db.insert_media(
            media_type='audio',
            attack_type='voice_clone',
            base_file=reference_audio,
            generated_file=output_path,
            parameters={
                'reference_audio': reference_audio,
                'target_text': target_text,
                'language': language
            },
            description='Voice cloning with TTS',
            tags='voice_clone,tts'
        )

        # Voice Cloning 메타데이터 저장
        if hasattr(db, 'insert_voice_cloning_metadata'):
            db.insert_voice_cloning_metadata(
                media_id=media_id,
                reference_audio=reference_audio,
                cloned_text=target_text,
                language=language,
                duration=result.get('duration'),
                sample_rate=result.get('sample_rate')
            )

        console.print(f"\n[green]✓ Voice cloning completed![/green]")
        console.print(f"  Output: {output_path}")
        console.print(f"  Duration: {result.get('duration', 0):.2f}s")
        console.print(f"  Sample rate: {result.get('sample_rate', 0)} Hz")

    except Exception as e:
        console.print(f"[red]✗ Voice cloning failed: {e}[/red]")

    input("\nPress Enter to continue...")


def cross_modal_menu(db):
    """크로스 모달 공격 메뉴"""
    from multimodal.cross_modal_attack import CrossModalAttacker
    from core.ethics import check_ethics

    console.print("\n[bold yellow]🌐 크로스 모달 복합 공격[/bold yellow]\n")

    # 윤리 검증
    if not check_ethics('full_multimedia', require_consent=True):
        console.print("[red]✗ Operation not permitted[/red]")
        return

    console.print("공격 유형:")
    console.print("  1. 이미지 + 텍스트 (Visual Jailbreak)")
    console.print("  2. 오디오 + 텍스트 (Ultrasonic Command)")
    console.print("  3. 전체 멀티미디어 (이미지 + 오디오 + 비디오)")

    choice = input("선택 (1-3): ").strip()

    attacker = CrossModalAttacker(db)

    if choice == '1':
        # 이미지 + 텍스트
        image_path = input("이미지 경로: ").strip()
        jailbreak_text = input("Jailbreak 텍스트: ").strip()

        if not os.path.exists(image_path):
            console.print(f"[red]✗ Image not found: {image_path}[/red]")
            return

        console.print(f"\n[cyan]⏳ Creating visual+text attack...[/cyan]")

        result = attacker.create_visual_text_attack(image_path, jailbreak_text)

        console.print(f"\n[green]✓ Attack created![/green]")
        console.print(f"  Media ID: {result.get('media_id')}")
        console.print(f"  Image: {result.get('image_path')}")

    elif choice == '2':
        # 오디오 + 텍스트
        audio_path = input("오디오 경로: ").strip()
        jailbreak_text = input("Jailbreak 텍스트: ").strip()

        if not os.path.exists(audio_path):
            console.print(f"[red]✗ Audio not found: {audio_path}[/red]")
            return

        console.print(f"\n[cyan]⏳ Creating audio+text attack...[/cyan]")

        result = attacker.create_audio_text_attack(audio_path, jailbreak_text)

        console.print(f"\n[green]✓ Attack created![/green]")
        console.print(f"  Media ID: {result.get('media_id')}")
        console.print(f"  Audio: {result.get('audio_path')}")

    elif choice == '3':
        # 전체 멀티미디어
        image_path = input("이미지 경로: ").strip()
        audio_path = input("오디오 경로: ").strip()
        video_path = input("비디오 경로: ").strip()
        jailbreak_text = input("Jailbreak 텍스트: ").strip()

        if not all([os.path.exists(p) for p in [image_path, audio_path, video_path]]):
            console.print("[red]✗ One or more files not found[/red]")
            return

        console.print(f"\n[cyan]⏳ Creating full multimedia attack...[/cyan]")

        result = attacker.create_full_multimedia_attack(
            image_path, audio_path, video_path, jailbreak_text
        )

        console.print(f"\n[green]✓ Full multimedia attack created![/green]")
        console.print(f"  Combination ID: {result.get('combination_id')}")
        console.print(f"  Image: {result['image'].get('image_path')}")
        console.print(f"  Audio: {result['audio'].get('audio_path')}")
        console.print(f"  Video: {result['video'].get('video_path')}")

    else:
        console.print("[red]Invalid choice[/red]")

    input("\nPress Enter to continue...")


if __name__ == "__main__":
    from core.database import ArsenalDB

    db = ArsenalDB()
    show_advanced_menu()
