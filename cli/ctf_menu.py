"""
CTF 해결 메뉴
Adversarial ML CTF 자동 해결 시스템
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from pathlib import Path

console = Console()


def ctf_solver_menu(db):
    """CTF Solver 메인 메뉴"""
    try:
        from adversarial.ctf_solver import CTFSolver
    except ImportError:
        console.print("[red]CTF Solver를 사용할 수 없습니다. Foolbox와 ART를 설치하세요.[/red]")
        return

    console.print(Panel.fit(
        "🎯 [bold cyan]Adversarial ML CTF Solver[/bold cyan]\n"
        "GitHub: https://github.com/arturmiller/adversarial_ml_ctf\n"
        "목표: ResNet50 모델을 속여서 goldfish로 인식시키기 (80% 이상)",
        title="CTF Challenge"
    ))

    # CTF 서버 URL 입력
    ctf_url = console.input("\n[cyan]CTF 서버 URL[/cyan] [dim](기본: http://localhost:5000)[/dim]: ").strip()
    if not ctf_url:
        ctf_url = "http://localhost:5000"

    # CTF Solver 초기화
    try:
        solver = CTFSolver(ctf_url=ctf_url)
    except Exception as e:
        console.print(f"[red]CTF Solver 초기화 실패: {e}[/red]")
        return

    # 메뉴
    while True:
        console.print("\n" + "="*60)
        table = Table(title="CTF Solver 메뉴", show_header=False)
        table.add_column("Option", style="cyan")
        table.add_column("Description")

        table.add_row("1", "🚀 자동 해결 (모든 전략 시도)")
        table.add_row("2", "⚡ Transfer Attack (PGD)")
        table.add_row("3", "⚡ Transfer Attack (C&W)")
        table.add_row("4", "🎯 Black-box Attack (HopSkipJump)")
        table.add_row("5", "🎯 Black-box Attack (SimBA)")
        table.add_row("6", "🎯 Black-box Attack (Square)")
        table.add_row("7", "📊 이미지 테스트 (CTF 서버에 전송)")
        table.add_row("8", "💾 Goldfish 샘플 다운로드")
        table.add_row("b", "◀ 뒤로 가기")

        console.print(table)

        choice = console.input("\n선택: ").strip().lower()

        if choice == 'b':
            break
        elif choice == '1':
            # 자동 해결
            goldfish_path = _get_goldfish_image_input()
            if goldfish_path:
                result = solver.solve(goldfish_image_path=goldfish_path)
                _display_result(result)
        elif choice == '2':
            # Transfer Attack (PGD)
            goldfish_path = _get_goldfish_image_input()
            if goldfish_path:
                result = solver.solve_with_transfer_attack(
                    goldfish_path,
                    attack_type='pgd'
                )
                _display_result(result)
        elif choice == '3':
            # Transfer Attack (C&W)
            goldfish_path = _get_goldfish_image_input()
            if goldfish_path:
                result = solver.solve_with_transfer_attack(
                    goldfish_path,
                    attack_type='cw'
                )
                _display_result(result)
        elif choice == '4':
            # Black-box (HopSkipJump)
            goldfish_path = _get_goldfish_image_input()
            if goldfish_path:
                max_iter = console.input("[cyan]최대 반복 횟수[/cyan] [dim](기본: 100)[/dim]: ").strip()
                max_iter = int(max_iter) if max_iter else 100

                result = solver.solve_with_blackbox_attack(
                    goldfish_path,
                    attack_type='hopskipjump',
                    max_iter=max_iter
                )
                _display_result(result)
        elif choice == '5':
            # Black-box (SimBA)
            goldfish_path = _get_goldfish_image_input()
            if goldfish_path:
                max_iter = console.input("[cyan]최대 반복 횟수[/cyan] [dim](기본: 100)[/dim]: ").strip()
                max_iter = int(max_iter) if max_iter else 100

                result = solver.solve_with_blackbox_attack(
                    goldfish_path,
                    attack_type='simba',
                    max_iter=max_iter
                )
                _display_result(result)
        elif choice == '6':
            # Black-box (Square)
            goldfish_path = _get_goldfish_image_input()
            if goldfish_path:
                max_iter = console.input("[cyan]최대 반복 횟수[/cyan] [dim](기본: 100)[/dim]: ").strip()
                max_iter = int(max_iter) if max_iter else 100

                result = solver.solve_with_blackbox_attack(
                    goldfish_path,
                    attack_type='square',
                    max_iter=max_iter
                )
                _display_result(result)
        elif choice == '7':
            # 이미지 테스트
            image_path = console.input("\n[cyan]테스트할 이미지 경로[/cyan]: ").strip()
            if image_path and Path(image_path).exists():
                result = solver.check_ctf_response(image_path)

                console.print(f"\n[bold]CTF 서버 응답:[/bold]")
                console.print(f"  Success: {result.get('success', False)}")
                console.print(f"  Similarity: {result.get('similarity', 'N/A')}")
                console.print(f"  Access: {result.get('access', 'N/A')}")
                console.print(f"  Text: {result.get('text', 'N/A')}")
            else:
                console.print("[red]이미지를 찾을 수 없습니다.[/red]")
        elif choice == '8':
            # Goldfish 샘플 다운로드
            try:
                goldfish_path = solver._get_goldfish_image()
                console.print(f"[green]✓ Goldfish 이미지 다운로드 완료: {goldfish_path}[/green]")
            except Exception as e:
                console.print(f"[red]다운로드 실패: {e}[/red]")
        else:
            console.print("[yellow]잘못된 선택입니다.[/yellow]")


def _get_goldfish_image_input() -> str:
    """Goldfish 이미지 경로 입력"""
    goldfish_path = console.input("\n[cyan]Goldfish 이미지 경로[/cyan] [dim](Enter: 샘플 다운로드)[/dim]: ").strip()

    if not goldfish_path:
        # 샘플 다운로드
        try:
            from adversarial.ctf_solver import CTFSolver
            solver_temp = CTFSolver()
            goldfish_path = solver_temp._get_goldfish_image()
            console.print(f"[green]✓ 샘플 다운로드 완료: {goldfish_path}[/green]")
        except Exception as e:
            console.print(f"[red]다운로드 실패: {e}[/red]")
            return None

    if not Path(goldfish_path).exists():
        console.print(f"[red]이미지를 찾을 수 없습니다: {goldfish_path}[/red]")
        return None

    return goldfish_path


def _display_result(result: dict):
    """결과 표시"""
    console.print("\n" + "="*60)

    if result.get('success', False):
        console.print(Panel.fit(
            f"[bold green]🎉 CTF SOLVED![/bold green]\n\n"
            f"Strategy: {result.get('strategy', 'N/A')}\n"
            f"Image: {result.get('best_image', 'N/A')}\n"
            f"Confidence: {result.get('best_confidence', 'N/A')}%\n"
            f"Attack Type: {result.get('attack_type', 'N/A')}",
            title="Success"
        ))
    else:
        console.print(Panel.fit(
            f"[bold red]❌ Failed to solve CTF[/bold red]\n\n"
            f"Strategy: {result.get('strategy', 'N/A')}\n"
            f"Best Confidence: {result.get('best_confidence', 0.0)}%\n"
            f"Try different strategies or parameters",
            title="Failed"
        ))

    console.print("="*60)
