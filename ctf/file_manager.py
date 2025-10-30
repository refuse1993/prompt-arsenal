"""
CTF Challenge File Manager
파일 다운로드 및 관리 시스템
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from urllib.parse import urlparse, urljoin

console = Console()


class ChallengeFileManager:
    """CTF 챌린지 파일 관리자"""

    def __init__(self, base_dir: Path = None):
        """
        Args:
            base_dir: 파일 저장 기본 디렉토리
        """
        if base_dir is None:
            # 프로젝트 루트의 ctf_files 디렉토리
            project_root = Path(__file__).parent.parent
            base_dir = project_root / "ctf_files"

        self.base_dir = base_dir
        self.base_dir.mkdir(exist_ok=True)

    async def detect_file_links(self, page, challenge_title: str = "") -> List[str]:
        """페이지에서 파일 다운로드 링크 감지"""
        try:
            console.print(f"[dim]  파일 링크 감지 중...[/dim]")

            # JavaScript로 다운로드 링크 찾기
            file_links = await page.evaluate('''() => {
                const links = [];
                const fileExtensions = [
                    '.zip', '.tar', '.tar.gz', '.tgz', '.rar', '.7z', '.gz',
                    '.exe', '.dll', '.so', '.elf', '.bin', '.out',
                    '.pcap', '.pcapng', '.cap',
                    '.pdf', '.doc', '.docx', '.txt',
                    '.png', '.jpg', '.jpeg', '.gif',
                    '.py', '.js', '.c', '.cpp', '.sh', '.php',
                    '.json', '.xml', '.csv', '.db', '.sqlite'
                ];

                // 모든 링크 검사
                document.querySelectorAll('a[href]').forEach(a => {
                    const href = a.href;
                    const text = a.innerText?.toLowerCase() || '';

                    // 파일 확장자 체크 (URL 끝에 확장자가 있어야 함)
                    const hasFileExt = fileExtensions.some(ext => {
                        const urlLower = href.toLowerCase();
                        // 확장자로 끝나거나, 확장자 뒤에 ? 또는 # 만 있는 경우
                        return urlLower.endsWith(ext) ||
                               urlLower.includes(ext + '?') ||
                               urlLower.includes(ext + '#');
                    });

                    // download 속성이 있는 경우만
                    const hasDownloadAttr = a.hasAttribute('download');

                    // 파일 확장자가 있거나 download 속성이 있는 경우만
                    if (hasFileExt || hasDownloadAttr) {
                        links.push({
                            url: href,
                            text: a.innerText?.trim() || 'Unnamed',
                            isDownloadAttr: hasDownloadAttr
                        });
                    }
                });

                // 버튼 검사 (onclick 이벤트로 다운로드하는 경우)
                document.querySelectorAll('button, div[role="button"]').forEach(btn => {
                    const text = btn.innerText?.toLowerCase() || '';
                    const onclick = btn.getAttribute('onclick') || '';

                    if ((text.includes('download') || text.includes('file')) && onclick) {
                        // onclick에서 URL 추출 시도
                        const urlMatch = onclick.match(/['"]([^'"]+\.(zip|tar|gz|exe|pcap|txt|pdf)[^'"]*)['"]/);
                        if (urlMatch) {
                            links.push({
                                url: urlMatch[1],
                                text: btn.innerText?.trim() || 'Button Download',
                                isDownloadAttr: false
                            });
                        }
                    }
                });

                return links;
            }''')

            if not file_links:
                return []

            # URL 정규화
            base_url = page.url
            normalized_urls = []

            for link in file_links:
                url = link['url']

                # 상대 경로 처리
                if not url.startswith('http'):
                    url = urljoin(base_url, url)

                # 중복 제거
                if url not in normalized_urls:
                    normalized_urls.append(url)
                    console.print(f"[dim]    • {link['text'][:50]}: {url}[/dim]")

            return normalized_urls

        except Exception as e:
            console.print(f"[yellow]파일 링크 감지 오류: {str(e)}[/yellow]")
            return []

    async def download_file(
        self,
        page,  # Playwright page
        url: str,
        challenge_id: int,
        challenge_title: str = ""
    ) -> Optional[Dict]:
        """
        파일 다운로드 (Playwright 사용, 브라우저 세션 유지)

        Returns:
            {
                'file_path': Path,
                'filename': str,
                'file_size': int,
                'download_url': str,
                'downloaded_at': datetime
            }
        """
        try:
            # 저장 디렉토리: ctf_files/{challenge_id}/
            save_dir = self.base_dir / str(challenge_id)
            save_dir.mkdir(exist_ok=True, parents=True)

            console.print(f"[cyan]  ⬇️  다운로드 중: {url}[/cyan]")

            # Playwright download 처리
            async with page.expect_download(timeout=60000) as download_info:
                # 링크 클릭 또는 navigate
                try:
                    # 먼저 페이지에서 링크 클릭 시도
                    link = await page.query_selector(f'a[href*="{url.split("/")[-1]}"]')
                    if link:
                        await link.click()
                    else:
                        # 직접 navigate
                        await page.goto(url)
                except:
                    # 실패하면 navigate
                    await page.goto(url)

            download = await download_info.value

            # 파일명 결정
            suggested_filename = download.suggested_filename
            if not suggested_filename or suggested_filename == 'download':
                # URL에서 추출
                parsed = urlparse(url)
                suggested_filename = Path(parsed.path).name or f"file_{challenge_id}"

            # 파일 저장
            file_path = save_dir / suggested_filename
            await download.save_as(file_path)

            file_size = file_path.stat().st_size
            console.print(f"[green]  ✓ 다운로드 완료: {suggested_filename} ({file_size:,} bytes)[/green]")

            return {
                'file_path': file_path,
                'filename': suggested_filename,
                'file_size': file_size,
                'download_url': url,
                'downloaded_at': datetime.now().isoformat()
            }

        except asyncio.TimeoutError:
            console.print(f"[yellow]  ⚠️  다운로드 타임아웃: {url}[/yellow]")
            return None
        except Exception as e:
            console.print(f"[red]  ❌ 다운로드 실패: {str(e)[:100]}[/red]")
            return None

    async def download_all_files(
        self,
        page,
        challenge_id: int,
        challenge_title: str = ""
    ) -> List[Dict]:
        """현재 페이지의 모든 파일 다운로드"""

        # 파일 링크 감지
        file_urls = await self.detect_file_links(page, challenge_title)

        if not file_urls:
            console.print(f"[dim]  첨부파일 없음[/dim]")
            return []

        console.print(f"[cyan]  📎 {len(file_urls)}개 파일 발견[/cyan]")

        # 다운로드
        downloaded_files = []

        for url in file_urls:
            file_info = await self.download_file(
                page,
                url,
                challenge_id,
                challenge_title
            )

            if file_info:
                downloaded_files.append(file_info)

            # Rate limiting
            await asyncio.sleep(1)

        return downloaded_files

    def get_challenge_dir(self, challenge_id: int) -> Path:
        """챌린지 파일 디렉토리 경로"""
        return self.base_dir / str(challenge_id)

    def list_files(self, challenge_id: int) -> List[Path]:
        """챌린지의 모든 파일 나열"""
        challenge_dir = self.get_challenge_dir(challenge_id)

        if not challenge_dir.exists():
            return []

        return list(challenge_dir.rglob('*'))

    async def cleanup_old_files(self, days: int = 30):
        """오래된 파일 정리"""
        import time
        from datetime import timedelta

        cutoff_time = time.time() - (days * 24 * 60 * 60)
        removed_count = 0

        for challenge_dir in self.base_dir.iterdir():
            if not challenge_dir.is_dir():
                continue

            # 디렉토리 수정 시간 체크
            if challenge_dir.stat().st_mtime < cutoff_time:
                try:
                    import shutil
                    shutil.rmtree(challenge_dir)
                    removed_count += 1
                    console.print(f"[dim]정리: {challenge_dir.name}[/dim]")
                except Exception as e:
                    console.print(f"[yellow]정리 실패 ({challenge_dir.name}): {str(e)}[/yellow]")

        if removed_count > 0:
            console.print(f"[green]✓ {removed_count}개 챌린지 파일 정리 완료 ({days}일 이상)[/green]")
        else:
            console.print(f"[dim]정리할 파일 없음[/dim]")


# 테스트용
async def test_file_manager():
    """파일 관리자 테스트"""
    manager = ChallengeFileManager()

    console.print(f"[cyan]파일 저장 경로: {manager.base_dir}[/cyan]")

    # 챌린지 디렉토리 생성 테스트
    test_challenge_id = 999
    test_dir = manager.get_challenge_dir(test_challenge_id)
    test_dir.mkdir(exist_ok=True)

    console.print(f"[green]✓ 테스트 디렉토리 생성: {test_dir}[/green]")

    # 파일 나열 테스트
    files = manager.list_files(test_challenge_id)
    console.print(f"[cyan]파일 개수: {len(files)}[/cyan]")


if __name__ == "__main__":
    asyncio.run(test_file_manager())
