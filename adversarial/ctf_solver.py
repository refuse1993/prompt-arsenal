"""
Adversarial ML CTF Solver
Transfer Attack + Black-box Attack 조합으로 CTF 해결

GitHub: https://github.com/arturmiller/adversarial_ml_ctf
목표: ResNet50 모델을 속여서 goldfish(class 1)로 인식시키기 (80% 이상)
"""

import numpy as np
from PIL import Image
import requests
import base64
from io import BytesIO
from typing import Dict, List, Optional, Tuple
import time
from pathlib import Path

# 선택적 임포트
try:
    from .foolbox_attacks import FoolboxAttacker
    from .art_attacks import ARTAttacker
    ATTACKERS_AVAILABLE = True
except ImportError:
    ATTACKERS_AVAILABLE = False


class CTFSolver:
    """
    Adversarial ML CTF 자동 해결 시스템

    전략:
    1. Transfer Attack (화이트박스 공격 → 블랙박스 전이)
    2. Black-box Attack (실패 시 폴백)
    3. Confidence 최적화 (80% 이상 달성)
    """

    def __init__(self, ctf_url: str = "http://localhost:5000"):
        """
        Args:
            ctf_url: CTF 서버 URL
        """
        if not ATTACKERS_AVAILABLE:
            raise ImportError("Foolbox and ART not available")

        self.ctf_url = ctf_url
        self.target_label = 1  # ImageNet class 1 = goldfish
        self.target_confidence = 80.0  # 성공 기준

        # 공격 도구 초기화
        self.foolbox = FoolboxAttacker(target_model='resnet50', device='cpu')
        self.art = ARTAttacker(target_model='resnet50', device='cpu')

        print(f"🎯 CTF Solver initialized")
        print(f"   Target: {ctf_url}")
        print(f"   Goal: Classify as goldfish (class {self.target_label}) with {self.target_confidence}% confidence")

    def check_ctf_response(self, image_path: str) -> Dict:
        """
        CTF 서버에 이미지 전송 및 응답 확인

        Args:
            image_path: 전송할 이미지 경로

        Returns:
            {
                'success': bool,
                'similarity': float,
                'access': str,
                'text': str,
                'redirected': bool
            }
        """
        # 이미지를 base64로 인코딩
        img = Image.open(image_path).convert('RGB')
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # CTF 서버로 POST 요청
        data_url = f"data:image/png;base64,{img_base64}"

        try:
            response = requests.post(
                f"{self.ctf_url}/check",
                data=data_url,
                headers={'Content-Type': 'application/x-www-form-urlencoded'},
                allow_redirects=False
            )

            # Redirect 확인 (80% 이상 성공)
            if response.status_code == 302:
                print(f"✅ SUCCESS! Redirected to hidden page (confidence > 80%)")
                return {
                    'success': True,
                    'similarity': None,
                    'access': 'granted',
                    'text': 'Authenticated! (Redirected)',
                    'redirected': True,
                    'status_code': response.status_code
                }

            # JSON 응답 파싱
            result = response.json()
            result['redirected'] = False
            result['status_code'] = response.status_code

            if result.get('similarity', 0) > self.target_confidence:
                print(f"✅ SUCCESS! Confidence: {result['similarity']:.2f}%")
                result['success'] = True
            else:
                print(f"❌ Failed. Confidence: {result.get('similarity', 0):.2f}%")
                result['success'] = False

            return result

        except Exception as e:
            print(f"⚠️ Error checking CTF response: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def solve_with_transfer_attack(self, goldfish_image_path: str,
                                   attack_type: str = 'pgd',
                                   epsilon_range: List[float] = None) -> Dict:
        """
        Transfer Attack으로 CTF 해결 시도

        로컬 ResNet50으로 adversarial 생성 → CTF 서버에 전송 (전이)

        Args:
            goldfish_image_path: 실제 goldfish 이미지 경로
            attack_type: 'pgd', 'cw', 'fgsm', 'deepfool'
            epsilon_range: 시도할 epsilon 값 리스트

        Returns:
            {
                'success': bool,
                'best_image': str,
                'best_confidence': float,
                'attack_params': dict
            }
        """
        if epsilon_range is None:
            epsilon_range = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1]

        print(f"\n🎯 Strategy 1: Transfer Attack ({attack_type.upper()})")
        print(f"   Testing {len(epsilon_range)} epsilon values...")

        best_result = None
        best_confidence = 0.0
        best_image_path = None

        for epsilon in epsilon_range:
            print(f"\n📊 Testing epsilon={epsilon}")

            # Foolbox로 타겟팅 공격 생성
            output_path = f"media/ctf_transfer_{attack_type}_eps{epsilon}.png"

            try:
                if attack_type == 'pgd':
                    result = self.foolbox.pgd_attack(
                        image_path=goldfish_image_path,
                        epsilon=epsilon,
                        steps=40,
                        target_label=self.target_label,
                        output_path=output_path
                    )
                elif attack_type == 'cw':
                    result = self.foolbox.cw_attack(
                        image_path=goldfish_image_path,
                        confidence=0.0,
                        steps=100,
                        target_label=self.target_label,
                        output_path=output_path
                    )
                elif attack_type == 'fgsm':
                    result = self.foolbox.fgsm_attack(
                        image_path=goldfish_image_path,
                        epsilon=epsilon,
                        target_label=self.target_label,
                        output_path=output_path
                    )
                elif attack_type == 'deepfool':
                    result = self.foolbox.deepfool_attack(
                        image_path=goldfish_image_path,
                        steps=50,
                        target_label=self.target_label,
                        output_path=output_path
                    )
                else:
                    print(f"Unknown attack type: {attack_type}")
                    continue

                # 생성된 이미지로 CTF 서버 테스트
                ctf_result = self.check_ctf_response(output_path)

                if ctf_result.get('success', False):
                    print(f"🎉 CTF SOLVED with Transfer Attack!")
                    return {
                        'success': True,
                        'best_image': output_path,
                        'best_confidence': ctf_result.get('similarity', 100.0),
                        'attack_type': attack_type,
                        'attack_params': {'epsilon': epsilon},
                        'strategy': 'transfer_attack'
                    }

                # Best 결과 추적
                similarity = ctf_result.get('similarity', 0.0)
                if similarity and similarity > best_confidence:
                    best_confidence = similarity
                    best_image_path = output_path
                    best_result = ctf_result

            except Exception as e:
                print(f"⚠️ Attack failed with epsilon={epsilon}: {e}")
                continue

        # 최고 결과 반환
        if best_image_path:
            return {
                'success': (best_confidence >= self.target_confidence),
                'best_image': best_image_path,
                'best_confidence': best_confidence,
                'attack_type': attack_type,
                'strategy': 'transfer_attack'
            }
        else:
            return {'success': False, 'strategy': 'transfer_attack'}

    def solve_with_blackbox_attack(self, goldfish_image_path: str,
                                   attack_type: str = 'hopskipjump',
                                   max_iter: int = 100) -> Dict:
        """
        Black-box Attack으로 CTF 해결 시도

        CTF 서버를 블랙박스로 사용 (쿼리 기반)

        Args:
            goldfish_image_path: 시작 이미지
            attack_type: 'hopskipjump', 'simba', 'square', 'boundary'
            max_iter: 최대 반복 횟수

        Returns:
            {
                'success': bool,
                'best_image': str,
                'best_confidence': float
            }
        """
        print(f"\n🎯 Strategy 2: Black-box Attack ({attack_type.upper()})")

        output_path = f"media/ctf_blackbox_{attack_type}.png"

        try:
            if attack_type == 'hopskipjump':
                result = self.art.hopskipjump_attack(
                    image_path=goldfish_image_path,
                    target_label=self.target_label,
                    max_iter=max_iter,
                    max_eval=10000,
                    output_path=output_path
                )
            elif attack_type == 'simba':
                result = self.art.simba_attack(
                    image_path=goldfish_image_path,
                    target_label=self.target_label,
                    max_iter=max_iter * 30,  # SimBA는 더 많은 쿼리 필요
                    epsilon=0.1,
                    output_path=output_path
                )
            elif attack_type == 'square':
                result = self.art.square_attack(
                    image_path=goldfish_image_path,
                    max_iter=max_iter * 50,
                    epsilon=0.05,
                    output_path=output_path
                )
            elif attack_type == 'boundary':
                result = self.art.boundary_attack(
                    image_path=goldfish_image_path,
                    target_label=self.target_label,
                    max_iter=max_iter * 50,
                    output_path=output_path
                )
            else:
                print(f"Unknown black-box attack: {attack_type}")
                return {'success': False}

            # CTF 서버 테스트
            ctf_result = self.check_ctf_response(output_path)

            if ctf_result.get('success', False):
                print(f"🎉 CTF SOLVED with Black-box Attack!")
                return {
                    'success': True,
                    'best_image': output_path,
                    'best_confidence': ctf_result.get('similarity', 100.0),
                    'attack_type': attack_type,
                    'strategy': 'blackbox_attack'
                }
            else:
                return {
                    'success': False,
                    'best_image': output_path,
                    'best_confidence': ctf_result.get('similarity', 0.0),
                    'attack_type': attack_type,
                    'strategy': 'blackbox_attack'
                }

        except Exception as e:
            print(f"⚠️ Black-box attack failed: {e}")
            return {'success': False, 'error': str(e)}

    def solve(self, goldfish_image_path: str = None) -> Dict:
        """
        CTF 자동 해결 (모든 전략 시도)

        Args:
            goldfish_image_path: Goldfish 이미지 경로 (없으면 샘플 다운로드)

        Returns:
            {
                'success': bool,
                'strategy': str,
                'image_path': str,
                'confidence': float
            }
        """
        print(f"\n{'='*60}")
        print(f"🚀 Starting Adversarial ML CTF Auto-Solver")
        print(f"{'='*60}")

        # Goldfish 이미지 확인
        if goldfish_image_path is None:
            goldfish_image_path = self._get_goldfish_image()

        # Strategy 1: Transfer Attack (PGD)
        print(f"\n{'='*60}")
        result = self.solve_with_transfer_attack(
            goldfish_image_path,
            attack_type='pgd',
            epsilon_range=[0.01, 0.02, 0.03, 0.05, 0.07, 0.1]
        )

        if result.get('success', False):
            print(f"\n🎉 CTF SOLVED!")
            print(f"   Strategy: Transfer Attack (PGD)")
            print(f"   Image: {result['best_image']}")
            print(f"   Confidence: {result.get('best_confidence', 'N/A')}")
            return result

        # Strategy 2: Transfer Attack (C&W)
        print(f"\n{'='*60}")
        result = self.solve_with_transfer_attack(
            goldfish_image_path,
            attack_type='cw'
        )

        if result.get('success', False):
            print(f"\n🎉 CTF SOLVED!")
            print(f"   Strategy: Transfer Attack (C&W)")
            return result

        # Strategy 3: Black-box (HopSkipJump)
        print(f"\n{'='*60}")
        result = self.solve_with_blackbox_attack(
            goldfish_image_path,
            attack_type='hopskipjump',
            max_iter=100
        )

        if result.get('success', False):
            print(f"\n🎉 CTF SOLVED!")
            print(f"   Strategy: Black-box (HopSkipJump)")
            return result

        # 실패
        print(f"\n{'='*60}")
        print(f"❌ Failed to solve CTF with all strategies")
        print(f"   Best confidence achieved: {result.get('best_confidence', 0.0):.2f}%")
        print(f"{'='*60}")

        return {
            'success': False,
            'best_result': result
        }

    def _get_goldfish_image(self) -> str:
        """
        샘플 goldfish 이미지 다운로드 또는 생성

        Returns:
            goldfish 이미지 경로
        """
        goldfish_path = "samples/images/goldfish.jpg"

        # 이미 존재하면 반환
        if Path(goldfish_path).exists():
            return goldfish_path

        # 샘플 다운로드 시도
        print(f"⬇️ Downloading sample goldfish image...")
        try:
            url = "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n01443537_goldfish.JPEG"
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            Path("samples/images").mkdir(parents=True, exist_ok=True)
            with open(goldfish_path, 'wb') as f:
                f.write(response.content)

            print(f"✓ Downloaded to {goldfish_path}")
            return goldfish_path

        except Exception as e:
            print(f"⚠️ Failed to download: {e}")
            print(f"Please provide a goldfish image manually")
            raise FileNotFoundError("Goldfish image not found")


# ========================================
# CLI 사용
# ========================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Adversarial ML CTF Solver")
    parser.add_argument('--url', type=str, default='http://localhost:5000',
                       help='CTF server URL')
    parser.add_argument('--image', type=str, default=None,
                       help='Goldfish image path')
    parser.add_argument('--strategy', type=str, choices=['transfer', 'blackbox', 'auto'],
                       default='auto', help='Attack strategy')

    args = parser.parse_args()

    solver = CTFSolver(ctf_url=args.url)

    if args.strategy == 'auto':
        result = solver.solve(goldfish_image_path=args.image)
    elif args.strategy == 'transfer':
        result = solver.solve_with_transfer_attack(args.image or solver._get_goldfish_image())
    elif args.strategy == 'blackbox':
        result = solver.solve_with_blackbox_attack(args.image or solver._get_goldfish_image())

    if result.get('success'):
        print(f"\n✅ CTF Solved Successfully!")
    else:
        print(f"\n❌ Failed to solve CTF")
