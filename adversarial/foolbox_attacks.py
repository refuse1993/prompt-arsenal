"""
Foolbox 기반 고급 이미지 Adversarial 공격
그래디언트 기반 화이트박스 공격 (FGSM, PGD, C&W, DeepFool 등)
"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import Optional, Dict, List
import warnings

# Foolbox 선택적 임포트
try:
    import foolbox as fb
    from foolbox import PyTorchModel
    from foolbox.attacks import (
        FGSM, LinfPGD, L2PGD, L2CarliniWagnerAttack,
        LinfDeepFoolAttack, L2DeepFoolAttack,
        BoundaryAttack, GaussianBlurAttack,
        SaltAndPepperNoiseAttack
    )
    FOOLBOX_AVAILABLE = True
except ImportError:
    FOOLBOX_AVAILABLE = False
    warnings.warn("Foolbox not installed. Install with: pip install foolbox")


class FoolboxAttacker:
    """
    Foolbox 기반 고급 Adversarial 이미지 공격

    지원 공격:
    - FGSM (Fast Gradient Sign Method): 빠른 단일 스텝
    - PGD (Projected Gradient Descent): 강력한 반복 공격
    - C&W (Carlini & Wagner): 최소 섭동 최적화
    - DeepFool: 결정 경계 최소화
    - Boundary Attack: 블랙박스 공격
    - Noise Attacks: Gaussian, Salt & Pepper
    """

    def __init__(self, target_model: str = 'resnet50', device: str = 'cpu'):
        """
        Args:
            target_model: 타겟 모델 (resnet50, vgg16, densenet121 등)
            device: 'cpu' 또는 'cuda'
        """
        if not FOOLBOX_AVAILABLE:
            raise ImportError("Foolbox not installed. Run: pip install foolbox")

        self.device = device
        self.target_model_name = target_model

        # 모델 로드
        self.model = self._load_model(target_model)
        self.model.eval()

        # Foolbox 모델 래핑
        preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
        self.fmodel = PyTorchModel(self.model, bounds=(0, 1), preprocessing=preprocessing)

        # 이미지 전처리
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

        # 공격 알고리즘 초기화
        self.attacks = {
            'fgsm': FGSM(),
            'pgd_linf': LinfPGD(),
            'pgd_l2': L2PGD(),
            'cw': L2CarliniWagnerAttack(steps=100),
            'deepfool_linf': LinfDeepFoolAttack(),
            'deepfool_l2': L2DeepFoolAttack(),
            'boundary': BoundaryAttack(),
            'gaussian_blur': GaussianBlurAttack(),
            'salt_pepper': SaltAndPepperNoiseAttack()
        }

    def _load_model(self, model_name: str):
        """사전 학습된 모델 로드"""
        if model_name not in models.__dict__:
            raise ValueError(f"Model {model_name} not found in torchvision.models")

        model = models.__dict__[model_name](pretrained=True)
        model.to(self.device)
        return model

    def _load_image(self, image_path: str) -> torch.Tensor:
        """이미지 로드 및 전처리"""
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        return img_tensor

    def _get_prediction(self, image_tensor: torch.Tensor) -> int:
        """모델 예측"""
        with torch.no_grad():
            output = self.model(image_tensor)
            pred = output.argmax(dim=1).item()
        return pred

    def fgsm_attack(self, image_path: str, epsilon: float = 0.03,
                   output_path: str = None) -> Dict:
        """
        FGSM (Fast Gradient Sign Method) 공격

        Args:
            image_path: 원본 이미지 경로
            epsilon: 섭동 크기 (L∞ norm)
            output_path: 출력 경로 (None이면 PIL Image 반환)

        Returns:
            {
                'adversarial_image': PIL Image,
                'original_pred': int,
                'adversarial_pred': int,
                'success': bool,
                'perturbation_norm': float,
                'epsilon': float
            }
        """
        return self._run_attack('fgsm', image_path, epsilon, output_path)

    def pgd_attack(self, image_path: str, epsilon: float = 0.03,
                  steps: int = 40, output_path: str = None,
                  norm: str = 'linf') -> Dict:
        """
        PGD (Projected Gradient Descent) 공격

        Args:
            image_path: 원본 이미지
            epsilon: 섭동 크기
            steps: 반복 횟수 (기본 40)
            output_path: 출력 경로
            norm: 'linf' 또는 'l2'
        """
        attack_name = 'pgd_linf' if norm == 'linf' else 'pgd_l2'

        # PGD 파라미터 설정
        if attack_name == 'pgd_linf':
            self.attacks[attack_name] = LinfPGD(steps=steps)
        else:
            self.attacks[attack_name] = L2PGD(steps=steps)

        return self._run_attack(attack_name, image_path, epsilon, output_path)

    def cw_attack(self, image_path: str, confidence: float = 0.0,
                 steps: int = 100, output_path: str = None) -> Dict:
        """
        C&W (Carlini & Wagner) 공격 - 최소 섭동 최적화

        Args:
            image_path: 원본 이미지
            confidence: 신뢰도 (0.0 = 낮은 신뢰도)
            steps: 최적화 스텝
        """
        # C&W 재설정
        self.attacks['cw'] = L2CarliniWagnerAttack(
            steps=steps,
            confidence=confidence
        )

        return self._run_attack('cw', image_path, epsilon=None, output_path=output_path)

    def deepfool_attack(self, image_path: str, steps: int = 50,
                       output_path: str = None, norm: str = 'l2') -> Dict:
        """
        DeepFool 공격 - 결정 경계 최소 섭동

        Args:
            steps: 최대 반복 횟수
            norm: 'linf' 또는 'l2'
        """
        attack_name = 'deepfool_linf' if norm == 'linf' else 'deepfool_l2'

        if attack_name == 'deepfool_linf':
            self.attacks[attack_name] = LinfDeepFoolAttack(steps=steps)
        else:
            self.attacks[attack_name] = L2DeepFoolAttack(steps=steps)

        return self._run_attack(attack_name, image_path, epsilon=None, output_path=output_path)

    def boundary_attack(self, image_path: str, steps: int = 1000,
                       output_path: str = None) -> Dict:
        """
        Boundary Attack - 블랙박스 공격 (그래디언트 불필요)

        Args:
            steps: 최대 스텝 (많을수록 정확하지만 느림)
        """
        self.attacks['boundary'] = BoundaryAttack(steps=steps)
        return self._run_attack('boundary', image_path, epsilon=None, output_path=output_path)

    def gaussian_noise_attack(self, image_path: str, std: float = 0.1,
                             output_path: str = None) -> Dict:
        """Gaussian 노이즈 공격"""
        return self._run_attack('gaussian_blur', image_path, std, output_path)

    def salt_pepper_attack(self, image_path: str, amount: float = 0.05,
                          output_path: str = None) -> Dict:
        """Salt & Pepper 노이즈 공격"""
        return self._run_attack('salt_pepper', image_path, amount, output_path)

    def _run_attack(self, attack_name: str, image_path: str,
                   epsilon: Optional[float], output_path: Optional[str] = None) -> Dict:
        """공격 실행 공통 로직"""
        # 이미지 로드
        image = self._load_image(image_path)

        # 원본 예측
        original_pred = self._get_prediction(image)

        # 공격 실행
        attack = self.attacks[attack_name]

        if epsilon is not None:
            # Epsilon 기반 공격 (FGSM, PGD)
            _, adversarial, success = attack(self.fmodel, image,
                                            torch.tensor([original_pred]).to(self.device),
                                            epsilons=[epsilon])
            adversarial = adversarial[0]
        else:
            # Epsilon 없는 공격 (C&W, DeepFool, Boundary)
            _, adversarial, success = attack(self.fmodel, image,
                                            torch.tensor([original_pred]).to(self.device),
                                            epsilons=None)

        # Adversarial 이미지 예측
        adversarial_pred = self._get_prediction(adversarial)

        # 섭동 계산
        perturbation = (adversarial - image).squeeze().cpu().detach().numpy()
        l2_norm = np.linalg.norm(perturbation)
        linf_norm = np.abs(perturbation).max()

        # PIL 이미지 변환
        adv_img_pil = self._tensor_to_pil(adversarial)

        # 저장
        if output_path:
            adv_img_pil.save(output_path)

        return {
            'adversarial_image': adv_img_pil,
            'original_pred': original_pred,
            'adversarial_pred': adversarial_pred,
            'success': bool(success.item()) if isinstance(success, torch.Tensor) else bool(success),
            'l2_norm': float(l2_norm),
            'linf_norm': float(linf_norm),
            'epsilon': epsilon,
            'attack_type': attack_name
        }

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Tensor를 PIL Image로 변환"""
        img = tensor.squeeze().cpu().detach().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(img)

    def batch_attack(self, image_path: str,
                    attack_types: List[str] = ['fgsm', 'pgd_linf', 'cw', 'deepfool_l2'],
                    epsilon: float = 0.03,
                    output_dir: str = 'media/foolbox_attacks') -> Dict[str, Dict]:
        """
        여러 공격을 한 번에 실행

        Args:
            image_path: 원본 이미지
            attack_types: 공격 유형 리스트
            epsilon: FGSM/PGD용 epsilon
            output_dir: 출력 디렉토리

        Returns:
            {attack_name: result_dict}
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        results = {}

        for attack_type in attack_types:
            output_path = f"{output_dir}/{attack_type}.png"

            try:
                if attack_type in ['fgsm', 'pgd_linf', 'pgd_l2']:
                    result = self._run_attack(attack_type, image_path, epsilon, output_path)
                else:
                    result = self._run_attack(attack_type, image_path, None, output_path)

                results[attack_type] = result
                print(f"✓ {attack_type}: Success={result['success']}, L2={result['l2_norm']:.4f}")

            except Exception as e:
                print(f"✗ {attack_type} failed: {e}")
                results[attack_type] = {'error': str(e)}

        return results


# ========================================
# 고수준 API
# ========================================

def generate_foolbox_attack(image_path: str, attack_type: str = 'pgd_linf',
                            epsilon: float = 0.03, output_path: str = None,
                            **kwargs) -> Dict:
    """
    간편한 Foolbox 공격 생성

    Args:
        image_path: 원본 이미지
        attack_type: 공격 유형 ('fgsm', 'pgd_linf', 'cw', 'deepfool_l2' 등)
        epsilon: 섭동 크기
        output_path: 출력 경로
        **kwargs: 추가 파라미터

    Returns:
        공격 결과 딕셔너리
    """
    if not FOOLBOX_AVAILABLE:
        return {
            'error': 'Foolbox not installed',
            'message': 'Install with: pip install foolbox torch torchvision'
        }

    attacker = FoolboxAttacker()

    if attack_type == 'fgsm':
        return attacker.fgsm_attack(image_path, epsilon, output_path)
    elif attack_type == 'pgd_linf':
        steps = kwargs.get('steps', 40)
        return attacker.pgd_attack(image_path, epsilon, steps, output_path, 'linf')
    elif attack_type == 'pgd_l2':
        steps = kwargs.get('steps', 40)
        return attacker.pgd_attack(image_path, epsilon, steps, output_path, 'l2')
    elif attack_type == 'cw':
        confidence = kwargs.get('confidence', 0.0)
        steps = kwargs.get('steps', 100)
        return attacker.cw_attack(image_path, confidence, steps, output_path)
    elif attack_type in ['deepfool_linf', 'deepfool_l2']:
        steps = kwargs.get('steps', 50)
        norm = attack_type.split('_')[1]
        return attacker.deepfool_attack(image_path, steps, output_path, norm)
    elif attack_type == 'boundary':
        steps = kwargs.get('steps', 1000)
        return attacker.boundary_attack(image_path, steps, output_path)
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")


if __name__ == "__main__":
    # 테스트
    if FOOLBOX_AVAILABLE:
        print("Testing Foolbox attacks...")

        # 샘플 이미지로 테스트 (실제 경로로 변경 필요)
        test_image = "media/test_image.png"

        attacker = FoolboxAttacker()

        # Batch attack
        results = attacker.batch_attack(
            test_image,
            attack_types=['fgsm', 'pgd_linf', 'cw'],
            epsilon=0.03
        )

        print("\n✓ Foolbox attacks completed!")
        for attack_name, result in results.items():
            if 'error' not in result:
                print(f"  {attack_name}: Success={result['success']}, L2={result['l2_norm']:.4f}")
    else:
        print("Foolbox not installed. Install with: pip install foolbox")
