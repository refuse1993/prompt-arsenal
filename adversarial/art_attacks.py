"""
ART (Adversarial Robustness Toolbox) - IBM Research
Universal Perturbation 및 고급 공격 기법
"""

import numpy as np
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from typing import Optional, Dict, List
import warnings

# ART 선택적 임포트
try:
    from art.attacks.evasion import (
        FastGradientMethod,
        ProjectedGradientDescent,
        CarliniL2Method,
        DeepFool,
        UniversalPerturbation,
        AdversarialPatch,
        HopSkipJump,
        SimBA,
        SquareAttack,
        BoundaryAttack,
        ZooAttack,
        SaliencyMapMethod,  # JSMA
        ThresholdAttack,
        PixelAttack
    )
    from art.estimators.classification import PyTorchClassifier
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False
    warnings.warn("ART not installed. Install with: pip install adversarial-robustness-toolbox")


class ARTAttacker:
    """
    IBM ART 기반 고급 공격

    화이트박스 공격:
    - Universal Perturbation: 모든 이미지에 적용 가능한 단일 섭동
    - Adversarial Patch: 물리적 패치 공격
    - FGSM, PGD, C&W, DeepFool: 그래디언트 기반 공격

    블랙박스 공격:
    - HopSkipJump: Decision-based, Query-efficient
    - SimBA: Simple Black-box Attack
    - Square Attack: Query-efficient boundary attack
    - Boundary Attack: Decision-based black-box
    - ZOO: Zeroth Order Optimization
    """

    def __init__(self, target_model: str = 'resnet50', device: str = 'cpu'):
        """
        Args:
            target_model: 타겟 모델명
            device: 'cpu' 또는 'cuda'
        """
        if not ART_AVAILABLE:
            raise ImportError("ART not installed. Run: pip install adversarial-robustness-toolbox")

        self.device = device
        self.target_model_name = target_model

        # 모델 로드
        self.model = self._load_model(target_model)
        self.model.eval()

        # ART Classifier 래핑
        self.classifier = self._create_classifier()

        # 이미지 전처리
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

    def _load_model(self, model_name: str):
        """사전 학습된 모델 로드"""
        if model_name not in models.__dict__:
            raise ValueError(f"Model {model_name} not found")

        model = models.__dict__[model_name](pretrained=True)
        model.to(self.device)
        return model

    def _create_classifier(self):
        """ART PyTorchClassifier 생성"""
        # 손실 함수
        criterion = torch.nn.CrossEntropyLoss()

        # 옵티마이저 (공격에는 사용되지 않지만 필수)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        # ART Classifier
        classifier = PyTorchClassifier(
            model=self.model,
            loss=criterion,
            optimizer=optimizer,
            input_shape=(3, 224, 224),
            nb_classes=1000,
            clip_values=(0, 1)
        )

        return classifier

    def _load_image(self, image_path: str) -> np.ndarray:
        """이미지 로드 및 전처리 (ART 형식)"""
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img)
        img_array = img_tensor.numpy()
        return img_array

    def _array_to_pil(self, array: np.ndarray) -> Image.Image:
        """Numpy array를 PIL Image로 변환"""
        img = np.transpose(array, (1, 2, 0))
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(img)

    def universal_perturbation(self, image_dataset: List[str],
                              max_iter: int = 10,
                              eps: float = 0.1,
                              output_path: str = None) -> Dict:
        """
        Universal Perturbation 생성

        단 하나의 섭동으로 대부분의 이미지를 공격 가능

        Args:
            image_dataset: 학습용 이미지 경로 리스트
            max_iter: 최대 반복 횟수
            eps: 섭동 크기 (L∞ norm)
            output_path: 섭동 저장 경로

        Returns:
            {
                'perturbation': Universal Perturbation (numpy array),
                'success_rate': 성공률,
                'fooling_rate': Fooling Rate
            }
        """
        # 이미지 데이터셋 로드
        images = []
        for img_path in image_dataset[:20]:  # 최대 20장으로 제한
            try:
                img_array = self._load_image(img_path)
                images.append(img_array)
            except Exception as e:
                print(f"Warning: Failed to load {img_path}: {e}")
                continue

        if len(images) == 0:
            raise ValueError("No valid images loaded")

        images = np.array(images)

        # Universal Perturbation 공격
        attack = UniversalPerturbation(
            classifier=self.classifier,
            max_iter=max_iter,
            eps=eps,
            norm=np.inf
        )

        # 공격 실행
        print(f"Generating Universal Perturbation (max_iter={max_iter}, eps={eps})...")
        perturbation = attack.generate(x=images)

        # 성공률 계산
        original_preds = self.classifier.predict(images).argmax(axis=1)
        adversarial_preds = self.classifier.predict(perturbation).argmax(axis=1)
        success_rate = (original_preds != adversarial_preds).mean()

        # 섭동 시각화 저장
        if output_path:
            # 섭동의 첫 번째 샘플 저장
            pert_img = self._array_to_pil(perturbation[0])
            pert_img.save(output_path)

        return {
            'perturbation': perturbation,
            'success_rate': float(success_rate),
            'fooling_rate': float(success_rate),  # Alias
            'num_images': len(images),
            'eps': eps,
            'max_iter': max_iter
        }

    def adversarial_patch(self, image_path: str,
                         patch_shape: tuple = (50, 50, 3),
                         learning_rate: float = 0.01,
                         max_iter: int = 500,
                         output_path: str = None) -> Dict:
        """
        Adversarial Patch 생성

        물리적으로 부착 가능한 패치 생성 (예: 스티커)

        Args:
            image_path: 원본 이미지
            patch_shape: 패치 크기 (height, width, channels)
            learning_rate: 학습률
            max_iter: 최대 반복
            output_path: 출력 경로

        Returns:
            {
                'patch': 패치 이미지,
                'patched_image': 패치가 적용된 이미지,
                'success': 성공 여부
            }
        """
        # 이미지 로드
        img_array = self._load_image(image_path).reshape(1, 3, 224, 224)

        # Adversarial Patch 공격
        attack = AdversarialPatch(
            classifier=self.classifier,
            patch_shape=patch_shape,
            learning_rate=learning_rate,
            max_iter=max_iter,
            batch_size=1
        )

        # 패치 생성
        print(f"Generating Adversarial Patch (max_iter={max_iter})...")
        patch, patched_images = attack.generate(x=img_array)

        # 결과 변환
        patch_pil = self._array_to_pil(patch[0])
        patched_pil = self._array_to_pil(patched_images[0])

        # 저장
        if output_path:
            patched_pil.save(output_path)
            patch_pil.save(output_path.replace('.png', '_patch.png'))

        # 성공 여부 확인
        original_pred = self.classifier.predict(img_array).argmax()
        adversarial_pred = self.classifier.predict(patched_images).argmax()
        success = (original_pred != adversarial_pred)

        return {
            'patch': patch_pil,
            'patched_image': patched_pil,
            'success': bool(success),
            'original_pred': int(original_pred),
            'adversarial_pred': int(adversarial_pred)
        }

    def fgsm_attack(self, image_path: str, eps: float = 0.03,
                   output_path: str = None) -> Dict:
        """FGSM 공격 (ART 버전)"""
        img_array = self._load_image(image_path).reshape(1, 3, 224, 224)

        attack = FastGradientMethod(estimator=self.classifier, eps=eps)
        adversarial = attack.generate(x=img_array)

        adv_pil = self._array_to_pil(adversarial[0])

        if output_path:
            adv_pil.save(output_path)

        return {
            'adversarial_image': adv_pil,
            'success': True,
            'eps': eps
        }

    def pgd_attack(self, image_path: str, eps: float = 0.03,
                  eps_step: float = 0.01, max_iter: int = 40,
                  output_path: str = None) -> Dict:
        """PGD 공격 (ART 버전)"""
        img_array = self._load_image(image_path).reshape(1, 3, 224, 224)

        attack = ProjectedGradientDescent(
            estimator=self.classifier,
            eps=eps,
            eps_step=eps_step,
            max_iter=max_iter
        )

        adversarial = attack.generate(x=img_array)
        adv_pil = self._array_to_pil(adversarial[0])

        if output_path:
            adv_pil.save(output_path)

        return {
            'adversarial_image': adv_pil,
            'success': True,
            'eps': eps,
            'max_iter': max_iter
        }

    def cw_attack(self, image_path: str, confidence: float = 0.0,
                 max_iter: int = 100, output_path: str = None) -> Dict:
        """C&W 공격 (ART 버전)"""
        img_array = self._load_image(image_path).reshape(1, 3, 224, 224)

        attack = CarliniL2Method(
            classifier=self.classifier,
            confidence=confidence,
            max_iter=max_iter
        )

        adversarial = attack.generate(x=img_array)
        adv_pil = self._array_to_pil(adversarial[0])

        if output_path:
            adv_pil.save(output_path)

        return {
            'adversarial_image': adv_pil,
            'success': True,
            'confidence': confidence,
            'max_iter': max_iter
        }

    def deepfool_attack(self, image_path: str, max_iter: int = 50,
                       output_path: str = None) -> Dict:
        """DeepFool 공격 (ART 버전)"""
        img_array = self._load_image(image_path).reshape(1, 3, 224, 224)

        attack = DeepFool(classifier=self.classifier, max_iter=max_iter)
        adversarial = attack.generate(x=img_array)
        adv_pil = self._array_to_pil(adversarial[0])

        if output_path:
            adv_pil.save(output_path)

        return {
            'adversarial_image': adv_pil,
            'success': True,
            'max_iter': max_iter
        }

    # ========================================
    # 블랙박스 공격 메서드
    # ========================================

    def hopskipjump_attack(self, image_path: str,
                          target_label: Optional[int] = None,
                          max_iter: int = 50,
                          max_eval: int = 10000,
                          init_eval: int = 100,
                          output_path: str = None) -> Dict:
        """
        HopSkipJump 블랙박스 공격 (Decision-based)

        CTF 해결에 최적화된 블랙박스 공격:
        - 모델 그래디언트 불필요 (블랙박스)
        - Query-efficient (수천 번의 쿼리)
        - Targeted attack 지원

        Args:
            image_path: 원본 이미지 경로
            target_label: 타겟 클래스 (None이면 untargeted)
            max_iter: 최대 반복 횟수
            max_eval: 최대 쿼리 횟수
            init_eval: 초기 평가 횟수
            output_path: 출력 경로

        Returns:
            {
                'adversarial_image': PIL Image,
                'success': bool,
                'original_pred': int,
                'adversarial_pred': int,
                'confidence': float,
                'num_queries': int,
                'target_label': int or None
            }
        """
        img_array = self._load_image(image_path).reshape(1, 3, 224, 224)

        # 원본 예측
        original_pred = self.classifier.predict(img_array).argmax()
        original_confidence = self.classifier.predict(img_array)[0, original_pred]

        # HopSkipJump 공격
        attack = HopSkipJump(
            classifier=self.classifier,
            targeted=(target_label is not None),
            max_iter=max_iter,
            max_eval=max_eval,
            init_eval=init_eval,
            norm=2,  # L2 norm
            verbose=True
        )

        # 타겟 공격 설정
        if target_label is not None:
            # One-hot encoding for target
            target_array = np.zeros((1, 1000))
            target_array[0, target_label] = 1
            print(f"Targeted HopSkipJump attack to class {target_label}...")
            adversarial = attack.generate(x=img_array, y=target_array)
        else:
            print(f"Untargeted HopSkipJump attack...")
            adversarial = attack.generate(x=img_array)

        # 결과 변환
        adv_pil = self._array_to_pil(adversarial[0])

        if output_path:
            adv_pil.save(output_path)

        # 성공 확인
        adv_pred = self.classifier.predict(adversarial).argmax()
        adv_confidence = self.classifier.predict(adversarial)[0, adv_pred]

        # 타겟 공격의 경우 타겟 클래스 confidence 확인
        if target_label is not None:
            target_confidence = self.classifier.predict(adversarial)[0, target_label]
            success = (adv_pred == target_label)
        else:
            target_confidence = None
            success = (original_pred != adv_pred)

        return {
            'adversarial_image': adv_pil,
            'success': success,
            'original_pred': int(original_pred),
            'original_confidence': float(original_confidence),
            'adversarial_pred': int(adv_pred),
            'adversarial_confidence': float(adv_confidence),
            'target_label': target_label,
            'target_confidence': float(target_confidence) if target_confidence is not None else None,
            'max_iter': max_iter,
            'max_eval': max_eval
        }

    def simba_attack(self, image_path: str,
                    target_label: Optional[int] = None,
                    max_iter: int = 3000,
                    epsilon: float = 0.1,
                    output_path: str = None) -> Dict:
        """
        SimBA (Simple Black-box Attack)

        매우 단순하고 효율적인 블랙박스 공격:
        - Random direction search
        - Query-efficient
        - Pixel-wise perturbation

        Args:
            image_path: 원본 이미지
            target_label: 타겟 클래스 (None이면 untargeted)
            max_iter: 최대 쿼리 횟수
            epsilon: 섭동 크기
            output_path: 출력 경로
        """
        img_array = self._load_image(image_path).reshape(1, 3, 224, 224)

        # 원본 예측
        original_pred = self.classifier.predict(img_array).argmax()

        # SimBA 공격
        attack = SimBA(
            classifier=self.classifier,
            attack='dct',  # DCT-based or 'px' for pixel-based
            max_iter=max_iter,
            epsilon=epsilon,
            targeted=(target_label is not None)
        )

        # 타겟 공격
        if target_label is not None:
            target_array = np.zeros((1, 1000))
            target_array[0, target_label] = 1
            print(f"Targeted SimBA attack to class {target_label}...")
            adversarial = attack.generate(x=img_array, y=target_array)
        else:
            print(f"Untargeted SimBA attack...")
            adversarial = attack.generate(x=img_array)

        # 결과
        adv_pil = self._array_to_pil(adversarial[0])
        if output_path:
            adv_pil.save(output_path)

        adv_pred = self.classifier.predict(adversarial).argmax()
        adv_confidence = self.classifier.predict(adversarial)[0, adv_pred]

        if target_label is not None:
            target_confidence = self.classifier.predict(adversarial)[0, target_label]
            success = (adv_pred == target_label)
        else:
            target_confidence = None
            success = (original_pred != adv_pred)

        return {
            'adversarial_image': adv_pil,
            'success': success,
            'original_pred': int(original_pred),
            'adversarial_pred': int(adv_pred),
            'adversarial_confidence': float(adv_confidence),
            'target_label': target_label,
            'target_confidence': float(target_confidence) if target_confidence is not None else None,
            'epsilon': epsilon,
            'max_iter': max_iter
        }

    def square_attack(self, image_path: str,
                     max_iter: int = 5000,
                     epsilon: float = 0.05,
                     output_path: str = None) -> Dict:
        """
        Square Attack (Query-efficient boundary attack)

        State-of-the-art 블랙박스 공격:
        - 매우 효율적 (적은 쿼리)
        - Random search based
        - L∞ norm perturbation

        Args:
            image_path: 원본 이미지
            max_iter: 최대 쿼리 횟수
            epsilon: 섭동 크기 (L∞)
            output_path: 출력 경로
        """
        img_array = self._load_image(image_path).reshape(1, 3, 224, 224)

        # 원본 예측
        original_pred = self.classifier.predict(img_array).argmax()

        # Square Attack
        attack = SquareAttack(
            estimator=self.classifier,
            max_iter=max_iter,
            eps=epsilon,
            norm='inf',  # L∞ norm
            verbose=True
        )

        print(f"Square Attack (max_iter={max_iter}, eps={epsilon})...")
        adversarial = attack.generate(x=img_array)

        # 결과
        adv_pil = self._array_to_pil(adversarial[0])
        if output_path:
            adv_pil.save(output_path)

        adv_pred = self.classifier.predict(adversarial).argmax()
        adv_confidence = self.classifier.predict(adversarial)[0, adv_pred]

        success = (original_pred != adv_pred)

        return {
            'adversarial_image': adv_pil,
            'success': success,
            'original_pred': int(original_pred),
            'adversarial_pred': int(adv_pred),
            'adversarial_confidence': float(adv_confidence),
            'epsilon': epsilon,
            'max_iter': max_iter
        }

    def boundary_attack(self, image_path: str,
                       target_label: Optional[int] = None,
                       max_iter: int = 5000,
                       delta: float = 0.01,
                       epsilon: float = 0.01,
                       output_path: str = None) -> Dict:
        """
        Boundary Attack (Decision-based black-box)

        결정 경계를 따라 최소 섭동 탐색:
        - 완전 블랙박스
        - 느리지만 효과적
        - Targeted 지원

        Args:
            image_path: 원본 이미지
            target_label: 타겟 클래스
            max_iter: 최대 반복
            delta: Step size
            epsilon: Perturbation size
            output_path: 출력 경로
        """
        img_array = self._load_image(image_path).reshape(1, 3, 224, 224)

        original_pred = self.classifier.predict(img_array).argmax()

        # Boundary Attack
        attack = BoundaryAttack(
            estimator=self.classifier,
            targeted=(target_label is not None),
            max_iter=max_iter,
            delta=delta,
            epsilon=epsilon,
            verbose=True
        )

        if target_label is not None:
            target_array = np.zeros((1, 1000))
            target_array[0, target_label] = 1
            print(f"Targeted Boundary attack to class {target_label}...")
            adversarial = attack.generate(x=img_array, y=target_array)
        else:
            print(f"Untargeted Boundary attack...")
            adversarial = attack.generate(x=img_array)

        adv_pil = self._array_to_pil(adversarial[0])
        if output_path:
            adv_pil.save(output_path)

        adv_pred = self.classifier.predict(adversarial).argmax()
        adv_confidence = self.classifier.predict(adversarial)[0, adv_pred]

        if target_label is not None:
            target_confidence = self.classifier.predict(adversarial)[0, target_label]
            success = (adv_pred == target_label)
        else:
            target_confidence = None
            success = (original_pred != adv_pred)

        return {
            'adversarial_image': adv_pil,
            'success': success,
            'original_pred': int(original_pred),
            'adversarial_pred': int(adv_pred),
            'adversarial_confidence': float(adv_confidence),
            'target_label': target_label,
            'target_confidence': float(target_confidence) if target_confidence is not None else None,
            'max_iter': max_iter
        }

    def jsma_attack(self, image_path: str,
                   target_label: int,
                   theta: float = 0.1,
                   gamma: float = 1.0,
                   output_path: str = None) -> Dict:
        """
        JSMA (Jacobian-based Saliency Map Attack)

        특징:
        - Gradient 기반 타겟 공격
        - 가장 영향력 있는 픽셀만 수정
        - 작은 L0 norm (적은 픽셀 수정)

        Args:
            image_path: 원본 이미지
            target_label: 타겟 클래스 (필수)
            theta: 픽셀당 최대 섭동
            gamma: 픽셀 선택 비율
            output_path: 출력 경로
        """
        img_array = self._load_image(image_path).reshape(1, 3, 224, 224)

        # 원본 예측
        original_pred = self.classifier.predict(img_array).argmax()

        # JSMA 공격 (항상 targeted)
        attack = SaliencyMapMethod(
            classifier=self.classifier,
            theta=theta,
            gamma=gamma
        )

        # 타겟 레이블
        target_array = np.zeros((1, 1000))
        target_array[0, target_label] = 1

        print(f"JSMA attack to class {target_label}...")
        adversarial = attack.generate(x=img_array, y=target_array)

        # 결과
        adv_pil = self._array_to_pil(adversarial[0])
        if output_path:
            adv_pil.save(output_path)

        adv_pred = self.classifier.predict(adversarial).argmax()
        adv_confidence = self.classifier.predict(adversarial)[0, adv_pred]
        target_confidence = self.classifier.predict(adversarial)[0, target_label]

        return {
            'adversarial_image': adv_pil,
            'success': (adv_pred == target_label),
            'original_pred': int(original_pred),
            'adversarial_pred': int(adv_pred),
            'adversarial_confidence': float(adv_confidence),
            'target_label': target_label,
            'target_confidence': float(target_confidence),
            'theta': theta,
            'gamma': gamma
        }

    def pixel_attack(self, image_path: str,
                    target_label: Optional[int] = None,
                    max_iter: int = 100,
                    th: Optional[int] = None,
                    output_path: str = None) -> Dict:
        """
        One-Pixel Attack (Pixel Attack)

        특징:
        - 극소수 픽셀만 수정 (1-5개)
        - Differential Evolution 알고리즘
        - 매우 효율적

        Args:
            image_path: 원본 이미지
            target_label: 타겟 클래스 (None이면 untargeted)
            max_iter: 최대 반복 횟수
            th: 수정할 픽셀 수 (None이면 자동)
            output_path: 출력 경로
        """
        img_array = self._load_image(image_path).reshape(1, 3, 224, 224)

        # 원본 예측
        original_pred = self.classifier.predict(img_array).argmax()

        # Pixel Attack
        attack = PixelAttack(
            classifier=self.classifier,
            th=th,
            max_iter=max_iter,
            targeted=(target_label is not None)
        )

        if target_label is not None:
            target_array = np.zeros((1, 1000))
            target_array[0, target_label] = 1
            print(f"Pixel attack to class {target_label} (max {th or 'auto'} pixels)...")
            adversarial = attack.generate(x=img_array, y=target_array)
        else:
            print(f"Untargeted Pixel attack...")
            adversarial = attack.generate(x=img_array)

        # 결과
        adv_pil = self._array_to_pil(adversarial[0])
        if output_path:
            adv_pil.save(output_path)

        adv_pred = self.classifier.predict(adversarial).argmax()
        adv_confidence = self.classifier.predict(adversarial)[0, adv_pred]

        if target_label is not None:
            target_confidence = self.classifier.predict(adversarial)[0, target_label]
            success = (adv_pred == target_label)
        else:
            target_confidence = None
            success = (original_pred != adv_pred)

        return {
            'adversarial_image': adv_pil,
            'success': success,
            'original_pred': int(original_pred),
            'adversarial_pred': int(adv_pred),
            'adversarial_confidence': float(adv_confidence),
            'target_label': target_label,
            'target_confidence': float(target_confidence) if target_confidence is not None else None,
            'pixels_modified': th or 'auto',
            'max_iter': max_iter
        }


# ========================================
# 고수준 API
# ========================================

def generate_universal_perturbation(image_dataset: List[str],
                                   max_iter: int = 10,
                                   eps: float = 0.1,
                                   output_path: str = None) -> Dict:
    """
    간편한 Universal Perturbation 생성

    Args:
        image_dataset: 이미지 경로 리스트
        max_iter: 최대 반복
        eps: 섭동 크기
        output_path: 출력 경로

    Returns:
        Universal Perturbation 결과
    """
    if not ART_AVAILABLE:
        return {
            'error': 'ART not installed',
            'message': 'Install with: pip install adversarial-robustness-toolbox'
        }

    attacker = ARTAttacker()
    return attacker.universal_perturbation(image_dataset, max_iter, eps, output_path)


def generate_adversarial_patch(image_path: str,
                              patch_shape: tuple = (50, 50, 3),
                              max_iter: int = 500,
                              output_path: str = None) -> Dict:
    """
    간편한 Adversarial Patch 생성

    Args:
        image_path: 원본 이미지
        patch_shape: 패치 크기
        max_iter: 최대 반복
        output_path: 출력 경로

    Returns:
        Adversarial Patch 결과
    """
    if not ART_AVAILABLE:
        return {
            'error': 'ART not installed',
            'message': 'Install with: pip install adversarial-robustness-toolbox'
        }

    attacker = ARTAttacker()
    return attacker.adversarial_patch(image_path, patch_shape, max_iter=max_iter, output_path=output_path)


if __name__ == "__main__":
    # 테스트
    if ART_AVAILABLE:
        print("Testing ART attacks...")

        # 샘플 이미지로 테스트
        test_images = ["media/test1.png", "media/test2.png"]

        attacker = ARTAttacker()

        # Universal Perturbation 테스트
        result = attacker.universal_perturbation(
            test_images,
            max_iter=5,
            eps=0.1,
            output_path="media/universal_pert.png"
        )

        print(f"\n✓ Universal Perturbation: Success Rate={result['success_rate']:.2%}")
    else:
        print("ART not installed. Install with: pip install adversarial-robustness-toolbox")
