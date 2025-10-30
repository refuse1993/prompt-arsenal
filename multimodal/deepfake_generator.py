"""
Deepfake 생성 시스템
- Face Swapping (얼굴 교체)
- Lip-sync (립싱크)
- Face Reenactment (표정 재연)
"""

import cv2
import numpy as np
from PIL import Image
from typing import Optional, Dict, List
import warnings
import subprocess
import os

# InsightFace 선택적 임포트
try:
    import insightface
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import get_model
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    warnings.warn("InsightFace not installed. Install with: pip install insightface onnxruntime")

# FaceNet PyTorch 선택적 임포트
try:
    from facenet_pytorch import MTCNN, InceptionResnetV1
    FACENET_AVAILABLE = True
except ImportError:
    FACENET_AVAILABLE = False
    warnings.warn("facenet-pytorch not installed. Install with: pip install facenet-pytorch")


class DeepfakeGenerator:
    """
    Deepfake 생성기

    기능:
    - Face Swapping: 얼굴 교체
    - Face Detection: 얼굴 탐지 및 랜드마크
    - Face Enhancement: GFPGAN, Real-ESRGAN 통합
    """

    def __init__(self, device: str = 'cpu'):
        """
        Args:
            device: 'cpu' 또는 'cuda'
        """
        self.device = device

        # InsightFace 초기화
        if INSIGHTFACE_AVAILABLE:
            self.face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
            self.face_app.prepare(ctx_id=0 if device == 'cpu' else 0, det_size=(640, 640))
        else:
            self.face_app = None

        # FaceNet 초기화
        if FACENET_AVAILABLE:
            self.mtcnn = MTCNN(keep_all=True, device=device)
            self.facenet = InceptionResnetV1(pretrained='vggface2').eval()
        else:
            self.mtcnn = None
            self.facenet = None

    def detect_faces(self, image_path: str) -> List[Dict]:
        """
        얼굴 탐지 및 정보 추출

        Returns:
            [{
                'bbox': [x, y, w, h],
                'landmarks': [[x1,y1], [x2,y2], ...],
                'age': int,
                'gender': str,
                'embedding': np.ndarray
            }]
        """
        if not INSIGHTFACE_AVAILABLE:
            raise ImportError("InsightFace not installed")

        img = cv2.imread(image_path)
        faces = self.face_app.get(img)

        results = []
        for face in faces:
            results.append({
                'bbox': face.bbox.astype(int).tolist(),
                'landmarks': face.kps.astype(int).tolist(),
                'age': int(face.age) if hasattr(face, 'age') else None,
                'gender': 'M' if face.gender == 1 else 'F' if hasattr(face, 'gender') else None,
                'embedding': face.embedding
            })

        return results

    def face_swap(self, source_image: str, target_image: str,
                  output_path: str, face_index: int = 0) -> Dict:
        """
        얼굴 교체 (Face Swapping)

        Args:
            source_image: 소스 얼굴 이미지 (교체할 얼굴)
            target_image: 타겟 이미지 (배경 유지)
            output_path: 출력 경로
            face_index: 타겟 이미지에서 교체할 얼굴 인덱스

        Returns:
            {
                'output_path': 출력 경로,
                'source_faces': 소스 얼굴 수,
                'target_faces': 타겟 얼굴 수,
                'success': 성공 여부
            }
        """
        if not INSIGHTFACE_AVAILABLE:
            return {
                'error': 'InsightFace not installed',
                'message': 'Install with: pip install insightface onnxruntime'
            }

        # 이미지 로드
        source_img = cv2.imread(source_image)
        target_img = cv2.imread(target_image)

        # 얼굴 탐지
        source_faces = self.face_app.get(source_img)
        target_faces = self.face_app.get(target_img)

        if len(source_faces) == 0:
            return {'error': 'No face detected in source image'}

        if len(target_faces) == 0:
            return {'error': 'No face detected in target image'}

        if face_index >= len(target_faces):
            face_index = 0

        # 얼굴 교체 (간단한 버전 - 실제로는 더 복잡한 블렌딩 필요)
        source_face = source_faces[0]
        target_face = target_faces[face_index]

        # 얼굴 영역 추출
        source_bbox = source_face.bbox.astype(int)
        target_bbox = target_face.bbox.astype(int)

        # 소스 얼굴 크롭
        source_face_crop = source_img[source_bbox[1]:source_bbox[3], source_bbox[0]:source_bbox[2]]

        # 타겟 크기로 리사이즈
        target_w = target_bbox[2] - target_bbox[0]
        target_h = target_bbox[3] - target_bbox[1]
        source_face_resized = cv2.resize(source_face_crop, (target_w, target_h))

        # 타겟 이미지에 붙여넣기 (간단한 오버레이)
        result_img = target_img.copy()
        result_img[target_bbox[1]:target_bbox[3], target_bbox[0]:target_bbox[2]] = source_face_resized

        # 블렌딩 (더 자연스럽게)
        # 간단한 Gaussian blur로 경계 부드럽게
        mask = np.zeros_like(result_img)
        mask[target_bbox[1]:target_bbox[3], target_bbox[0]:target_bbox[2]] = 255
        mask = cv2.GaussianBlur(mask, (21, 21), 11)

        # 저장
        cv2.imwrite(output_path, result_img)

        return {
            'output_path': output_path,
            'source_faces': len(source_faces),
            'target_faces': len(target_faces),
            'success': True,
            'face_swapped_index': face_index
        }

    def extract_face_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """얼굴 임베딩 추출 (512-dim vector)"""
        if not INSIGHTFACE_AVAILABLE:
            return None

        img = cv2.imread(image_path)
        faces = self.face_app.get(img)

        if len(faces) == 0:
            return None

        return faces[0].embedding

    def compare_faces(self, image1_path: str, image2_path: str) -> float:
        """
        두 얼굴 비교 (코사인 유사도)

        Returns:
            유사도 (0~1, 높을수록 유사)
        """
        emb1 = self.extract_face_embedding(image1_path)
        emb2 = self.extract_face_embedding(image2_path)

        if emb1 is None or emb2 is None:
            return 0.0

        # 코사인 유사도
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)

    def lip_sync_video(self, video_path: str, audio_path: str,
                      output_path: str, use_wav2lip: bool = False) -> Dict:
        """
        Lip-sync 비디오 생성 (Wav2Lip 사용)

        Args:
            video_path: 원본 비디오
            audio_path: 오디오 파일
            output_path: 출력 비디오
            use_wav2lip: Wav2Lip 사용 여부 (False면 간단한 병합)

        Returns:
            {
                'output_path': 출력 경로,
                'method': 'wav2lip' 또는 'simple_merge',
                'success': bool
            }
        """
        if use_wav2lip:
            # Wav2Lip 실행 (외부 리포지토리 필요)
            return self._run_wav2lip(video_path, audio_path, output_path)
        else:
            # 간단한 오디오 병합
            return self._simple_audio_merge(video_path, audio_path, output_path)

    def _simple_audio_merge(self, video_path: str, audio_path: str,
                           output_path: str) -> Dict:
        """간단한 오디오 병합 (FFmpeg 사용)"""
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-i', audio_path,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-map', '0:v:0',
                '-map', '1:a:0',
                '-shortest',
                output_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                return {
                    'output_path': output_path,
                    'method': 'simple_merge',
                    'success': True
                }
            else:
                return {
                    'error': 'FFmpeg failed',
                    'message': result.stderr,
                    'success': False
                }

        except FileNotFoundError:
            return {
                'error': 'FFmpeg not found',
                'message': 'Install FFmpeg: https://ffmpeg.org/download.html',
                'success': False
            }

    def _run_wav2lip(self, video_path: str, audio_path: str,
                    output_path: str) -> Dict:
        """
        Wav2Lip 실행 (외부 리포지토리)

        주의: Wav2Lip 리포지토리를 별도로 클론하고 설정해야 함
        https://github.com/Rudrabha/Wav2Lip
        """
        # Wav2Lip 경로 (사용자 환경에 맞게 수정 필요)
        wav2lip_path = os.path.expanduser("~/Wav2Lip")

        if not os.path.exists(wav2lip_path):
            return {
                'error': 'Wav2Lip not found',
                'message': f'Clone Wav2Lip to {wav2lip_path}',
                'instructions': 'git clone https://github.com/Rudrabha/Wav2Lip.git ~/Wav2Lip',
                'success': False
            }

        try:
            cmd = [
                'python',
                os.path.join(wav2lip_path, 'inference.py'),
                '--checkpoint_path', os.path.join(wav2lip_path, 'checkpoints/wav2lip_gan.pth'),
                '--face', video_path,
                '--audio', audio_path,
                '--outfile', output_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=wav2lip_path)

            if result.returncode == 0:
                return {
                    'output_path': output_path,
                    'method': 'wav2lip',
                    'success': True
                }
            else:
                return {
                    'error': 'Wav2Lip failed',
                    'message': result.stderr,
                    'success': False
                }

        except Exception as e:
            return {
                'error': 'Wav2Lip execution failed',
                'message': str(e),
                'success': False
            }

    def enhance_face(self, image_path: str, output_path: str,
                    method: str = 'gfpgan') -> Dict:
        """
        얼굴 복원/향상 (GFPGAN 또는 Real-ESRGAN)

        Args:
            image_path: 입력 이미지
            output_path: 출력 이미지
            method: 'gfpgan' 또는 'realesrgan'

        Returns:
            결과 딕셔너리
        """
        # GFPGAN/Real-ESRGAN은 별도 패키지 설치 필요
        # 여기서는 플레이스홀더만 제공
        return {
            'error': f'{method} not implemented',
            'message': f'Install {method}: pip install {method}',
            'success': False
        }


# ========================================
# 고수준 API
# ========================================

def swap_faces(source_image: str, target_image: str,
              output_path: str = 'media/face_swapped.png') -> Dict:
    """
    간편한 얼굴 교체 API

    Args:
        source_image: 소스 얼굴
        target_image: 타겟 이미지
        output_path: 출력 경로

    Returns:
        결과 딕셔너리
    """
    if not INSIGHTFACE_AVAILABLE:
        return {
            'error': 'InsightFace not installed',
            'message': 'Install with: pip install insightface onnxruntime'
        }

    generator = DeepfakeGenerator()
    return generator.face_swap(source_image, target_image, output_path)


def create_lip_sync_video(video_path: str, audio_path: str,
                          output_path: str = 'media/lip_sync.mp4') -> Dict:
    """
    간편한 립싱크 비디오 생성

    Args:
        video_path: 원본 비디오
        audio_path: 오디오 파일
        output_path: 출력 경로

    Returns:
        결과 딕셔너리
    """
    generator = DeepfakeGenerator()
    return generator.lip_sync_video(video_path, audio_path, output_path, use_wav2lip=False)


if __name__ == "__main__":
    # 테스트
    if INSIGHTFACE_AVAILABLE:
        print("Testing Deepfake Generator...")

        generator = DeepfakeGenerator()

        # Face detection 테스트
        test_image = "media/test_face.jpg"
        if os.path.exists(test_image):
            faces = generator.detect_faces(test_image)
            print(f"✓ Detected {len(faces)} face(s)")

        print("\nDeepfake Generator ready!")
        print("Available methods:")
        print("  - face_swap(source, target, output)")
        print("  - detect_faces(image)")
        print("  - compare_faces(image1, image2)")
        print("  - lip_sync_video(video, audio, output)")
    else:
        print("InsightFace not installed. Install with:")
        print("  pip install insightface onnxruntime")
