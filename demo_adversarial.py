#!/usr/bin/env python3
"""
Adversarial Attack Demo - 원본 vs 공격 이미지 비교
"""
import sys
sys.path.insert(0, '/Users/brownkim/Downloads/ACDC/prompt_arsenal')

from adversarial.foolbox_attacks import FoolboxAttack
from PIL import Image, ImageDraw, ImageFont
import numpy as np

print("🎨 Adversarial Attack 시각화 데모\n")

# Foolbox 공격 실행
foolbox = FoolboxAttack()

print("1️⃣  원본 이미지 로드: samples/images/sample.jpg")
original = Image.open("samples/images/sample.jpg")

print("2️⃣  FGSM 공격 수행 (epsilon=0.03)...")
fgsm_result = foolbox.fgsm_attack("samples/images/sample.jpg", epsilon=0.03)

print("3️⃣  PGD 공격 수행 (epsilon=0.03, steps=40)...")
pgd_result = foolbox.pgd_attack("samples/images/sample.jpg", epsilon=0.03, steps=40)

print("4️⃣  차이 계산...")

# 원본과 공격 이미지 차이 시각화
original_array = np.array(original.resize((512, 512)))
fgsm_array = np.array(fgsm_result)
pgd_array = np.array(pgd_result)

# 차이 계산 (절대값)
fgsm_diff = np.abs(original_array.astype(float) - fgsm_array.astype(float))
pgd_diff = np.abs(original_array.astype(float) - pgd_array.astype(float))

# 차이 증폭 (사람 눈에 보이도록)
fgsm_diff_amplified = np.clip(fgsm_diff * 10, 0, 255).astype(np.uint8)
pgd_diff_amplified = np.clip(pgd_diff * 10, 0, 255).astype(np.uint8)

# 이미지로 변환
fgsm_diff_img = Image.fromarray(fgsm_diff_amplified)
pgd_diff_img = Image.fromarray(pgd_diff_amplified)

# 비교 이미지 생성 (2x3 그리드)
width, height = 512, 512
comparison = Image.new('RGB', (width * 3, height * 2), (255, 255, 255))

# 배치
comparison.paste(original.resize((width, height)), (0, 0))
comparison.paste(fgsm_result, (width, 0))
comparison.paste(fgsm_diff_img, (width * 2, 0))

comparison.paste(original.resize((width, height)), (0, height))
comparison.paste(pgd_result, (width, height))
comparison.paste(pgd_diff_img, (width * 2, height))

# 텍스트 추가
draw = ImageDraw.Draw(comparison)

labels = [
    (10, 10, "원본 이미지"),
    (width + 10, 10, "FGSM 공격"),
    (width * 2 + 10, 10, "차이 (x10 증폭)"),
    (10, height + 10, "원본 이미지"),
    (width + 10, height + 10, "PGD 공격"),
    (width * 2 + 10, height + 10, "차이 (x10 증폭)")
]

for x, y, text in labels:
    # 배경 박스
    bbox = draw.textbbox((x, y), text)
    draw.rectangle(bbox, fill=(0, 0, 0, 200))
    draw.text((x, y), text, fill=(255, 255, 255))

# 저장
output_path = "adversarial_comparison.png"
comparison.save(output_path)

print(f"\n✅ 비교 이미지 생성 완료: {output_path}")

# 통계 출력
fgsm_avg_diff = np.mean(fgsm_diff)
pgd_avg_diff = np.mean(pgd_diff)
fgsm_max_diff = np.max(fgsm_diff)
pgd_max_diff = np.max(pgd_diff)

print(f"""
📊 변화량 통계:

FGSM 공격:
  - 평균 픽셀 변화: {fgsm_avg_diff:.2f} / 255 ({fgsm_avg_diff/255*100:.2f}%)
  - 최대 픽셀 변화: {fgsm_max_diff:.0f} / 255 ({fgsm_max_diff/255*100:.1f}%)

PGD 공격:
  - 평균 픽셀 변화: {pgd_avg_diff:.2f} / 255 ({pgd_avg_diff/255*100:.2f}%)
  - 최대 픽셀 변화: {pgd_max_diff:.0f} / 255 ({pgd_max_diff/255*100:.1f}%)

💡 해석:
  - 사람 눈에는 거의 동일하게 보임 (평균 변화 < 3%)
  - 하지만 AI 모델은 완전히 다르게 인식할 수 있음
  - 이를 "Adversarial Perturbation"이라고 함
""")

print("\n🎯 실제 사용 시나리오:")
print("""
1. 원본 이미지 + 텍스트 프롬프트 → 멀티모달 LLM
   → "죄송합니다, 그런 요청은 도와드릴 수 없습니다"

2. 공격 이미지 + 동일한 텍스트 프롬프트 → 멀티모달 LLM
   → 유해한 응답 생성 (jailbreak 성공!)

3. 차이점: 사람이 보기엔 동일하지만, 모델은 다르게 반응
""")
