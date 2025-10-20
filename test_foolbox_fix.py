#!/usr/bin/env python3
"""
Test Foolbox PGD attack fix
"""
import sys
sys.path.insert(0, '/Users/brownkim/Downloads/ACDC/prompt_arsenal')

print("Step 1: Importing FoolboxAttack...")
from adversarial.foolbox_attacks import FoolboxAttack

print("Step 2: Creating FoolboxAttack instance...")
foolbox = FoolboxAttack()

print("Step 3: Running PGD attack on sample image...")
try:
    result = foolbox.pgd_attack("samples/images/sample.jpg", epsilon=0.03)
    print(f"✓ PGD attack successful! Image size: {result.size}")

    # Save result
    result.save("test_pgd_result.png")
    print("✓ Result saved to: test_pgd_result.png")

except Exception as e:
    print(f"✗ PGD attack failed: {e}")
    import traceback
    traceback.print_exc()

print("\nStep 4: Testing FGSM attack...")
try:
    result = foolbox.fgsm_attack("samples/images/sample.jpg", epsilon=0.03)
    print(f"✓ FGSM attack successful! Image size: {result.size}")

    result.save("test_fgsm_result.png")
    print("✓ Result saved to: test_fgsm_result.png")

except Exception as e:
    print(f"✗ FGSM attack failed: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ All tests completed!")
