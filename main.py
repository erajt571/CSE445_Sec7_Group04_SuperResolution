"""Main entry point for the CSE445 SRCNN super-resolution project."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SUPPORT_DIR = PROJECT_ROOT / 'support'
sys.path.insert(0, str(SUPPORT_DIR))

from evaluate_results import evaluate_folder


if __name__ == '__main__':
    print('CSE445 Section 07 - Group 04')
    print('Machine Learning-Based Single-Image Super-Resolution with SRCNN')
    print('\nRunning the custom evaluation set...\n')
    evaluate_folder(
        PROJECT_ROOT / 'data' / 'evaluation_set' / 'HR_256',
        PROJECT_ROOT / 'data' / 'evaluation_set' / 'LR_x4',
        SUPPORT_DIR / 'best_srcnn_x4.pth',
        SUPPORT_DIR / 'new_results' / 'custom_evaluation',
        title_prefix='Custom Stress Test',
    )
