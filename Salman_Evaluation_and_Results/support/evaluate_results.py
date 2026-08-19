"""Evaluate SRCNN against bicubic interpolation and save comparison figures."""
from pathlib import Path
import csv

import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from model import SRCNN

VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg'}


def psnr(prediction, target):
    mse = F.mse_loss(prediction.clamp(0, 1), target.clamp(0, 1))
    return 10 * torch.log10(1.0 / (mse + 1e-12))


def matching_image(folder: Path, stem: str):
    for ext in VALID_EXTENSIONS:
        candidate = folder / f'{stem}{ext}'
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f'No image found for {stem} in {folder}')


def evaluate_folder(hr_dir: Path, lr_dir: Path, checkpoint: Path, output_dir: Path, title_prefix='Evaluation'):
    hr_dir, lr_dir, checkpoint, output_dir = map(Path, (hr_dir, lr_dir, checkpoint, output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SRCNN().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()

    rows = []
    hr_files = sorted(p for p in hr_dir.iterdir() if p.suffix.lower() in VALID_EXTENSIONS)
    for hr_path in hr_files:
        stem = hr_path.stem
        lr_path = matching_image(lr_dir, stem)

        hr_img = Image.open(hr_path).convert('RGB')
        lr_img = Image.open(lr_path).convert('RGB')
        lr_tensor = TF.to_tensor(lr_img).unsqueeze(0).to(device)
        hr_tensor = TF.to_tensor(hr_img).unsqueeze(0).to(device)
        bic_tensor = F.interpolate(lr_tensor, scale_factor=4, mode='bicubic', align_corners=False).clamp(0, 1)
        bicubic = TF.to_pil_image(bic_tensor.squeeze(0).cpu())
        with torch.no_grad():
            sr_tensor = model(bic_tensor).clamp(0, 1)

        sr_img = TF.to_pil_image(sr_tensor.squeeze(0).cpu())
        bicubic_psnr = psnr(bic_tensor, hr_tensor).item()
        srcnn_psnr = psnr(sr_tensor, hr_tensor).item()
        gain = srcnn_psnr - bicubic_psnr

        sr_img.save(output_dir / f'{stem}_SRCNN.png')
        bicubic.save(output_dir / f'{stem}_BICUBIC.png')

        figure, axes = plt.subplots(1, 4, figsize=(12, 3.0))
        panels = [
            (lr_img.resize(hr_img.size, Image.Resampling.NEAREST), f'LR Input\n{lr_img.width}x{lr_img.height}'),
            (bicubic, f'Bicubic x4\n{bicubic_psnr:.2f} dB'),
            (sr_img, f'SRCNN x4\n{srcnn_psnr:.2f} dB ({gain:+.2f})'),
            (hr_img, f'HR Reference\n{hr_img.width}x{hr_img.height}'),
        ]
        for ax, (image, title) in zip(axes, panels):
            ax.imshow(image)
            ax.set_title(title, fontsize=9)
            ax.axis('off')
        figure.suptitle(f'{title_prefix}: {stem.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        figure.savefig(output_dir / f'{stem}_COMPARE.png', dpi=180, bbox_inches='tight')
        plt.close(figure)

        rows.append((stem, bicubic_psnr, srcnn_psnr, gain))
        print(f'{stem:20s} Bicubic {bicubic_psnr:6.2f} dB | SRCNN {srcnn_psnr:6.2f} dB | gain {gain:+.2f} dB')

    metrics_path = output_dir / 'evaluation_metrics.csv'
    with metrics_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['image', 'bicubic_psnr', 'srcnn_psnr', 'gain'])
        writer.writerows(rows)

    if rows:
        avg_b = sum(r[1] for r in rows) / len(rows)
        avg_s = sum(r[2] for r in rows) / len(rows)
        print(f'Average: Bicubic {avg_b:.2f} dB | SRCNN {avg_s:.2f} dB | gain {avg_s-avg_b:+.2f} dB')
    return rows


if __name__ == '__main__':
    project_root = Path(__file__).resolve().parents[1]
    evaluate_folder(
        project_root / 'data' / 'evaluation_set' / 'HR_256',
        project_root / 'data' / 'evaluation_set' / 'LR_x4',
        project_root / 'support' / 'best_srcnn_x4.pth',
        project_root / 'support' / 'new_results' / 'custom_evaluation',
        title_prefix='Custom Stress Test',
    )
