"""Prepare 256x256 HR and 64x64 LR image pairs for x4 super-resolution."""
from pathlib import Path
from PIL import Image, ImageFilter, ImageOps

VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}


def prepare_pairs(source_dir: Path, hr_dir: Path, lr_dir: Path, size: int = 256, scale: int = 4):
    source_dir = Path(source_dir)
    hr_dir = Path(hr_dir)
    lr_dir = Path(lr_dir)
    hr_dir.mkdir(parents=True, exist_ok=True)
    lr_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in source_dir.iterdir() if p.suffix.lower() in VALID_EXTENSIONS)
    if not files:
        raise FileNotFoundError(f'No images found in {source_dir}')

    lr_size = size // scale
    prepared = []
    for path in files:
        image = Image.open(path).convert('RGB')
        hr = ImageOps.fit(image, (size, size), method=Image.Resampling.LANCZOS)
        lr = hr.resize((lr_size, lr_size), Image.Resampling.BICUBIC)
        lr = lr.filter(ImageFilter.GaussianBlur(radius=0.5))

        out_name = f'{path.stem}.png'
        hr.save(hr_dir / out_name)
        lr.save(lr_dir / out_name)
        prepared.append(out_name)

    return prepared


if __name__ == '__main__':
    project_root = Path(__file__).resolve().parents[1]
    evaluation_root = project_root / 'data' / 'evaluation_set'
    items = prepare_pairs(
        evaluation_root / 'source_images',
        evaluation_root / 'HR_256',
        evaluation_root / 'LR_x4',
    )
    print(f'Prepared {len(items)} evaluation image pairs.')
