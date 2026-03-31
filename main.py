import os
import time

from PIL import Image
from pathlib import Path

# ------------------------- UTILS ------------------------- #
def get_size_format(b, factor=1024, suffix="B"):
    """
    Scale bytes to its proper byte format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if b < factor:
            return f"{b:6.2f}{unit}{suffix}"
        b /= factor
    return f"{b:6.2f}Y{suffix}"

def count_files(dir_path: Path) -> int:
    return sum(1 for p in dir_path.iterdir() if p.is_file())

def dir_size(dir_path: Path) -> int:
    return sum(p.stat().st_size for p in dir_path.iterdir() if p.is_file())

def print_loading_bar(val_act:int, nb_total:int, bar_length:int=100,) -> None:
    val_act = min(val_act, nb_total - 1)

    progress = (val_act + 1) / nb_total if nb_total else 0

    filled = int(bar_length * progress)
    empty = bar_length - filled
    bar = ("█" * filled) + ("░" * empty)

    percent = round(progress * 100)

    if val_act >= nb_total:
        percent = 100
        bar = "█" * bar_length

    print(f"{bar} {percent:03d}% ({(val_act+1):03d}/{nb_total:03d})", end="\r", flush=True)
# --------------------------------------------------------- #

def compress_img(image_name,
                 new_size_ratio: float = 1.0,
                 quality: int = 90,
                 width: int = 224,
                 height: int = 224,
                 to_jpg: bool = True,
                 compressed_suf: bool = True,
                 output_dir: str = "_out"
    ) -> None:

    image_path = Path(image_name)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(image_path)

    if new_size_ratio < 1.0:
        img = img.resize((int(img.size[0] * new_size_ratio), int(img.size[1] * new_size_ratio)), Image.LANCZOS)
    elif width and height:
        img = img.resize((width, height), Image.LANCZOS)

    # nom sans dossier
    stem = image_path.stem          # ex: "img_in_2500x1250"
    ext = image_path.suffix         # ex: ".png" / ".jpg"

    if to_jpg:
        out_ext = ".jpg"
    else:
        out_ext = ext

    suffix = "_compressed" if compressed_suf else ""
    out_name = f"{stem}{suffix}{out_ext}"

    saved_path = out_dir / out_name

    try:
        img.save(saved_path, quality=quality, optimize=True)
    except OSError:
        img = img.convert("RGB")
        img.save(saved_path, quality=quality, optimize=True)

def compress_all_img_in_file(in_dir: Path, output_dir="_out") -> None:
    valid_ext = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

    files = [
        f for f in in_dir.iterdir()
        if f.is_file() and f.suffix.lower() in valid_ext
    ]
    nb_total = len(files)

    init_dir_size = dir_size(in_dir)
    #print(f"[*] Dir init size: {get_size_format(init_dir_size)}")

    print_loading_bar(0, nb_total)

    for i, img_path in enumerate(files, start=1):
        compress_img(img_path, new_size_ratio=1, quality=1, to_jpg=True, output_dir=output_dir)
        print_loading_bar(i, nb_total)

    print()  # (¯\_(ツ)_/¯)

    print(f"[*] Dir init size: {get_size_format(init_dir_size)}")

    new_dir_size = dir_size(Path(output_dir))
    print(f"[*] New Dir size : {get_size_format(new_dir_size)}")

    saving_diff = new_dir_size - init_dir_size
    print(f"[+] New Dir size: {100+(saving_diff/init_dir_size*100):.2f}% of the original Dir size.")


if __name__ == "__main__":
    dir_path_str = "_in"
    dir_path = Path(dir_path_str)

    compress_all_img_in_file(dir_path)
