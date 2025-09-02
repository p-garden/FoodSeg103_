import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image


class FoodSeg103(Dataset):
    IMG_EXTS  = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    MASK_EXTS = {'.png', '.bmp', '.tif', '.tiff'}  # 라벨은 보통 png/bmp/tiff

    def __init__(self, img_dir, mask_dir, img_transform=None, mask_transform=None,
                 size=(256, 256),joint_transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_transform = img_transform
        self.mask_transform = mask_transform  # (주의) 라벨 변환 시 수치 변형 금지
        self.joint_transform = joint_transform    
        self.size = size

        # 1) 파일 수집 + 확장자/숨김파일 필터링
        imgs = [f for f in os.listdir(img_dir)
                if not f.startswith('.') and os.path.splitext(f)[1].lower() in self.IMG_EXTS]
        masks = [f for f in os.listdir(mask_dir)
                 if not f.startswith('.') and os.path.splitext(f)[1].lower() in self.MASK_EXTS]

        # 2) stem 기준 매핑
        def to_map(files):
            d = {}
            for f in files:
                stem, ext = os.path.splitext(f)
                d.setdefault(stem, []).append(f)  # 동일 stem에 여러 확장자 있을 수 있음
            return d

        img_map  = to_map(imgs)
        mask_map = to_map(masks)

        # 3) 공통 stem만 페어링 (여러 파일인 경우 사전순 첫 번째 사용)
        common_stems = sorted(set(img_map.keys()) & set(mask_map.keys()))
        if not common_stems:
            raise RuntimeError(
                f"No matching image/mask pairs found.\n"
                f"- img_dir: {img_dir}\n- mask_dir: {mask_dir}\n"
                f"Check filenames share the same stems (e.g., 0001.jpg & 0001.png)."
            )

        self.data = []
        for stem in common_stems:
            img_file  = sorted(img_map[stem])[0]
            mask_file = sorted(mask_map[stem])[0]
            self.data.append((os.path.join(img_dir, img_file),
                              os.path.join(mask_dir, mask_file)))

        # 4) 기본 이미지 텐서 변환기
        self._to_tensor = ToTensor()

    def __len__(self):
        return len(self.data)

    def _read_image_rgb(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _read_mask_gray(self, path):
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Failed to read mask: {path}")
        return mask

    def __getitem__(self, idx):
        img_path, mask_path = self.data[idx]

        # --- Read ---
        img  = self._read_image_rgb(img_path)
        mask = self._read_mask_gray(mask_path)
        if self.joint_transform is not None:
            out = self.joint_transform(image=img, mask=mask)
            img, mask = out["image"], out["mask"]  # Albumentations 결과

            # ToTensorV2를 joint_transform 안에서 썼다면 이미 torch.Tensor임
            if isinstance(img, np.ndarray):
                img = self._to_tensor(img)  # CxHxW float32 [0,1]
            if isinstance(mask, np.ndarray):
                mask = torch.as_tensor(mask, dtype=torch.long)

            return img, mask

        # --- Resize (이미지: bilinear, 마스크: nearest) ---
        if self.size is not None:
            w, h = self.size  # (width, height)
            img  = cv2.resize(img,  (w, h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # --- Optional transforms ---
        #   * img_transform은 PIL/ndarray 모두 지원 가능한 것으로 가정
        #   * mask_transform은 라벨값 보존되는 변환만(회전/크롭 등) 사용 권장
        if self.img_transform is not None:
            # torchvision 변환 호환 위해 PIL로 변환 후 적용
            img_pil = Image.fromarray(img)
            img = self.img_transform(img_pil)
        else:
            # 기본: ToTensor (0~1 float, CxHxW)
            img = self._to_tensor(img)

        if self.mask_transform is not None:
            # PIL로 변환 후, 최근접 보간 유지 위해 mode='L' 사용 권장
            mask_pil = Image.fromarray(mask)
            mask = self.mask_transform(mask_pil)
            # 변환 결과가 텐서가 아닐 수 있어 numpy로 보장
            if isinstance(mask, Image.Image):
                mask = np.array(mask)
            elif torch.is_tensor(mask):
                # (H,W) 보장 및 long으로 변환
                mask = mask.squeeze().to(dtype=torch.long)
                return img, mask

        # --- Tensorize mask (라벨 보존) ---
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)

        return img, mask
