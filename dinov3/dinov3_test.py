from dinov3.hub.backbones import dinov3_vits16
import cv2
import torch

device = torch.device("cuda:0")

encoder_path = "/home/renchengwei/GDR-Net/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
ckpt = torch.load(encoder_path, map_location="cpu", weights_only=False)
image_encoder = dinov3_vits16(pretrained=False, weights=encoder_path)
image_encoder.load_state_dict(ckpt, strict=True)
image_encoder.to(device)
# print(image_encoder)

img_bgr = cv2.imread("/home/renchengwei/GDR-Net/image_with_bbox.png")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img = torch.from_numpy(img_rgb).float() / 255.0   # (H, W, 3)
img = img.permute(2, 0, 1).unsqueeze(0) # (1, 3, H, W)

# ImageNet normalization (DINOv3 uses this)
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

img = (img - mean) / std
# copy batch dimension
img = torch.cat([img] * 8, dim=0)
img = img.to(device)


with torch.inference_mode():
    with torch.autocast(device.type, dtype=torch.bfloat16):
        features = image_encoder.forward_features(img)


print(features['x_norm_patchtokens'].shape) # (B, L, C)
# check VRAM
import pdb; pdb.set_trace()