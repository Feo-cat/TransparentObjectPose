import torch

def test_roi_token_restore():
    torch.manual_seed(0)

    # ====== 假设一个 patch grid ======
    Ph, Pw = 4, 4
    C = 3
    N = Ph * Pw

    # 假设 ViT / DINO 的 token： [N, C]
    tokens = torch.arange(N * C).float().reshape(N, C)

    print("Tokens (index -> value):")
    for i in range(N):
        print(f"{i:2d}: {tokens[i].tolist()}")

    # ====== 构造一个“非矩形”的 ROI mask ======
    mask = torch.zeros((Ph, Pw), dtype=torch.bool)
    mask[1, 1] = True
    mask[1, 2] = True
    mask[2, 2] = True   # L 形

    print("\nROI mask:")
    print(mask.int())

    # ====== flatten mask & 选 token ======
    mask_flat = mask.view(-1)
    roi_tokens = tokens[mask_flat]   # [K, C]

    print("\nSelected token indices:")
    print(torch.nonzero(mask_flat).squeeze(-1).tolist())

    print("\nroi_tokens:")
    print(roi_tokens)

    # ====== ❌ 错误做法：直接 reshape ======
    try:
        fake = roi_tokens.reshape(2, 2, C)
        print("\n❌ Fake reshape result [H, W, C]:")
        print(fake)
    except Exception as e:
        print("Reshape failed:", e)

    # ====== ✅ 正确做法：按 (y, x) 放回 ======
    ys, xs = torch.where(mask)

    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    roi_h = y1 - y0 + 1
    roi_w = x1 - x0 + 1

    roi_feat = torch.zeros((C, roi_h, roi_w))
    roi_valid = torch.zeros((roi_h, roi_w), dtype=torch.bool)

    for k in range(len(ys)):
        y = ys[k] - y0
        x = xs[k] - x0
        roi_feat[:, y, x] = roi_tokens[k]
        roi_valid[y, x] = True

    print("\n✅ Correct ROI feature map [C, H, W]:")
    print(roi_feat)

    print("\nROI valid mask:")
    print(roi_valid.int())


if __name__ == "__main__":
    test_roi_token_restore()