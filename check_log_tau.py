checkpoint = torch.load('checkpoint_276.pth')['model']
for k, v in checkpoint.items():
    if 'log_tau' in k:
        print(f"{k}: {v.item():.4f}  →  τ = {v.exp().item():.2f}")

# 2. Check α của tất cả blocks  
for k, v in checkpoint.items():
    if 'alpha' in k:
        print(f"{k}: raw={v.item():.4f}  →  σ(α) = {v.sigmoid().item():.3f}")