import glob

import jcamp
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.integrate import trapezoid

# ==========================================
# 0. GPUの設定
# ==========================================
# GPUが利用可能なら 'cuda'、利用不可なら 'cpu' を自動選択します
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用中のデバイス: {device}")


# ==========================================
# 1. 全 .jdx ファイルの読み込みと前処理
# ==========================================
# 全スペクトルを統一波数グリッドへ補間してから面積正規化する
TARGET_X = np.linspace(600, 4000, 1671)


def load_spectrum(filename):
    dx = jcamp.jcamp_readfile(filename)
    x = np.array(dx["x"])
    y = np.array(dx["y"])

    if y.max() > 1.0:
        y = 2.0 - np.log10(y)
    else:
        y = -np.log10(y)
    # 透過率0（測定限界）による飽和ピークを2.0でクランプ
    y = np.nan_to_num(y, nan=0.0, posinf=2.0, neginf=0.0)
    y = np.clip(y, 0.0, 2.0)

    if x[0] > x[-1]:
        x, y = x[::-1], y[::-1]

    y_interp = np.interp(TARGET_X, x, y, left=0.0, right=0.0)

    area = np.abs(trapezoid(y_interp, TARGET_X))
    return y_interp / area if area > 0 else y_interp


files = sorted(glob.glob("./data-dx/*.jdx"))
spectra, filenames = [], []
for f in files:
    try:
        spectra.append(load_spectrum(f))
        filenames.append(f)
    except Exception as e:
        print(f"スキップ: {f} ({e})")

print(f"読み込み成功: {len(spectra)} ファイル")

idx_ben = filenames.index("./data-dx/benzene.jdx")
idx_tol = filenames.index("./data-dx/toluene.jdx")

INPUT_DIM = len(TARGET_X)
LATENT_DIM = 64

# ==========================================
# 2. PyTorchテンソル作成 ＆ GPUへ転送
# ==========================================
train_data = np.vstack(spectra).astype(np.float32)
input_tensor = torch.tensor(train_data).to(device)


# ==========================================
# 3. 1D CNN オートエンコーダの定義
# ==========================================
# エンコーダのConv後サイズ: 1671→418→105→27 (stride=4, kernel=15, padding=7)
_CONV_OUT = 27
_CONV_CH  = 256


class CNN_AE(nn.Module):

    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(1, 64,  kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, _CONV_CH, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(_CONV_CH), nn.ReLU(),
        )
        self.fc_mu      = nn.Linear(_CONV_CH * _CONV_OUT, latent_dim)
        self.fc_log_var = nn.Linear(_CONV_CH * _CONV_OUT, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, _CONV_CH * _CONV_OUT)

        # Upsample + reflect padding Conv1d でエッジアーチファクトを防ぐ
        # ConvTranspose1dは境界でリンギングが発生するため使用しない
        self.dec_block1 = nn.Sequential(
            nn.Conv1d(_CONV_CH, 128, kernel_size=15, padding=7, padding_mode='reflect'),
            nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.dec_block2 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=15, padding=7, padding_mode='reflect'),
            nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.dec_out = nn.Conv1d(64, 1, kernel_size=15, padding=7, padding_mode='reflect')

    def encode(self, x):
        h = self.encoder_conv(x.unsqueeze(1)).flatten(1)
        return self.fc_mu(h), self.fc_log_var(h)

    def forward(self, x):
        mu, log_var = self.encode(x)
        h = F.relu(self.decoder_fc(mu)).view(-1, _CONV_CH, _CONV_OUT)
        h = F.interpolate(h, scale_factor=4, mode='linear', align_corners=False)
        h = self.dec_block1(h)
        h = F.interpolate(h, scale_factor=4, mode='linear', align_corners=False)
        h = self.dec_block2(h)
        h = F.interpolate(h, size=self.input_dim, mode='linear', align_corners=False)
        recon = self.dec_out(h).squeeze(1)
        return recon, mu, log_var


def vae_loss(reconstructed, x, mu, log_var, beta):
    recon_loss = F.l1_loss(reconstructed, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


model = CNN_AE(input_dim=INPUT_DIM, latent_dim=LATENT_DIM).to(device)

MODEL_PATH = "autoencoder.pth"
loss_history = []

# ==========================================
# 4. モデルのロードまたは学習
# ==========================================
if torch.os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    loss_history = checkpoint.get("loss_history", [])
    print(f"モデルをロードしました: {MODEL_PATH}")
else:
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1500, gamma=0.3)
    EPOCHS = 6000
    # 復元品質を最優先するためβ=0（KL損失なし）
    # μベクトルはF.normalizeなしで自由に分布するため、コサイン類似度・L2距離ともに有効
    BETA = 0.0

    model.train()
    for epoch in range(1, EPOCHS + 1):
        reconstructed, mu, log_var = model(input_tensor)
        loss, recon, kl = vae_loss(reconstructed, input_tensor, mu, log_var, BETA)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        loss_history.append(loss.item())

        if epoch % 500 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch [{epoch}/{EPOCHS}], Loss (L1): {loss.item():.8f}, LR: {lr:.5f}")

    torch.save({"model_state_dict": model.state_dict(), "loss_history": loss_history}, MODEL_PATH)
    print(f"モデルを保存しました: {MODEL_PATH}")
# ==========================================
# 5. μベクトルでの類似度評価
# ==========================================
model.eval()
with torch.no_grad():
    final_reconstructed, mu_all, _ = model(input_tensor)

    mu_ben = mu_all[idx_ben]
    mu_tol = mu_all[idx_tol]

    l2_dist = (mu_ben - mu_tol).norm().item()
    cos_sim = F.cosine_similarity(mu_ben.unsqueeze(0), mu_tol.unsqueeze(0)).item()

    y_ben_pred = final_reconstructed[idx_ben].cpu().numpy()
    y_tol_pred = final_reconstructed[idx_tol].cpu().numpy()

print("\n==== 類似度評価結果 (μベクトル) ====")
print(f"ベンゼン vs トルエン  L2距離: {l2_dist:.4f}  コサイン類似度: {cos_sim:.4f}")


# ==========================================
# 7. グラフ描画
# ==========================================
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.plot(loss_history, color="purple")
plt.title("Learning Curve")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(TARGET_X, spectra[idx_ben], label="Original", color="black")
plt.plot(TARGET_X, y_ben_pred, label="Reconstructed", color="blue", linestyle="--")
plt.title("Benzene")
plt.xlabel("Wavenumber (cm-1)")
plt.gca().invert_xaxis()
plt.grid(True)
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(TARGET_X, spectra[idx_tol], label="Original", color="black")
plt.plot(TARGET_X, y_tol_pred, label="Reconstructed", color="red", linestyle="--")
plt.title("Toluene")
plt.xlabel("Wavenumber (cm-1)")
plt.gca().invert_xaxis()
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("result.png", dpi=150)
print("グラフを result.png へ保存しました")
plt.show()