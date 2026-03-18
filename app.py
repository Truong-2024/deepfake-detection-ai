from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.fft as fft
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import timm
from PIL import Image
import io
import base64
import numpy as np
import os
import cv2
import tempfile
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from scipy.stats import entropy, kurtosis, skew
from scipy import signal
import pywt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
from io import BytesIO
from flask import render_template
app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# ------------------------
# CONFIG
# ------------------------
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
print("Device:", DEVICE)

# Classes (5 real + 10 fake)
REAL_CLASSES = ["afhq", "celebahq", "coco", "ffhq", "imagenet"]
FAKE_CLASSES = [
    "stylegan1", "stylegan2", "stylegan3", "big_gan", "cycle_gan",
    "pro_gan", "projected_gan", "latent_diffusion", "stable_diffusion", "vq_diffusion"
]

# Feature map per fake class (10 classes) - Updated to match original
FEATURE_MAP = {
    "stylegan1": ["PRNU", "High-Freq", "PSD", "Wavelet", "Spectral Entropy"],
    "stylegan2": ["High-Freq", "PSD", "Wavelet", "Spectral Centroid", "Gradient"],
    "stylegan3": ["Spectral Entropy", "Autocorrelation", "PSD", "Wavelet", "Gradient"],
    "big_gan": ["PSD", "Spectral Centroid", "High-Freq", "Gradient", "PRNU"],
    "cycle_gan": ["GLCM", "Wavelet", "PSD", "Spectral Entropy", "Denoise Residual"],
    "pro_gan": ["PRNU", "High-Freq", "PSD", "Autocorrelation", "Wavelet"],
    "projected_gan": ["Spectral Centroid", "PSD", "Gradient", "Wavelet", "PRNU"],
    "latent_diffusion": ["Spectral Entropy", "PSD", "Denoise Residual", "Wavelet", "Gradient"],
    "stable_diffusion": ["Denoise Residual", "PSD", "Spectral Centroid", "Gradient", "Wavelet"],
    "vq_diffusion": ["Spectral Centroid", "PSD", "Wavelet", "Spectral Entropy", "GLCM"]
}

# 4 key features for real images (baseline: PSD, High-Freq, Spectral Entropy, Wavelet)
REAL_KEY_FEATURES = ["PSD", "High-Freq", "Spectral Entropy", "Wavelet"]

ALL_FEATURES = list(set().union(*FEATURE_MAP.values()))

# Detailed descriptions for features (based on forensics research) - Expanded
FEATURE_DESCRIPTIONS = {
    "PRNU": "Photo Response Non-Uniformity (PRNU): Dấu vân tay cảm biến từ nhiễu ảnh, thường bị thiếu hoặc không tự nhiên trong deepfakes do thiếu dữ liệu thực tế.",
    "High-Freq": "High-Frequency Map: Bản đồ tần số cao thể hiện mức độ chi tiết nhỏ (biên, kết cấu); deepfakes thường thiếu thành phần high-frequency do quá trình sinh ảnh làm mịn.",
    "PSD": "Power Spectral Density (PSD): Phân bố công suất tín hiệu theo tần số; các bất thường trong phổ tần số giúp phát hiện artifacts như nén, làm mịn hoặc mất chi tiết trong deepfakes.",
    "Wavelet": "Wavelet Transform: Phân tích đa thang đo (multi-scale) trong miền không gian và tần số, giúp phát hiện các sai lệch tinh vi và khác biệt ở chi tiết tần số cao.",
    "Spectral Entropy": "Spectral Entropy: Đo mức độ ngẫu nhiên (randomness) của phổ tần số; ảnh sinh từ GANs thường có entropy thấp hơn do thiếu nhiễu tự nhiên.",
    "Autocorrelation": "Autocorrelation: Đo sự tương quan tự thân của ảnh, giúp phát hiện các mẫu lặp lại nhân tạo hoặc thiếu tính ngẫu nhiên trong deepfakes.",
    "GLCM": "Gray Level Co-occurrence Matrix (GLCM): Phân tích kết cấu (texture) dựa trên mối tương quan giữa các mức xám lân cận; giúp phát hiện vùng có kết cấu nhân tạo hoặc quá đồng nhất trong ảnh giả.",
    "Gradient": "Gradient Map: Bản đồ gradient thể hiện biên cạnh và thay đổi cường độ; deepfakes thường có gradient không tự nhiên do quá trình học.",
    "Spectral Centroid": "Spectral Centroid: 'Trọng tâm' của phổ tần số, biểu thị vị trí tập trung năng lượng phổ; deepfakes thường lệch về vùng tần số thấp do kết cấu mịn và thiếu chi tiết.",
    "Denoise Residual": "Denoise Residual: Phần dư sau khi khử nhiễu, thể hiện các mẫu nhiễu bất thường hoặc thiếu nhiễu cảm biến tự nhiên – dấu hiệu đặc trưng của deepfakes."
}

# ------------------------
# MODEL (DualStreamEffNetB3)
# ------------------------
class DualStreamEffNetB3(nn.Module):
    def __init__(self, num_binary_classes=2, num_fake_classes=10):
        super().__init__()
        self.spatial_cnn = timm.create_model("efficientnet_b3", pretrained=False, num_classes=0)
        self.freq_cnn = timm.create_model("efficientnet_b3", pretrained=False, num_classes=0)
        self.feature_dim = self.spatial_cnn.num_features
        self.fusion = nn.Linear(self.feature_dim * 2, self.feature_dim)
        self.binary_head = nn.Linear(self.feature_dim, num_binary_classes)
        self.multi_class_head = nn.Linear(self.feature_dim, num_fake_classes)  # 10 fake classes

    def forward(self, spatial_x, freq_x):
        s_feat = self.spatial_cnn(spatial_x)
        f_feat = self.freq_cnn(freq_x)
        fused = torch.cat((s_feat, f_feat), dim=1)
        fused = torch.relu(self.fusion(fused))
        binary_out = self.binary_head(fused)
        multi_out = self.multi_class_head(fused)
        return binary_out, multi_out

# ------------------------
# XGrad-CAM implementation
# ------------------------
class XGradCAMDual:
    def __init__(self, model, img_size=224, device=None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.img_size = img_size
        self.model.eval()

    def _find_target_layer(self, cnn_branch):
        children = list(cnn_branch.children())
        for m in reversed(children):
            if isinstance(m, nn.Sequential):
                submods = list(m)
                for sub in reversed(submods):
                    if isinstance(sub, nn.Conv2d):
                        return sub
            elif isinstance(m, nn.Conv2d):
                return m
        for m in reversed(list(cnn_branch.modules())):
            if isinstance(m, nn.Conv2d):
                return m
        raise ValueError("No Conv2d layer found in branch")

    def _register_hooks(self, layer):
        activations, gradients = {}, {}
        def fwd_hook(module, inp, out):
            activations["value"] = out.detach()
        def bwd_hook(module, grad_in, grad_out):
            gradients["value"] = grad_out[0].detach()
        fwd = layer.register_forward_hook(fwd_hook)
        bwd = layer.register_full_backward_hook(bwd_hook)  # Sử dụng full backward hook
        return activations, gradients, fwd, bwd

    def _compute_xgradcam(self, acts, grads):
        eps = 1e-8
        numerator = (grads * acts).mean(dim=(2,3), keepdim=True)
        denominator = 2.0 * (acts ** 2).mean(dim=(2,3), keepdim=True) + eps
        weights = numerator / denominator
        cam = torch.relu((weights * acts).sum(dim=1))
        cam0 = cam[0].cpu().numpy()
        cam0 -= cam0.min()
        if cam0.max() > 0:
            cam0 /= cam0.max()
        cam_resized = cv2.resize(cam0, (self.img_size, self.img_size))
        return cam_resized

    def generate(self, img_spatial, img_freq, branch="fused", target=("multi", None)):
        img_spatial = img_spatial.to(self.device)
        img_freq = img_freq.to(self.device)

        def compute_for(branch_name):
            cnn_branch = self.model.spatial_cnn if branch_name == "spatial" else self.model.freq_cnn
            target_layer = self._find_target_layer(cnn_branch)
            acts, grads, fwd, bwd = self._register_hooks(target_layer)
            self.model.zero_grad()
            b_out, m_out = self.model(img_spatial, img_freq)
            if target[0] == "multi":
                class_idx = target[1] if target[1] is not None else int(m_out.argmax(dim=1).item())
                score = m_out[:, class_idx]
            else:
                class_idx = target[1] if target[1] is not None else int(b_out.argmax(dim=1).item())
                score = b_out[:, class_idx]
            score.sum().backward(retain_graph=True)
            cam = self._compute_xgradcam(acts["value"], grads["value"])
            fwd.remove(); bwd.remove()
            self.model.zero_grad()
            return cam

        if branch == "spatial":
            return {"spatial_cam": compute_for("spatial")}
        elif branch == "frequency":
            return {"freq_cam": compute_for("frequency")}
        else:
            s = compute_for("spatial")
            f = compute_for("frequency")
            fused = (s + f) / 2.0
            fused -= fused.min()
            if fused.max() > 0:
                fused /= fused.max()
            return {"spatial_cam": s, "freq_cam": f, "fused_cam": fused}

    @staticmethod
    def overlay(image, activation, alpha=0.5):
        img = np.float32(image)
        # Đảm bảo activation có cùng số chiều với img
        if len(img.shape) == 2:  # Ảnh xám
            img = np.stack([img, img, img], axis=-1)  # Chuyển thành RGB
        heat = np.uint8(255 * activation)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB) / 255.0
        out = (1 - alpha) * img + alpha * heat
        out = out / (out.max() + 1e-8)
        return out

# ------------------------
# IMAGE & FFT helpers
# ------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def image_to_freq_tensor_prep(pil_img, img_size=IMG_SIZE):
    t = transforms.ToTensor()(pil_img)
    f = torch.fft.fft2(t)
    fshift = torch.fft.fftshift(f)
    mag = torch.log1p(torch.abs(fshift))  # Match original: log1p
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
    mag = TF.resize(mag, (img_size, img_size))
    mag = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])(mag)
    return mag

def preprocess_for_model(pil_img):
    img_spatial = transform(pil_img).unsqueeze(0)
    img_freq = image_to_freq_tensor_prep(pil_img, img_size=IMG_SIZE).unsqueeze(0)
    original_resized_pil = pil_img.resize((IMG_SIZE, IMG_SIZE))
    return img_spatial, img_freq, original_resized_pil

def fft_image(pil_img):
    img_np = np.array(pil_img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
    fshift = np.fft.fftshift(np.fft.fft2(gray))
    mag = 20 * np.log(np.abs(fshift) + 1e-8)
    mag = np.clip((mag - mag.min()) / (mag.max() - mag.min()), 0, 1)
    mag_pil = Image.fromarray((mag * 255).astype(np.uint8))
    return mag_pil

def generate_cams(model, img_spatial, img_freq, binary_label, b_pred, m_pred):
    cammer = XGradCAMDual(model, img_size=IMG_SIZE, device=DEVICE)
    if binary_label == "real":
        cams = cammer.generate(img_spatial, img_freq, branch="fused", target=("binary", b_pred))
    else:
        cams = cammer.generate(img_spatial, img_freq, branch="fused", target=("multi", m_pred))
    return cams

def overlay_heatmap_on_image(pil_img, heatmap_np, alpha=0.5):
    img_np = np.array(pil_img).astype(np.float32) / 255.0
    # Đảm bảo img_np là RGB
    if len(img_np.shape) == 2:  # Nếu là ảnh xám
        img_np = np.stack([img_np, img_np, img_np], axis=-1)
    overlay = XGradCAMDual.overlay(img_np, heatmap_np, alpha=alpha)
    return Image.fromarray((overlay * 255).astype(np.uint8))

def pil_to_base64(pil_img):
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def create_feature_vis_heatmap(data_np, cmap_name='JET', title='Feature Map', size=(224, 224)):
    """Tạo heatmap visualization dùng cv2 cho data 2D."""
    if len(data_np.shape) > 2:
        data_np = np.mean(data_np, axis=-1)  # Nếu RGB, average to gray
    data_norm = cv2.normalize(data_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if cmap_name == 'HOT':
        colormap = cv2.COLORMAP_HOT
    elif cmap_name == 'INFERNO':
        colormap = cv2.COLORMAP_JET  # Approx, cv2 không có inferno, dùng JET
    else:
        colormap = cv2.COLORMAP_JET
    colored = cv2.applyColorMap(data_norm, colormap)
    pil_img = Image.fromarray(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))
    pil_img = pil_img.resize(size)
    return pil_img

def create_feature_vis_plot(fig, size=(224, 224)):
    """Tạo plot visualization dùng matplotlib, save to PIL."""
    buffered = BytesIO()
    fig.savefig(buffered, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buffered.seek(0)
    pil_img = Image.open(buffered)
    pil_img = pil_img.resize(size)
    return pil_img

def feature_PRNU(image_np):
    residual = cv2.GaussianBlur((image_np*255).astype(np.uint8), (5,5), 0).astype(np.float32)/255.0
    prnu = image_np - residual
    val = float(np.mean(np.abs(prnu)))
    # Vis: Grayscale prnu
    prnu_vis = Image.fromarray((np.abs(prnu) * 255).astype(np.uint8)).resize((224, 224))
    prnu_b64 = pil_to_base64(prnu_vis)
    return {"prnu": {"value": val, "image": f"data:image/png;base64,{prnu_b64}", "explanation": "PRNU (mean abs)", "description": FEATURE_DESCRIPTIONS["PRNU"]}}

def feature_highfreq(gray):  # gray float [0,255]
    fshift = np.fft.fftshift(np.fft.fft2(gray))
    mag = 20 * np.log(np.abs(fshift) + 1e-8)
    high_freq_mean = float(np.mean(mag))
    # Vis: Heatmap mag with inferno approx (use JET)
    high_vis = create_feature_vis_heatmap(mag, 'INFERNO', 'High-Frequency Map')
    high_b64 = pil_to_base64(high_vis)
    return {"high_freq": {"value": high_freq_mean, "image": f"data:image/png;base64,{high_b64}", "explanation": "High-Frequency Map (mean)", "description": FEATURE_DESCRIPTIONS["High-Freq"]}}

def feature_spectral_entropy(freq_np):
    p = np.abs(freq_np)**2
    se = float(entropy(p.flatten()))
    # Vis: Bar plot like original
    fig, ax = plt.subplots(figsize=(4,3))
    ax.bar(["Spectral Entropy"], [se], color='orange')
    ax.set_title("Spectral Entropy Value")
    ax.set_ylim(0, 10)
    ent_vis = create_feature_vis_plot(fig)
    ent_b64 = pil_to_base64(ent_vis)
    return {"spectral_entropy": {"value": se, "image": f"data:image/png;base64,{ent_b64}", "explanation": "Spectral Entropy", "description": FEATURE_DESCRIPTIONS["Spectral Entropy"]}}

def feature_psd(gray):  # gray float [0,255]
    f_psd, Pxx = signal.welch(gray.flatten())
    psd_mean = float(np.mean(Pxx))
    # Vis: PSD plot
    fig, ax = plt.subplots(figsize=(6,4))
    ax.semilogy(f_psd, Pxx)
    ax.set_title("Power Spectral Density (PSD)")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Power")
    ax.grid(True)
    psd_vis = create_feature_vis_plot(fig)
    psd_b64 = pil_to_base64(psd_vis)
    return {"psd": {"value": psd_mean, "image": f"data:image/png;base64,{psd_b64}", "explanation": "PSD Plot (mean)", "description": FEATURE_DESCRIPTIONS["PSD"]}}

def feature_wavelet(gray):  # gray float [0,255]
    coeffs2 = pywt.dwt2(gray, 'haar')
    _, (cH, cV, cD) = coeffs2
    e = float(np.mean(np.abs(cH) + np.abs(cV) + np.abs(cD)))
    # Vis: For fake, show 3 subplots; for simplicity, show energy map
    energy_map = np.abs(cH) + np.abs(cV) + np.abs(cD)
    wavelet_vis = create_feature_vis_heatmap(energy_map, 'JET', 'Wavelet Energy Map')
    wavelet_b64 = pil_to_base64(wavelet_vis)
    return {"wavelet": {"value": e, "image": f"data:image/png;base64,{wavelet_b64}", "explanation": "Wavelet Detail Energy (mean)", "description": FEATURE_DESCRIPTIONS["Wavelet"]}}

def feature_autocorrelation(gray):  # gray float [0,255], but matchTemplate needs uint8? Cast inside
    gray_uint8 = gray.astype(np.uint8)
    res = cv2.matchTemplate(gray_uint8, gray_uint8, cv2.TM_CCORR_NORMED)
    val = float(np.mean(res))
    # Vis: Heatmap res with viridis approx (use JET)
    auto_vis = create_feature_vis_heatmap(res, 'JET', 'Autocorrelation Map')
    auto_b64 = pil_to_base64(auto_vis)
    return {"autocorrelation": {"value": val, "image": f"data:image/png;base64,{auto_b64}", "explanation": "Autocorrelation (mean)", "description": FEATURE_DESCRIPTIONS["Autocorrelation"]}}

def feature_glcm(gray):  # gray uint8 [0,255]
    img8 = gray  # Already uint8
    g = graycomatrix(img8, [1], [0, np.pi/4], levels=256, symmetric=True, normed=True)
    contrast = float(graycoprops(g, 'contrast').mean())
    corr = float(graycoprops(g, 'correlation').mean())
    # Vis: Heatmap g[:,:,0,0]
    glcm_mat = g[:, :, 0, 0]
    glcm_vis = create_feature_vis_heatmap(glcm_mat, 'HOT', 'GLCM Matrix')
    glcm_b64 = pil_to_base64(glcm_vis)
    return {"glcm": {"value_contrast": contrast, "value_correlation": corr, "image": f"data:image/png;base64,{glcm_b64}", "explanation": "GLCM (contrast, correlation)", "description": FEATURE_DESCRIPTIONS["GLCM"]}}

def feature_gradient(gray):  # gray float [0,255]
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(gx**2 + gy**2)
    val = float(np.mean(grad))
    # Vis: Normalize grad to [0,255]
    grad_norm = (grad / np.max(grad + 1e-8) * 255).astype(np.uint8)
    grad_vis = Image.fromarray(grad_norm).resize((224, 224))
    grad_b64 = pil_to_base64(grad_vis)
    return {"gradient": {"value": val, "image": f"data:image/png;base64,{grad_b64}", "explanation": "Gradient Map (mean)", "description": FEATURE_DESCRIPTIONS["Gradient"]}}

def feature_spectral_centroid(gray):  # gray float [0,255]
    f_welch, Pxx = signal.welch(gray.flatten())
    centroid = np.sum(f_welch * Pxx) / (np.sum(Pxx) + 1e-8)
    centroid_val = float(centroid)
    # Vis: Bar plot like original
    fig, ax = plt.subplots(figsize=(4,3))
    ax.bar(["Spectral Centroid"], [centroid_val], color='purple')
    ax.set_title("Spectral Centroid Value")
    ax.set_ylim(0, 0.5)
    cent_vis = create_feature_vis_plot(fig)
    cent_b64 = pil_to_base64(cent_vis)
    return {"spectral_centroid": {"value": centroid_val, "image": f"data:image/png;base64,{cent_b64}", "explanation": "Spectral Centroid", "description": FEATURE_DESCRIPTIONS["Spectral Centroid"]}}

def feature_denoise_residual(gray):  # gray float [0,255]
    # MedianBlur on float ok, but cast uint8 for cv2
    gray_uint8 = gray.astype(np.uint8)
    blur = cv2.medianBlur(gray_uint8, 5).astype(np.float32)
    residual = gray - blur
    res_mean = float(np.mean(np.abs(residual)))
    # Vis: Grayscale residual
    res_vis = Image.fromarray((np.abs(residual) / np.max(np.abs(residual) + 1e-8) * 255).astype(np.uint8)).resize((224, 224))
    res_b64 = pil_to_base64(res_vis)
    return {"denoise_residual": {"value": res_mean, "image": f"data:image/png;base64,{res_b64}", "explanation": "Denoise Residual (mean)", "description": FEATURE_DESCRIPTIONS["Denoise Residual"]}}

def extract_features_full(image_np, img_freq_tensor, features):
    gray_uint8 = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)  # uint8 [0,255]
    gray_float = gray_uint8.astype(np.float32)  # float [0,255] - Match Kaggle for FFT/Welch
    freq_np = img_freq_tensor.squeeze().cpu().numpy() if img_freq_tensor is not None else None
    feature_results = {}
    for f in features:
        try:
            if f == "PRNU":
                feat_dict = feature_PRNU(image_np)
                feature_results.update(feat_dict)
            elif f == "High-Freq":
                feat_dict = feature_highfreq(gray_float)
                feature_results.update(feat_dict)
            elif f == "Spectral Entropy":
                if freq_np is not None:
                    feat_dict = feature_spectral_entropy(freq_np)
                else:
                    feat_dict = {"spectral_entropy": {"value": 0.0, "image": None, "explanation": "N/A", "description": FEATURE_DESCRIPTIONS["Spectral Entropy"]}}
                feature_results.update(feat_dict)
            elif f == "PSD":
                feat_dict = feature_psd(gray_float)
                feature_results.update(feat_dict)
            elif f == "Wavelet":
                feat_dict = feature_wavelet(gray_float)
                feature_results.update(feat_dict)
            elif f == "Autocorrelation":
                feat_dict = feature_autocorrelation(gray_float)
                feature_results.update(feat_dict)
            elif f == "GLCM":
                feat_dict = feature_glcm(gray_uint8)
                feature_results.update(feat_dict)
            elif f == "Gradient":
                feat_dict = feature_gradient(gray_float)
                feature_results.update(feat_dict)
            elif f == "Spectral Centroid":
                feat_dict = feature_spectral_centroid(gray_float)
                feature_results.update(feat_dict)
            elif f == "Denoise Residual":
                feat_dict = feature_denoise_residual(gray_float)
                feature_results.update(feat_dict)
            else:
                pass
        except Exception as e:
            print(f"Warning: Error in feature {f}: {e}")
    return feature_results

# ===== Tải mô hình và checkpoint =====
model = None
try:
    model = DualStreamEffNetB3(num_binary_classes=2, num_fake_classes=len(FAKE_CLASSES)).to(DEVICE)
    CHECKPOINT_PATH = 'models/checkpoint_epoch_eff.pth'
    print(f"Đang cố gắng tải checkpoint từ: {CHECKPOINT_PATH}")
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Không tìm thấy file checkpoint: {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print("Mô hình DualStreamEffNetB3 đã được tải thành công!")
except Exception as e:
    print(f"Lỗi khi tải mô hình hoặc checkpoint: {e}")
    raise Exception(f"Không thể tải mô hình từ {CHECKPOINT_PATH}. Vui lòng kiểm tra file và cấu hình.")

# Route phục vụ index.html
@app.route('/')
def serve_index():
    return render_template('index.html')

# Route xử lý dự đoán
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Mô hình chưa được tải hoặc lỗi cấu hình"}), 500

    if 'image' not in request.files:
        return jsonify({"error": "Không tìm thấy file ảnh"}), 400

    file = request.files['image']
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
        return jsonify({"error": "Định dạng file không hợp lệ. Vui lòng tải lên ảnh (png, jpg, jpeg, gif)."}), 400
    try:
        image_pil = Image.open(io.BytesIO(file.read())).convert("RGB")

        img_spatial, img_freq_tensor, original_resized_pil = preprocess_for_model(image_pil)        
        img_spatial = img_spatial.to(DEVICE)
        img_freq_tensor = img_freq_tensor.to(DEVICE)

        # --- 1. Dự đoán ---
        with torch.no_grad():
            b_out, m_out = model(img_spatial, img_freq_tensor)
            b_pred = int(b_out.argmax(dim=1).item())
            m_pred = int(m_out.argmax(dim=1).item())
            probs_multi = torch.softmax(m_out, dim=1).cpu().numpy()[0]
            main_confidence = min(99, float(probs_multi[m_pred] * 100))  # Cap confidence at 99%
            
            binary_label = "real" if b_pred == 0 else "fake"
            if binary_label == "real":
                multi_label = "real"
            else:
                multi_label = FAKE_CLASSES[m_pred]
            overall_label = binary_label if binary_label == "real" else f"fake ({multi_label})"

        cams = generate_cams(model, img_spatial, img_freq_tensor, binary_label, b_pred, m_pred) 

        # --- 3. Tạo ảnh FFT (để hiển thị) ---
        fft_display_pil = fft_image(original_resized_pil)
        
        # --- 4. Overlay CAMs ---
        grad_cam_on_rgb_pil = overlay_heatmap_on_image(original_resized_pil, cams["spatial_cam"])
        grad_cam_on_fft_pil = overlay_heatmap_on_image(fft_display_pil, cams["freq_cam"])
        grad_cam_fused_pil = overlay_heatmap_on_image(original_resized_pil, cams["fused_cam"])

        # Chuyển đổi các ảnh kết quả sang Base64
        def pil_to_base64(pil_img):
            buffered = io.BytesIO()
            pil_img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        original_img_base64 = pil_to_base64(original_resized_pil)
        fft_img_base64 = pil_to_base64(fft_display_pil)
        grad_cam_on_fft_base64 = pil_to_base64(grad_cam_on_fft_pil)
        grad_cam_on_rgb_base64 = pil_to_base64(grad_cam_on_rgb_pil)
        grad_cam_fused_base64 = pil_to_base64(grad_cam_fused_pil)

        # --- 5. Detailed confidences (với cap 99%) ---
        detailed_confidences = []
        sorted_idx = probs_multi.argsort()[::-1]
        for idx in sorted_idx:
            class_name = FAKE_CLASSES[idx]
            confidence = min(99, float(probs_multi[idx] * 100))  # Cap tại 99
            detailed_confidences.append({
                "class_name": class_name,
                "confidence": round(confidence, 2)
            })

        features = {}
        image_np = np.array(original_resized_pil).astype(np.float32) / 255.0
        if binary_label == "fake" and multi_label in FEATURE_MAP:
            features = extract_features_full(image_np, img_freq_tensor, FEATURE_MAP[multi_label])
        elif binary_label == "real":
            features = extract_features_full(image_np, img_freq_tensor, ALL_FEATURES)  # All 10 features for real

        return jsonify({
            "overall_prediction": overall_label,
            "binary_prediction": binary_label,
            "multi_prediction": multi_label,
            "main_confidence": round(main_confidence, 2),
            "detailed_confidences": detailed_confidences,
            "features": features,
            "original_image_url": f"data:image/png;base64,{original_img_base64}",
            "fft_image_url": f"data:image/png;base64,{fft_img_base64}",
            "grad_cam_spatial_url": f"data:image/png;base64,{grad_cam_on_rgb_base64}",
            "grad_cam_freq_url": f"data:image/png;base64,{grad_cam_on_fft_base64}",
            "grad_cam_fused_url": f"data:image/png;base64,{grad_cam_fused_base64}"
        })

    except Exception as e:
        print(f"Lỗi trong quá trình xử lý: {e}")
        return jsonify({"error": f"Lỗi server nội bộ: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
