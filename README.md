# 🛡️ DeepFake & Synthetic Image Detection Web App

Dự án nghiên cứu và triển khai hệ thống phân loại ảnh thực (Real) và ảnh giả lập (Fake) từ các mô hình AI (GANs, Diffusion Models).Và nếu là ảnh giả sẽ nhận diện được do mô hình AI nào tạo ra. Ứng dụng kết hợp phân tích đa miền (Không gian & Tần số) và cung cấp khả năng giải thích mô hình thông qua XGrad-CAM.

## 🚀 Điểm nổi bật kỹ thuật (Technical Highlights)

* **Kiến trúc Dual-Stream:** Sử dụng đồng thời hai nhánh **EfficientNet-B3** (via `timm`) để trích xuất đặc trưng từ ảnh gốc (Spatial) và phổ tần số (FFT Magnitude).
* **Explainable AI (XAI):** Triển khai thuật toán **XGrad-CAM** để trực quan hóa các vùng "artifacts" mà mô hình tập trung vào để đưa ra quyết định.
* **Phân tích Pháp chứng (Digital Forensics):** Tích hợp bộ trích xuất đặc trưng chuyên sâu bao gồm:
    * **PRNU** (Photo Response Non-Uniformity) - Dấu vân tay cảm biến.
    * **Wavelet Transform** & **FFT Analysis** - Phân tích nhiễu tần số cao.
    * **GLCM** & **Local Binary Pattern** - Phân tích cấu trúc bề mặt (Texture).
* **Hỗ trợ đa dạng nguồn giả lập:** Nhận diện 10 loại ảnh fake từ StyleGAN (1, 2, 3), BigGAN, CycleGAN, Stable Diffusion, Latent Diffusion, v.v.

## 🛠️ Công nghệ sử dụng (Tech Stack)

* **Backend:** Python, Flask, Gunicorn.
* **Deep Learning:** PyTorch, Torchvision, TIMM (PyTorch Image Models).
* **Image Processing:** OpenCV, Scikit-image, Scipy, PyWavelets.
* **Visualization:** Matplotlib (Agg backend), XGrad-CAM.
* **Frontend:** HTML5, CSS3, JavaScript (AJAX/Fetch API).

## 📁 Cấu trúc thư mục

```text
├── app/               # Logic xử lý chính
├── models/            # Chứa checkpoint .pth (Xem hướng dẫn tải bên dưới)
├── static/            # CSS, JS và ảnh giao diện
├── templates/         # Giao diện HTML (index.html)
├── app.py             # File khởi chạy Flask server
├── Procfile           # Cấu hình cho deployment (Heroku/Render)
└── requirements.txt   # Danh sách thư viện cần thiết
