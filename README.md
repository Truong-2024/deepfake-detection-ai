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

## 🖼️ Demo & Kết quả thực nghiệm

### Giao diện hệ thống
Ứng dụng Web cho phép người dùng tải lên hình ảnh và nhận kết quả phân tích thời gian thực.
![Giao diện chính](screenshots/giaodien.png)
![tải ảnh lên](screenshots/taianh.png)
![Kết quả phân tích](screenshots/phantich.png)
### Kết quả phân loại nguồn gốc
Dựa trên kiến trúc Dual-Branch, mô hình có khả năng phân loại chính xác nguồn gốc ảnh từ nhiều cấu trúc AI khác nhau.
![Kết quả phân tích chi tiết](screenshots/phantichanhAI.png)
### Để tăng tính giải thích, ứng dụng hiển thị các phân tích đặc trưng bổ sung dựatrên các chỉ số đã tính toán trong mô hình và ý nghĩa của từng đặc trưng, tùy thuộc vào quá trình phân loại và biết được do mô hình AI
![Bằng chứng kết quả](screenshots/bangchung.png)
![Bằng chứng kết quả bản đồ trực quan](screenshots/bangchung2.png)
### Phân tích trực quan với XGrad-CAM
Hệ thống sử dụng phương pháp giải thích mô hình (Explainable AI) để làm nổi bật các vùng đặc trưng giúp nhận diện ảnh giả lập.
![Kết quả Grad-CAM](screenshots/gradcam.png)

## Kết quả thực nghiệm
### Hiệu suất phân loại nhị phân của Dual-Stream EfficientNet-B3
![Kết quả nhị phân](ketquathucnghiem/NhiphanEfficientNet-B3.png)
### Hiệu suất phân loại đa lớp tập Train EfficientNet-B3
![Kết quả đa lớp Train](ketquathucnghiem/TrainEfficientNet-B3.png)
### Hiệu suất phân loại đa lớp tập Val EfficientNet-B3
![Kết quả đa lớp Train](ketquathucnghiem/ValidationEfficientNet-B3.png)
### Hiệu suất phân loại đa lớp tập Test EfficientNet-B3
![Kết quả đa lớp Train](ketquathucnghiem/ketquathucnghiem/TestValidationEfficientNet-B3.png)

### Trực quan hóa tính năng và dấu vết số
## Đặc trưng PRNU
![Kết quả PRUN](ketquathucnghiem/PRNU.png)

## Đặc trưng PSD
![Kết quả PSD](ketquathucnghiem/PSD.png)

## Đặc trưng Wavelet
![Kết quả Wavelet](ketquathucnghiem/Wavelet.png)

## 📁 Tài nguyên dự án (Project Resources)
* **Model Weights:** [Tải file checkpoint_epoch_eff.pth tại đây](https://drive.google.com/file/d/1jey48XBsVM5ETkW4nfVAw8evInzsTkKH/view?usp=drive_link)

## 📁 Cấu trúc thư mục

```text
├── app/               # Logic xử lý chính
├── models/            # Chứa checkpoint .pth (Xem hướng dẫn tải bên dưới)
├── static/            # CSS, JS và ảnh giao diện
├── templates/         # Giao diện HTML (index.html)
├── app.py             # File khởi chạy Flask server
├── Procfile           # Cấu hình cho deployment (Heroku/Render)
└── requirements.txt   # Danh sách thư viện cần thiết





