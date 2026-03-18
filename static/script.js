// script.js
document.addEventListener('DOMContentLoaded', () => {
    // --- 1. Lấy các phần tử DOM ---
    const dropArea = document.getElementById('dropArea');
    const fileInput = document.getElementById('fileInput');
    const imagePreview = document.getElementById('imagePreview');
    const analyzeButton = document.getElementById('analyzeButton');
    const errorMessage = document.getElementById('errorMessage');
    
    const uploadSection = document.getElementById('uploadSection');
    const loadingSection = document.getElementById('loadingSection');
    const resultsSection = document.getElementById('resultsSection');
    
    const overallPredictionText = document.getElementById('overallPredictionText');
    const mainConfidenceValue = document.getElementById('mainConfidenceValue');
    const detailedConfidencesBody = document.getElementById('detailedConfidencesBody');
    const featuresSection = document.getElementById('featuresSection');
    const featuresList = document.getElementById('featuresList');

    const originalImageResult = document.getElementById('originalImageResult');
    const fftImageResult = document.getElementById('fftImageResult');
    const gradCamSpatialImageResult = document.getElementById('gradCamSpatialImageResult');
    const gradCamFreqImageResult = document.getElementById('gradCamFreqImageResult');
    const gradCamFusedImageResult = document.getElementById('gradCamFusedImageResult');
    
    const analyzeAnotherButton = document.getElementById('analyzeAnotherButton');

    // Thêm mới cho features visualization
    const featuresVisualization = document.getElementById('featuresVisualization');
    const featuresGrid = document.getElementById('featuresGrid');

    // Thêm mới cho analysis summary
    const analysisSummarySection = document.getElementById('analysisSummarySection');
    const explanationText = document.getElementById('explanationText');
    const fakeSignalsList = document.getElementById('fakeSignalsList');
    const confidenceEstimateValue = document.getElementById('confidenceEstimateValue');

    // Thêm elements cho titles (nếu chưa có trong HTML, JS sẽ tạo động)
    let featuresTitle = document.getElementById('featuresTitle');
    if (!featuresTitle) {
        featuresTitle = document.createElement('h3');
        featuresTitle.id = 'featuresTitle';
        featuresSection.insertBefore(featuresTitle, featuresList);
    }
    let featuresVizTitle = document.getElementById('featuresVizTitle');
    if (!featuresVizTitle) {
        featuresVizTitle = document.createElement('h3');
        featuresVizTitle.id = 'featuresVizTitle';
        featuresVisualization.insertBefore(featuresVizTitle, featuresGrid);
    }

    let selectedFile = null;
    const API_URL = '/predict'; 

    // --- 2. Xử lý Drag & Drop và File Input ---
    dropArea.addEventListener('click', () => {
        if (imagePreview.classList.contains('hidden')) {
            fileInput.click();
        }
    });

    dropArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropArea.classList.add('highlight');
    });

    dropArea.addEventListener('dragleave', () => {
        dropArea.classList.remove('highlight');
    });

    dropArea.addEventListener('drop', (e) => {
        e.preventDefault();
        dropArea.classList.remove('highlight');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // --- 3. Hàm xử lý file được chọn/thả ---
    function handleFile(file) {
        if (file.type.startsWith('image/')) {
            selectedFile = file;
            errorMessage.classList.add('hidden');
            
            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.src = e.target.result;
                imagePreview.classList.remove('hidden');
            };
            reader.readAsDataURL(file);
            
            analyzeButton.disabled = false;
            dropArea.querySelector('p').classList.add('hidden');
        } else {
            selectedFile = null;
            imagePreview.classList.add('hidden');
            analyzeButton.disabled = true;
            errorMessage.textContent = 'Định dạng file không hợp lệ. Vui lòng chọn ảnh.';
            errorMessage.classList.remove('hidden');
            dropArea.querySelector('p').classList.remove('hidden');
        }
    }

    // --- 4. Hàm chuyển trạng thái UI ---
    function switchSection(sectionToShow) {
        [uploadSection, loadingSection, resultsSection].forEach(section => {
            section.classList.add('hidden');
        });
        sectionToShow.classList.remove('hidden');
    }

    // --- 5. Xử lý nút Phân tích ---
    analyzeButton.addEventListener('click', async () => {
        if (!selectedFile) return;

        switchSection(loadingSection);
        
        const formData = new FormData();
        formData.append('image', selectedFile);

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || `Lỗi ${response.status} không xác định từ server.`);
            }

            displayResults(data);

        } catch (error) {
            console.error('Lỗi khi gọi API:', error);
            errorMessage.textContent = `Lỗi phân tích: ${error.message}`;
            errorMessage.classList.remove('hidden');
            switchSection(uploadSection);
            dropArea.querySelector('p').classList.remove('hidden');
        }
    });

    // --- 6. Hàm hiển thị kết quả ---
    function displayResults(data) {
        // Cập nhật Dự đoán Chung và Confidence
        overallPredictionText.textContent = data.overall_prediction.toUpperCase();
        mainConfidenceValue.textContent = `${data.main_confidence}%`;
        
        // Cập nhật màu sắc dự đoán
        overallPredictionText.className = '';
        const predictionBox = overallPredictionText.closest('.prediction-box');
        predictionBox.className = 'prediction-box';
        
        if (data.binary_prediction === 'fake') {
            overallPredictionText.classList.add('prediction-fake');
            predictionBox.classList.add('prediction-fake');
        } else {
            overallPredictionText.classList.add('prediction-real');
            predictionBox.classList.add('prediction-real');
        }

        // Cập nhật Bảng độ tin cậy chi tiết
        detailedConfidencesBody.innerHTML = '';
        const realClasses = ["afhq", "celebahq", "coco", "ffhq", "imagenet"]; // Lấy từ REAL_CLASSES trong app.py
        data.detailed_confidences.forEach(item => {
            const row = detailedConfidencesBody.insertRow();
            const isMainPrediction = item.class_name === data.multi_prediction;
            const isReal = realClasses.includes(item.class_name.toLowerCase());

            let className = isReal ? 'class-real' : 'class-fake';
            if (isMainPrediction) {
                className = data.binary_prediction === 'fake' ? 'class-fake' : 'class-real';
                row.classList.add('main-confidence-row');
            }
            
            const cellName = row.insertCell(0);
            const cellConf = row.insertCell(1);

            cellName.textContent = item.class_name.toUpperCase();
            cellConf.textContent = `${item.confidence}%`;
            
            cellName.classList.add(className);
        });
        
        // Cập nhật Hình ảnh kết quả
        originalImageResult.src = data.original_image_url;
        fftImageResult.src = data.fft_image_url;
        gradCamSpatialImageResult.src = data.grad_cam_spatial_url;
        gradCamFreqImageResult.src = data.grad_cam_freq_url;
        gradCamFusedImageResult.src = data.grad_cam_fused_url;

        // Cập nhật Đặc trưng (text values) - Show cho cả real/fake
        featuresList.innerHTML = '';
        if (Object.keys(data.features).length > 0) {
            featuresSection.classList.remove('hidden');
            const sectionTitle = data.binary_prediction === 'fake' ? 'Fake Artifacts' : 'Natural Features';
            featuresTitle.textContent = sectionTitle;
            
            for (const [key, feat] of Object.entries(data.features)) {
                const dt = document.createElement('dt');
                dt.textContent = (feat.explanation || key.replace(/_/g, ' ').toUpperCase()) + ':';
                
                const dd = document.createElement('dd');
                
                // Xử lý value: single number, object (GLCM/Stat), hoặc direct value_ keys
                let valueText = '';
                if (feat.value !== undefined) {
                    if (typeof feat.value === 'number') {
                        valueText = feat.value.toFixed(3);
                    } else if (typeof feat.value === 'object' && feat.value !== null) {
                        valueText = Object.entries(feat.value).filter(([k]) => !['image', 'explanation', 'description'].includes(k)).map(([k, v]) => `${k}: ${v.toFixed(3)}`).join(', ');
                    }
                } else {
                    // Fallback cho direct value_ keys (e.g., glcm: {value_contrast: num})
                    const valueEntries = Object.entries(feat).filter(([k]) => (k.startsWith('value_') || k === 'value') && k !== 'value_image' && k !== 'value_explanation');
                    valueText = valueEntries.map(([k, v]) => `${k.replace('value_', '')}: ${v.toFixed(3)}`).join(', ');
                }
                dd.textContent = valueText || 'N/A';
                
                featuresList.appendChild(dt);
                featuresList.appendChild(dd);
            }
        } else {
            featuresSection.classList.add('hidden');
        }

        // Thêm mới: Trực quan hóa Đặc trưng (images) - Show cho cả real/fake
        featuresGrid.innerHTML = '';
        if (Object.keys(data.features).length > 0) {
            const gridTitle = data.binary_prediction === 'fake' ? 'Fake Feature Visualizations' : 'Real Feature Visualizations';
            featuresVizTitle.textContent = gridTitle;
            
            for (const [key, feat] of Object.entries(data.features)) {
                if (feat.image) {
                    // Xử lý valueText (reuse code từ trên)
                    let valueText = '';
                    if (feat.value !== undefined) {
                        if (typeof feat.value === 'number') {
                            valueText = feat.value.toFixed(3);
                        } else if (typeof feat.value === 'object' && feat.value !== null) {
                            valueText = Object.entries(feat.value).filter(([k]) => !['image', 'explanation', 'description'].includes(k)).map(([k, v]) => `${k}: ${v.toFixed(3)}`).join(', ');
                        }
                    } else {
                        const valueEntries = Object.entries(feat).filter(([k]) => (k.startsWith('value_') || k === 'value') && k !== 'value_image' && k !== 'value_explanation');
                        valueText = valueEntries.map(([k, v]) => `${k.replace('value_', '')}: ${v.toFixed(3)}`).join(', ');
                    }
                    
                    const card = document.createElement('div');
                    card.className = 'result-card';
                    card.innerHTML = `
                        <h4>${feat.explanation || key.replace(/_/g, ' ').toUpperCase()}</h4>
                        <img src="${feat.image}" alt="${key} visualization" style="max-width: 100%; height: auto;">
                        <p class="explanation-text">Giá trị: ${valueText || 'N/A'}</p>
                        ${feat.description ? `<p class="feature-description" style="font-size: 0.8em; color: #666; margin-top: 5px; font-style: italic;">${feat.description}</p>` : ''}
                    `;
                    featuresGrid.appendChild(card);
                }
            }
            featuresVisualization.classList.remove('hidden');
        } else {
            featuresVisualization.classList.add('hidden');
        }

        // THÊM MỚI: Xử lý Analysis Summary (chỉ khi fake và có analysis_summary)
        if (data.binary_prediction === 'fake' && data.analysis_summary) {
            explanationText.textContent = data.analysis_summary.explanation;
            fakeSignalsList.innerHTML = '';
            data.analysis_summary.fake_signals.forEach(signal => {
                const li = document.createElement('li');
                li.textContent = signal;
                fakeSignalsList.appendChild(li);
            });
            confidenceEstimateValue.textContent = `${data.analysis_summary.confidence_estimate}%`;
            analysisSummarySection.classList.remove('hidden');
        } else {
            analysisSummarySection.classList.add('hidden');
        }

        switchSection(resultsSection);
    }
    
    // --- 7. Xử lý nút Phân tích ảnh khác (Reset) ---
    analyzeAnotherButton.addEventListener('click', resetUI);

    function resetUI() {
        selectedFile = null;
        fileInput.value = '';
        imagePreview.src = '#';
        imagePreview.classList.add('hidden');
        analyzeButton.disabled = true;
        errorMessage.classList.add('hidden');
        featuresSection.classList.add('hidden');
        featuresVisualization.classList.add('hidden');
        analysisSummarySection.classList.add('hidden'); // Thêm mới
        featuresGrid.innerHTML = '';
        fakeSignalsList.innerHTML = ''; // Thêm mới
        
        dropArea.querySelector('p').classList.remove('hidden');
        overallPredictionText.className = '';
        overallPredictionText.closest('.prediction-box').className = 'prediction-box';

        switchSection(uploadSection);
    }
    
    resetUI();
});