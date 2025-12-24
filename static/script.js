const uploadBox = document.getElementById('uploadBox');
const videoInput = document.getElementById('videoInput');
const fileName = document.getElementById('fileName');
const loading = document.getElementById('loading');
const resultsSection = document.getElementById('resultsSection');
const errorMessage = document.getElementById('errorMessage');
const framesGrid = document.getElementById('framesGrid');

// Click to upload
uploadBox.addEventListener('click', () => {
    videoInput.click();
});

// File input change
videoInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

// Drag and drop
uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.classList.add('dragover');
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.classList.remove('dragover');
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.classList.remove('dragover');
    
    if (e.dataTransfer.files.length > 0) {
        const file = e.dataTransfer.files[0];
        if (file.type.startsWith('video/')) {
            videoInput.files = e.dataTransfer.files;
            handleFile(file);
        } else {
            showError('Please upload a video file');
        }
    }
});

function handleFile(file) {
    // Validate file type
    const allowedTypes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska', 'video/webm'];
    if (!allowedTypes.some(type => file.type.includes(type.split('/')[1]))) {
        showError('Invalid file type. Please upload a video file (MP4, AVI, MOV, MKV, WEBM)');
        return;
    }

    // Validate file size (500MB max)
    if (file.size > 500 * 1024 * 1024) {
        showError('File size exceeds 500MB limit');
        return;
    }

    fileName.textContent = `Selected: ${file.name}`;
    fileName.style.display = 'block';
    errorMessage.style.display = 'none';
    resultsSection.style.display = 'none';
    
    uploadVideo(file);
}

function uploadVideo(file) {
    const formData = new FormData();
    formData.append('video', file);

    loading.style.display = 'block';
    resultsSection.style.display = 'none';

    fetch('/api/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        loading.style.display = 'none';
        
        if (data.error) {
            showError(data.error);
            return;
        }

        if (data.success) {
            displayResults(data);
        }
    })
    .catch(error => {
        loading.style.display = 'none';
        showError('An error occurred while processing the video: ' + error.message);
        console.error('Error:', error);
    });
}

function displayResults(data) {
    // Update overall result
    const resultLabel = document.getElementById('resultLabel');
    const resultConfidence = document.getElementById('resultConfidence');
    const fakeCount = document.getElementById('fakeCount');
    const realCount = document.getElementById('realCount');
    const totalCount = document.getElementById('totalCount');
    const resultBadge = document.getElementById('resultBadge');

    resultLabel.textContent = data.overall_result;
    resultConfidence.textContent = `${data.overall_confidence}% Confidence`;
    fakeCount.textContent = data.fake_frames;
    realCount.textContent = data.real_frames;
    totalCount.textContent = data.total_frames_analyzed;
    
    // Update displayed frames count
    const displayedCount = document.getElementById('displayedCount');
    if (displayedCount) {
        displayedCount.textContent = data.frames_displayed || data.frames.length;
    }

    // Update badge color
    resultBadge.className = 'result-badge';
    if (data.overall_result === 'FAKE') {
        resultBadge.style.background = 'rgba(255, 107, 107, 0.3)';
    } else {
        resultBadge.style.background = 'rgba(81, 207, 102, 0.3)';
    }

    // Clear previous frames
    framesGrid.innerHTML = '';

    // Display frames
    data.frames.forEach((frame, index) => {
        const frameCard = createFrameCard(frame, index);
        framesGrid.appendChild(frameCard);
    });

    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function createFrameCard(frame, index) {
    const card = document.createElement('div');
    card.className = 'frame-card';

    const isFake = frame.label === 'FAKE';
    const labelClass = isFake ? 'fake' : 'real';

    card.innerHTML = `
        <img src="${frame.image}" alt="Frame ${frame.frame_index}" class="frame-image">
        <div class="frame-info">
            <span class="frame-label ${labelClass}">${frame.label}</span>
            <div class="frame-details">
                <span class="frame-index">Frame #${frame.frame_index}</span>
                <span class="confidence-text">${frame.confidence}%</span>
            </div>
            <div class="confidence-bar">
                <div class="confidence-fill ${labelClass}" style="width: ${frame.confidence}%"></div>
            </div>
        </div>
    `;

    return card;
}

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
    loading.style.display = 'none';
    resultsSection.style.display = 'none';
}

