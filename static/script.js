// Lottie animation
lottie.loadAnimation({
  container: document.getElementById('animation'),
  renderer: 'svg',
  loop: true,
  autoplay: true,
  path: 'animation.json'
});

// Element references
const uploadArea = document.getElementById('uploadArea');
const audioFileInput = document.getElementById('audioFile');
const convertBtn = document.getElementById('convertBtn');
const convertText = document.getElementById('convertText');
const convertSpinner = document.getElementById('convertSpinner');
const fileNameSpan = document.getElementById('fileName');
const progressSection = document.getElementById('progressSection');
const progressBar = document.getElementById('progressBar');
const progressPercent = document.getElementById('progressPercent');
const progressStep = document.getElementById('progressStep');
const youtubeUrlInput = document.getElementById('youtubeUrl');
// Drag & drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('audio/')) {
        audioFileInput.files = e.dataTransfer.files;
        fileNameSpan.innerHTML = `<i class="fas fa-check mr-1"></i> Selected file: ${file.name}`;
    } else {
        alert('Please drop a valid audio file.');
    }
});

// Display selected file name
audioFileInput.addEventListener('change', () => {
    const file = audioFileInput.files[0];
    if (file) {
        fileNameSpan.innerHTML = `<i class="fas fa-check mr-1"></i> Selected file: ${file.name}`;
    }
});

// Conversion
convertBtn.addEventListener('click', async () => {
    const file = audioFileInput.files[0];
    const youtubeUrl = document.getElementById('youtubeUrl')?.value?.trim();
    const format = document.querySelector('input[name="output"]:checked').value;
    const formData = new FormData();

    if (file) {
        formData.append('audio', file);
    } else if (youtubeUrl) {
        formData.append('youtube_url', youtubeUrl);
    } else {
        alert('Please select an audio file or enter a YouTube link.');
        return;
    }
    formData.append('format', format);
    // const format = document.querySelector('input[name="output"]:checked').value;
    // const formData = new FormData();
    // formData.append('audio', file);
    // formData.append('format', format);

    // Show loading state and disable button
    convertText.classList.add('hidden');
    convertSpinner.classList.remove('hidden');
    convertBtn.disabled = true;

    try {
        const headers = new Headers();
        headers.append('ngrok-skip-browser-warning', 'true');

        // Start the conversion process
        const response = await fetch('/convert', {
            method: 'POST',
            headers: headers,
            body: formData
        });

        if (!response.ok) {
            throw new Error('Conversion failed to start');
        }

        const result = await response.json();
        const sessionId = result.session_id;

        // Show progress section and hide convert button
        progressSection.classList.remove('hidden');
        convertBtn.classList.add('hidden');
        
        // Start listening for progress updates
        listenForProgress(sessionId, format);

    } catch (error) {
        console.error(error);
        alert('An error occurred during conversion.');
        resetUI();
    }
});
youtubeUrlInput.addEventListener('input', () => {
    if (youtubeUrlInput.value.trim() !== '') {
        // Clear file input if a YouTube URL is being used
        audioFileInput.value = null;
        fileNameSpan.textContent = '';
    }
});
function listenForProgress(sessionId, format) {
    const eventSource = new EventSource(`/progress/${sessionId}`);
    
    eventSource.onmessage = function(event) {
        try {
            const data = JSON.parse(event.data);
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            // Update progress bar
            const percentage = Math.round((data.progress / data.total) * 100);
            progressBar.style.width = `${percentage}%`;
            progressPercent.textContent = `${percentage}%`;
            progressStep.textContent = data.current_step;
            
            // Check if completed
            if (data.status === 'completed') {
                eventSource.close();
                downloadFile(sessionId, format);
            } else if (data.status === 'error') {
                eventSource.close();
                throw new Error(data.current_step);
            }
            
        } catch (error) {
            eventSource.close();
            console.error('Progress update error:', error);
            alert('An error occurred during processing: ' + error.message);
            resetUI();
        }
    };
    
    eventSource.onerror = function(event) {
        eventSource.close();
        console.error('EventSource failed');
        alert('Connection to server lost during processing.');
        resetUI();
    };
}

async function downloadFile(sessionId, format) {
    try {
        const response = await fetch(`/download/${sessionId}`);
        
        if (!response.ok) {
            throw new Error('Download failed');
        }
        
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        
        // Let the server determine the filename via Content-Disposition header
        // Extract filename from response headers if available
        const contentDisposition = response.headers.get('Content-Disposition');
        if (contentDisposition) {
            const filenameMatch = contentDisposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/);
            if (filenameMatch && filenameMatch[1]) {
                a.download = filenameMatch[1].replace(/['"]/g, '');
            }
        }
        
        a.click();
        window.URL.revokeObjectURL(url);
        
        resetUI();
        
    } catch (error) {
        console.error('Download error:', error);
        alert('Error downloading the converted file.');
        resetUI();
    }
}

function resetUI() {
    convertText.classList.remove('hidden');
    convertSpinner.classList.add('hidden');
    convertBtn.disabled = false;
    convertBtn.classList.remove('hidden');
    progressSection.classList.add('hidden');
    progressBar.style.width = '0%';
    progressPercent.textContent = '0%';
    progressStep.textContent = 'Initializing...';
}

// GSAP animation
gsap.from('h1', { y: -50, opacity: 0, duration: 1, ease: 'power2.out' });
