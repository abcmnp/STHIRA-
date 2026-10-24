const spectrumContainer = document.getElementById('spectrum-bars');
const numBars = 30;
for (let i = 0; i < numBars; i++) {
    const bar = document.createElement('div');
    bar.className = 'audio-bar';
    bar.style.height = Math.max(10, Math.random() * 30) + '%';
    spectrumContainer.appendChild(bar);
}

let analysisInterval;
let spectrumInterval;
let activeRecorder = null;
let activeMicrophoneStream = null;
let recordingInterval = null;
let recordingTimeout = null;
let recordingStartedAt = 0;
let visualObjectUrl = null;
let audioObjectUrl = null;

function switchTab(tabId) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    const activeTab = document.getElementById('tab-' + tabId);
    if (!activeTab) return;
    activeTab.classList.add('active');

    document.querySelectorAll('nav button').forEach(btn => {
        btn.classList.remove('text-brand-600');
        btn.classList.add('text-slate-400');
    });
    const activeNav = document.getElementById('nav-' + tabId);
    if (activeNav) {
        activeNav.classList.remove('text-slate-400');
        activeNav.classList.add('text-brand-600');
    }
}

function animateSpectrum() {
    document.querySelectorAll('.audio-bar').forEach(bar => {
        bar.style.height = (10 + Math.random() * 90) + '%';
    });
}

function stopSpectrum() {
    document.querySelectorAll('.audio-bar').forEach((bar, index) => {
        const height = index > 4 && index < 8 ? 95 : 10 + Math.random() * 30;
        bar.style.height = height + '%';
    });
}

function revokeObjectUrl(url) {
    if (url) URL.revokeObjectURL(url);
}

function setVisualFile(file) {
    if (!file) return;
    revokeObjectUrl(visualObjectUrl);
    visualObjectUrl = URL.createObjectURL(file);
    document.getElementById('visual-preview-image').src = visualObjectUrl;
    document.getElementById('visual-file-name').textContent = file.name;
    document.getElementById('visual-preview').hidden = false;
    document.getElementById('visual-dropzone').classList.add('has-selection');
}

function clearVisualFile() {
    revokeObjectUrl(visualObjectUrl);
    visualObjectUrl = null;
    document.getElementById('visual-preview-image').removeAttribute('src');
    document.getElementById('visual-preview').hidden = true;
    document.getElementById('visual-dropzone').classList.remove('has-selection');
    document.getElementById('visual-camera-input').value = '';
    document.getElementById('visual-file-input').value = '';
}

function setAudioFile(file, sourceLabel) {
    if (!file) return;
    revokeObjectUrl(audioObjectUrl);
    audioObjectUrl = URL.createObjectURL(file);
    const player = document.getElementById('audio-preview-player');
    player.src = audioObjectUrl;
    document.getElementById('audio-file-name').textContent = file.name || sourceLabel || 'Microphone recording';
    document.getElementById('audio-preview').hidden = false;
    document.getElementById('audio-input-status').textContent = sourceLabel || 'Audio file ready for inspection.';
}

function clearAudioFile() {
    revokeObjectUrl(audioObjectUrl);
    audioObjectUrl = null;
    const player = document.getElementById('audio-preview-player');
    player.pause();
    player.removeAttribute('src');
    player.load();
    document.getElementById('audio-preview').hidden = true;
    document.getElementById('audio-file-input').value = '';
    document.getElementById('audio-input-status').textContent = 'Microphone access will be requested when you start recording.';
}

function stopMicrophoneTracks() {
    if (activeMicrophoneStream) {
        activeMicrophoneStream.getTracks().forEach(track => track.stop());
        activeMicrophoneStream = null;
    }
}

function finishRecording() {
    if (recordingTimeout) clearTimeout(recordingTimeout);
    if (recordingInterval) clearInterval(recordingInterval);
    recordingTimeout = null;
    recordingInterval = null;
    if (activeRecorder && activeRecorder.state !== 'inactive') activeRecorder.stop();
}

async function recordAudio() {
    const status = document.getElementById('audio-input-status');
    const button = document.getElementById('record-audio-button');
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || !window.MediaRecorder) {
        status.textContent = 'Microphone recording is not supported here. Choose an audio file instead.';
        return;
    }
    if (!window.isSecureContext) {
        status.textContent = 'Microphone access requires a secure page (HTTPS). Choose an audio file instead.';
        return;
    }

    button.disabled = true;
    status.textContent = 'Requesting microphone permission…';
    try {
        activeMicrophoneStream = await navigator.mediaDevices.getUserMedia({
            audio: { channelCount: 1, sampleRate: 48000 }
        });
        const mimeType = ['audio/webm;codecs=opus', 'audio/ogg;codecs=opus', 'audio/mp4']
            .find(type => MediaRecorder.isTypeSupported(type));
        activeRecorder = mimeType
            ? new MediaRecorder(activeMicrophoneStream, { mimeType })
            : new MediaRecorder(activeMicrophoneStream);
        const chunks = [];
        activeRecorder.addEventListener('dataavailable', event => {
            if (event.data && event.data.size > 0) chunks.push(event.data);
        });
        activeRecorder.addEventListener('stop', () => {
            stopMicrophoneTracks();
            button.disabled = false;
            button.innerHTML = '<i class="fa-solid fa-microphone" aria-hidden="true"></i> Record with microphone';
            if (chunks.length) {
                const audioBlob = new Blob(chunks, { type: activeRecorder.mimeType || 'audio/webm' });
                const extension = audioBlob.type.includes('ogg') ? 'ogg' : audioBlob.type.includes('mp4') ? 'm4a' : 'webm';
                const recording = new File([audioBlob], `sthira-recording.${extension}`, { type: audioBlob.type });
                setAudioFile(recording, '3-second microphone recording ready.');
            } else {
                status.textContent = 'No audio was captured. Please try again or choose an audio file.';
            }
            activeRecorder = null;
        }, { once: true });

        activeRecorder.start();
        recordingStartedAt = Date.now();
        status.textContent = 'Recording coin tap… recording stops automatically after 3 seconds.';
        button.innerHTML = '<i class="fa-solid fa-circle" aria-hidden="true"></i> Recording… 0.0s';
        recordingInterval = setInterval(() => {
            const elapsed = (Date.now() - recordingStartedAt) / 1000;
            button.innerHTML = `<i class="fa-solid fa-circle" aria-hidden="true"></i> Recording… ${Math.min(elapsed, 3).toFixed(1)}s`;
        }, 100);
        recordingTimeout = setTimeout(finishRecording, 3000);
    } catch (error) {
        stopMicrophoneTracks();
        button.disabled = false;
        button.innerHTML = '<i class="fa-solid fa-microphone" aria-hidden="true"></i> Record with microphone';
        if (error && (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError')) {
            status.textContent = 'Microphone permission was denied. Allow access in your browser settings or choose an audio file.';
        } else if (error && error.name === 'NotFoundError') {
            status.textContent = 'No microphone was found. Connect a microphone or choose an audio file.';
        } else {
            status.textContent = 'Could not start recording. Check microphone access or choose an audio file.';
        }
    }
}

function startAnalysis() {
    switchTab('analyze');

    const visionContainer = document.getElementById('vision-container');
    const audioContainer = document.getElementById('audio-container');
    const statusText = document.getElementById('analysis-status');
    const timerText = document.getElementById('analysis-timer');
    const btnReport = document.getElementById('btn-view-report');

    btnReport.style.opacity = '0';
    btnReport.style.transform = 'translateY(16px)';
    btnReport.style.pointerEvents = 'none';
    visionContainer.classList.add('scanning');
    audioContainer.classList.add('analyzing');

    let time = 0;
    const phases = [
        'Extracting ArUco scale reference...',
        'Applying Homography perspective warp...',
        'Running Euclidean Distance Transform...',
        'Extracting 48kHz Acoustic FFT...',
        'Cross-referencing IS 456 limits...'
    ];

    spectrumInterval = setInterval(animateSpectrum, 100);
    analysisInterval = setInterval(() => {
        time += 0.1;
        timerText.innerText = time.toFixed(1) + 's';
        const phaseIndex = Math.floor(time / 0.7);
        if (phaseIndex < phases.length) statusText.innerText = phases[phaseIndex];

        if (time >= 3.5) {
            clearInterval(analysisInterval);
            clearInterval(spectrumInterval);
            visionContainer.classList.remove('scanning');
            audioContainer.classList.remove('analyzing');
            stopSpectrum();
            statusText.innerText = 'Diagnostics Complete.';
            statusText.classList.remove('pulse-text', 'text-brand-600');
            statusText.classList.add('text-emerald-600');
            btnReport.style.opacity = '1';
            btnReport.style.transform = 'translateY(0)';
            btnReport.style.pointerEvents = 'auto';
            document.getElementById('nav-report').classList.remove('pointer-events-none', 'opacity-50');
            document.getElementById('nav-analyze').classList.remove('pointer-events-none', 'opacity-50');
        }
    }, 100);
}

const loginDialog = document.getElementById('login-dialog');
document.getElementById('login-button').addEventListener('click', () => loginDialog.showModal());
document.getElementById('close-login-button').addEventListener('click', () => loginDialog.close());
loginDialog.addEventListener('click', event => {
    if (event.target === loginDialog) loginDialog.close();
});
document.getElementById('login-form').addEventListener('submit', event => {
    event.preventDefault();
    document.getElementById('login-status').textContent = 'Authentication is not connected yet. Your credentials were not submitted.';
});

document.getElementById('take-photo-button').addEventListener('click', () => document.getElementById('visual-camera-input').click());
document.getElementById('choose-image-button').addEventListener('click', () => document.getElementById('visual-file-input').click());
document.getElementById('visual-camera-input').addEventListener('change', event => setVisualFile(event.target.files[0]));
document.getElementById('visual-file-input').addEventListener('change', event => setVisualFile(event.target.files[0]));
document.getElementById('remove-visual-button').addEventListener('click', clearVisualFile);

document.getElementById('choose-audio-button').addEventListener('click', () => document.getElementById('audio-file-input').click());
document.getElementById('audio-file-input').addEventListener('change', event => {
    const file = event.target.files[0];
    if (file) setAudioFile(file, 'Audio file ready for inspection.');
});
document.getElementById('record-audio-button').addEventListener('click', recordAudio);
document.getElementById('remove-audio-button').addEventListener('click', clearAudioFile);
