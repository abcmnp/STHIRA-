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
let completedInspectionId = null;

// api-config.js can set the public API origin for GitHub Pages. Local browser
// use retains localhost, and the forwarded-host branch supports the sandbox check.
const API_BASE_URL = (() => {
    const configuredBase = String(window.STHIRA_API_BASE_URL || '').trim().replace(/\/+$/, '');
    if (configuredBase) return configuredBase;
    const host = window.location.hostname;
    if (/^3000-/.test(host)) {
        return `${window.location.protocol}//${host.replace(/^3000-/, '8000-')}`;
    }
    return 'http://127.0.0.1:8000';
})();
let registrationMode = false;

async function apiRequest(path, options = {}) {
    const response = await fetch(`${API_BASE_URL}${path}`, options);
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
        const detail = typeof body.detail === 'string' ? body.detail : `Request failed (${response.status})`;
        throw new Error(detail);
    }
    return body;
}

function populateBuildingSelect(buildings) {
    const select = document.getElementById('building-select');
    const status = document.getElementById('building-load-status');
    select.replaceChildren();
    if (!buildings.length) {
        const option = document.createElement('option');
        option.value = '';
        option.textContent = 'No buildings available';
        select.appendChild(option);
        select.disabled = true;
        status.textContent = 'The backend has no buildings yet. An admin must add one before inspections can be uploaded.';
        return;
    }
    const placeholder = document.createElement('option');
    placeholder.value = '';
    placeholder.textContent = 'Choose a building';
    select.appendChild(placeholder);
    buildings.forEach(building => {
        const option = document.createElement('option');
        option.value = building.id;
        option.textContent = `${building.name} (#${building.id})`;
        select.appendChild(option);
    });
    select.disabled = false;
    status.textContent = `${buildings.length} building${buildings.length === 1 ? '' : 's'} loaded from the backend.`;
}

async function checkBackendConnection() {
    const status = document.getElementById('backend-status');
    const dot = document.getElementById('backend-status-dot');
    try {
        const health = await apiRequest('/health');
        status.textContent = `Connected · v${health.version}`;
        dot.className = 'w-2 h-2 rounded-full bg-emerald-500';
        document.getElementById('backend-detail-status').textContent = `Connected · API ${health.version} · database ${health.database || 'unknown'}`;
        console.info('[STHIRA] Backend health response:', health);
    } catch (error) {
        status.textContent = 'Backend offline';
        dot.className = 'w-2 h-2 rounded-full bg-red-500';
        document.getElementById('backend-detail-status').textContent = `Offline · ${error.message}`;
        console.error('[STHIRA] Backend health check failed:', error);
    }
}

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
    const analysisPhoto = document.getElementById('analysis-photo');
    analysisPhoto.src = visualObjectUrl;
    analysisPhoto.hidden = false;
    document.getElementById('analysis-photo-placeholder').hidden = true;
    document.getElementById('visual-file-name').textContent = file.name;
    document.getElementById('visual-preview').hidden = false;
    document.getElementById('visual-dropzone').classList.add('has-selection');
}

function clearVisualFile() {
    revokeObjectUrl(visualObjectUrl);
    visualObjectUrl = null;
    document.getElementById('visual-preview-image').removeAttribute('src');
    const analysisPhoto = document.getElementById('analysis-photo');
    analysisPhoto.removeAttribute('src');
    analysisPhoto.hidden = true;
    document.getElementById('analysis-photo-placeholder').hidden = false;
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

function renderInspectionResult(result) {
    document.getElementById('report-verdict-title').textContent = result.verdict?.label || 'Analysis completed';
    document.getElementById('report-verdict-reason').textContent = result.verdict?.reason || 'The backend completed processing but returned no verdict details.';
    document.getElementById('report-crack-width').textContent = result.crack ? Number(result.crack.max_width_mm).toFixed(2) : '—';
    document.getElementById('report-crack-length').textContent = result.crack ? Number(result.crack.length_mm).toFixed(1) : '—';
    const referenceType = result.scale?.reference_type;
    document.getElementById('report-scale-reference').textContent = referenceType === 'id_card'
        ? '6 × 9 cm ID card'
        : referenceType === 'aruco' ? 'STHIRA ArUco marker' : 'No calibrated reference';
    document.getElementById('report-scale-confidence').textContent = result.scale?.confidence !== undefined
        ? `Calibration confidence: ${(Number(result.scale.confidence) * 100).toFixed(0)}%`
        : 'Calibration confidence unavailable';
    document.getElementById('report-crack-limit').textContent = result.exceeds_is456_limit === null || result.exceeds_is456_limit === undefined
        ? 'No limit comparison returned'
        : result.exceeds_is456_limit ? 'Exceeds configured IS 456 comparison' : 'Within configured IS 456 comparison';
    document.getElementById('report-crack-angle').textContent = result.crack ? Number(result.crack.angle_deg).toFixed(0) : '—';
    document.getElementById('report-audio-peak').textContent = result.audio?.dominant_hz ? Number(result.audio.dominant_hz).toFixed(0) : '—';
    const audioClass = result.audio?.classification || 'No audio result';
    document.getElementById('report-crack-depth').textContent = 'Not measured';
    document.getElementById('report-crack-depth-note').textContent = audioClass === 'void' || audioClass === 'solid'
        ? `Optional tap response: ${audioClass}. This may indicate a hollow/solid response, but it does not measure crack depth.`
        : 'A single 2D photo cannot measure subsurface depth. Use a validated depth sensor or separate NDT inspection.';
    const audioBadge = document.getElementById('report-audio-classification');
    audioBadge.textContent = audioClass;
    audioBadge.className = `px-2 py-1 rounded text-[10px] font-medium border ${audioClass === 'void' ? 'bg-amber-50 text-amber-700 border-amber-100' : 'bg-slate-50 text-slate-600 border-slate-100'}`;

    const costList = document.getElementById('cost-items');
    costList.replaceChildren();
    const formatter = new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 });
    if (result.cost?.items?.length) {
        result.cost.items.forEach(item => {
            const row = document.createElement('div');
            row.className = 'flex justify-between items-start text-xs border-b border-slate-100 pb-3';
            const description = document.createElement('span');
            description.className = 'font-medium text-slate-700';
            description.textContent = item.description;
            const amount = document.createElement('span');
            amount.className = 'font-medium text-slate-800 font-mono';
            amount.textContent = formatter.format(item.amount_rs);
            row.append(description, amount);
            costList.appendChild(row);
        });
        document.getElementById('report-cost-total').textContent = formatter.format(result.cost.total_rs);
    } else {
        const empty = document.createElement('p');
        empty.className = 'text-xs text-slate-500';
        empty.textContent = 'No estimate returned for this inspection.';
        costList.appendChild(empty);
        document.getElementById('report-cost-total').textContent = '—';
    }
    console.info('[STHIRA] Completed inspection result:', result);
}

async function downloadInspectionReport() {
    const status = document.getElementById('report-download-status');
    const token = sessionStorage.getItem('sthira_access_token');
    if (!token || !completedInspectionId) {
        status.textContent = 'Complete an inspection and sign in before downloading its PDF.';
        return;
    }
    const button = document.getElementById('download-report-button');
    button.disabled = true;
    status.textContent = 'Preparing the backend PDF report…';
    try {
        const response = await fetch(`${API_BASE_URL}/inspections/${completedInspectionId}/report.pdf`, {
            headers: { Authorization: `Bearer ${token}` }
        });
        if (!response.ok) {
            const body = await response.json().catch(() => ({}));
            throw new Error(typeof body.detail === 'string' ? body.detail : `Download failed (${response.status})`);
        }
        const url = URL.createObjectURL(await response.blob());
        const link = document.createElement('a');
        link.href = url;
        link.download = `STHIRA-inspection-${completedInspectionId}.pdf`;
        document.body.appendChild(link);
        link.click();
        link.remove();
        setTimeout(() => URL.revokeObjectURL(url), 1500);
        status.textContent = 'PDF report downloaded.';
    } catch (error) {
        status.textContent = `PDF download failed: ${error.message}`;
    } finally {
        button.disabled = false;
    }
}

async function startAnalysis() {
    const statusText = document.getElementById('analysis-status');
    const submitStatus = document.getElementById('inspection-submit-status');
    const token = sessionStorage.getItem('sthira_access_token');
    const photo = document.getElementById('visual-camera-input').files[0] || document.getElementById('visual-file-input').files[0];
    const buildingId = document.getElementById('building-select').value;
    const audio = document.getElementById('audio-file-input').files[0];

    if (!token) {
        submitStatus.textContent = 'Sign in first so the backend can save this inspection.';
        loginDialog.showModal();
        return;
    }
    if (!photo) {
        submitStatus.textContent = 'Choose a crack photo before submitting the inspection.';
        return;
    }
    if (!buildingId) {
        submitStatus.textContent = 'Select a building returned by the backend before submitting.';
        document.getElementById('building-load-status').textContent = 'Select a building before submitting.';
        return;
    }
    if (!document.getElementById('inspection-floor').value.trim() || !document.getElementById('inspection-location').value.trim()) {
        submitStatus.textContent = 'Enter both a floor and a location so the backend can record this inspection.';
        return;
    }
    if (audio && !audio.name.toLowerCase().endsWith('.wav')) {
        submitStatus.textContent = 'The backend accepts uncompressed WAV recordings only. Choose a .wav file or remove this audio file.';
        return;
    }

    submitStatus.textContent = 'Sending this photo and inspection metadata to the backend…';
    switchTab('analyze');
    const runButton = document.getElementById('run-diagnostics-button');
    const reportButton = document.getElementById('btn-view-report');
    completedInspectionId = null;
    document.getElementById('download-report-button').classList.add('hidden');
    document.getElementById('report-download-status').textContent = '';
    const visionContainer = document.getElementById('vision-container');
    const audioContainer = document.getElementById('audio-container');
    audioContainer.hidden = !audio;
    reportButton.style.opacity = '0';
    reportButton.style.transform = 'translateY(16px)';
    reportButton.style.pointerEvents = 'none';
    document.getElementById('nav-report').classList.add('pointer-events-none', 'opacity-50');
    document.getElementById('analysis-status').classList.remove('text-emerald-600', 'text-red-600');
    document.getElementById('analysis-status').classList.add('text-brand-600', 'pulse-text');
    visionContainer.classList.add('scanning');
    audioContainer.classList.add('analyzing');
    runButton.disabled = true;
    const startedAt = Date.now();
    const timer = setInterval(() => {
        document.getElementById('analysis-timer').textContent = `${((Date.now() - startedAt) / 1000).toFixed(1)}s`;
    }, 100);
    if (audio) spectrumInterval = setInterval(animateSpectrum, 100);
    else stopSpectrum();

    try {
        statusText.textContent = 'Uploading photo and metadata to backend…';
        const form = new FormData();
        form.append('photo', photo, photo.name);
        form.append('building_id', buildingId);
        form.append('floor', document.getElementById('inspection-floor').value.trim());
        form.append('location', document.getElementById('inspection-location').value.trim());
        form.append('exposure_class', document.getElementById('inspection-exposure').value);
        if (audio) form.append('wav', audio, audio.name);

        let result = await apiRequest('/inspections', {
            method: 'POST',
            headers: { Authorization: `Bearer ${token}` },
            body: form
        });
        statusText.textContent = `Inspection #${result.inspection_id} accepted by backend; status: ${result.status}.`;
        const deadline = Date.now() + 120000;
        while (['pending', 'processing'].includes(result.status) && Date.now() < deadline) {
            await new Promise(resolve => setTimeout(resolve, 1800));
            result = await apiRequest(`/inspections/${result.inspection_id}`, {
                headers: { Authorization: `Bearer ${token}` }
            });
            statusText.textContent = `Backend processing inspection #${result.inspection_id}… status: ${result.status}.`;
        }
        if (result.status === 'failed') throw new Error(result.error_message || 'The backend could not process this photo.');
        if (result.status !== 'completed') throw new Error('The backend is still processing. Open Analyze again in a moment to check the inspection.');

        renderInspectionResult(result);
        completedInspectionId = result.inspection_id;
        document.getElementById('download-report-button').classList.remove('hidden');
        document.getElementById('report-download-status').textContent = `PDF report for inspection #${result.inspection_id} is ready.`;
        statusText.textContent = `Backend analysis complete · inspection #${result.inspection_id}`;
        submitStatus.textContent = `Inspection #${result.inspection_id} completed successfully on the backend.`;
        statusText.classList.remove('pulse-text', 'text-brand-600');
        statusText.classList.add('text-emerald-600');
        reportButton.style.opacity = '1';
        reportButton.style.transform = 'translateY(0)';
        reportButton.style.pointerEvents = 'auto';
        document.getElementById('nav-report').classList.remove('pointer-events-none', 'opacity-50');
        document.getElementById('nav-analyze').classList.remove('pointer-events-none', 'opacity-50');
    } catch (error) {
        statusText.textContent = `Backend request failed: ${error.message}`;
        submitStatus.textContent = `Backend returned an error: ${error.message}`;
        statusText.classList.remove('pulse-text', 'text-brand-600');
        statusText.classList.add('text-red-600');
        console.error('[STHIRA] Inspection request failed:', error);
    } finally {
        clearInterval(timer);
        clearInterval(spectrumInterval);
        visionContainer.classList.remove('scanning');
        audioContainer.classList.remove('analyzing');
        stopSpectrum();
        runButton.disabled = false;
    }
}

const loginDialog = document.getElementById('login-dialog');
document.getElementById('login-button').addEventListener('click', () => loginDialog.showModal());
document.getElementById('close-login-button').addEventListener('click', () => loginDialog.close());
loginDialog.addEventListener('click', event => {
    if (event.target === loginDialog) loginDialog.close();
});
const authModeToggle = document.getElementById('auth-mode-toggle');
authModeToggle.addEventListener('click', () => {
    registrationMode = !registrationMode;
    document.getElementById('login-title').textContent = registrationMode ? 'Create account' : 'Sign in';
    document.querySelector('.login-submit').textContent = registrationMode ? 'Create account' : 'Continue';
    authModeToggle.textContent = registrationMode ? 'Already have an account? Sign in' : 'Create an inspector account';
    document.getElementById('login-status').textContent = registrationMode
        ? 'New accounts are created with the inspector role.'
        : 'Enter your local inspector credentials.';
});
document.getElementById('login-form').addEventListener('submit', async event => {
    event.preventDefault();
    const status = document.getElementById('login-status');
    const submitButton = document.querySelector('.login-submit');
    const username = document.getElementById('login-email').value.trim();
    const password = document.getElementById('login-password').value;
    submitButton.disabled = true;
    status.textContent = registrationMode ? 'Creating inspector account…' : 'Signing in…';
    try {
        if (registrationMode) {
            await apiRequest('/auth/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password })
            });
            registrationMode = false;
            document.getElementById('login-title').textContent = 'Sign in';
            submitButton.textContent = 'Continue';
            authModeToggle.textContent = 'Create an inspector account';
        }
        const credentials = new URLSearchParams({ username, password });
        const tokenResult = await apiRequest('/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: credentials
        });
        sessionStorage.setItem('sthira_access_token', tokenResult.access_token);
        const buildings = await apiRequest('/buildings', {
            headers: { Authorization: `Bearer ${tokenResult.access_token}` }
        });
        populateBuildingSelect(buildings);
        status.textContent = `Signed in. ${buildings.length} building${buildings.length === 1 ? '' : 's'} available.`;
        console.info('[STHIRA] Authenticated building list:', buildings);
        document.getElementById('login-button').innerHTML = '<i class="fa-solid fa-user-check" aria-hidden="true"></i><span>Signed in</span>';
        setTimeout(() => loginDialog.close(), 900);
    } catch (error) {
        status.textContent = error.message || 'Could not connect to the backend.';
        console.error('[STHIRA] Authentication request failed:', error);
    } finally {
        submitButton.disabled = false;
    }
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

document.getElementById('run-diagnostics-button').addEventListener('click', startAnalysis);
document.getElementById('refresh-backend-button').addEventListener('click', checkBackendConnection);
document.getElementById('download-report-button').addEventListener('click', downloadInspectionReport);

async function restoreAuthSession() {
    const token = sessionStorage.getItem('sthira_access_token');
    if (!token) return;
    try {
        const buildings = await apiRequest('/buildings', { headers: { Authorization: `Bearer ${token}` } });
        populateBuildingSelect(buildings);
        document.getElementById('login-button').innerHTML = '<i class="fa-solid fa-user-check" aria-hidden="true"></i><span>Signed in</span>';
    } catch (error) {
        sessionStorage.removeItem('sthira_access_token');
        document.getElementById('building-load-status').textContent = 'Your session expired. Sign in again to load buildings.';
        console.warn('[STHIRA] Saved session could not be restored:', error);
    }
}

checkBackendConnection();
restoreAuthSession();
