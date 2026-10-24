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

function switchTab(tabId) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.getElementById('tab-' + tabId).classList.add('active');

    document.querySelectorAll('nav button').forEach(btn => {
        btn.classList.remove('text-brand-600');
        btn.classList.add('text-slate-400');
    });
    document.getElementById('nav-' + tabId).classList.remove('text-slate-400');
    document.getElementById('nav-' + tabId).classList.add('text-brand-600');
}

function animateSpectrum() {
    const bars = document.querySelectorAll('.audio-bar');
    bars.forEach(bar => {
        bar.style.height = (10 + Math.random() * 90) + '%';
    });
}

function stopSpectrum() {
    const bars = document.querySelectorAll('.audio-bar');
    bars.forEach((bar, index) => {
        let height = 20;
        if (index > 4 && index < 8) height = 95; 
        else height = 10 + Math.random() * 30;
        bar.style.height = height + '%';
    });
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
        "Extracting ArUco scale reference...",
        "Applying Homography perspective warp...",
        "Running Euclidean Distance Transform...",
        "Extracting 48kHz Acoustic FFT...",
        "Cross-referencing IS 456 limits..."
    ];

    spectrumInterval = setInterval(animateSpectrum, 100);

    analysisInterval = setInterval(() => {
        time += 0.1;
        timerText.innerText = time.toFixed(1) + 's';
        
        let phaseIndex = Math.floor(time / 0.7);
        if (phaseIndex < phases.length) {
            statusText.innerText = phases[phaseIndex];
        }

        if (time >= 3.5) {
            clearInterval(analysisInterval);
            clearInterval(spectrumInterval);
            
            visionContainer.classList.remove('scanning');
            audioContainer.classList.remove('analyzing');
            stopSpectrum();
            
            statusText.innerText = "Diagnostics Complete.";
            statusText.classList.remove('pulse-text');
            statusText.classList.remove('text-brand-600');
            statusText.classList.add('text-emerald-600');

            btnReport.style.opacity = '1';
            btnReport.style.transform = 'translateY(0)';
            btnReport.style.pointerEvents = 'auto';

            const navReport = document.getElementById('nav-report');
            navReport.classList.remove('pointer-events-none', 'opacity-50');
            const navAnalyze = document.getElementById('nav-analyze');
            navAnalyze.classList.remove('pointer-events-none', 'opacity-50');
        }
    }, 100);
}