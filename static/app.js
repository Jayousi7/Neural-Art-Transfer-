const contentInput = document.getElementById('contentInput');
const styleInput = document.getElementById('styleInput');
const contentDrop = document.getElementById('contentDrop');
const styleDrop = document.getElementById('styleDrop');
const contentPreview = document.getElementById('contentPreview');
const stylePreview = document.getElementById('stylePreview');
const contentPlaceholder = document.getElementById('contentPlaceholder');
const stylePlaceholder = document.getElementById('stylePlaceholder');
const alphaSlider = document.getElementById('alphaSlider');
const alphaValue = document.getElementById('alphaValue');
const generateBtn = document.getElementById('generateBtn');
const btnText = document.getElementById('btnText');
const btnLoader = document.getElementById('btnLoader');
const resultImage = document.getElementById('resultImage');
const downloadBtn = document.getElementById('downloadBtn');
const emptyState = document.getElementById('emptyState');
const presetBtns = document.querySelectorAll('.preset-btn');

let presetStyleFile = null;

function setupDropzone(dropzone, input, preview, placeholder) {
    ['dragenter', 'dragover'].forEach(evt => {
        dropzone.addEventListener(evt, e => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(evt => {
        dropzone.addEventListener(evt, e => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('dragover');
        });
    });

    dropzone.addEventListener('drop', e => {
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            const dt = new DataTransfer();
            dt.items.add(file);
            input.files = dt.files;
            showPreview(file, preview, placeholder, dropzone);
            if (input === styleInput) {
                presetStyleFile = null;
                presetBtns.forEach(b => b.classList.remove('active'));
            }
        }
    });

    input.addEventListener('change', () => {
        if (input.files[0]) {
            showPreview(input.files[0], preview, placeholder, dropzone);
            if (input === styleInput) {
                presetStyleFile = null;
                presetBtns.forEach(b => b.classList.remove('active'));
            }
        }
    });
}

function showPreview(file, preview, placeholder, dropzone) {
    const reader = new FileReader();
    reader.onload = e => {
        preview.src = e.target.result;
        preview.classList.remove('hidden');
        placeholder.classList.add('hidden');
        dropzone.classList.add('has-image');
    };
    reader.readAsDataURL(file);
}

function showPreviewFromUrl(url, preview, placeholder, dropzone) {
    preview.src = url;
    preview.classList.remove('hidden');
    placeholder.classList.add('hidden');
    dropzone.classList.add('has-image');
}

setupDropzone(contentDrop, contentInput, contentPreview, contentPlaceholder);
setupDropzone(styleDrop, styleInput, stylePreview, stylePlaceholder);

presetBtns.forEach(btn => {
    btn.addEventListener('click', async () => {
        const stylePath = btn.dataset.style;

        presetBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        showPreviewFromUrl(stylePath, stylePreview, stylePlaceholder, styleDrop);

        const response = await fetch(stylePath);
        const blob = await response.blob();
        const fileName = stylePath.split('/').pop();
        presetStyleFile = new File([blob], fileName, { type: blob.type });
    });
});

alphaSlider.addEventListener('input', e => {
    alphaValue.textContent = `${e.target.value}%`;
});

generateBtn.addEventListener('click', async () => {
    const styleFile = presetStyleFile || (styleInput.files[0] || null);

    if (!contentInput.files[0] || !styleFile) {
        alert('Please upload a content image and select a style.');
        return;
    }

    const formData = new FormData();
    formData.append('content', contentInput.files[0]);
    formData.append('style', styleFile);
    formData.append('alpha', (alphaSlider.value / 100).toFixed(2));


    btnText.classList.add('hidden');
    btnLoader.classList.remove('hidden');

    generateBtn.disabled = true;

    const progressTrack = document.getElementById('progressTrack');
    if (progressTrack) progressTrack.classList.remove('hidden');
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/style`);

    ws.onopen = async () => {
        ws.send(await contentInput.files[0].arrayBuffer());
        ws.send(await styleFile.arrayBuffer());
        ws.send((alphaSlider.value / 100).toFixed(2));
    };

    ws.onmessage = (event) => {
        if (typeof event.data === 'string'){
            const currentStep = parseInt(event.data);
            const totalSteps = 500;
            const precent = Math.round((currentStep/totalSteps)*100);

            const progressBar = document.getElementById('progressBar');
            if (progressBar){
                progressBar.style.width = `${precent}%`;
            }

            const progressText = document.getElementById('progressText')
            if(progressText){
                progressText.innerText = `${precent}%`;
            }
        }
        else {
        // 3. Every time a new frame arrives, update the image source instantly!
        const url = URL.createObjectURL(event.data);

        resultImage.onload = () => {
            resultImage.classList.remove('hidden');
            downloadBtn.classList.remove('hidden');
            emptyState.classList.add('hidden');
        };

        resultImage.src = url;
        downloadBtn.href = url;
    }

    ws.onclose = () => {
        // 4. Reset the UI when the generation is complete
        btnText.classList.remove('hidden');
        btnLoader.classList.add('hidden');
        generateBtn.disabled = false;
        if (progressTrack) progressTrack.classList.add('hidden');

    };

    ws.onerror = (error) => {
        console.error('WebSocket Error:', error);
        alert('Connection failed.');
        btnText.classList.remove('hidden');
        btnLoader.classList.add('hidden');
        generateBtn.disabled = false;
        if (progressTrack) progressTrack.classList.add('hidden');

    }}});