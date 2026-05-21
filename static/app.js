const contentInput = document.getElementById('contentInput');
const styleInput = document.getElementById('styleInput');
const thresholdSlider = document.getElementById('thresholdSlider');
const sliderValue = document.getElementById('sliderValue');
const generateBtn = document.getElementById('generateBtn');
const loader = document.getElementById('loader');
const canvas = document.getElementById('studioCanvas');
const ctx = canvas.getContext('2d');

let originalImage = new Image();
let stylizedImage = new Image();
let hasStylizedImage = false;

contentInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (event) => {
            originalImage.onload = () => {
                // Resize canvas to match the uploaded image dimensions
                canvas.width = originalImage.width;
                canvas.height = originalImage.height;
                hasStylizedImage = false; // Reset style state
                renderCanvas();
            };
            originalImage.src = event.target.result;
        };
        reader.readAsDataURL(file);
    }
});

function renderCanvas() {
    if (!originalImage.src) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.globalAlpha = 1.0;
    ctx.drawImage(originalImage, 0, 0);

    if (hasStylizedImage) {
        const alpha = thresholdSlider.value / 100;
        ctx.globalAlpha = alpha;
        ctx.drawImage(stylizedImage, 0, 0);
    }
}

thresholdSlider.addEventListener('input', (e) => {
    sliderValue.textContent = `${e.target.value}%`;
    renderCanvas(); 
});

generateBtn.addEventListener('click', async () => {
    if (!contentInput.files[0] || !styleInput.files[0]) {
        alert("Please select both a Content image and a Style reference image first!");
        return;
    }

    const formData = new FormData();
    formData.append("content", contentInput.files[0]);
    formData.append("style", styleInput.files[0]);

    loader.classList.remove('hidden');
    generateBtn.disabled = true;

    try {
        const response = await fetch('http://localhost:8000/api/style', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error("Server processed package with an error status.");

        const blob = await response.blob();
        stylizedImage.onload = () => {
            hasStylizedImage = true;
            renderCanvas();
            loader.classList.add('hidden');
            generateBtn.disabled = false;
        };
        stylizedImage.src = URL.createObjectURL(blob);

    } catch (error) {
        console.error("API Pipeline broken:", error);
        alert("Could not connect to the ML backend. Check console logs.");
        loader.classList.add('hidden');
        generateBtn.disabled = false;
    }
});