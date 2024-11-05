const dropArea = document.getElementById('drop-area');
const fileElem = document.getElementById('fileElem');
const preview = document.getElementById('preview');
const submitBtn = document.getElementById('submitBtn');
const resultContainer = document.getElementById('result-container');
const resultText = document.getElementById('result-text');
const confidence = document.getElementById('confidence');
const errorMessage = document.getElementById('error-message');

// Image size limit in bytes (2 MB)
const MAX_FILE_SIZE = 2 * 1024 * 1024;

// Trigger the hidden input field when clicking "Browse"
document.getElementById('browse').addEventListener('click', function () {
    fileElem.click();
});

// Handle file upload
function handleFiles(files) {
    if (files.length > 0) {
        const file = files[0];

        // Check file size
        if (file.size > MAX_FILE_SIZE) {
            errorMessage.textContent = 'File size exceeds 2MB. Please upload a smaller image.';
            preview.innerHTML = '';
            submitBtn.style.display = 'none';
            return;
        } else {
            errorMessage.textContent = '';
        }

        const reader = new FileReader();

        reader.onload = function (e) {
            const img = document.createElement('img');
            img.src = e.target.result;
            preview.innerHTML = '';
            preview.appendChild(img);
            submitBtn.style.display = 'block'; // Show the classify button
            submitBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }
}

// Prevent form submission on button click
submitBtn.addEventListener('click', async function (event) {
    event.preventDefault(); // Prevent any form submission behavior
    submitBtn.disabled = true;
    submitBtn.innerText = 'Classifying...';

    // Get the uploaded file from fileElem
    const file = fileElem.files[0];

    // Create a FormData object to send the file
    const formData = new FormData();
    formData.append("file", file);

    try {
        // Send a POST request to the FastAPI /predict endpoint
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error("Prediction failed. Please try again.");
        }

        const data = await response.json();

        // Display the result from the API response
        resultText.innerText = data.predictions; // assuming the result is in data.predictions
        confidence.innerText = `Confidence: ${data.confidence}%`; // assuming confidence is in the API response

        resultContainer.style.display = 'block';
    
    } 
    catch (error) {
        errorMessage.textContent = error.message;
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerText = 'Classify Image';
    }
});

// Reset the page for a new image upload
function resetPage() {
    preview.innerHTML = '';
    submitBtn.style.display = 'none';
    submitBtn.disabled = false;
    resultContainer.style.display = 'none';
    errorMessage.textContent = '';
    submitBtn.innerText = 'Classify Image';
}

// Optional: Add reset button functionality to start over
//document.getElementById('resetBtn').addEventListener('click', resetPage);
