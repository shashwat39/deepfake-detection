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

// Prevent default refresh behavior on button click
function classifyImage(event) {
    
    event.preventDefault(); // Important to prevent any default behavior

    submitBtn.disabled = true;
    submitBtn.innerText = 'Classifying...';

    // Check if there’s a file selected before proceeding
    const file = fileElem.files[0];
    if (!file) {
        console.log("No file")
        errorMessage.textContent = 'Please upload a file before classifying.';
        submitBtn.disabled = false;
        submitBtn.innerText = 'Classify Image';
        return;
    }

    // Create a FormData object to send the file
    const formData = new FormData();
    formData.append("file", file);

    // Fetch request to FastAPI
    fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error("Prediction failed. Please try again.");
        }
        return response.json();
    })
    .then(data => {
        console.log("Data received from API:", data);
        resultText.innerText = data.label; // Display the label
        confidence.innerText = `Confidence: ${data.confidence.toFixed(2)}%`; // Display the confidence
        resultContainer.style.display = 'block';

        submitBtn.style.display = 'none';

    })
    .catch(error => {
        // Handle any errors here
        console.log("Error occured")
        errorMessage.textContent = error.message;
    });
}

submitBtn.addEventListener('click', classifyImage);


// Reset the page for a new image upload
function resetPage() {
    location.reload();
}

// Optional: Add reset button functionality to start over
document.getElementById('resetBtn').addEventListener('click', resetPage);
