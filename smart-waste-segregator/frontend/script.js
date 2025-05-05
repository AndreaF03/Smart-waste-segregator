document.getElementById('uploadForm').addEventListener('submit', async function (e) {
    e.preventDefault();

    const imageInput = document.getElementById('imageInput');
    const file = imageInput.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('image', file);

    try {
        const response = await fetch('http://127.0.0.1:5000/segment', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        document.getElementById('segmentedImage').src = data.segmented_image;
        document.getElementById('classificationLabel').textContent = data.classification;
        document.getElementById('result').style.display = 'block';
    } catch (error) {
        alert('Error uploading or processing image');
        console.error(error);
    }
});
