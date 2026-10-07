const videoInput = document.getElementById("videoInput");
const analyzeButton = document.getElementById("analyzeButton");
const status = document.getElementById("status");

videoInput.addEventListener("change", () => {
    if (videoInput.files.length > 0) {
        status.textContent = "Video selected";
    } else {
        status.textContent = "";
    }
});

analyzeButton.addEventListener("click", async () => {

    if (videoInput.files.length === 0) {
        status.textContent = "Please choose a video first.";
        return;
    }

    const file = videoInput.files[0];

    const formData = new FormData();
    formData.append("video", file);

    status.textContent = "Uploading and processing video...";
    analyzeButton.disabled = true;

    try {
        const response = await fetch("/upload", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error("Upload failed");
        }

        const data = await response.json();

        if (data.success) {
            status.textContent = "Video processed successfully!";
        } else {
            status.textContent = data.message || "Processing failed.";
        }

    } catch (error) {
        console.error(error);
        status.textContent = "Something went wrong while processing the video.";
    }

    analyzeButton.disabled = false;
});