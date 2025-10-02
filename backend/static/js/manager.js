const uploadInput = document.getElementById("photo-upload");
const captureTrigger = document.getElementById("capture-trigger");
const dialog = document.getElementById("camera-dialog");
const preview = document.getElementById("camera-preview");
const canvas = document.getElementById("camera-canvas");
const toast = document.getElementById("toast");

let mediaStream = null;

function showToast(message, isError = false) {
  if (!toast) return;
  toast.textContent = message;
  toast.classList.toggle("error", isError);
  toast.classList.add("show");
  window.setTimeout(() => toast.classList.remove("show"), 3600);
}

async function uploadBlob(blob, filename = "capture.jpg") {
  const formData = new FormData();
  formData.append("image", blob, filename);

  const response = await fetch("/upload_face", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    const message = data.message || data.error || "上傳失敗，請稍後再試";
    throw new Error(message);
  }

  return response.json();
}

async function handleFileUpload(event) {
  const [file] = event.target.files || [];
  if (!file) return;

  showToast("上傳中，請稍候...");

  try {
    const data = await uploadBlob(file, file.name);
    showToast("上傳完成，重新整理頁面...");
    if (data.member_id) {
      const params = new URLSearchParams(window.location.search);
      params.set("member_id", data.member_id);
      window.location.search = params.toString();
    }
  } catch (error) {
    console.error(error);
    showToast(error.message, true);
  } finally {
    event.target.value = "";
  }
}

async function openCameraDialog() {
  if (!dialog) return;
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
    preview.srcObject = mediaStream;
    dialog.showModal();
  } catch (error) {
    console.error(error);
    showToast("無法啟動攝影機：" + (error.message || "請確認權限"), true);
  }
}

function closeCameraDialog() {
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
  }
  mediaStream = null;
  if (dialog.open) {
    dialog.close();
  }
}

async function captureAndUpload() {
  if (!mediaStream || !canvas || !preview) {
    showToast("攝影機尚未就緒", true);
    return;
  }

  const videoTrack = mediaStream.getVideoTracks()[0];
  const settings = videoTrack.getSettings();
  const width = settings.width || preview.videoWidth || 1280;
  const height = settings.height || preview.videoHeight || 720;
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  context.drawImage(preview, 0, 0, width, height);

  const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.9));
  if (!blob) {
    showToast("截圖失敗，請重試", true);
    return;
  }

  showToast("上傳攝影機截圖中...");

  try {
    const data = await uploadBlob(blob, "esp32cam.jpg");
    showToast("辨識完成，重新整理頁面...");
    if (data.member_id) {
      const params = new URLSearchParams(window.location.search);
      params.set("member_id", data.member_id);
      window.location.search = params.toString();
    }
  } catch (error) {
    console.error(error);
    showToast(error.message, true);
  } finally {
    closeCameraDialog();
  }
}

if (uploadInput) {
  uploadInput.addEventListener("change", handleFileUpload);
}

if (captureTrigger) {
  captureTrigger.addEventListener("click", openCameraDialog);
}

if (dialog) {
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) {
      closeCameraDialog();
    }
  });
  dialog.querySelectorAll("[data-action='close']").forEach((button) => {
    button.addEventListener("click", closeCameraDialog);
  });
  dialog.querySelectorAll("[data-action='capture']").forEach((button) => {
    button.addEventListener("click", captureAndUpload);
  });
}

window.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeCameraDialog();
  }
});
