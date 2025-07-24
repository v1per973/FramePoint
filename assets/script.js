// script.js
window.addEventListener("DOMContentLoaded", async () => {
  await tf.setBackend('webgl');
  await tf.ready();

  const $ = id => document.getElementById(id);
  const video = $("video"), canvas = $("canvas"), ctx = canvas.getContext("2d");
  const info = $("info");
  const cameraSelect = $("cameraSelect");
  const settingsPanel = $("settingsPanel"), settingsToggleIcon = $("settingsToggleIcon");

  const fixedCanvasWidth = 1920;
  const fixedCanvasHeight = 1080;
  canvas.width = fixedCanvasWidth;
  canvas.height = fixedCanvasHeight;

  settingsPanel.style.display = "none";
  settingsToggleIcon.textContent = "⚙";

  settingsToggleIcon.onclick = () => {
    settingsPanel.style.display =
      settingsPanel.style.display === "none" ? "block" : "none";
    settingsToggleIcon.textContent =
      settingsPanel.style.display === "none" ? "⚙" : "✖";
  };

  $("fullscreenBtn").onclick = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
    } else {
      document.exitFullscreen();
    }
  };

  let pose = null;
  let currentDeviceId = null;
  let lastInfoUpdate = 0;
  let lastStatUpdate = 0;
  let lastCPU = "nm", lastFPS = "nm", lastTTF = "nm";

  async function setupCamera(deviceId) {
    if (window.stream) {
      window.stream.getTracks().forEach(t => t.stop());
      video.srcObject = null;
    }
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        deviceId: deviceId ? { exact: deviceId } : undefined,
        width: { ideal: 1920, min: 1280 },
        height: { ideal: 1080, min: 720 }
      }
    });
    video.srcObject = stream;
    window.stream = stream;
    await video.play();
  }

  const devices = await navigator.mediaDevices.enumerateDevices();
  const cams = devices
    .filter(d => d.kind === "videoinput")
    .sort((a, b) => (a.label || "").toLowerCase().localeCompare((b.label || "").toLowerCase()));
  cams.forEach((d, i) => {
    const o = document.createElement("option");
    o.value = d.deviceId;
    o.text = d.label || `Camera ${i + 1}`;
    cameraSelect.appendChild(o);
  });

  const urlParams = new URLSearchParams(window.location.search);
  const requestedCamera = urlParams.get("camera");
  const foundCamera = cams.find(c => c.deviceId === requestedCamera);

  async function loadModels() {
    pose = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, {
      modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING
    });
  }

  async function initCameraAndModels(deviceId) {
    currentDeviceId = deviceId;
    info.innerText = "Switching camera and reloading models...";
    await setupCamera(deviceId);
    await loadModels();
    info.innerText = "Detecting...";
  }

  cameraSelect.onchange = async () => {
    if (window.stream) {
      window.stream.getTracks().forEach(t => t.stop());
    }
    const selectedId = cameraSelect.value;
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.set("camera", selectedId);
    const newUrl = `${window.location.pathname}?${searchParams.toString()}`;
    window.location.href = newUrl;
  };

  if (cams.length > 0) {
    const startCam = foundCamera ? foundCamera.deviceId : cams[0].deviceId;
    cameraSelect.value = startCam;
    await initCameraAndModels(startCam);
  }

  let lastDetection = performance.now();
  let lastTime = performance.now();
  let logicalTarget = { x: 0.5, y: 0.5 };
  let lastDrawnFrameTime = 0;
  const minFrameInterval = 1000 / 70;

  function drawVideoFrame(video, ctx, cw, ch) {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, cw, ch);
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    const videoRatio = vw / vh;
    const canvasRatio = cw / ch;
    let dw = cw, dh = ch, dx = 0, dy = 0;

    if (videoRatio > canvasRatio) {
      dh = cw / videoRatio;
      dy = (ch - dh) / 2;
    } else {
      dw = ch * videoRatio;
      dx = (cw - dw) / 2;
    }

    ctx.drawImage(video, 0, 0, vw, vh, dx, dy, dw, dh);
    return { dx, dy, dw, dh };
  }

  function getFaceBoundingBox(keypoints) {
    const relevant = ["left_eye", "right_eye", "nose", "left_ear", "right_ear"]
      .map(name => keypoints.find(k => k.name === name && k.score > 0.2))
      .filter(Boolean);
    if (relevant.length < 2) return null;

    const xs = relevant.map(k => k.x);
    const ys = relevant.map(k => k.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);

    return {
      w: maxX - minX,
      h: maxY - minY
    };
  }

  async function detect(timestamp) {
    const vw = video.videoWidth;
    const vh = video.videoHeight;

    if (!pose || timestamp - lastDrawnFrameTime < minFrameInterval || !vw || !vh || !video.srcObject || video.readyState < 2) {
      requestAnimationFrame(detect);
      return;
    }

    lastDrawnFrameTime = timestamp;
    const start = performance.now();
    const cw = fixedCanvasWidth;
    const ch = fixedCanvasHeight;

    const poses = await pose.estimatePoses(video, { maxPoses: 1 });
    let detected = false;
    let keyX = 0.5;
    let keyY = 0.5;
    let dotColor = "#f00";

    if (poses.length > 0 && poses[0].keypoints) {
      const keypoints = poses[0].keypoints;
      const leftEye = keypoints.find(k => k.name === "left_eye" && k.score > 0.2);
      const rightEye = keypoints.find(k => k.name === "right_eye" && k.score > 0.2);

      if (leftEye && rightEye) {
        keyX = (leftEye.x + rightEye.x) / 2 / vw;
        keyY = (leftEye.y + rightEye.y) / 2 / vh;
        logicalTarget.x = keyX;
        logicalTarget.y = keyY;
        dotColor = "#0f0";
        detected = true;
        lastDetection = performance.now();
      } else {
        const rightWrist = keypoints.find(k => k.name === "right_wrist" && k.score > 0.2);
        const leftWrist = keypoints.find(k => k.name === "left_wrist" && k.score > 0.2);
        const wrist = rightWrist || leftWrist;
        if (wrist) {
          keyX = wrist.x / vw;
          keyY = wrist.y / vh;
          logicalTarget.x = keyX;
          logicalTarget.y = keyY;
          dotColor = rightWrist ? "#00f" : "#f00";
          detected = true;
          lastDetection = performance.now();
        }
      }
    }

    const drawResult = drawVideoFrame(video, ctx, cw, ch);
    const x = Math.round(drawResult.dx + logicalTarget.x * drawResult.dw);
    const y = Math.round(drawResult.dy + logicalTarget.y * drawResult.dh);

    ctx.beginPath();
    ctx.arc(x, y, 6, 0, 2 * Math.PI);
    ctx.fillStyle = dotColor;
    ctx.fill();

    const faceBox = getFaceBoundingBox(poses[0].keypoints);
    const z = faceBox ? Math.round(2000 / faceBox.w) : 40;
    const now = performance.now();

    if (now - lastInfoUpdate > 100) {
      info.innerText = detected ? `x:${x} y:${y} z:${z}` : "No detection";
      lastInfoUpdate = now;
    }

    const duration = now - start;
    const delta = now - lastTime;
    lastTime = now;

    if (now - lastStatUpdate > 250) {
      lastCPU = `${Math.round((duration / delta) * 100)}%`;
      lastFPS = `${Math.round(1000 / delta)}`;
      lastTTF = `${Math.round(duration)}ms`;
      lastStatUpdate = now;
      $("cpuUsage").textContent = lastCPU;
      $("fpsDisplay").textContent = lastFPS;
      $("detectTime").textContent = lastTTF;
    }

    requestAnimationFrame(detect);
  }

  requestAnimationFrame(detect);
});
