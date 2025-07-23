window.addEventListener("DOMContentLoaded", async () => {
  const $ = id => document.getElementById(id);
  const video = $("video"), canvas = $("canvas"), ctx = canvas.getContext("2d");
  const info = $("info");
  const cameraSelect = $("cameraSelect");
  const settingsPanel = $("settingsPanel"), settingsToggleIcon = $("settingsToggleIcon");

  const fixedCanvasWidth = 3840;
  const fixedCanvasHeight = 2160;
  canvas.width = fixedCanvasWidth;
  canvas.height = fixedCanvasHeight;

  settingsPanel.style.display = "none";
  settingsToggleIcon.textContent = "⚙";

  let lastStatUpdate = 0;
  let lastInfoUpdate = 0;
  let lastCPU = "nm", lastFPS = "nm", lastTTF = "nm";

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

  let coco = null;
  let pose = null;
  let currentDeviceId = null;

  async function setupCamera(deviceId) {
    if (window.stream) {
      window.stream.getTracks().forEach(t => t.stop());
      video.srcObject = null;
    }
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        deviceId: deviceId ? { exact: deviceId } : undefined,
        width: { ideal: 3840, min: 1920 },
        height: { ideal: 2160, min: 1080 }
      }
    });
    video.srcObject = stream;
    window.stream = stream;
    await video.play();
  }

  const devices = await navigator.mediaDevices.enumerateDevices();
  const cams = devices.filter(d => d.kind === "videoinput").sort((a, b) => (b.label || "").localeCompare(a.label || ""));
  cams.forEach((d, i) => {
    const o = document.createElement("option");
    o.value = d.deviceId;
    o.text = d.label || `Camera ${i + 1}`;
    cameraSelect.appendChild(o);
  });

  async function loadModels() {
    coco = await cocoSsd.load();
    pose = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, {
      modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING
    });
  }

  async function initCameraAndModels(deviceId) {
    currentDeviceId = deviceId;
    info.innerText = "Switching camera and reloading models...";
    await setupCamera(deviceId);
    if (!coco || !pose) await loadModels();
    info.innerText = "Detecting...";
  }

  cameraSelect.onchange = async () => {
    const selectedId = cameraSelect.value;
    if (selectedId !== currentDeviceId) {
      await initCameraAndModels(selectedId);
    }
  };

  if (cams.length > 0) {
    await initCameraAndModels(cams[0].deviceId);
  }

  let lastDetection = performance.now();
  let lastTime = performance.now();
  let logicalTarget = { x: 0.5, y: 0.5 };

  function drawVideoFrame(video, ctx, cw, ch) {
    ctx.clearRect(0, 0, cw, ch);
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
    const relevant = ["left_eye", "right_eye", "nose", "left_ear", "right_ear"].map(name =>
      keypoints.find(k => k.name === name && k.score > 0.4)
    ).filter(Boolean);
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

  function round1(v) {
    return Math.round(v);
  }

  async function detect() {
    const start = performance.now();
    const cw = fixedCanvasWidth;
    const ch = fixedCanvasHeight;

    const vw = video.videoWidth;
    const vh = video.videoHeight;

    if (!vw || !vh || !video.srcObject || video.readyState < 2) {
      requestAnimationFrame(detect);
      return;
    }

    const poses = await pose.estimatePoses(video, { maxPoses: 1 });
    let detected = false;
    let keyX = 0.5;
    let keyY = 0.5;

    if (poses.length > 0 && poses[0].keypoints) {
      const leftEye = poses[0].keypoints.find(k => k.name === "left_eye" && k.score > 0.4);
      const rightEye = poses[0].keypoints.find(k => k.name === "right_eye" && k.score > 0.4);
      if (leftEye && rightEye) {
        keyX = (leftEye.x + rightEye.x) / 2 / vw;
        keyY = (leftEye.y + rightEye.y) / 2 / vh;
        logicalTarget.x = keyX;
        logicalTarget.y = keyY;

        lastDetection = performance.now();
        detected = true;
      }
    }

    const drawResult = drawVideoFrame(video, ctx, cw, ch);

    if (detected) {
      const x = round1(drawResult.dx + logicalTarget.x * drawResult.dw);
      const y = round1(drawResult.dy + logicalTarget.y * drawResult.dh);
      const faceBox = getFaceBoundingBox(poses[0].keypoints);
      const rawDistance = faceBox ? (faceBox.w + faceBox.h) : 40;
      const z = round1(5000 / rawDistance);

      ctx.beginPath();
      ctx.arc(x, y, 6, 0, 2 * Math.PI);
      ctx.fillStyle = "#0f0";
      ctx.fill();

      const now = performance.now();
      if (now - lastInfoUpdate > 100) {
        info.innerText = `x:${x} y:${y} z:${z}`;
        lastInfoUpdate = now;
      }
    } else {
      const now = performance.now();
      if (now - lastInfoUpdate > 100) {
        info.innerText = "No detection (holding position)";
        lastInfoUpdate = now;
      }
    }

    const duration = performance.now() - start;
    const delta = performance.now() - lastTime;
    lastTime = performance.now();

    const now = performance.now();
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

  detect();
});
