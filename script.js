/*
==================================================
 YOLOv11 Fashion Detection using ONNX Runtime Web
 คลาส: shirt, skirt, trousers
==================================================
*/

// ==================================================
// 1. Initialize HTML Elements
// ==================================================
const imageInput = document.getElementById("imageInput");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const labelDiv = document.getElementById("label");

const dropZone = document.getElementById("dropZone");
const changeBtn = document.getElementById("changeBtn");
const confSlider = document.getElementById("confSlider");
const confValue = document.getElementById("confValue");
const statusDot = document.getElementById("statusDot");
const detectionList = document.getElementById("detectionList");
const timeInfo = document.getElementById("timeInfo");

// สีของแต่ละคลาส (ให้ตรงกับ style.css)  shirt, skirt, trousers
const COLORS = ["#ff5c7a", "#3b82f6", "#f59e0b"];
const ICONS = ["👕", "👗", "👖"];
const IOU_THRESHOLD = 0.45;

let session = null;
let labels = [];
let currentImage = null;   // รูปล่าสุด
let rawBoxes = [];         // ผลจากโมเดล (ยังไม่กรอง) เก็บไว้ให้ slider ใช้ซ้ำ
let hasResult = false;     // รูปปัจจุบันผ่านโมเดลแล้วหรือยัง

function setStatus(text, state) {
    labelDiv.textContent = text;
    statusDot.className = "dot " + state;   // loading | ready | error
}

// ย่อรูปแบบคงสัดส่วน แล้วเติมขอบสีเทา (เหมือนตอนเทรน)
function drawLetterbox(context, img, size) {
    const scale = Math.min(size / img.width, size / img.height);
    const w = img.width * scale;
    const h = img.height * scale;
    const x = (size - w) / 2;
    const y = (size - h) / 2;
    context.fillStyle = "rgb(114,114,114)";
    context.fillRect(0, 0, size, size);
    context.drawImage(img, x, y, w, h);
}

// ==================================================
// 2. Load Model และ Labels
// ==================================================
async function loadModel() {
    // โหลดไฟล์ fashion.onnx (wasm = ใช้ CPU ของ Browser)
    session = await ort.InferenceSession.create("fashion.onnx", {
        executionProviders: ["wasm"]
    });

    // โหลดชื่อคลาส
    const response = await fetch("labels.json");
    labels = await response.json();

    setStatus("โมเดลพร้อมแล้ว เลือกรูปได้เลย", "ready");
    console.log("YOLOv11 Fashion Model Loaded");

    // ถ้าเลือกรูปไว้ก่อนโมเดลโหลดเสร็จ ให้ตรวจจับรูปนั้นทันที
    if (currentImage) runDetection(currentImage);
}

setStatus("กำลังโหลดโมเดล...", "loading");
loadModel().catch(err => {
    setStatus("โหลดโมเดลไม่สำเร็จ: " + err.message, "error");
    console.error(err);
});

// ==================================================
// 3. Image Upload (คลิก / ลากวาง) & Detection
// ==================================================
imageInput.addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (file) handleFile(file);
});

["dragenter", "dragover"].forEach(evt =>
    dropZone.addEventListener(evt, e => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    })
);

["dragleave", "drop"].forEach(evt =>
    dropZone.addEventListener(evt, e => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
    })
);

dropZone.addEventListener("drop", e => {
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) handleFile(file);
});

// เมื่อมีรูปแล้ว คลิกที่รูปจะไม่เปิดเลือกไฟล์ ให้ใช้ปุ่ม "เปลี่ยนรูป" แทน
dropZone.addEventListener("click", e => {
    // บล็อกเฉพาะคลิกบนรูป ไม่บล็อกคลิกที่มาจากปุ่ม "เปลี่ยนรูป" (ซึ่งส่งไปที่ imageInput)
    if (e.target === imageInput) return;
    if (dropZone.classList.contains("has-image")) e.preventDefault();
});

changeBtn.addEventListener("click", () => {
    imageInput.value = "";
    imageInput.click();
});

// เลื่อน slider แล้ววาดใหม่ทันที โดยไม่ต้องรันโมเดลซ้ำ
confSlider.addEventListener("input", () => {
    confValue.textContent = confSlider.value + "%";
    if (currentImage && hasResult) render();
});

function handleFile(file) {
    const img = new Image();

    img.onload = () => {
        currentImage = img;
        hasResult = false;
        rawBoxes = [];
        timeInfo.textContent = "";
        dropZone.classList.add("has-image");
        changeBtn.hidden = false;

        // แสดงรูปต้นฉบับแบบคงสัดส่วน (YOLOv11 ใช้ 640x640)
        canvas.width = 640;
        canvas.height = 640;
        drawLetterbox(ctx, img, 640);

        if (!session) {
            // ยังโหลดโมเดลไม่เสร็จ -> loadModel() จะเรียก runDetection ให้เองเมื่อเสร็จ
            setStatus("รอโมเดลโหลดเสร็จ แล้วจะตรวจจับให้อัตโนมัติ...", "loading");
            return;
        }

        runDetection(img);
    };

    img.src = URL.createObjectURL(file);
}

async function runDetection(img) {
    try {
        setStatus("กำลังตรวจจับ...", "loading");
        const t0 = performance.now();

        // เตรียม Tensor
        const tensor = preprocessImage(img);

        // รันโมเดล
        const feeds = {};
        feeds[session.inputNames[0]] = tensor;
        const result = await session.run(feeds);
        const outputTensor = result[session.outputNames[0]];
        console.log("Output shape:", outputTensor.dims);

        // ถ้าระหว่างรันมีการเปลี่ยนรูป ให้ทิ้งผลเก่า
        if (img !== currentImage) return;

        // เก็บกล่องทั้งหมดที่คะแนน >= 5% ไว้ก่อน แล้วค่อยกรองตาม slider
        rawBoxes = postprocessOutput(outputTensor, 0.05);
        hasResult = true;

        const best = rawBoxes.reduce((m, b) => Math.max(m, b.score), 0);
        console.log(`จำนวนกล่องที่คะแนน >= 5%: ${rawBoxes.length}, คะแนนสูงสุด: ${(best * 100).toFixed(1)}%`);

        const ms = performance.now() - t0;
        timeInfo.textContent = `ใช้เวลาประมวลผล ${ms.toFixed(0)} ms`;

        render();
    } catch (err) {
        setStatus("ตรวจจับไม่สำเร็จ: " + err.message, "error");
        console.error(err);
    }
}

// กรองตามค่า slider -> NMS -> วาด -> แสดงผล
function render() {
    const threshold = confSlider.value / 100;
    let boxes = rawBoxes
        .filter(b => b.score >= threshold)
        .map(b => ({ ...b }));
    boxes = nonMaxSuppression(boxes, IOU_THRESHOLD);

    ctx.clearRect(0, 0, 640, 640);
    drawLetterbox(ctx, currentImage, 640);
    drawBoxes(boxes);
    showResult(boxes);
}

// ==================================================
// 4. Image Preprocessing (Image -> Tensor [1,3,640,640])
// ==================================================
function preprocessImage(img) {
    const size = 640;
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = size;
    tempCanvas.height = size;
    const tempCtx = tempCanvas.getContext("2d");
    drawLetterbox(tempCtx, img, size);

    const data = tempCtx.getImageData(0, 0, size, size).data;
    const float32 = new Float32Array(3 * size * size);

    for (let i = 0; i < size * size; i++) {
        float32[i] = data[i * 4] / 255;                         // R
        float32[i + size * size] = data[i * 4 + 1] / 255;       // G
        float32[i + 2 * size * size] = data[i * 4 + 2] / 255;   // B
    }

    return new ort.Tensor("float32", float32, [1, 3, size, size]);
}

// ==================================================
// 5. YOLO Output Processing
// Output: [1, 7, 8400]  (7 = x,y,w,h + 3 คลาส)
// ==================================================
function postprocessOutput(output, threshold = 0.35) {
    const data = output.data;
    const numAttrs = output.dims[1];
    const numBoxes = output.dims[2];
    const boxes = [];

    for (let i = 0; i < numBoxes; i++) {
        const x = data[i];
        const y = data[numBoxes + i];
        const w = data[numBoxes * 2 + i];
        const h = data[numBoxes * 3 + i];

        // หาคลาสที่คะแนนสูงสุด
        let bestScore = 0;
        let classId = 0;
        for (let c = 4; c < numAttrs; c++) {
            const score = data[numBoxes * c + i];
            if (score > bestScore) {
                bestScore = score;
                classId = c - 4;
            }
        }

        if (bestScore < threshold) continue;

        boxes.push({
            x: x - w / 2,
            y: y - h / 2,
            width: w,
            height: h,
            score: bestScore,
            classId: classId
        });
    }
    return boxes;
}

// ==================================================
// 6. Non-Max Suppression (ลบกล่องที่ซ้อนกัน)
// ==================================================
function nonMaxSuppression(boxes, iouThreshold = 0.45) {
    boxes.sort((a, b) => b.score - a.score);
    const selected = [];

    while (boxes.length) {
        const current = boxes.shift();
        selected.push(current);
        boxes = boxes.filter(box =>
            box.classId !== current.classId ||
            calculateIoU(current, box) < iouThreshold
        );
    }
    return selected;
}

function calculateIoU(a, b) {
    const x1 = Math.max(a.x, b.x);
    const y1 = Math.max(a.y, b.y);
    const x2 = Math.min(a.x + a.width, b.x + b.width);
    const y2 = Math.min(a.y + a.height, b.y + b.height);
    const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const union = a.width * a.height + b.width * b.height - intersection;
    return intersection / union;
}

// ==================================================
// 7. Draw Bounding Boxes
// ==================================================
function roundRect(c, x, y, w, h, r) {
    c.beginPath();
    c.moveTo(x + r, y);
    c.arcTo(x + w, y, x + w, y + h, r);
    c.arcTo(x + w, y + h, x, y + h, r);
    c.arcTo(x, y + h, x, y, r);
    c.arcTo(x, y, x + w, y, r);
    c.closePath();
}

function drawBoxes(boxes) {
    ctx.lineWidth = 3;
    ctx.font = "500 16px Prompt, Arial";
    ctx.textBaseline = "middle";

    boxes.forEach(box => {
        const color = COLORS[box.classId] || "#ff5c7a";
        const text = `${labels[box.classId]} ${(box.score * 100).toFixed(0)}%`;

        // กรอบ + พื้นโปร่งแสงด้านใน
        ctx.strokeStyle = color;
        ctx.fillStyle = color + "22";
        roundRect(ctx, box.x, box.y, box.width, box.height, 8);
        ctx.fill();
        ctx.stroke();

        // ป้ายชื่อ
        const textWidth = ctx.measureText(text).width;
        const tagH = 26;
        const tagY = box.y > tagH + 2 ? box.y - tagH - 2 : box.y + 2;
        const tagX = Math.min(box.x, 640 - textWidth - 16);
        ctx.fillStyle = color;
        roundRect(ctx, tagX, tagY, textWidth + 16, tagH, 6);
        ctx.fill();

        ctx.fillStyle = "white";
        ctx.fillText(text, tagX + 8, tagY + tagH / 2 + 1);
    });
}

// ==================================================
// 8. แสดงผลด้านขวา
// ==================================================
function showResult(boxes) {
    // นับจำนวนแต่ละคลาส
    const counts = [0, 0, 0];
    boxes.forEach(b => counts[b.classId]++);
    counts.forEach((n, i) => {
        const el = document.getElementById("count-" + i);
        if (el) el.textContent = n;
    });

    detectionList.innerHTML = "";

    if (boxes.length === 0) {
        setStatus("ไม่พบเสื้อผ้าในภาพ ลองลดค่าความมั่นใจดูนะ", "ready");
        detectionList.innerHTML = '<li class="empty">ไม่พบวัตถุที่ความมั่นใจนี้</li>';
        return;
    }

    setStatus(`ตรวจพบ ${boxes.length} ชิ้น`, "ready");

    boxes.forEach(b => {
        const pct = (b.score * 100).toFixed(1);
        const li = document.createElement("li");
        li.className = "det-item";
        li.style.setProperty("--c", COLORS[b.classId]);
        li.innerHTML = `
            <div class="det-top">
                <span class="det-name">${ICONS[b.classId] || ""} ${labels[b.classId]}</span>
                <span>${pct}%</span>
            </div>
            <div class="bar"><span style="width:${pct}%"></span></div>
        `;
        detectionList.appendChild(li);
    });
}