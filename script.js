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

let session = null;
let labels = [];

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

    labelDiv.textContent = "โหลดโมเดลเสร็จแล้ว เลือกรูปได้เลย";
    console.log("YOLOv11 Fashion Model Loaded");
}

labelDiv.textContent = "กำลังโหลดโมเดล...";
loadModel().catch(err => {
    labelDiv.textContent = "โหลดโมเดลไม่สำเร็จ: " + err.message;
    console.error(err);
});

// ==================================================
// 3. Image Upload & Detection
// ==================================================
imageInput.addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const img = new Image();

    img.onload = async () => {
        // แสดงรูปต้นฉบับแบบคงสัดส่วน (YOLOv11 ใช้ 640x640)
        canvas.width = 640;
        canvas.height = 640;
        drawLetterbox(ctx, img, 640);

        if (!session) {
            labelDiv.textContent = "โมเดลยังโหลดไม่เสร็จ รอสักครู่แล้วลองใหม่";
            return;
        }

        labelDiv.textContent = "กำลังตรวจจับ...";

        // เตรียม Tensor
        const tensor = preprocessImage(img);

        // รันโมเดล
        const feeds = {};
        feeds[session.inputNames[0]] = tensor;
        const result = await session.run(feeds);
        const outputTensor = result[session.outputNames[0]];

        // แปลงผลเป็นกล่อง แล้วตัดกล่องซ้ำ
        let boxes = postprocessOutput(outputTensor, 0.35);
        boxes = nonMaxSuppression(boxes, 0.35);

        // วาดผล
        ctx.clearRect(0, 0, 640, 640);
        drawLetterbox(ctx, img, 640);
        drawBoxes(boxes);
        showResult(boxes);
    };

    img.src = URL.createObjectURL(file);
});

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
function nonMaxSuppression(boxes, iouThreshold = 0.35) {
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
function drawBoxes(boxes) {
    const colors = ["red", "blue", "orange"]; // shirt, skirt, trousers
    ctx.lineWidth = 3;
    ctx.font = "18px Arial";

    boxes.forEach(box => {
        const color = colors[box.classId] || "red";
        const text = `${labels[box.classId]} ${(box.score * 100).toFixed(1)}%`;

        ctx.strokeStyle = color;
        ctx.strokeRect(box.x, box.y, box.width, box.height);

        // พื้นหลังป้ายชื่อ
        const textWidth = ctx.measureText(text).width;
        const labelY = box.y > 25 ? box.y - 25 : box.y;
        ctx.fillStyle = color;
        ctx.fillRect(box.x, labelY, textWidth + 10, 25);

        ctx.fillStyle = "white";
        ctx.fillText(text, box.x + 5, labelY + 18);
    });
}

// ==================================================
// 8. แสดงผลใต้รูป
// ==================================================
function showResult(boxes) {
    if (boxes.length === 0) {
        labelDiv.textContent = "ไม่พบเสื้อผ้าในภาพ";
        return;
    }
    labelDiv.textContent = "ผลลัพธ์: " + boxes
        .map(b => `${labels[b.classId]} ${(b.score * 100).toFixed(1)}%`)
        .join(", ");
}