/**
 * script.js — Air Canvas Web App Logic
 * 
 * Replicates the Python desktop app's features in the browser:
 * - Real-time hand tracking via MediaPipe.
 * - Draw, Hover, Erase, and Fist-Hold-to-Clear gestures.
 * - Multi-layer Neon Glow Brush & Stable Starry Sparkles.
 * - Brush size backup/restore and pinch-to-resize.
 * - Shape snapping (circle, rectangle, line).
 * - Doodle recognition.
 * - Clean composite image exporting.
 */

// BGR to RGB / CSS colors mapping
const COLORS = {
    "Red": "rgb(255, 76, 59)",
    "Orange": "rgb(255, 153, 0)",
    "Yellow": "rgb(255, 220, 0)",
    "Green": "rgb(121, 217, 100)",
    "Cyan": "rgb(73, 224, 230)",
    "Blue": "rgb(0, 120, 255)",
    "Purple": "rgb(160, 90, 222)",
    "Pink": "rgb(255, 105, 180)",
    "White": "rgb(245, 245, 245)"
};
const COLOR_ORDER = ["Red", "Orange", "Yellow", "Green", "Cyan", "Blue", "Purple", "Pink", "White"];

// --- DOM Elements ---
const videoElement = document.getElementById("webcamVideo");
const uiOverlayCanvas = document.getElementById("uiOverlayCanvas");
const drawingCanvas = document.getElementById("drawingCanvas");
const loadingOverlay = document.getElementById("loadingOverlay");
const topBar = document.getElementById("topBar");
const colorPalette = document.getElementById("colorPalette");
const btnClear = document.getElementById("btnClear");
const btnUndo = document.getElementById("btnUndo");
const btnRedo = document.getElementById("btnRedo");
const btnSizeDec = document.getElementById("btnSizeDec");
const btnSizeInc = document.getElementById("btnSizeInc");
const btnOverlay = document.getElementById("btnOverlay");
const btnSave = document.getElementById("btnSave");
const sizeIndicatorWidget = document.getElementById("sizeIndicatorWidget");
const sizeDot = document.getElementById("sizeDot");
const sizeText = document.getElementById("sizeText");

const gestureHud = document.getElementById("gestureHud");
const hudStatusDot = document.getElementById("hudStatusDot");
const hudGestureText = document.getElementById("hudGestureText");
const fpsText = document.getElementById("fpsText");
const fpsDot = document.getElementById("fpsDot");
const doodleCard = document.getElementById("doodleCard");
const doodleLabel = document.getElementById("doodleLabel");
const saveSuccessCard = document.getElementById("saveSuccessCard");

// --- Canvas Contexts ---
const uiCtx = uiOverlayCanvas.getContext("2d");
const drawCtx = drawingCanvas.getContext("2d");

// Internal resolution coordinates (matching webcam 640x360 for high-performance stabilization)
const CANVAS_WIDTH = 640;
const CANVAS_HEIGHT = 360;

uiOverlayCanvas.width = CANVAS_WIDTH;
uiOverlayCanvas.height = CANVAS_HEIGHT;
drawingCanvas.width = CANVAS_WIDTH;
drawingCanvas.height = CANVAS_HEIGHT;

// --- State Variables ---
let activeColorName = "Blue";
let activeColor = COLORS[activeColorName];
let brushThickness = 5;
let drawThickness = 5; // Saved backup
let eraserMode = false;
let showOverlay = true;

// Stroke structures
let strokes = []; // { points: [], color: "", thickness: 5, shape: null }
let redoStack = [];
let activeStrokePoints = [];
let smoothingBuffer = [];
const SMOOTHING_WINDOW = 5;

// Gestures
let prevGesture = "none";
let currentGesture = "none";
let gestureHistory = [];
const GESTURE_HISTORY_MAX = 5;

// Fist Hold state
let fistStartTime = null;
let clearTriggered = false;
const FIST_HOLD_DURATION = 2000; // ms

// Top bar selection cooldown (frames)
let topBarCooldown = 0;

// FPS
let lastFrameTime = performance.now();
let frameCount = 0;
let fps = 0;

// Doodle Recognition timer
let doodleCardTimeout = null;

// --- Initialize Color Swatches ---
COLOR_ORDER.forEach(name => {
    const swatch = document.createElement("div");
    swatch.className = `swatch ${name === activeColorName ? 'active' : ''}`;
    swatch.style.backgroundColor = COLORS[name];
    swatch.dataset.colorName = name;
    colorPalette.appendChild(swatch);
});

// Update active swatch
function updateActiveColor(name) {
    activeColorName = name;
    activeColor = COLORS[name];
    
    // Update border indicator
    topBar.style.borderBottomColor = activeColor;
    
    // Update active class
    document.querySelectorAll(".swatch").forEach(s => {
        s.classList.toggle("active", s.dataset.colorName === name);
    });
}

// Update brush size widget
function updateBrushSize(size) {
    brushThickness = Math.max(2, Math.min(size, 30));
    sizeText.textContent = `${brushThickness}px`;
    sizeDot.style.width = `${brushThickness}px`;
    sizeDot.style.height = `${brushThickness}px`;
}

// Toggle Eraser Mode safely preserving custom thickness
function setEraserMode(active) {
    if (active === eraserMode) return;
    eraserMode = active;
    if (active) {
        drawThickness = brushThickness;
        updateBrushSize(40); // Eraser thickness
    } else {
        updateBrushSize(drawThickness); // Restore custom drawing thickness
    }
}

// --- Setup MediaPipe Hands ---
const hands = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
});

hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.7,
    minTrackingConfidence: 0.7
});

hands.onResults(onHandResults);

// Start camera stream
const camera = new Camera(videoElement, {
    onFrame: async () => {
        await hands.send({ image: videoElement });
    },
    width: CANVAS_WIDTH,
    height: CANVAS_HEIGHT
});

camera.start().then(() => {
    loadingOverlay.style.opacity = "0";
    setTimeout(() => loadingOverlay.style.display = "none", 500);
});

// --- Helper Functions: Math & Geometry ---
function distance(p1, p2) {
    return Math.hypot(p2.x - p1.x, p2.y - p1.y);
}

// Check if a finger is extended (compared to MCP joint)
function isFingerUp(landmarks, tipId, mcpId) {
    return landmarks[tipId].y < landmarks[mcpId].y;
}

// Check if thumb is extended horizontally
function isThumbUp(landmarks) {
    const wrist = landmarks[0];
    const mcp = landmarks[9];
    const tip = landmarks[4];
    const ip = landmarks[3];
    
    if (wrist.x < mcp.x) {
        // Mirrored right hand: thumb tip left of IP joint
        return tip.x < ip.x;
    } else {
        // Mirrored left hand: thumb tip right of IP joint
        return tip.x > ip.x;
    }
}

// Temporal majority vote for gesture stability
function stabilizeGesture(raw) {
    gestureHistory.push(raw);
    if (gestureHistory.length > GESTURE_HISTORY_MAX) {
        gestureHistory.shift();
    }
    
    // Count frequencies
    const counts = {};
    let maxCount = 0;
    let winner = "none";
    
    gestureHistory.forEach(g => {
        counts[g] = (counts[g] || 0) + 1;
        if (counts[g] > maxCount) {
            maxCount = counts[g];
            winner = g;
        }
    });
    
    return winner;
}

// --- Shape Detection (JS Port) ---
function detectShape(points) {
    if (points.length < 15) return null;
    
    const start = points[0];
    const end = points[points.length - 1];
    
    // Calculate path length
    let pathLength = 0;
    for (let i = 1; i < points.length; i++) {
        pathLength += distance(points[i-1], points[i]);
    }
    if (pathLength < 50) return null;
    
    const directDist = distance(start, end);
    const closureRatio = directDist / pathLength;
    
    // 1. Line Snapping: open strokes that are straight
    if (closureRatio > 0.25) {
        const straightResult = checkLine(points, start, end, pathLength);
        if (straightResult) return straightResult;
        return null;
    }
    
    // Bounding Box
    let xMin = Infinity, yMin = Infinity, xMax = -Infinity, yMax = -Infinity;
    points.forEach(p => {
        if (p.x < xMin) xMin = p.x;
        if (p.y < yMin) yMin = p.y;
        if (p.x > xMax) xMax = p.x;
        if (p.y > yMax) yMax = p.y;
    });
    const w = xMax - xMin;
    const h = yMax - yMin;
    
    // Center point
    const cx = (xMin + xMax) / 2;
    const cy = (yMin + yMax) / 2;
    
    // 2. Circle Snapping: check variance of distances from center
    const radii = points.map(p => Math.hypot(p.x - cx, p.y - cy));
    const meanRadius = radii.reduce((a, b) => a + b, 0) / radii.length;
    const variance = radii.reduce((a, b) => a + Math.pow(b - meanRadius, 2), 0) / radii.length;
    const stdDev = Math.sqrt(variance);
    const circularityRatio = stdDev / meanRadius;
    
    if (circularityRatio < 0.12) {
        return { type: "circle", center: { x: cx, y: cy }, radius: meanRadius };
    }
    
    // 3. Rectangle Snapping: check bounding box aspect and area fill
    const rectArea = w * h;
    if (rectArea > 100) {
        // Approximate area of hand drawn path using shoelace formula
        let drawnArea = 0;
        for (let i = 0; i < points.length; i++) {
            const j = (i + 1) % points.length;
            drawnArea += points[i].x * points[j].y - points[j].x * points[i].y;
        }
        drawnArea = Math.abs(drawnArea) / 2;
        const fillRatio = drawnArea / rectArea;
        
        if (fillRatio > 0.7 && (w / h) < 6.0 && (h / w) < 6.0) {
            return { type: "rectangle", pt1: { x: xMin, y: yMin }, pt2: { x: xMax, y: yMax } };
        }
    }
    
    return null;
}

function checkLine(points, start, end, pathLength) {
    const directDist = distance(start, end);
    if (directDist < 40) return null;
    
    // Max perpendicular deviation
    let maxDev = 0;
    const vx = end.x - start.x;
    const vy = end.y - start.y;
    const len = Math.hypot(vx, vy);
    const nx = -vy / len;
    const ny = vx / len;
    
    points.forEach(p => {
        const dx = p.x - start.x;
        const dy = p.y - start.y;
        const dev = Math.abs(dx * nx + dy * ny);
        if (dev > maxDev) maxDev = dev;
    });
    
    const straightness = maxDev / directDist;
    const efficiency = directDist / pathLength;
    
    if (straightness < 0.08 && efficiency > 0.85) {
        return { type: "line", pt1: start, pt2: end };
    }
    
    return null;
}

// --- Doodle Recognition heuristics (JS Port) ---
function classifyDoodle(points) {
    if (points.length < 10) return null;
    
    // Check bounding box
    let xMin = Infinity, yMin = Infinity, xMax = -Infinity, yMax = -Infinity;
    points.forEach(p => {
        if (p.x < xMin) xMin = p.x;
        if (p.y < yMin) yMin = p.y;
        if (p.x > xMax) xMax = p.x;
        if (p.y > yMax) yMax = p.y;
    });
    const w = xMax - xMin;
    const h = yMax - yMin;
    const cx = (xMin + xMax) / 2;
    const cy = (yMin + yMax) / 2;
    
    // Lengths
    const start = points[0];
    const end = points[points.length - 1];
    let pathLen = 0;
    for (let i = 1; i < points.length; i++) {
        pathLen += distance(points[i-1], points[i]);
    }
    
    const closureDist = distance(start, end);
    const closureRatio = closureDist / pathLen;
    
    if (closureRatio > 0.22) {
        // Open doodles
        const directDist = distance(start, end);
        const vx = end.x - start.x;
        const vy = end.y - start.y;
        const len = Math.hypot(vx, vy);
        const nx = -vy / len;
        const ny = vx / len;
        let maxDev = 0;
        points.forEach(p => {
            const dev = Math.abs((p.x - start.x) * nx + (p.y - start.y) * ny);
            if (dev > maxDev) maxDev = dev;
        });
        
        const straightness = maxDev / directDist;
        if (straightness < 0.08 && directDist > 40) return "Line 📏";
        
        // Zigzag: check sign changes in diffs
        let xChanges = 0;
        let yChanges = 0;
        let lastDx = 0;
        let lastDy = 0;
        for (let i = 1; i < points.length; i++) {
            const dx = points[i].x - points[i-1].x;
            const dy = points[i].y - points[i-1].y;
            if (i > 1) {
                if (Math.sign(dx) !== Math.sign(lastDx) && Math.abs(dx) > 1) xChanges++;
                if (Math.sign(dy) !== Math.sign(lastDy) && Math.abs(dy) > 1) yChanges++;
            }
            lastDx = dx;
            lastDy = dy;
        }
        if (Math.max(xChanges, yChanges) >= 4 && pathLen > 80) return "Zigzag ⚡";
        
        // Arrow: straight-ish shaft + hook at tail
        if (straightness < 0.16 && directDist > 60) {
            const tailStart = Math.floor(points.length * 0.85);
            const tailPoints = points.slice(tailStart);
            if (tailPoints.length > 2) {
                let tailDev = 0;
                const tvx = tailPoints[tailPoints.length - 1].x - tailPoints[0].x;
                const tvy = tailPoints[tailPoints.length - 1].y - tailPoints[0].y;
                const tlen = Math.hypot(tvx, tvy);
                const tnx = -tvy / tlen;
                const tny = tvx / tlen;
                tailPoints.forEach(p => {
                    const dev = Math.abs((p.x - tailPoints[0].x) * tnx + (p.y - tailPoints[0].y) * tny);
                    if (dev > tailDev) tailDev = dev;
                });
                if (tailDev > 8) return "Arrow ➡️";
            }
        }
        
        return null;
    }
    
    // Closed shapes:
    // Shoelace area
    let area = 0;
    for (let i = 0; i < points.length; i++) {
        const j = (i + 1) % points.length;
        area += points[i].x * points[j].y - points[j].x * points[i].y;
    }
    area = Math.abs(area) / 2;
    
    // Circularity
    const circularity = (4 * Math.PI * area) / (pathLen * pathLen);
    if (circularity >= 0.76) return "Circle ⭕";
    
    // Rectangle check
    const rectArea = w * h;
    const fillRatio = area / rectArea;
    if (fillRatio > 0.65 && (w / h) < 6.0 && (h / w) < 6.0) return "Rectangle ⬛";
    
    // Heart Check: top dip, wider upper half
    const upperHalf = points.filter(p => p.y < cy);
    const lowerHalf = points.filter(p => p.y >= cy);
    if (upperHalf.length > 4 && lowerHalf.length > 4) {
        let uwMin = Infinity, uwMax = -Infinity;
        upperHalf.forEach(p => { if (p.x < uwMin) uwMin = p.x; if (p.x > uwMax) uwMax = p.x; });
        let lwMin = Infinity, lwMax = -Infinity;
        lowerHalf.forEach(p => { if (p.x < lwMin) lwMin = p.x; if (p.x > lwMax) lwMax = p.x; });
        
        const uw = uwMax - uwMin;
        const lw = lwMax - lwMin;
        
        if (uw > lw * 0.9) {
            const upperLeft = upperHalf.filter(p => p.x < cx);
            const upperRight = upperHalf.filter(p => p.x >= cx);
            if (upperLeft.length > 1 && upperRight.length > 1) {
                let centerBand = upperHalf.filter(p => Math.abs(p.x - cx) < w * 0.15);
                if (centerBand.length > 0) {
                    const leftTop = Math.min(...upperLeft.map(p => p.y));
                    const rightTop = Math.min(...upperRight.map(p => p.y));
                    const centerTop = Math.min(...centerBand.map(p => p.y));
                    if (centerTop - Math.min(leftTop, rightTop) > h * 0.05) {
                        return "Heart ❤️";
                    }
                }
            }
        }
    }
    
    // Star Check: low density/solidity, high vertices count
    // Radial peaks/valleys
    const distances = points.map(p => Math.hypot(p.x - cx, p.y - cy));
    const meanD = distances.reduce((a,b)=>a+b,0) / distances.length;
    let peaks = 0, valleys = 0;
    let above = distances[0] > meanD;
    for (let i = 1; i < distances.length; i++) {
        const curAbove = distances[i] > meanD;
        if (curAbove && !above) valleys++;
        else if (!curAbove && above) peaks++;
        above = curAbove;
    }
    if (peaks >= 4 && valleys >= 4) return "Star ⭐";
    
    // Default Triangle: roughly 3 vertices
    if (area / (w * h) < 0.6) return "Triangle 🔺";
    
    return null;
}

// --- Rendering canvas strokes ---
function drawStrokePrimitive(ctx, stroke, thickness, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = thickness;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    
    if (stroke.shape) {
        const s = stroke.shape;
        if (s.type === "circle") {
            ctx.beginPath();
            ctx.arc(s.center.x, s.center.y, s.radius, 0, 2 * Math.PI);
            ctx.stroke();
        } else if (s.type === "rectangle") {
            ctx.beginPath();
            ctx.rect(s.pt1.x, s.pt1.y, s.pt2.x - s.pt1.x, s.pt2.y - s.pt1.y);
            ctx.stroke();
        } else if (s.type === "line") {
            ctx.beginPath();
            ctx.moveTo(s.pt1.x, s.pt1.y);
            ctx.lineTo(s.pt2.x, s.pt2.y);
            ctx.stroke();
        }
    } else {
        const pts = stroke.points;
        if (pts.length < 2) return;
        ctx.beginPath();
        ctx.moveTo(pts[0].x, pts[0].y);
        for (let i = 1; i < pts.length; i++) {
            ctx.lineTo(pts[i].x, pts[i].y);
        }
        ctx.stroke();
    }
}

// Draw starry sparkles (deterministic offset coordinate hashing)
function drawStrokeSparkles(ctx, stroke) {
    const pts = stroke.points;
    if (pts.length < 2 && !stroke.shape) return;
    
    function drawStar(imgCtx, cx, cy, size, color) {
        imgCtx.strokeStyle = color;
        imgCtx.lineWidth = 1;
        imgCtx.lineJoin = "miter";
        
        // Primary cross spikes
        imgCtx.beginPath();
        imgCtx.moveTo(cx - size, cy); imgCtx.lineTo(cx + size, cy);
        imgCtx.moveTo(cx, cy - size); imgCtx.lineTo(cx, cy + size);
        imgCtx.stroke();
        
        // Secondary diagonal spikes for larger stars
        if (size > 4) {
            const dSize = Math.floor(size * 0.6);
            imgCtx.beginPath();
            imgCtx.moveTo(cx - dSize, cy - dSize); imgCtx.lineTo(cx + dSize, cy + dSize);
            imgCtx.moveTo(cx + dSize, cy - dSize); imgCtx.lineTo(cx - dSize, cy + dSize);
            imgCtx.stroke();
        }
        
        // Bright white core center dot
        imgCtx.fillStyle = "#ffffff";
        imgCtx.beginPath();
        imgCtx.arc(cx, cy, 1, 0, 2 * Math.PI);
        imgCtx.fill();
    }
    
    // Parse color to BGR/RGB components to lighten it
    let match = stroke.color.match(/\d+/g);
    if (!match) return;
    let r = parseInt(match[0]), g = parseInt(match[1]), b = parseInt(match[2]);
    let sparkleColor = `rgb(${Math.min(r+60, 255)}, ${Math.min(g+60, 255)}, ${Math.min(b+60, 255)})`;
    
    // Assemble point array for sparkles
    let sparklePts = [];
    if (stroke.shape) {
        const s = stroke.shape;
        if (s.type === "circle") {
            const numSp = Math.max(4, Math.floor(s.radius / 15));
            for (let i = 0; i < numSp; i++) {
                const angle = (i * 2 * Math.PI / numSp);
                sparklePts.push({
                    x: Math.floor(s.center.x + s.radius * Math.cos(angle)),
                    y: Math.floor(s.center.y + s.radius * Math.sin(angle))
                });
            }
        } else if (s.type === "rectangle") {
            const x1 = s.pt1.x, y1 = s.pt1.y, x2 = s.pt2.x, y2 = s.pt2.y;
            const stepsW = Math.max(2, Math.floor(Math.abs(x2 - x1) / 40));
            const stepsH = Math.max(2, Math.floor(Math.abs(y2 - y1) / 40));
            for(let i=0; i<stepsW; i++) {
                let t = i / stepsW;
                sparklePts.push({ x: Math.floor(x1 + t*(x2-x1)), y: y1 });
                sparklePts.push({ x: Math.floor(x1 + t*(x2-x1)), y: y2 });
            }
            for(let i=0; i<stepsH; i++) {
                let t = i / stepsH;
                sparklePts.push({ x: x1, y: Math.floor(y1 + t*(y2-y1)) });
                sparklePts.push({ x: x2, y: Math.floor(y1 + t*(y2-y1)) });
            }
        } else if (s.type === "line") {
            const numSp = Math.max(2, Math.floor(distance(s.pt1, s.pt2) / 25));
            for (let i = 0; i <= numSp; i++) {
                let t = i / numSp;
                sparklePts.push({
                    x: Math.floor(s.pt1.x + t * (s.pt2.x - s.pt1.x)),
                    y: Math.floor(s.pt1.y + t * (s.pt2.y - s.pt1.y))
                });
            }
        }
    } else {
        sparklePts = pts;
    }
    
    for (let i = 0; i < sparklePts.length; i += 12) {
        const pt = sparklePts[i];
        
        // Deterministic offset coordinates hashing (prevents flickering)
        const hashX = ((pt.x * 7 + pt.y * 13) % 7) - 3;
        const hashY = ((pt.x * 11 + pt.y * 3) % 7) - 3;
        const cx = pt.x + hashX;
        const cy = pt.y + hashY;
        const size = 2 + ((pt.x + pt.y) % 3);
        
        drawStar(ctx, cx, cy, size, sparkleColor);
    }
}

// Render complete canvas matching the premium Neon Glow effect
function drawStroke(ctx, stroke) {
    if (stroke.color !== "rgb(0, 0, 0)") {
        // Neon Glow Paint Layering
        
        // 1. Outer Glow (thickest, lowest opacity)
        let match = stroke.color.match(/\d+/g);
        let r = parseInt(match[0]), g = parseInt(match[1]), b = parseInt(match[2]);
        let outerColor = `rgba(${r}, ${g}, ${b}, 0.15)`;
        drawStrokePrimitive(ctx, stroke, stroke.thickness + 12, outerColor);
        
        // 2. Medium Glow
        let mediumColor = `rgba(${r}, ${g}, ${b}, 0.4)`;
        drawStrokePrimitive(ctx, stroke, stroke.thickness + 6, mediumColor);
        
        // 3. Core Line (full color)
        drawStrokePrimitive(ctx, stroke, stroke.thickness, stroke.color);
        
        // 4. Center Bright Core (thinnest, off-white high contrast line)
        let coreColor = `rgba(${Math.min(255, Math.floor(r + (255 - r) * 0.6))}, ${Math.min(255, Math.floor(g + (255 - g) * 0.6))}, ${Math.min(255, Math.floor(b + (255 - b) * 0.6))}, 0.95)`;
        drawStrokePrimitive(ctx, stroke, Math.max(1, Math.floor(stroke.thickness * 0.25)), coreColor);
        
        // 5. Starry Sparkles overlay
        drawStrokeSparkles(ctx, stroke);
    } else {
        // Clean Smart Eraser strokes (black)
        drawStrokePrimitive(ctx, stroke, stroke.thickness, stroke.color);
    }
}

// Redraw canvas from scratch
function redrawCanvas() {
    drawCtx.fillStyle = "#08080a";
    drawCtx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    
    strokes.forEach(s => drawStroke(drawCtx, s));
    
    // Also render the currently active stroke if any
    if (activeStrokePoints.length > 0) {
        const activeStroke = {
            points: activeStrokePoints,
            color: eraserMode ? "rgb(0, 0, 0)" : activeColor,
            thickness: brushThickness,
            shape: null
        };
        drawStroke(drawCtx, activeStroke);
    }
}

// --- MediaPipe hand tracking frame results loop ---
function onHandResults(results) {
    // 1. Calculate FPS
    frameCount++;
    const now = performance.now();
    if (now - lastFrameTime >= 1000) {
        fps = Math.round((frameCount * 1000) / (now - lastFrameTime));
        frameCount = 0;
        lastFrameTime = now;
        
        // Update FPS Pill
        fpsText.textContent = `${fps} FPS`;
        fpsDot.className = `hud-status-dot ${fps >= 20 ? 'green' : (fps >= 10 ? 'orange' : 'red')}`;
    }
    
    // Clear overlay canvas
    uiCtx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    
    // Check if hand is detected
    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        const landmarks = results.multiHandLandmarks[0];
        
        // Get finger up states
        const thumbUp = isThumbUp(landmarks);
        const indexUp = isFingerUp(landmarks, 8, 5);
        const middleUp = isFingerUp(landmarks, 12, 9);
        const ringUp = isFingerUp(landmarks, 16, 13);
        const pinkyUp = isFingerUp(landmarks, 20, 17);
        
        const nonThumbFingersUp = [indexUp, middleUp, ringUp, pinkyUp].filter(Boolean).length;
        
        // Gesture classification
        let rawGesture = "none";
        if (nonThumbFingersUp === 0) {
            // Fist: Clear gesture countdown
            const timeNow = Date.now();
            if (fistStartTime === null) {
                fistStartTime = timeNow;
                clearTriggered = false;
            }
            const elapsed = timeNow - fistStartTime;
            if (elapsed >= FIST_HOLD_DURATION && !clearTriggered) {
                clearTriggered = true;
                rawGesture = "clear";
            } else {
                rawGesture = "none";
            }
            
            // Draw wrist fist progress arc
            const progress = Math.min(elapsed / FIST_HOLD_DURATION, 1.0);
            if (progress > 0) {
                const wrist = landmarks[0];
                const wx = (1 - wrist.x) * CANVAS_WIDTH;
                const wy = wrist.y * CANVAS_HEIGHT;
                
                // Draw wrist progress arc
                uiCtx.beginPath();
                uiCtx.arc(wx, wy, 25, 0, 2 * Math.PI);
                uiCtx.strokeStyle = "rgba(80, 80, 80, 0.5)";
                uiCtx.lineWidth = 2;
                uiCtx.stroke();
                
                uiCtx.beginPath();
                uiCtx.arc(wx, wy, 25, -Math.PI / 2, -Math.PI / 2 + (2 * Math.PI * progress));
                uiCtx.strokeStyle = "rgb(255, 76, 59)";
                uiCtx.lineWidth = 4;
                uiCtx.stroke();
                
                // Overlay text
                uiCtx.fillStyle = "rgb(255, 76, 59)";
                uiCtx.font = "bold 8px Outfit";
                uiCtx.textAlign = "center";
                uiCtx.textBaseline = "middle";
                uiCtx.fillText("CLEAR", wx, wy);
            }
        } else {
            fistStartTime = null;
            clearTriggered = false;
            
            if (indexUp && middleUp && ringUp && !pinkyUp) {
                rawGesture = "erase";
            } else if (middleUp && !ringUp && !pinkyUp) {
                rawGesture = "hover"; // Allows index finger to bend for pinch sizing/clicks
            } else if (indexUp && !middleUp && !ringUp && !pinkyUp) {
                rawGesture = "draw";
            }
        }
        
        // Stabilize gesture (fist-clear bypasses temporal filter for instant clear)
        if (rawGesture === "clear") {
            gestureHistory = [];
            currentGesture = "clear";
        } else {
            currentGesture = stabilizeGesture(rawGesture);
        }
        
        // Get index tip coordinates (mirrored horizontally)
        const indexTip = landmarks[8];
        const ix = Math.floor((1 - indexTip.x) * CANVAS_WIDTH);
        const iy = Math.floor(indexTip.y * CANVAS_HEIGHT);
        
        // --- Draw landmarks on overlay ---
        drawMediaPipeLandmarks(uiCtx, landmarks);
        
        // --- Transition / Action triggers ---
        // Pen-up detection
        if ((prevGesture === "draw" || prevGesture === "erase") && 
            (currentGesture !== "draw" && currentGesture !== "erase")) {
            commitActiveStroke();
        }
        
        // Action: Clear
        if (currentGesture === "clear" && prevGesture !== "clear") {
            clearCanvas();
        }
        
        // Action: Draw/Erase
        if (currentGesture === "draw") {
            setEraserMode(false);
            
            // Start stroke
            if (prevGesture !== "draw") {
                activeStrokePoints = [];
                smoothingBuffer = [];
            }
            
            // Add point with smoothing
            addSmoothedPoint(ix, iy);
            
        } else if (currentGesture === "erase") {
            setEraserMode(true);
            
            // Start erase stroke
            if (prevGesture !== "erase") {
                activeStrokePoints = [];
                smoothingBuffer = [];
            }
            
            addSmoothedPoint(ix, iy);
            
        } else if (currentGesture === "hover") {
            setEraserMode(false);
            
            // Brush resizing via Pinch distance (only when index is over the SizeIndicator Widget)
            const thumbTip = landmarks[4];
            const rawPinch = Math.hypot((indexTip.x - thumbTip.x) * 640, (indexTip.y - thumbTip.y) * 360);
            
            // Calculate relative palm length (wrist #0 to middle MCP #9)
            const palmLen = Math.hypot((landmarks[9].x - landmarks[0].x) * 640, (landmarks[9].y - landmarks[0].y) * 360);
            const pinchDist = rawPinch / palmLen;
            
            // Draw hover sizing indicator
            drawBrushIndicatorCircle(uiCtx, ix, iy);
            
            // Get size indicator element bounding client rect relative to canvas container
            const sizeWidgetRect = sizeIndicatorWidget.getBoundingClientRect();
            const canvasRect = drawingCanvas.getBoundingClientRect();
            
            // Map widget coordinate bounds to 640x360 coordinate space
            const wx1 = ((sizeWidgetRect.left - canvasRect.left) / canvasRect.width) * CANVAS_WIDTH;
            const wy1 = ((sizeWidgetRect.top - canvasRect.top) / canvasRect.height) * CANVAS_HEIGHT;
            const wx2 = wx1 + (sizeWidgetRect.width / canvasRect.width) * CANVAS_WIDTH;
            const wy2 = wy1 + (sizeWidgetRect.height / canvasRect.height) * CANVAS_HEIGHT;
            
            if (ix >= wx1 && ix <= wx2 && iy >= wy1 && iy <= wy2) {
                // Pinch to resize brush dynamically
                if (pinchDist < 0.45) {
                    const ratio = Math.max(0, Math.min((pinchDist - 0.2) / 0.6, 1.0));
                    const targetSize = Math.round(2 + ratio * 28);
                    
                    // Exponential Moving Average smoothing
                    updateBrushSize(Math.round(0.15 * targetSize + 0.85 * brushThickness));
                }
            }
            
            // Pinch-to-click selections
            if (topBarCooldown > 0) {
                topBarCooldown--;
            } else if (pinchDist < 0.45) {
                // Check if hovering over any buttons
                const clickedBtn = checkTopBarClick(ix, iy);
                if (clickedBtn) {
                    handleButtonClick(clickedBtn);
                    topBarCooldown = 15; // COOLDOWN 0.5s at 30fps
                }
            }
        }
        
        prevGesture = currentGesture;
    } else {
        // No hand tracked
        fistStartTime = null;
        clearTriggered = false;
        
        if (prevGesture === "draw" || prevGesture === "erase") {
            commitActiveStroke();
        }
        currentGesture = "none";
        prevGesture = "none";
        gestureHistory = [];
    }
    
    // Update floating HUD indicators
    updateFloatingHuds();
    
    // Render drawing canvas
    redrawCanvas();
    
    // Blend canvas onto camera left feed if overlay is checked
    if (showOverlay) {
        uiCtx.globalAlpha = 0.85;
        uiCtx.drawImage(drawingCanvas, 0, 0);
        uiCtx.globalAlpha = 1.0;
    }
}

// MediaPipe visual landmarks drawing helper
function drawMediaPipeLandmarks(ctx, landmarks) {
    // Draw connections
    ctx.strokeStyle = "rgba(255, 255, 255, 0.4)";
    ctx.lineWidth = 1;
    
    // Helper connection lines
    const CONNECTIONS = [
        [0,1],[1,2],[2,3],[3,4], // Thumb
        [0,5],[5,6],[6,7],[7,8], // Index
        [9,10],[10,11],[11,12],  // Middle
        [13,14],[14,15],[15,16], // Ring
        [0,17],[17,18],[18,19],[19,20], // Pinky
        [5,9],[9,13],[13,17]     // Palm knuckles
    ];
    
    CONNECTIONS.forEach(([i1, i2]) => {
        ctx.beginPath();
        ctx.moveTo((1 - landmarks[i1].x) * CANVAS_WIDTH, landmarks[i1].y * CANVAS_HEIGHT);
        ctx.lineTo((1 - landmarks[i2].x) * CANVAS_WIDTH, landmarks[i2].y * CANVAS_HEIGHT);
        ctx.stroke();
    });
    
    // Draw joints
    landmarks.forEach((lm, idx) => {
        const x = (1 - lm.x) * CANVAS_WIDTH;
        const y = lm.y * CANVAS_HEIGHT;
        ctx.fillStyle = idx === 8 ? "rgb(0, 120, 255)" : "rgb(121, 217, 100)";
        ctx.beginPath();
        ctx.arc(x, y, idx === 8 ? 5 : 3, 0, 2 * Math.PI);
        ctx.fill();
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1;
        ctx.stroke();
    });
}

// Draw brush indicator
function drawBrushIndicatorCircle(ctx, x, y) {
    ctx.strokeStyle = activeColor;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(x, y, brushThickness, 0, 2 * Math.PI);
    ctx.stroke();
    
    // Brush size text
    ctx.fillStyle = "rgba(30, 30, 30, 0.85)";
    ctx.beginPath();
    ctx.roundRect(x + brushThickness + 5, y - 10, 35, 16, 4);
    ctx.fill();
    ctx.strokeStyle = "rgb(100, 100, 100)";
    ctx.lineWidth = 1;
    ctx.stroke();
    
    ctx.fillStyle = "#ffffff";
    ctx.font = "600 9px Outfit";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(`${brushThickness}px`, x + brushThickness + 22, y - 2);
}

// Add point to stroke smoothing buffer
function addSmoothedPoint(x, y) {
    smoothingBuffer.push({ x, y });
    if (smoothingBuffer.length > SMOOTHING_WINDOW) {
        smoothingBuffer.shift();
    }
    
    // Average coordinate points
    const avgX = Math.round(smoothingBuffer.reduce((a, b) => a + b.x, 0) / smoothingBuffer.length);
    const avgY = Math.round(smoothingBuffer.reduce((a, b) => a + b.y, 0) / smoothingBuffer.length);
    
    activeStrokePoints.push({ x: avgX, y: avgY });
}

// Commit stroke to main history stack on pen-up
function commitActiveStroke() {
    if (activeStrokePoints.length < 2) {
        activeStrokePoints = [];
        return;
    }
    
    // Shape snapping
    const snappedShape = detectShape(activeStrokePoints);
    
    const stroke = {
        points: activeStrokePoints,
        color: eraserMode ? "rgb(0, 0, 0)" : activeColor,
        thickness: brushThickness,
        shape: snappedShape
    };
    
    strokes.push(stroke);
    
    // Clear redo history
    redoStack = [];
    
    // Classify doodle on pen up (if not erasing)
    if (!eraserMode) {
        const label = classifyDoodle(activeStrokePoints);
        if (label) {
            triggerDoodleCard(label);
        }
    }
    
    activeStrokePoints = [];
    smoothingBuffer = [];
}

// --- Menu clicks / selectors ---
function checkTopBarClick(ix, iy) {
    if (iy > 80) return null; // Bar height is 80px
    
    // Map button bounds dynamically by element bounding rects
    const buttons = [
        { id: "btnClear", el: btnClear },
        { id: "btnUndo", el: btnUndo },
        { id: "btnRedo", el: btnRedo },
        { id: "btnSizeDec", el: btnSizeDec },
        { id: "btnSizeInc", el: btnSizeInc },
        { id: "btnOverlay", el: btnOverlay },
        { id: "btnSave", el: btnSave }
    ];
    
    const canvasRect = drawingCanvas.getBoundingClientRect();
    
    for (let btn of buttons) {
        const r = btn.el.getBoundingClientRect();
        const bx1 = ((r.left - canvasRect.left) / canvasRect.width) * CANVAS_WIDTH;
        const by1 = ((r.top - canvasRect.top) / canvasRect.height) * CANVAS_HEIGHT;
        const bx2 = bx1 + (r.width / canvasRect.width) * CANVAS_WIDTH;
        const by2 = by1 + (r.height / canvasRect.height) * CANVAS_HEIGHT;
        
        if (ix >= bx1 && ix <= bx2 && iy >= by1 && iy <= by2) {
            return btn.id;
        }
    }
    
    // Color swatches click check
    const swatches = document.querySelectorAll(".swatch");
    for (let sw of swatches) {
        const r = sw.getBoundingClientRect();
        const bx1 = ((r.left - canvasRect.left) / canvasRect.width) * CANVAS_WIDTH;
        const by1 = ((r.top - canvasRect.top) / canvasRect.height) * CANVAS_HEIGHT;
        const bx2 = bx1 + (r.width / canvasRect.width) * CANVAS_WIDTH;
        const by2 = by1 + (r.height / canvasRect.height) * CANVAS_HEIGHT;
        
        if (ix >= bx1 && ix <= bx2 && iy >= by1 && iy <= by2) {
            return sw.dataset.colorName;
        }
    }
    
    return null;
}

function handleButtonClick(actionId) {
    if (actionId === "btnClear") {
        clearCanvas();
    } else if (actionId === "btnUndo") {
        undo();
    } else if (actionId === "btnRedo") {
        redo();
    } else if (actionId === "btnSizeDec") {
        updateBrushSize(brushThickness - 3);
    } else if (actionId === "btnSizeInc") {
        updateBrushSize(brushThickness + 3);
    } else if (actionId === "btnOverlay") {
        toggleOverlay();
    } else if (actionId === "btnSave") {
        saveDrawing();
    } else if (COLORS[actionId]) {
        updateActiveColor(actionId);
    }
}

// --- Actions Implementation ---
function clearCanvas() {
    strokes = [];
    redoStack = [];
    activeStrokePoints = [];
    redrawCanvas();
}

function undo() {
    if (strokes.length > 0) {
        const popped = strokes.pop();
        redoStack.push(popped);
        redrawCanvas();
    }
}

function redo() {
    if (redoStack.length > 0) {
        const popped = redoStack.pop();
        strokes.push(popped);
        redrawCanvas();
    }
}

function toggleOverlay() {
    showOverlay = !showOverlay;
    btnOverlay.classList.toggle("active", showOverlay);
}

function saveDrawing() {
    // Save composite: Merges camera frame and canvas drawings (original background brightness)
    const mergeCanvas = document.createElement("canvas");
    mergeCanvas.width = CANVAS_WIDTH;
    mergeCanvas.height = CANVAS_HEIGHT;
    const mergeCtx = mergeCanvas.getContext("2d");
    
    // Draw mirrored video frame
    mergeCtx.save();
    mergeCtx.translate(CANVAS_WIDTH, 0);
    mergeCtx.scale(-1, 1);
    mergeCtx.drawImage(videoElement, 0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    mergeCtx.restore();
    
    // Draw drawing canvas elements with 85% opacity
    mergeCtx.globalAlpha = 0.85;
    mergeCtx.drawImage(drawingCanvas, 0, 0);
    mergeCtx.globalAlpha = 1.0;
    
    // Download composite
    const link = document.createElement("a");
    const dateStr = new Date().toISOString().slice(0, 19).replace(/[-T:]/g, "");
    link.download = `composite_${dateStr}.png`;
    link.href = mergeCanvas.toDataURL("image/png");
    link.click();
    
    // Display saved notification HUD card
    triggerSaveSuccessNotification();
}

// --- HUDs Display Controllers ---
function updateFloatingHuds() {
    // Update gesture label pill
    const labelMap = {
        "draw": { text: "DRAW", class: "green" },
        "hover": { text: "HOVER", class: "orange" },
        "erase": { text: "ERASE", class: "purple" },
        "clear": { text: "CLEARING", class: "blue" },
        "none": { text: "---", class: "" }
    };
    
    const config = labelMap[currentGesture] || labelMap["none"];
    hudGestureText.textContent = `MODE: ${config.text}`;
    hudStatusDot.className = `hud-status-dot ${config.class}`;
}

function triggerDoodleCard(label) {
    if (doodleCardTimeout) clearTimeout(doodleCardTimeout);
    
    doodleLabel.textContent = `Detected: ${label}`;
    doodleCard.classList.add("visible");
    
    doodleCardTimeout = setTimeout(() => {
        doodleCard.classList.remove("visible");
    }, 2000);
}

function triggerSaveSuccessNotification() {
    saveSuccessCard.classList.add("visible");
    setTimeout(() => {
        saveSuccessCard.classList.remove("visible");
    }, 1500);
}

// --- Add Traditional Click Listeners for standard Mouse/Touch fallbacks ---
btnClear.addEventListener("click", () => handleButtonClick("btnClear"));
btnUndo.addEventListener("click", () => handleButtonClick("btnUndo"));
btnRedo.addEventListener("click", () => handleButtonClick("btnRedo"));
btnSizeDec.addEventListener("click", () => handleButtonClick("btnSizeDec"));
btnSizeInc.addEventListener("click", () => handleButtonClick("btnSizeInc"));
btnOverlay.addEventListener("click", () => handleButtonClick("btnOverlay"));
btnSave.addEventListener("click", () => handleButtonClick("btnSave"));

// Populate manual color click selectors
document.querySelectorAll(".swatch").forEach(sw => {
    sw.addEventListener("click", (e) => {
        handleButtonClick(e.target.dataset.colorName);
    });
});
