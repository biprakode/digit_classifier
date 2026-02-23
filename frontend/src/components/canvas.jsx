import { useRef, useState, useEffect } from "react";

export default function DrawCanvas({onPredict , onClear}) {
    const canvasRef = useRef(null);
    const [isDrawing , setDrawing] = useState(false);

    useEffect (() => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        ctx.lineWidth = 8;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.strokeStyle = "black";
        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    } , [])

    const startDraw = (e) => {
        const ctx = canvasRef.current.getContext('2d');
        ctx.beginPath();
        ctx.lineWidth = 8;
        ctx.moveTo(e.nativeEvent.offsetX , e.nativeEvent.offsetY);
        setDrawing(true);
    }

    const draw = (e) => {
        if(!isDrawing) return;
        const ctx = canvasRef.current.getContext('2d');
        ctx.lineTo(e.nativeEvent.offsetX , e.nativeEvent.offsetY);
        ctx.stroke();
    }

    const endDraw = () => {
        setDrawing(false);
    }

    const clearCanvas = () => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = 'white'
        ctx.fillRect(0 , 0 , canvas.width , canvas.height);
        onClear(true);
    }

    /**
     * Convert the drawn canvas to MNIST-compatible 28x28 grayscale pixels.
     *
     * MNIST preprocessing:
     *  1. White digits on black background (inverted from canvas)
     *  2. Grayscale 0-255 (not binary)
     *  3. Digit centered in the 28x28 frame, scaled to ~20x20 content area
     */
    const handlePredict = () => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");
        const w = canvas.width;
        const h = canvas.height;

        // Get full-resolution image data
        const fullData = ctx.getImageData(0, 0, w, h).data;

        // Convert to grayscale & invert (MNIST: white digit on black bg)
        const gray = new Float32Array(w * h);
        for (let i = 0; i < w * h; i++) {
            const r = fullData[i * 4];
            const g = fullData[i * 4 + 1];
            const b = fullData[i * 4 + 2];
            gray[i] = 255 - (r + g + b) / 3; // invert: black bg, white digit
        }

        // Find bounding box of the drawn digit
        let minX = w, minY = h, maxX = 0, maxY = 0;
        const threshold = 30; // noise threshold
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                if (gray[y * w + x] > threshold) {
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                }
            }
        }

        // If nothing drawn, send zeros
        if (maxX <= minX || maxY <= minY) {
            onPredict(new Array(784).fill(0));
            return;
        }

        // Extract bounding box and scale to ~20x20 (centered in 28x28)
        const bw = maxX - minX + 1;
        const bh = maxY - minY + 1;

        // Create a temp canvas for the cropped digit
        const cropCanvas = document.createElement("canvas");
        cropCanvas.width = bw;
        cropCanvas.height = bh;
        const cropCtx = cropCanvas.getContext("2d");
        cropCtx.drawImage(canvas, minX, minY, bw, bh, 0, 0, bw, bh);

        // Scale to fit within 20x20 while preserving aspect ratio
        const targetSize = 20;
        const scale = targetSize / Math.max(bw, bh);
        const scaledW = Math.round(bw * scale);
        const scaledH = Math.round(bh * scale);

        // Place centered in 28x28
        const outCanvas = document.createElement("canvas");
        outCanvas.width = 28;
        outCanvas.height = 28;
        const outCtx = outCanvas.getContext("2d");
        outCtx.fillStyle = "white"; // white background (will be inverted)
        outCtx.fillRect(0, 0, 28, 28);

        // Enable smooth downscaling
        outCtx.imageSmoothingEnabled = true;
        outCtx.imageSmoothingQuality = "high";

        const offsetX = Math.round((28 - scaledW) / 2);
        const offsetY = Math.round((28 - scaledH) / 2);
        outCtx.drawImage(cropCanvas, 0, 0, bw, bh, offsetX, offsetY, scaledW, scaledH);

        // Extract grayscale pixels, inverted (MNIST format)
        const outData = outCtx.getImageData(0, 0, 28, 28).data;
        const pixels = [];
        for (let i = 0; i < 28 * 28; i++) {
            const r = outData[i * 4];
            const g = outData[i * 4 + 1];
            const b = outData[i * 4 + 2];
            const avg = (r + g + b) / 3;
            pixels.push(Math.round(255 - avg)); // invert: 0=black bg, 255=white digit
        }

        onPredict(pixels);
    };

    return (
        <div className="mt-10 flex flex-col items-center">
            <canvas
                ref = {canvasRef}
                width={285}
                height={285}
                onMouseDown={startDraw}
                onMouseMove={draw}
                onMouseUp={endDraw}
                onMouseLeave={endDraw}
                className="border bg-white rounded-xl shadow-lg"
            />
            <div className="flex gap-4 mt-4">
                <button
                onClick={clearCanvas}
                className="px-4 py-2 bg-gray-200 rounded-lg hover:bg-gray-300"
                >Clear</button>

                <button
                onClick={handlePredict}
                className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
                >Predict</button>
            </div>
        </div>
    )
}