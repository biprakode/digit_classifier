import { useRef, useState, useEffect } from "react";

export default function DrawCanvas({onPredict}) {
    const canvasRef = useRef(null);
    const [isDrawing , setDrawing] = useState(false);

    useEffect (() => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d'); // drawer
        ctx.lineWidth = 12;
        ctx.lineCap = "round";
        ctx.strokeStyle = "black"; // black drawing
        ctx.fillStyle = "white"; // white background
        ctx.fillRect(0, 0, canvas.width, canvas.height); // fill canvas white
    } , []) // initialize canvas once

    const startDraw = (e) => {
        const ctx = canvasRef.current.getContext('2d'); // initialize drawer
        ctx.beginPath(); // start drawer
        ctx.lineWidth = 10
        ctx.moveTo(e.nativeEvent.offsetX , e.nativeEvent.offsetY); // move drawer to mouse position
        setDrawing(true);
    }

    const draw = (e) => {
        if(!isDrawing) return;
        const ctx = canvasRef.current.getContext('2d');
        ctx.lineTo(e.nativeEvent.offsetX , e.nativeEvent.offsetY);
        ctx.stroke(); // draw line to mouse pos
    }

    const endDraw = () => {
        setDrawing(false);
    }

    const clearCanvas = () => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = 'white'
        ctx.fillRect(0 , 0 , canvas.height , canvas.width); // fill canvas white
    }

    const handlePredict = () => {
        const dataUrl = canvasRef.current.toDataURL('image/png');
        onPredict(dataUrl); // send image to predict for backend compute
    }

    return (
        <div className="mt-10 flex flex-col items-center">
            <canvas 
                ref = {canvasRef}
                width={285}
                height={285}
                // event listeners
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