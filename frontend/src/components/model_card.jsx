export default function ModelCard({name , model, selected , onClick}) {
    return (
        <div
            onClick={onClick}
            className={`rounded-2xl shadow-md p-5 w-72 cursor-pointer border-2 transition ${selected ? "border-blue-500 scale-105 bg-sky-50" : "border-gray-200 hover:scale-105 bg-white"}`}
        >
            <h3 className="text-xl font-semibold text-gray-800">{name}</h3>
            <div className="mt-3 text-sm text-gray-600">
                <p>F1 score: <span className="font-medium">{model.F1}</span></p>
                <p>Complexity: <span className="break-keep font-medium">{model.complexity}</span></p>
            </div>
        </div>
    );  
}