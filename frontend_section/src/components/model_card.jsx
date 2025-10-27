export default function ModelCard({name , model, selected , onClick}) {
    return (
        <div
            onClick={onClick}
            className={`rounded-2xl shadow-md p-4 w-64 cursor-pointer border transition ${selected ? "border-blue-500 scale-105" : "border-gray-200 hover:scale-105"}`}
        >
            <h3 className="text-lg font-semibold text-gray-800">{name}</h3>
            <div className="mt-2 text-sm text-gray-600">
                <p>F1 score: <span className="font-medium">{model.F1}</span></p>
                <p>Complexity: <span className="break-keep font-medium">{model.complexity}</span></p>
            </div>
        </div>
    );  
}