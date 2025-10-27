import { useState } from 'react'
import ModelCard from './components/model_card'
import ModelCarousel from './components/model_carousel'
import DrawCanvas from './components/canvas'

// will change complexity to model params later
function App() {
    const custom_models = [
        { id:'custom_logreg' , name: "Logistic Regression", F1: 0.8955, complexity: 'LR=>0.1 || Max-iter=>2000' },
        { name: "Decision Tree", F1: 0., complexity: 'Max Depth=>100' }, // TODO
        { id:'custom_nb' , name: "Naive Bayes", F1: 0.7112, complexity: 'Multinomial NB || eps=>1e-2' },
        { id:'custom_RF' , name: "Random Forest", F1: 0.9574, complexity: '50 Trees || Max-Depth=>20' },
        { id: 'custom_GB' , name: "Gradient Boost", F1: 0.9460, complexity: '100 Trees || LR=>0.1' },
        { id: 'custom_KNN' , name: "K Nearest Neighbours" , F1: 0. , complexity: 'amar aar kichu bolar nei'},
        { id: 'custom_DNN_1' , name: "Deep Neural Network 1", F1: 0.8762, complexity: '784x128x64x10 || Epochs=>250 || SGD' },
        { id: 'custom_DNN_2' , name: "Deep Neural Network 2", F1: 0.8831 , complexity: '784x512x256x128x64x10 || Epochs=>250 || Momentum' },
        { id: 'custom_DNN_3' , name: "Deep Neural Network 3", F1: 0.8824 , complexity: '784x1024x512x256x128x64x10 || Epochs=>250 || Adam' },
    ];

    const sklearn_models = [
        { id:'sklearn_logreg' , name: "Logistic Regression", F1: 0.9175, complexity: 'LR=>SAG || Max-iter=>2000' },
        { id: 'sklearn_DT' , name: "Decision Tree", F1: 0.8420, complexity: 'Max Depth=>100' },
        { id:'sklearn_SVM' , name: "Kernelized SVM", F1: 0., complexity: 40 },
        { id:'sklearn_nb' , name: "Naive Bayes", F1: 0.8235, complexity: 'Multinomial NB || eps=>1e-2' },
        { name: "Random Forest", F1: 0.9571, complexity: '50 Decision Trees || Max-Depth=>100'},
        { id:'sklearn_GB' , name: "Gradient Boost", F1: 0.9399, complexity: '100 Trees || LR=>0.1' },
        { id: 'sklearn_KNN' , name: "K Nearest Neighbours" , F1: 0.9655 , complexity: ''},
        { name: "Deep Neural Network 1", F1: 0., complexity: 85 },
        { name: "Deep Neural Network 2", F1: 0., complexity: 85 },
        { name: "Deep Neural Network 3", F1: 0., complexity: 85 },
        { name: "Convolution Network 3", F1: 0., complexity: 85 },
        { name: "Efficient_Net Network 3", F1: 0., complexity: 85 }
    ];

    const [pred , set_pred] = useState('-1')
    const [isLoading, setIsLoading] = useState(false);

    const handlePredict = async(pixels) => { // send pixel data & model to server
        try {
            const response = await fetch ('https://localhost-8000/' , {
                method: 'POST',
                headers: {"Content-Type" : "application/json"},
                body: JSON.stringify ({
                    pixels,
                    model_id : selectedModel.data.id
                })
            })
            if (response.ok) {
                set_pred(await response.json)
                console.log(pred);
            }else{
                set_pred('Error')
            }
        } catch (error) {
            console.error("Prediction failed:", error);
            set_pred('Error');
        } finally {
            setIsLoading(false)
        }
    }

const isPredictionValid = pred !== '-1' && pred !== '' && pred !== 'Error';

    //smart
    const [selectedModel, setSelectedModel] = useState({
    type: "custom",
    data: custom_models[0],
  });

    return (
        <div className="min-h-screen bg-gray-50 flex flex-col items-center p-6">
            <h1 className=" text-gray-800 text-3xl font-bold mb-2 text-left">Digit Classifier Playground</h1>
            <p className="text-gray-600 mb-6">Experiment with machine learning models</p>
            <div className="w-full max-w-7xl flex flex-col lg:flex-row lg:items-start lg:space-x-8 ">
                <div className="text-gray-800 flex-1 space-y-8 min-w-0">
                    <ModelCarousel
                        title={"Custom Models (from scratch)"}
                        models={custom_models}
                        onSelect={(model) => {
                            setSelectedModel({ type: "custom", data: model })
                        }}
                        selectedModel={selectedModel.type === "custom" ? selectedModel.data : null}     
                    />
                    <ModelCarousel
                        title={"Scikit-learn Models"}
                        models={sklearn_models}
                        onSelect={(model) => {
                            setSelectedModel({ type: "sklearn", data: model })
                        }}
                        selectedModel={selectedModel.type === "sklearn" ? selectedModel.data : null}
                    />
                </div>
                <div className="w-full lg:w-96 lg:mt-2 mt-2 flex flex-col items-center justify-center p-2 border border-gray-200 rounded-lg shadow-md bg-white">
                    <h2 className="text-gray-800 text-2xl font-bold pt-4">Draw a Digit</h2>
                    <DrawCanvas onPredict={handlePredict} /> 
                    <div className="w-full text-center mt-6 p-4 rounded-xl">
                        <h2 className="text-gray-700 text-xl font-semibold flex items-center justify-center space-x-3">
                            <span className="text-2xl">
                                Result:
                            </span>

                            {isLoading && (
                                <span className="ml-3 text-2xl font-extrabold text-indigo-500 animate-pulse">
                                    Predicting...
                                </span>
                            )}
                            
                            {!isLoading && (
                                <>
                                    {isPredictionValid ? (
                                        <span className="ml-3 px-6 py-3 text-4xl font-extrabold text-white bg-green-600 rounded-xl shadow-2xl transition-all duration-300 transform scale-105">
                                            {pred}
                                        </span>
                                    ) : pred === 'Error' ? (
                                        <span className="ml-3 px-4 py-2 text-2xl font-bold text-red-700 bg-red-100 border border-red-500 rounded-lg flex items-center space-x-2">
                                            <XCircle size={20} />
                                            <span>Error</span>
                                        </span>
                                    ) : (
                                        <span className="ml-3 px-4 py-2 text-3xl font-extrabold text-gray-400">
                                            ?
                                        </span>
                                    )}
                                </>
                            )}
                        </h2>
                    </div>
                </div>
            </div>
        </div>
    );
}


export default App
