// embla'd by LLMs
import ModelCard from './model_card';
import { useEffect, useState } from 'react';
import { DotButton, useDotButton } from './EmblaCarouselDotButton'
import {
  PrevButton,
  NextButton,
  usePrevNextButtons
} from './EmblaCarouselArrowButtons'
import useEmblaCarousel from 'embla-carousel-react';

export default function ModelCarousel({ title, models, onSelect , selectedModel }) {
    const [emblaRef, emblaApi] = useEmblaCarousel({ loop: false, align: 'center' }); // embla by Gemini
    
    // 2. Logging/Re-initialization (Good, but maybe simplify)
    useEffect(() => {    
        if (emblaApi) {   
            // emblaApi.reInit(); // Generally not needed unless the model list changes dynamically
            console.log(emblaApi.slideNodes()); 
        }  
    }, [emblaApi]);

    const { 
        prevBtnDisabled, 
        nextBtnDisabled, 
        onPrevButtonClick, 
        onNextButtonClick 
    } = usePrevNextButtons(emblaApi);

    const { 
        selectedIndex, 
        scrollSnaps, 
        onDotButtonClick 
    } = useDotButton(emblaApi);

    return (
        <div className="flex flex-col items-center mt-6 w-full max-w-5xl"> {/* Added max-w-5xl here for better centering */}
            <h2 className="text-2xl font-bold mb-4">{title}</h2>
            
            <section className="embla w-full">
                {/* Embla Viewport (ref container) */}
                <div className="embla__viewport overflow-hidden" ref={emblaRef}>
                    
                    {/* Embla Container */}
                    <div className="embla__container flex">
                        {models.map((model) => 
                            // Embla Slide (applying padding/margin and flex properties)
                            <div 
                                className="embla__slide flex-none px-4 py-2" 
                                style={{minWidth: '280px'}} 
                                key={model.name}
                            >
                                <ModelCard 
                                    name={model.name} // risky (names unique?) but who cares?
                                    model={model}  
                                    selected={selectedModel && model.name === selectedModel.name}
                                    onClick={() => {
                                        onSelect(model);
                                    }}
                                />
                            </div>
                        )}
                    </div>
                </div>

                {/* 3. Controls Section */}
                <div className="embla__controls flex justify-center items-center mt-4 space-x-4">
                    
                    {/* Arrow Buttons */}
                    <div className="embla__buttons flex space-x-2">
                        <PrevButton onClick={onPrevButtonClick} disabled={prevBtnDisabled} />
                        <NextButton onClick={onNextButtonClick} disabled={nextBtnDisabled} />
                    </div>

                    {/* Dot Navigation */}
                    <div className="embla__dots flex space-x-2">
                        {scrollSnaps.map((_, index) => (
                            <DotButton
                                key={index}
                                onClick={() => onDotButtonClick(index)} 
                                className={`embla__dot w-3 h-3 rounded-full transition-colors ${
                                    index === selectedIndex ? 'bg-gray-700' : 'bg-gray-300 hover:bg-gray-400'
                                }`}
                            />
                        ))}
                    </div>
                </div>
            </section>
        </div>
    );
}