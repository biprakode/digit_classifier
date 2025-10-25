const carousel = document.getElementById("carousel");
const cards = document.querySelectorAll(".model-card");
const nextBtn = document.getElementById("nextBtn");
const prevBtn = document.getElementById("prevBtn");
const selectButtons = document.querySelectorAll(".select-btn");

let currentIndex = 0;
const cardWidth = cards[0].offsetWidth + 24; // width + spacing

function updateCarousel() {
  const offset = -currentIndex * cardWidth;
  carousel.style.transform = `translateX(${offset}px)`;
}

nextBtn.addEventListener("click", () => {
  if (currentIndex < cards.length - 1) currentIndex++;
  updateCarousel();
});

prevBtn.addEventListener("click", () => {
  if (currentIndex > 0) currentIndex--;
  updateCarousel();
});

selectButtons.forEach((btn) => {
  btn.addEventListener("click", (e) => {
    document.querySelectorAll(".model-card").forEach(card => {
      card.classList.remove("border-blue-400", "scale-105");
    });
    e.target.closest(".model-card").classList.add("border-blue-400", "scale-105");
  });
});
