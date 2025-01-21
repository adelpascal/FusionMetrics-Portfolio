document.addEventListener('DOMContentLoaded', () => { // Wait for DOM to load
    const cards = document.querySelectorAll('.market-trends-card');

    cards.forEach(card => {
        const button = card.querySelector('.read-more-btn');
        const content = card.querySelector('.content');

        button.addEventListener('click', () => {
            content.classList.toggle('expanded'); // Toggle the 'expanded' class
            button.textContent = content.classList.contains('expanded') ? 'Read Less' : 'Read More';
        });
    });
});