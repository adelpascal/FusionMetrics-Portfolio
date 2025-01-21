const readMoreLinks = document.querySelectorAll('.read-more-link');

readMoreLinks.forEach(link => {
  link.addEventListener('click', function(event) {
    event.preventDefault(); // Prevent the default link behavior

    const expandContent = this.parentNode.querySelector('.expand-content');
    const icon = this.querySelector('i'); // Get the arrow icon

    expandContent.style.display = expandContent.style.display === 'block' ? 'none' : 'block';

    // Toggle the icon (optional):
    icon.classList.toggle('fa-arrow-right');
    icon.classList.toggle('fa-arrow-down');
  });
});