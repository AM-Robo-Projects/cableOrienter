document.addEventListener('DOMContentLoaded', function() {
  const backToTopBtn = document.createElement('button');
  backToTopBtn.id = 'back-to-top';
  backToTopBtn.innerHTML = '<i class="fas fa-chevron-up"></i>';
  backToTopBtn.title = 'Back to top';
  document.body.appendChild(backToTopBtn);

  window.addEventListener('scroll', function() {
    if (window.pageYOffset > 300) {
      backToTopBtn.classList.add('show');
    } else {
      backToTopBtn.classList.remove('show');
    }
  });

  backToTopBtn.addEventListener('click', function() {
    window.scrollTo({
      top: 0,
      behavior: 'smooth'
    });
  });
});
