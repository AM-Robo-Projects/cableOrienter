document.addEventListener('DOMContentLoaded', function() {
  // Initialize all carousels
  var carousels = bulmaCarousel.attach('.carousel', {
    slidesToScroll: 1,
    slidesToShow: 1,
    infinite: true,
    autoplay: true
  });

  // Handle navbar burger menu
  const $navbarBurgers = Array.prototype.slice.call(document.querySelectorAll('.navbar-burger'), 0);
  if ($navbarBurgers.length > 0) {
    $navbarBurgers.forEach(el => {
      el.addEventListener('click', () => {
        const target = el.dataset.target;
        const $target = document.getElementById(target);
        el.classList.toggle('is-active');
        $target.classList.toggle('is-active');
      });
    });
  }

  // Initialize any sliders
  bulmaSlider.attach();

  // Smooth scroll for links with hash
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      e.preventDefault();
      document.querySelector(this.getAttribute('href')).scrollIntoView({
        behavior: 'smooth'
      });
    });
  });
});
