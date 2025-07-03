document.addEventListener('DOMContentLoaded', function() {
  const modal = document.createElement('div');
  modal.className = 'image-modal';
  
  const closeBtn = document.createElement('span');
  closeBtn.className = 'close-modal';
  closeBtn.innerHTML = '&times;';
  
  const modalImg = document.createElement('img');
  modalImg.className = 'modal-content';
  
  modal.appendChild(closeBtn);
  modal.appendChild(modalImg);
  document.body.appendChild(modal);
  
  const images = document.querySelectorAll('.clickable-image');
  
  images.forEach(img => {
    img.addEventListener('click', function() {
      modal.classList.add('active');
      modalImg.src = this.src;
      document.body.style.overflow = 'hidden';
    });
  });
  
  closeBtn.addEventListener('click', function() {
    modal.classList.remove('active');
    document.body.style.overflow = '';
  });
  
  modal.addEventListener('click', function(e) {
    if (e.target === modal) {
      modal.classList.remove('active');
      document.body.style.overflow = '';
    }
  });
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape' && modal.classList.contains('active')) {
      modal.classList.remove('active');
      document.body.style.overflow = '';
    }
  });
});
