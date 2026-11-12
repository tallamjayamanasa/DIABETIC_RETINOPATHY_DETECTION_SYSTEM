/**
 * Diabetic Retinopathy Detection System - Form Validation
 * Supplements Django server-side validation with instant client-side feedback.
 */

document.addEventListener('DOMContentLoaded', function () {
  // 1. Password Match and Strength Check in Registration Form
  const regForm = document.querySelector('form.needs-validation');
  const passwordInput = document.getElementById('register-password');
  const confirmInput = document.getElementById('register-confirm-password');

  if (passwordInput && confirmInput) {
    function checkPasswordsMatch() {
      if (confirmInput.value.length === 0) {
        confirmInput.setCustomValidity('');
        return;
      }
      if (passwordInput.value !== confirmInput.value) {
        confirmInput.setCustomValidity('Passwords do not match');
      } else {
        confirmInput.setCustomValidity('');
      }
    }

    passwordInput.addEventListener('input', checkPasswordsMatch);
    confirmInput.addEventListener('input', checkPasswordsMatch);
  }

  // 2. Show / Hide Password Toggle
  const toggleButtons = document.querySelectorAll('.toggle-password-visibility');
  toggleButtons.forEach(btn => {
    btn.addEventListener('click', function () {
      const targetId = this.getAttribute('data-target');
      const input = document.getElementById(targetId);
      const icon = this.querySelector('i');
      if (input) {
        if (input.type === 'password') {
          input.type = 'text';
          if (icon) {
            icon.classList.remove('bi-eye');
            icon.classList.add('bi-eye-slash');
          }
        } else {
          input.type = 'password';
          if (icon) {
            icon.classList.remove('bi-eye-slash');
            icon.classList.add('bi-eye');
          }
        }
      }
    });
  });

  // 3. Generic Bootstrap Form Validation enforcer
  const forms = document.querySelectorAll('.needs-validation');
  Array.from(forms).forEach(form => {
    form.addEventListener('submit', event => {
      if (!form.checkValidity()) {
        event.preventDefault();
        event.stopPropagation();
      }
      form.classList.add('was-validated');
    }, false);
  });
});
