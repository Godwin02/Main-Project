// Get references to the validation message elements
const usernameValidationMessage = document.getElementById('username-validation-message');
const usertypeValidationMessage = document.getElementById('usertype-validation-message');
const emailValidationMessage = document.getElementById('email-validation-message');
const passwordValidationMessage = document.getElementById('password-validation-message');
const confirmPasswordValidationMessage = document.getElementById('confirm-password-validation-message');

// Get a reference to the form
const form = document.getElementById('forms');

// Function to validate the form
function validateForm(event) {
  event.preventDefault(); // Prevent form submission initially

  let isValid = true;

  // Validate username (no whitespace, only underscores allowed)
  const usernameInput = document.getElementById('username-input');
  const usernameValue = usernameInput.value.trim();
  if (usernameValue === '') {
    isValid = false;
    usernameValidationMessage.textContent = 'Please enter your name.';
  } else if (/\s/.test(usernameValue)) {
    isValid = false;
    usernameValidationMessage.textContent = 'Username cannot contain spaces.';
  } else if (!/^[a-zA-Z0-9_]*$/.test(usernameValue)) {
    isValid = false;
    usernameValidationMessage.textContent = 'Username can only contain letters, numbers, and underscores.';
  }

  // Validate user type
  const userTypeInput = document.querySelector('select.form-control');
  const userTypeValue = userTypeInput.value;
  if (userTypeValue === '') {
    isValid = false;
    usertypeValidationMessage.textContent = 'Please select a user type.';
  }

  // Validate email
  const emailInput = document.querySelector('input[placeholder="Enter your Email-id"]');
  const emailValue = emailInput.value.trim();
  if (emailValue === '') {
    isValid = false;
    emailValidationMessage.textContent = 'Please enter your email.';
  } else if (!/^\S+@\S+\.\S+$/.test(emailValue)) {
    isValid = false;
    emailValidationMessage.textContent = 'Please enter a valid email address.';
  } else if (/\.com\.com/.test(emailValue)) {
    isValid = false;
    emailValidationMessage.textContent = 'Repeated ".com" is not allowed in the email.';
  }

  // Validate password
  const passwordInput = document.querySelector('input[placeholder="Create a password"]');
  const passwordValue = passwordInput.value;
  if (passwordValue === '') {
    isValid = false;
    passwordValidationMessage.textContent = 'Please create a password.';
  }

  // Validate confirm password
  const confirmPasswordInput = document.querySelector('input[placeholder="Create a password"]');
  const confirmPasswordValue = confirmPasswordInput.value;
  if (confirmPasswordValue === '') {
    isValid = false;
    confirmPasswordValidationMessage.textContent = 'Please confirm your password.';
  } else if (confirmPasswordValue !== passwordValue) {
    isValid = false;
    confirmPasswordValidationMessage.textContent = 'Passwords do not match.';
  }

  // Clear previous validation messages
  if (isValid) {
    usernameValidationMessage.textContent = '';
    usertypeValidationMessage.textContent = '';
    emailValidationMessage.textContent = '';
    passwordValidationMessage.textContent = '';
    confirmPasswordValidationMessage.textContent = '';
  }
  
  // Display success message
  if (isValid) {
    validationMessage.textContent = 'Good to go! 😃';
    validationMessage.style.color = 'green';
  }
}

// Attach the validateForm function to the form's submit event
form.addEventListener('submit', validateForm);