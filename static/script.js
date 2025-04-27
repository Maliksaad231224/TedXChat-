function sendMessage() {
    let userInput = document.getElementById("userInput").value;
    if (userInput.trim() === "") return;
  
    let selectedModel = document.getElementById("model-dropdown").value || "cohere";

  
    let chatBox = document.getElementById("chatBox");
    let userMessage = document.createElement("p");
    userMessage.className = "user-message";
    userMessage.textContent = userInput;
    chatBox.appendChild(userMessage);
  
    // Send request to backend
    fetch("/get", {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded"
      },
      body: new URLSearchParams({
        msg: userInput,
        selected_model: selectedModel
      })
    })
    .then(res => res.json())
    .then(data => {
      let botMessage = document.createElement("p");
      botMessage.className = "bot-message";
      botMessage.textContent = data.answer;
      chatBox.appendChild(botMessage);
      chatBox.scrollTop = chatBox.scrollHeight;
    })
    .catch(err => {
      let botMessage = document.createElement("p");
      botMessage.className = "bot-message";
      botMessage.textContent = "🤖 Sorry, something went wrong.";
      chatBox.appendChild(botMessage);
    });
  
    document.getElementById("userInput").value = "";
    console.log(selectedModel);
  }
// Trigger sendMessage on Enter key
document.addEventListener("DOMContentLoaded", function () {
    // Load particles
    particlesJS.load('particles-js', '/static/particles-config.json');

    // Add Enter key support
    const inputField = document.getElementById("userInput");
    inputField.addEventListener("keydown", function (event) {
        if (event.key === "Enter") {
            event.preventDefault(); // Prevent newline
            sendMessage(); // Call the same sendMessage function
        }
    });
});

const examplePrompts = document.querySelectorAll('.example-prompt');

// Add event listener to each prompt
examplePrompts.forEach((prompt) => {
  prompt.addEventListener('click', () => {
    // Get the input field
    const inputField = document.getElementById('userInput');
    
    // Populate the input field with the selected prompt
    inputField.value = prompt.textContent;
  });
});