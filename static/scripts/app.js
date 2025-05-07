let speaking = false;
let speech;
let recognition;

// Initialize Speech Recognition
if ('webkitSpeechRecognition' in window) {
    recognition = new webkitSpeechRecognition();
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onresult = function(event) {
        const transcript = event.results[0][0].transcript;
        document.getElementById('query').value = transcript;
        sendQuery();
    };

    recognition.onerror = function(event) {
        console.error('Speech recognition error:', event.error);
    };
}

// Send query on button click
document.getElementById('send-btn').addEventListener('click', sendQuery);

// Enable pressing Enter to send the query
document.getElementById('query').addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
        sendQuery();
    }
});

// Start Speech Recognition
document.getElementById('start-stt-btn').addEventListener('click', function() {
    if (recognition) {
        recognition.start();
    } else {
        alert('Speech recognition is not supported in this browser.');
    }
});

// Start Text-to-Speech
document.getElementById('start-tts-btn').addEventListener('click', function() {
    const responseText = document.querySelector('.chatbot-message:last-child')?.textContent;

    if (responseText && window.speechSynthesis) {
        speech = new SpeechSynthesisUtterance(responseText);
        speech.rate = 1;
        speech.pitch = 1;
        speech.volume = 1;
        speech.onend = function() {
            speaking = false;
        };
        window.speechSynthesis.speak(speech);
        speaking = true;
    } else {
        alert('No response available for TTS.');
    }
});

// Stop Text-to-Speech
document.getElementById('stop-tts-btn').addEventListener('click', function() {
    if (speaking) {
        window.speechSynthesis.cancel();
        speaking = false;
    }
});

// Function to send query to the server
function sendQuery() {
    const query = document.getElementById('query').value;
    if (query.trim() === '') {
        alert('Please enter a question.');
        return;
    }

    addMessage(query, 'user-message');
    document.getElementById('query').value = '';

    fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ query: query })
    })
    .then(response => response.json())
    .then(data => {
        addMessage(data.response, 'chatbot-message');
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

// Function to add message to the chat
function addMessage(message, className) {
    const messagesContainer = document.getElementById('messages');
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', className);

    // Use Marked.js to render Markdown content
    messageDiv.innerHTML = marked.parse(message); // Updated line

    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Theme toggle functionality
const themeToggleButton = document.getElementById('theme-toggle');
const body = document.body;

// Check the current theme and apply it
if (localStorage.getItem('theme') === 'dark') {
    body.classList.add('dark-theme');
}

// Add event listener to toggle the theme
themeToggleButton.addEventListener('click', () => {
    body.classList.toggle('dark-theme');

    // Save the current theme in localStorage
    if (body.classList.contains('dark-theme')) {
        localStorage.setItem('theme', 'dark');
    } else {
        localStorage.setItem('theme', 'light');
    }
});
