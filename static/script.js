// DOM Elements
const chatMessages = document.getElementById('chatMessages');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const modelSelect = document.getElementById('modelSelect');
const modeSelect = document.getElementById('modeSelect');

// API Configuration
const API_BASE_URL = 'http://127.0.0.1:8000';

// Event Listeners
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Scroll to bottom of chat
function scrollToBottom() {
    chatMessages.scrollTo({
        top: chatMessages.scrollHeight,
        behavior: 'smooth'
    });
}

// Create user avatar SVG
function getUserAvatarSVG() {
    return `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M20 21V19C20 17.9391 19.5786 16.9217 18.8284 16.1716C18.0783 15.4214 17.0609 15 16 15H8C6.93913 15 5.92172 15.4214 5.17157 16.1716C4.42143 16.9217 4 17.9391 4 19V21" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        <circle cx="12" cy="7" r="4" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>`;
}

// Create assistant avatar SVG
function getAssistantAvatarSVG() {
    return `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="currentColor" stroke-width="2"/>
        <path d="M2 17L12 22L22 17" stroke="currentColor" stroke-width="2"/>
        <path d="M2 12L12 17L22 12" stroke="currentColor" stroke-width="2"/>
    </svg>`;
}

// Add message to chat
function addMessage(content, isUser = false, metadata = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user' : 'assistant'}`;
    
    // Format content - convert markdown-like syntax to HTML
    let formattedContent = content
        // Handle bold text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        // Convert newlines to <br>
        .replace(/\n/g, '<br>')
        // Handle bullet points with * (Gemini style)
        .replace(/\* (.*?)(?=<br>|$)/g, '<li>$1</li>')
        // Handle bullet points with - (ChatGPT style)
        .replace(/- (.*?)(?=<br>|$)/g, '<li>$1</li>');
    
    // Wrap consecutive list items in ul tags and clean up extra <br> around lists
    if (formattedContent.includes('<li>')) {
        formattedContent = formattedContent
            .replace(/<br>(<li>)/g, '$1')
            .replace(/(<\/li>)<br>/g, '$1')
            .replace(/(<li>.*?<\/li>)+/g, '<ul>$&</ul>');
    }
    
    // Add metadata badge for assistant messages
    let metadataBadge = '';
    if (!isUser && metadata) {
        const modelLabel = metadata.model_used === 'gemini' ? '🟢 Gemini' : '🔵 ChatGPT';
        const modeLabel = metadata.mode_used.includes('rag') ? '⚡ RAG' : '📚 Full';
        metadataBadge = `<div class="message-meta"><span class="meta-badge">${modelLabel}</span><span class="meta-badge">${modeLabel}</span></div>`;
    }
    
    messageDiv.innerHTML = `
        <div class="message-avatar">
            ${isUser ? getUserAvatarSVG() : getAssistantAvatarSVG()}
        </div>
        <div class="message-content">
            ${metadataBadge}
            <p>${formattedContent}</p>
        </div>
    `;
    
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

// Show typing indicator
function showTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message assistant';
    typingDiv.id = 'typingIndicator';
    
    const model = modelSelect.value;
    const mode = modeSelect.value;
    const statusText = `Using ${model === 'gemini' ? 'Gemini' : 'ChatGPT'} in ${mode === 'rag' ? 'RAG' : 'Full Context'} mode...`;
    
    typingDiv.innerHTML = `
        <div class="message-avatar">
            ${getAssistantAvatarSVG()}
        </div>
        <div class="message-content">
            <div class="typing-status">${statusText}</div>
            <div class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;
    
    chatMessages.appendChild(typingDiv);
    scrollToBottom();
}

// Remove typing indicator
function removeTypingIndicator() {
    const typingDiv = document.getElementById('typingIndicator');
    if (typingDiv) {
        typingDiv.remove();
    }
}

// Send message to backend
async function sendMessage() {
    const message = userInput.value.trim();
    
    if (!message) return;
    
    // Add user message to chat
    addMessage(message, true);
    
    // Clear input
    userInput.value = '';
    
    // Disable input while processing
    userInput.disabled = true;
    sendBtn.disabled = true;
    
    // Show typing indicator
    showTypingIndicator();
    
    try {
        const response = await fetch(`${API_BASE_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                model: modelSelect.value,
                mode: modeSelect.value
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Remove typing indicator and add response
        removeTypingIndicator();
        addMessage(data.response, false, {
            model_used: data.model_used,
            mode_used: data.mode_used
        });
        
    } catch (error) {
        console.error('Error:', error);
        removeTypingIndicator();
        addMessage(`Sorry, I encountered an error: ${error.message}. Please check your API keys and try again.`);
    } finally {
        // Re-enable input
        userInput.disabled = false;
        sendBtn.disabled = false;
        userInput.focus();
    }
}

// Send suggestion chip message
function sendSuggestion(text) {
    userInput.value = text;
    sendMessage();
}

// Focus input on load
window.addEventListener('load', () => {
    userInput.focus();
});
