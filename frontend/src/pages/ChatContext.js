import React, { createContext, useState, useEffect } from 'react';

// Create a context for chat history
const ChatContext = createContext();

// ChatProvider component to wrap around the app
export const ChatProvider = ({ children }) => {
    const [messages, setMessages] = useState([]);

    // Load chat history from localStorage on initial render
    useEffect(() => {
        const savedHistory = localStorage.getItem('chatHistory');
        if (savedHistory) {
            try {
                const parsedHistory = JSON.parse(savedHistory);
                if (Array.isArray(parsedHistory)) {
                    setMessages(parsedHistory);
                }
            } catch (e) {
                console.error('Error parsing chat history:', e);
                localStorage.removeItem('chatHistory');
            }
        }
    }, []);

    // Save chat history to localStorage whenever messages change
    useEffect(() => {
        localStorage.setItem('chatHistory', JSON.stringify(messages));
    }, [messages]);

    return (
        <ChatContext.Provider value={{ messages, setMessages }}>
            {children}
        </ChatContext.Provider>
    );
};

export default ChatContext;