import React, { useState, useRef, useEffect, useContext } from 'react';
import axios from 'axios';
import styled, { keyframes } from 'styled-components';
import ReactMarkdown from 'react-markdown';
import ChatContext from './ChatContext';

// Theme configuration (unchanged)
const theme = {
    primary: '#000000',
    botBackground: '#FFFFFF', 
    userText: '#FFFFFF', 
    botText: '#000000', 
    background: '#F9FAFF', 
    inputBackground: '#FFFFFF',
    error: '#D32F2F',
    errorBg: '#FFEBEE',
    borderRadius: '30px', 
    shadow: '0 4px 12px rgba(0, 0, 0, 0.08)', 
    shadowHover: '0 6px 16px rgba(0, 0, 0, 0.12)', 
    transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)', 
    navbarHeight: '75px'
};

// Animations (unchanged)
const spin = keyframes`
  to { transform: rotate(360deg); }
`;

// Styled components (unchanged)
const ChatContainer = styled.div`
    position: fixed;
    top: ${theme.navbarHeight};
    left: 0;
    right: 0;
    bottom: 0;
    display: grid;
    grid-template-rows: auto 1fr auto auto;
    background: white;
    z-index: 999;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
`;

const Header = styled.header`
    padding: 16px 24px;
    height: ${theme.navbarHeight};
    background-color: ${theme.primary};
    color: white;
    height:58px;
    display: flex;
    align-items: center;
    gap: 12px;
    box-shadow: ${theme.shadow};
`;

const MessagesContainer = styled.main`
    overflow-y: auto;
    padding: 24px;
    display: flex;
    flex-direction: column;
    gap: 16px;
    background: ${theme.background};
`;

const MessageContainer = styled.div`
    max-width: 75%;
    min-width: 30%;
    width: fit-content;
    text-align: left;
    padding: 16px;
    border-radius: ${props => props.role === 'user' 
        ? `${theme.borderRadius} ${theme.borderRadius} 4px ${theme.borderRadius}`
        : `4px ${theme.borderRadius} ${theme.borderRadius} ${theme.borderRadius}`};
    background: ${props => props.role === 'user' ? theme.primary : theme.botBackground};
    color: ${props => props.role === 'user' ? theme.userText : theme.botText};
    align-self: ${props => props.role === 'user' ? 'flex-end' : 'flex-start'};
    box-shadow: ${theme.shadow};
    transition: ${theme.transition};
    margin: 8px 0;
    word-break: break-word;
    display: flex;
    flex-direction: column;
    gap: 8px;
    
    &:hover {
        box-shadow: ${theme.shadowHover};
        transform: translateY(-1px);
    }
`;

const TimeStamp = styled.time`
    align-self: flex-end;
    font-size: 0.75rem;
    opacity: 0.7;
    margin-top: 8px;
    display: block;
    width: 100%;
    text-align: right;
`;

const MarkdownWrapper = styled.div`
    font-size: 0.9375rem;
    line-height: 1.6;
    
    strong {
        color: ${theme.primary};
        font-weight: 600;
    }
    
    em {
        color: ${theme.primary};
        font-style: italic;
    }
    
    ul, ol {
        padding-left: 24px;
        margin: 8px 0;
    }
    
    li {
        margin: 4px 0;
    }
    
    code {
        background: rgba(0,0,0,0.1);
        padding: 2px 4px;
        border-radius: 4px;
        font-family: monospace;
    }
    
    pre {
        background: ${theme.botBackground};
        padding: 12px;
        border-radius: ${theme.borderRadius};
        overflow-x: auto;
        margin: 12px 0;
    }
`;

const InputArea = styled.div`
    padding: 16px 24px;
    background-color: ${theme.inputBackground};
    box-shadow: 0 -4px 12px rgba(0,0,0,0.05);
    position: relative;
`;

const LoadingIndicator = styled.div`
    align-self: flex-start;
    padding: 12px 16px;
    border-radius: ${theme.borderRadius};
    background-color: ${theme.botBackground};
    box-shadow: ${theme.shadow};
    display: flex;
    align-items: center;
    gap: 8px;
`;

const Spinner = styled.div`
    width: 16px;
    height: 16px;
    border: 2px solid ${theme.primary};
    border-top-color: transparent;
    border-radius: 50%;
    animation: ${spin} 0.8s linear infinite;
`;

const ErrorMessage = styled.div`
    padding: 12px 24px;
    background-color: ${theme.errorBg};
    color: ${theme.error};
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.875rem;
`;

const Chatbot = () => {
  const { messages, setMessages } = useContext(ChatContext); // Use ChatContext
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);
  const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';

  const scrollToBottom = () => {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
      scrollToBottom();
  }, [messages]);

  const formatMessageContent = (content, role) => {
      if (role === 'bot') {
          return (
              <MarkdownWrapper>
                  <ReactMarkdown
                      components={{
                          strong: ({ node, ...props }) => (
                              <strong style={{ color: theme.primary }} {...props} />
                          ),
                          em: ({ node, ...props }) => (
                              <em style={{ color: theme.primary }} {...props} />
                          ),
                          code: ({ node, ...props }) => (
                              <code style={{ 
                                  background: `${theme.primary}10`, 
                                  color: theme.primary 
                              }} {...props} />
                          )
                      }}
                  >
                      {content}
                  </ReactMarkdown>
              </MarkdownWrapper>
          );
      }
      
      return content.split(/(r\d+|\(\d+,\s\d+\))/g).map((part, index) =>
          /(r\d+|\(\d+,\s\d+\))/.test(part) ? (
              <strong key={index} style={{ 
                  color: theme.primary,
                  padding: '2px 4px',
                  borderRadius: '4px',
                  backgroundColor: `${theme.primary}10`
              }}>
                  {part}
              </strong>
          ) : part
      );
  };

  const handleSendMessage = async () => {
      if (!inputMessage.trim() || isLoading) return;

      try {
          const userMessage = {
              content: inputMessage,
              role: 'user',
              timestamp: new Date().toISOString()
          };

          setMessages(prev => [...prev, userMessage]);
          setInputMessage('');
          setIsLoading(true);
          setError(null);

          const response = await axios.post(`${API_BASE_URL}/api/chat`, {
              message: inputMessage
          });

          if (response.status !== 200) {
              throw new Error(response.data.error || 'API request failed');
          }

          const botMessage = {
              content: response.data.response,
              role: 'bot',
              timestamp: new Date().toISOString(),
              stats: response.data.stats
          };

          setMessages(prev => [...prev, botMessage]);
      } catch (err) {
          setError(err.message || 'Failed to get response. Please try again.');
          setMessages(prev => prev.filter(msg => msg.content !== inputMessage));
      } finally {
          setIsLoading(false);
      }
  };

  return (
      <ChatContainer>
          <Header>
              <div className="logo">
                  <span>🤖</span>
              </div>
              <div>
                  <h1>Requirements Assistant</h1>
                  <p>Powered by Gemini AI</p>
              </div>
              <button 
                  onClick={() => {
                      setMessages([]);
                      localStorage.removeItem('chatHistory');
                  }}
                  className="clear-button"
              >
                  Clear Chat
              </button>
          </Header>

          <MessagesContainer>
              {messages.map((message, index) => (
                  <MessageContainer 
                      key={`${message.timestamp}-${index}`}
                      role={message.role}
                  >
                      <div>
                          {formatMessageContent(message.content, message.role)}
                          {message.stats && (
                              <button
                                  aria-label="System stats"
                                  className="stats-button"
                                  title={`Approved: ${message.stats.approved}\nIn Review: ${message.stats.inReview}\nDisapproved: ${message.stats.disapproved}`}
                              >
                                  ⓘ
                              </button>
                          )}
                      </div>
                      <TimeStamp>
                          {new Date(message.timestamp).toLocaleTimeString([], {
                              hour: '2-digit',
                              minute: '2-digit'
                          })}
                      </TimeStamp>
                  </MessageContainer>
              ))}

              {isLoading && (
                  <LoadingIndicator>
                      <Spinner />
                      <span>Analyzing requirements...</span>
                  </LoadingIndicator>
              )}
              <div ref={messagesEndRef} />
          </MessagesContainer>

          <InputArea>
              <div className="input-wrapper">
                  <input
                      type="text"
                      aria-label="Type your message"
                      placeholder="Ask about requirements..."
                      value={inputMessage}
                      onChange={(e) => setInputMessage(e.target.value)}
                      onKeyPress={(e) => e.key === 'Enter' && !isLoading && handleSendMessage()}
                      disabled={isLoading}
                  />
                  <button
                      onClick={handleSendMessage}
                      disabled={!inputMessage.trim() || isLoading}
                      className="send-button"
                  >
                      <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                          <path
                              d="M3 12L21 12M21 12L12.6 3.6M21 12L12.6 20.4"
                              stroke="white"
                              strokeWidth="2"
                              strokeLinecap="round"
                              strokeLinejoin="round"
                          />
                      </svg>
                  </button>
              </div>
          </InputArea>

          {error && (
              <ErrorMessage role="alert">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill={theme.error}>
                      <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z" />
                  </svg>
                  {error}
              </ErrorMessage>
          )}

          <style jsx>{`
              .logo {
                  background: rgba(255,255,255,0.2);
                  border-radius: 50%;
                  width: 40px;
                  height: 40px;
                  display: flex;
                  align-items: center;
                  justify-content: center;
                  font-size: 33px;
              }

               .clear-button {
                /* Basic styling */
                background-color:rgb(0, 0, 0); /* Primary color */
                color: #FFFFFF; /* Text color */
                padding: 12px 24px; /* Padding for better spacing */
                border: none; /* Remove default border */
                border-radius: 8px; /* Rounded corners */
                margin-left: auto;
                font-size: 16px; /* Font size */
                font-weight: 500; /* Medium font weight */
                cursor: pointer; /* Pointer cursor on hover */
                position: right; /* Center button */
                background-color:rgb(255, 58, 58); /* Primary color */
                box-shadow: 0 4px 6px rgba(255, 255, 255, 0.1); /* Subtle shadow */
                transition: ${theme.transition};
            }

            /* Hover effect */
            .clear-button:hover {
                background-color:rgb(255, 58, 58); /* Darker shade for hover */
                box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15); /* Larger shadow on hover */
                transform: translateY(-2px); /* Slight lift effect */
            }

            /* Active effect (when clicked) */
            .clear-button:active {
                background-color:rgb(255, 255, 255); /* Even darker shade for active state */
                box-shadow: 0 3px 6px rgba(0, 0, 0, 0.1); /* Smaller shadow when pressed */
                transform: translateY(0); /* Reset lift effect */
            }

            /* Focus effect (for accessibility) */
            .clear-button:focus {
                outline: none; /* Remove default outline */
                box-shadow: 0 0 0 3px rgba(255, 255, 255, 0.5); /* Custom focus ring */
            }

            .custom-button:disabled {
    background-color: #CCCCCC; /* Grayed out */
    color: #666666; /* Dimmed text */
    cursor: not-allowed; /* Disabled cursor */
    box-shadow: none; /* No shadow */
    transform: none; /* No transform */
}


              .input-wrapper {
                  display: flex;
                  gap: 12px;
                  align-items: center;
                  position: relative;
              }

              input {
                  flex: 1;
                  padding: 12px 20px;
                  border-radius: 28px;
                  border: 1px solid ${theme.primary}30;
                  background: ${theme.inputBackground};
                  font-size: 0.9375rem;
                  transition: ${theme.transition};
                  box-shadow: ${theme.shadow};
                  outline: none;
              }

              input:focus {
                  border-color: ${theme.primary}60;
                  box-shadow: 0 0 0 3px ${theme.primary}20;
              }

              .send-button {
                  width: 48px;
                  height: 48px;
                  display: flex;
                  align-items: center;
                  justify-content: center;
                  background: ${theme.primary};
                  border: none;
                  border-radius: 50%;
                  cursor: pointer;
                  transition: ${theme.transition};
                  opacity: ${isLoading ? 0.6 : 1};
              }

              .send-button:hover {
                  transform: ${!inputMessage.trim() || isLoading ? 'none' : 'scale(1.05)'};
              }

              .stats-button {
                  margin-left: 8px;
                  background: none;
                  border: none;
                  color: ${theme.primary};
                  cursor: help;
                  padding: 0;
                  transition: ${theme.transition};
              }

              .stats-button:hover {
                  opacity: 0.8;
              }
          `}</style>
      </ChatContainer>
  );
};

export default Chatbot;