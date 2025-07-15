import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Send, Bot, User, Copy, Check, BrainCircuit, PanelLeft } from 'lucide-react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { CopyToClipboard } from 'react-copy-to-clipboard';

// Component สำหรับแสดง Code Block พร้อมปุ่มคัดลอก
const CodeBlock = ({ language, value }) => {
    const [isCopied, setIsCopied] = useState(false);
    const handleCopy = () => {
        setIsCopied(true);
        setTimeout(() => setIsCopied(false), 2000);
    };
    return (
        <div className="relative my-2 text-sm font-mono">
            <div className="flex items-center justify-between bg-gray-800 text-gray-300 px-4 py-1.5 rounded-t-md">
                <span className="text-xs">{language || 'code'}</span>
                <CopyToClipboard text={value} onCopy={handleCopy}>
                    <button className="flex items-center gap-1.5 text-xs hover:text-white transition-colors">
                        {isCopied ? <Check size={14} className="text-green-500" /> : <Copy size={14} />}
                        {isCopied ? 'Copied!' : 'Copy code'}
                    </button>
                </CopyToClipboard>
            </div>
            <SyntaxHighlighter language={language} style={oneDark} customStyle={{ margin: 0, borderRadius: '0 0 0.375rem 0.375rem', padding: '1rem' }}>
                {String(value).trim()}
            </SyntaxHighlighter>
        </div>
    );
};

// Component สำหรับจัดการแสดงผลข้อความและ Code Block
const MessageContent = ({ text }) => {
    const codeBlockRegex = /```(\w+)?\n([\s\S]+?)\n```/g;
    const parts = text.split(codeBlockRegex);

    return (
        <div>
            {parts.map((part, index) => {
                if (index % 3 === 2) { // นี่คือส่วนของโค้ด
                    const language = parts[index - 1] || 'plaintext';
                    return <CodeBlock key={index} language={language} value={part} />;
                } else if (index % 3 === 0) { // นี่คือส่วนของข้อความธรรมดา
                    return <p key={index} className="whitespace-pre-wrap">{part}</p>;
                }
                return null;
            })}
        </div>
    );
};

// Component สำหรับแสดงแต่ละข้อความ (ของ User และ Bot)
const Message = ({ text, sender }) => {
    const isUser = sender === 'user';
    return (
        <div className={`flex items-start gap-3 ${isUser ? 'justify-end' : ''} my-4`}>
            {!isUser && (
                <div className="flex-shrink-0 w-9 h-9 rounded-full flex items-center justify-center bg-gray-700 text-white">
                    <Bot size={20} />
                </div>
            )}
            <div className={`max-w-2xl px-5 py-3 rounded-xl shadow-sm break-words ${isUser ? 'bg-blue-600 text-white' : 'bg-white text-gray-800 border'}`}>
                <MessageContent text={text} />
            </div>
            {isUser && (
                 <div className="flex-shrink-0 w-9 h-9 rounded-full flex items-center justify-center bg-blue-600 text-white">
                    <User size={20} />
                </div>
            )}
        </div>
    );
};

function App() {
    const [messages, setMessages] = useState([
        { text: 'Hello! I am Panya, your AI assistant for PLCnext. How can I help you today?', sender: 'bot' },
    ]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [isSidebarOpen, setIsSidebarOpen] = useState(true);
    const chatEndRef = useRef(null);

    useEffect(() => {
        chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    const handleSendMessage = async (e) => {
        e.preventDefault();
        if (!input.trim() || isLoading) return;

        const userMessage = { text: input, sender: 'user' };
        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setIsLoading(true);

        try {
            // --- จุดเชื่อมต่อ Backend ที่สำคัญ ---
            // ใช้ /api/chat ซึ่งตรงกับ proxy ใน vite.config.js
            // และส่ง payload เป็น { message: input } ซึ่งตรงกับที่ backend ต้องการ
            const response = await axios.post('/api/chat', { message: input });
            const botMessage = { text: response.data.reply, sender: 'bot' };
            setMessages(prev => [...prev, botMessage]);
        } catch (error) {
            console.error("API Error: ", error);
            const errorMessageText = error.response?.data?.detail || 'ขออภัยครับ, เกิดข้อผิดพลาดในการเชื่อมต่อกับเซิร์ฟเวอร์';
            const errorMessage = { text: errorMessageText, sender: 'bot' };
            setMessages(prev => [...prev, errorMessage]);
        } finally {
            setIsLoading(false);
        }
    };

    return (
      <div className="flex h-screen bg-white text-gray-800 font-sans">
        <aside className={`bg-gray-50 border-r border-gray-200 flex flex-col transition-all duration-300 ease-in-out ${isSidebarOpen ? 'w-72 p-4' : 'w-0 p-0'}`}>
          <div className={`flex-shrink-0 mb-6 flex items-center gap-3 overflow-hidden ${isSidebarOpen ? 'opacity-100' : 'opacity-0'}`}>
            <div className="bg-blue-600 p-2 rounded-lg">
              <img src="/src/assets/logo.png" alt="PLCnext Logo" className="w-8 h-8 object-contain" />
            </div>
            <div><h1 className="text-xl font-bold text-gray-900">PLCnext AI</h1></div>
          </div>
          {/* พื้นที่สำหรับแสดงประวัติการแชทในอนาคต */}
        </aside>

        <div className="flex-1 flex flex-col bg-gray-100">
          <header className="flex items-center p-2 bg-white border-b border-gray-200 shadow-sm">
            <button onClick={() => setIsSidebarOpen(!isSidebarOpen)} className="p-2 text-gray-500 hover:text-gray-900 hover:bg-gray-100 rounded-md transition-colors"><PanelLeft size={20} /></button>
            <h2 className="ml-2 font-semibold text-gray-700">Smart Assistant for PLCnext</h2>
          </header>

          <main className="flex-1 p-6 overflow-y-auto">
            <div className="max-w-4xl mx-auto">
              {messages.map((msg, index) => (
                <Message key={index} text={msg.text} sender={msg.sender} />
              ))}
              {isLoading && (
                <div className="flex items-start gap-3 my-4">
                  <div className="flex-shrink-0 w-9 h-9 rounded-full flex items-center justify-center bg-gray-700 text-white"><Bot size={20} /></div>
                  <div className="max-w-lg px-5 py-4 rounded-xl shadow-sm bg-white border">
                    <div className="flex items-center space-x-2">
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-pulse [animation-delay:-0.3s]"></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-pulse [animation-delay:-0.15s]"></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-pulse"></div>
                    </div>
                  </div>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>
          </main>
          
          <footer className="p-4 bg-gray-100/80 backdrop-blur-sm">
            <div className="max-w-4xl mx-auto">
              <form onSubmit={handleSendMessage} className="flex items-center space-x-2 bg-white border border-gray-300 rounded-full p-2 shadow-sm focus-within:ring-2 focus-within:ring-blue-500 transition-all">
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="ถามคำถามเกี่ยวกับ PLCnext..."
                  className="flex-1 bg-transparent focus:outline-none px-2 text-gray-800 placeholder-gray-500"
                  disabled={isLoading}
                />
                <button type="submit" className="bg-blue-600 text-white p-2.5 rounded-full font-semibold hover:bg-blue-700 transition-colors shadow-sm disabled:bg-blue-300 disabled:cursor-not-allowed flex-shrink-0" disabled={isLoading || !input.trim()} aria-label="Send message"><Send size={20} /></button>
              </form>
            </div>
          </footer>
        </div>
      </div>
    );
}

export default App;