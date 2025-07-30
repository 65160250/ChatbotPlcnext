import { useState, useEffect, useRef } from "react";
import axios from "axios";
import {
  Send,
  Bot,
  User,
  Copy,
  Check,
  PanelLeft,
  Plus,
  MessageSquareText,
  Trash2,
  Paperclip,
Mic,
} from "lucide-react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import { CopyToClipboard } from "react-copy-to-clipboard";

// --- Helper: คำนวณเวลาที่ผ่านไป ---
const timeAgo = (date) => {
  const seconds = Math.floor((new Date() - new Date(date)) / 1000);
  let interval = seconds / 31536000;
  if (interval > 1) return Math.floor(interval) + " years ago";
  interval = seconds / 2592000;
  if (interval > 1) return Math.floor(interval) + " months ago";
  interval = seconds / 86400;
  if (interval > 1) return Math.floor(interval) + " days ago";
  interval = seconds / 3600;
  if (interval > 1) return Math.floor(interval) + " hours ago";
  interval = seconds / 60;
  if (interval > 1) return Math.floor(interval) + " minutes ago";
  return "Just now";
};

const CodeBlock = ({ language, value }) => {
  const [isCopied, setIsCopied] = useState(false);
  const handleCopy = () => {
    setIsCopied(true);
    setTimeout(() => setIsCopied(false), 2000);
  };
  return (
    <div className="relative my-2 text-sm font-mono">
      <div className="flex items-center justify-between bg-gray-800 text-gray-300 px-4 py-1.5 rounded-t-md">
        <span className="text-xs">{language || "code"}</span>
        <CopyToClipboard text={value} onCopy={handleCopy}>
          <button className="flex items-center gap-1.5 text-xs hover:text-white transition-colors">
            {isCopied ? (
              <Check size={14} className="text-green-500" />
            ) : (
              <Copy size={14} />
            )}
            {isCopied ? "Copied!" : "Copy code"}
          </button>
        </CopyToClipboard>
      </div>
      <SyntaxHighlighter
        language={language}
        style={oneDark}
        customStyle={{
          margin: 0,
          borderRadius: "0 0 0.375rem 0.375rem",
          padding: "1rem",
        }}
      >
        {String(value).trim()}
      </SyntaxHighlighter>
    </div>
  );
};

const MessageContent = ({ text }) => {
  const codeBlockRegex = /```(\w+)?\n([\s\S]+?)\n```/g;
  const parts = text.split(codeBlockRegex);

  return (
    <div>
      {parts.map((part, index) => {
        if (index % 3 === 2) {
          const language = parts[index - 1] || "plaintext";
          return <CodeBlock key={index} language={language} value={part} />;
        } else if (index % 3 === 0) {
          return (
            <p key={index} className="whitespace-pre-wrap">
              {part}
            </p>
          );
        }
        return null;
      })}
    </div>
  );
};

const Message = ({ text, sender, image }) => {
  const isUser = sender === "user";
  return (
    <div className={`flex items-start gap-3 ${isUser ? "justify-end" : ""} my-4`}>
      {!isUser && (
        <div className="flex-shrink-0 w-9 h-9 rounded-full flex items-center justify-center bg-gray-700 text-white">
          <Bot size={20} />
        </div>
      )}
      <div className={`max-w-2xl px-5 py-3 rounded-xl shadow-sm break-words ${isUser ? "bg-blue-600 text-white" : "bg-white text-gray-800 border"}`}>
        {image && (
          <img src={image} alt="upload" className="mb-2 max-h-40 rounded border" />
        )}
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
  const [chatHistory, setChatHistory] = useState([]);
  const [activeChatId, setActiveChatId] = useState(null);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [confirmDeleteId, setConfirmDeleteId] = useState(null); 
  const [imageFile, setImageFile] = useState(null);


  // ------ สำหรับ voice to text ------
  const [isListening, setIsListening] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  // ----------------------------------

  const chatEndRef = useRef(null);

  const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";

  useEffect(() => {
    try {
      const savedHistory = localStorage.getItem("plcnextChatHistory");
      const thirtyDaysAgo = Date.now() - 30 * 24 * 60 * 60 * 1000;
      let history = savedHistory ? JSON.parse(savedHistory) : [];
      const recentHistory = history.filter(
        (chat) => new Date(chat.createdAt).getTime() > thirtyDaysAgo
      );
      setChatHistory(recentHistory);
      if (recentHistory.length > 0) {
        setActiveChatId(recentHistory[0].id);
      } else {
        handleNewChat();
      }
    } catch (error) {
      handleNewChat();
    }
  }, []);

  useEffect(() => {
    if (chatHistory.length > 0) {
      localStorage.setItem("plcnextChatHistory", JSON.stringify(chatHistory));
    }
  }, [chatHistory]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory, activeChatId]);

  // ฟังก์ชันสำหรับปุ่มที่ยังไม่ implement จริง
  const handleFeatureNotImplemented = (feature) => {
    alert(`${feature} feature is not implemented yet.`);
  };

    // 1. เริ่มฟังเสียง
  const handleStartListening = async () => {
    setIsListening(true);
    setIsTranscribing(false);
    setInput(""); // clear input
    audioChunksRef.current = [];
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorderRef.current = new window.MediaRecorder(stream, { mimeType: "audio/webm" });
    mediaRecorderRef.current.ondataavailable = (e) => {
      if (e.data.size > 0) {
        audioChunksRef.current.push(e.data);
      }
    };
    mediaRecorderRef.current.start();
  };

  // 2. หยุดฟัง → ส่งไฟล์ไป backend
  const handleStopListening = async () => {
    setIsListening(false);
    setIsTranscribing(true);

    // หยุดบันทึกเสียงและรอ audio
    mediaRecorderRef.current.stop();
    mediaRecorderRef.current.stream.getTracks().forEach((track) => track.stop());

    // รอให้ ondataavailable ทำงานเสร็จ
    mediaRecorderRef.current.onstop = async () => {
      const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" });
      // ส่งไป backend
      try {
        const formData = new FormData();
        formData.append("file", audioBlob, "audio.webm");
        const res = await axios.post(`${API_URL}/api/transcribe`, formData, {
          headers: { "Content-Type": "multipart/form-data" },
        });
        setInput(res.data.text || "");
      } catch (err) {
        setInput("");
        alert("ถอดเสียงไม่สำเร็จ: " + err.message);
      } finally {
        setIsTranscribing(false);
      }
    };
  };

  const handleNewChat = () => {
    const newChat = {
      id: Date.now().toString(),
      title: "New Chat",
      createdAt: new Date().toISOString(),
      messages: [
        {
          text: "Hello! I am Panya, your AI assistant for PLCnext. How can I help you today?",
          sender: "bot",
        },
      ],
    };
    setChatHistory((prev) => [newChat, ...prev]);
    setActiveChatId(newChat.id);
    setInput("");
  };

  const handleSelectChat = (chatId) => {
    setActiveChatId(chatId);
  };

  // (เปลี่ยนให้แค่ setConfirmDeleteId)
  const handleDeleteChat = (e, chatIdToDelete) => {
    e.stopPropagation();
    setConfirmDeleteId(chatIdToDelete);
  };

  // ฟังก์ชันที่ใช้เมื่อกดยืนยันลบ (Yes)
  const confirmDeleteChat = () => {
    setChatHistory((prev) =>
      prev.filter((chat) => chat.id !== confirmDeleteId)
    );
    if (activeChatId === confirmDeleteId) {
      const remainingChats = chatHistory.filter(
        (chat) => chat.id !== confirmDeleteId
      );
      if (remainingChats.length > 0) {
        setActiveChatId(remainingChats[0].id);
      } else {
        handleNewChat();
      }
    }
    setConfirmDeleteId(null);
  };

  // ฟังก์ชันที่ใช้เมื่อกดไม่ลบ (No)
  const cancelDeleteChat = () => {
    setConfirmDeleteId(null);
  };

  const handleSendMessage = async (e) => {
  e.preventDefault();
  if ((!input.trim() && !imageFile) || isLoading || !activeChatId) return;

  const userMessage = {
    text: input,
    sender: "user",
    image: imageFile ? URL.createObjectURL(imageFile) : null,
  };
  setInput("");
  setIsLoading(true);

  // push user message ก่อน
  setChatHistory((prev) => {
    const newHistory = [...prev];
    const activeChatIndex = newHistory.findIndex(
      (chat) => chat.id === activeChatId
    );
    if (activeChatIndex !== -1) {
      newHistory[activeChatIndex].messages.push(userMessage);
      const userMessages = newHistory[activeChatIndex].messages.filter(
        (m) => m.sender === "user"
      );
      if (userMessages.length === 1) {
        newHistory[activeChatIndex].title =
          input.length > 30 ? `${input.substring(0, 27)}...` : input;
      }
    }
    return newHistory;
  });

  try {
    const formData = new FormData();
    formData.append("message", input);
    if (imageFile) {
      formData.append("file", imageFile);
    }

    // ส่งแบบ POST ไปที่ /api/agent-chat
    const response = await axios.post(
      `${API_URL}/api/agent-chat`,
      formData,
      { headers: { "Content-Type": "multipart/form-data" } }
    );
    const botMessage = { text: response.data.answer, sender: "bot" };
    setChatHistory((prev) => {
      const newHistory = [...prev];
      const activeChatIndex = newHistory.findIndex(
        (chat) => chat.id === activeChatId
      );
      if (activeChatIndex !== -1) {
        newHistory[activeChatIndex].messages.push(botMessage);
      }
      return newHistory;
    });
  } catch (error) {
    const errorMessageText =
      error.response?.data?.detail ||
      "Sorry, there was an error connecting to the server.";
    const errorMessage = { text: errorMessageText, sender: "bot" };
    setChatHistory((prev) => {
      const newHistory = [...prev];
      const activeChatIndex = newHistory.findIndex(
        (chat) => chat.id === activeChatId
      );
      if (activeChatIndex !== -1) {
        newHistory[activeChatIndex].messages.push(errorMessage);
      }
      return newHistory;
    });
  } finally {
    setIsLoading(false);
    setImageFile(null); // ล้างไฟล์หลังส่ง
  }
};


  const activeChat = chatHistory.find((chat) => chat.id === activeChatId);
  const messagesToDisplay = activeChat ? activeChat.messages : [];

  return (
    <div className="flex h-screen bg-white text-gray-800 font-sans">
      {/* Sidebar */}
      <aside
        className={`bg-gray-50 border-r border-gray-200 flex flex-col transition-all duration-300 ease-in-out ${
          isSidebarOpen ? "w-72 p-4" : "w-0 p-0"
        }`}
      >
        <div
          className={`flex-shrink-0 mb-4 flex items-center justify-between overflow-hidden transition-opacity duration-200 ${
            isSidebarOpen ? "opacity-100" : "opacity-0"
          }`}
        >
          <div className="flex items-center gap-3">
            <div className="bg-blue-600 p-1 rounded-full">
              <img
                src="/src/assets/logo.png"
                alt="PLCnext Logo"
                className="w-10 h-10 object-cover rounded-full border-2 border-white shadow"
              />
            </div>
            <h1 className="text-xl font-bold text-gray-900">Panya</h1>
          </div>
        </div>

        <button
          className={`flex items-center justify-center gap-2 w-full p-2.5 mb-4 bg-blue-600 text-white hover:bg-blue-700 rounded-lg transition-colors text-sm font-semibold mx-auto overflow-hidden ${
            isSidebarOpen ? "opacity-100" : "opacity-0"
          }`}
          onClick={handleNewChat}
        >
          <Plus size={18} /> New Chat
        </button>

        {/* Chat History List */}
        <div
          className={`flex-1 overflow-y-auto space-y-2 transition-opacity duration-200 ${
            isSidebarOpen ? "opacity-100" : "opacity-0"
          }`}
        >
          {chatHistory.map((chat) => (
            <div
              key={chat.id}
              onClick={() => handleSelectChat(chat.id)}
              className={`group relative flex items-center justify-between w-full p-2.5 rounded-lg cursor-pointer transition-colors ${
                activeChatId === chat.id
                  ? "bg-blue-100 text-blue-800"
                  : "hover:bg-gray-200"
              }`}
            >
              <div className="flex items-center gap-3">
                <MessageSquareText size={16} className="text-gray-500" />
                <div className="flex flex-col">
                  <span className="text-sm font-medium truncate w-40">
                    {chat.title}
                  </span>
                  <span className="text-xs text-gray-400">
                    {timeAgo(chat.createdAt)}
                  </span>
                </div>
              </div>
              <button
                onClick={(e) => handleDeleteChat(e, chat.id)}
                className="absolute right-2 top-1/2 -translate-y-1/2 p-1 text-gray-400 hover:text-red-500 opacity-0 group-hover:opacity-100 transition-opacity"
              >
                <Trash2 size={16} />
              </button>
            </div>
          ))}
        </div>
        {/* --- Pop up ยืนยันการลบแชท --- */}
        {confirmDeleteId && (
          <div className="fixed inset-0 z-40 bg-black bg-opacity-40 flex items-center justify-center">
            <div className="bg-white rounded-lg shadow-xl p-6 w-80 flex flex-col items-center">
              <p className="text-lg font-semibold text-gray-800 mb-4 text-center">
                คุณต้องการลบแชทนี้หรือไม่?
              </p>
              <div className="flex gap-4">
                <button
                  className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded"
                  onClick={confirmDeleteChat}
                >
                  ใช่, ลบ
                </button>
                <button
                  className="bg-gray-200 hover:bg-gray-300 text-gray-800 px-4 py-2 rounded"
                  onClick={cancelDeleteChat}
                >
                  ไม่ลบ
                </button>
              </div>
            </div>
          </div>
        )}
      </aside>
      {/* Main Content */}
      <div className="flex-1 flex flex-col bg-gray-100">
        <header className="flex items-center p-2 bg-white border-b border-gray-200 shadow-sm">
          <button
            onClick={() => setIsSidebarOpen(!isSidebarOpen)}
            className="p-2 text-gray-500 hover:text-gray-900 hover:bg-gray-100 rounded-md transition-colors"
          >
            <PanelLeft size={20} />
          </button>
          <h2 className="ml-2 font-semibold text-gray-700">
            {activeChat?.title || "Smart Assistant"}
          </h2>
        </header>

        <main className="flex-1 p-6 overflow-y-auto">
          <div className="max-w-4xl mx-auto">
            {messagesToDisplay.map((msg, index) => (
              <Message key={index} text={msg.text} sender={msg.sender} image={msg.image}/>
            ))}
            {isLoading && (
              <div className="flex items-start gap-3 my-4">
                <div className="flex-shrink-0 w-9 h-9 rounded-full flex items-center justify-center bg-gray-700 text-white">
                  <Bot size={20} />
                </div>
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
    {/* ฟุตเตอร์ 3 โหมด */}
    {!isListening && !isTranscribing && (
      <form
              onSubmit={handleSendMessage}
              className="flex items-center space-x-2 bg-white border border-gray-300 rounded-full p-2 shadow-sm focus-within:ring-2 focus-within:ring-blue-500 transition-all"
            >
              {/* ปุ่มแนบไฟล์ (image เท่านั้น) */}
              <label className="p-2 text-gray-500 hover:text-blue-600 hover:bg-gray-100 rounded-full transition-colors cursor-pointer">
                <Paperclip size={20} />
                <input
                  type="file"
                  accept="image/*"
                  style={{ display: "none" }}
                  onChange={e => {
                    if (e.target.files[0]) setImageFile(e.target.files[0]);
                  }}
                  disabled={isLoading}
                />
              </label>
              {/* preview file */}
              {imageFile && (
                <div className="ml-2 text-xs text-gray-500 flex items-center">
                  <img src={URL.createObjectURL(imageFile)} alt="preview" className="w-16 h-12 object-contain mr-2 rounded border" />
                    <span>{imageFile.name}</span>
                  <button
                    onClick={() => setImageFile(null)}
                    type="button"
                    className="ml-1 text-red-500 hover:text-red-700"
                  >
                    ×
                  </button>
                </div>
              )}

              {/* ปุ่มไมค์/voice (เหมือนเดิม) */}
              <button
                type="button"
                onClick={handleStartListening}
                className="p-2 text-gray-500 hover:text-blue-600 hover:bg-gray-100 rounded-full transition-colors"
                aria-label="Use microphone"
                disabled={isLoading}
              >
                <Mic size={20} />
              </button>
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask something about PLCnext..."
                className="flex-1 bg-transparent focus:outline-none px-4 text-gray-800 placeholder-gray-500"
                disabled={isLoading}
              />
              <button
                type="submit"
                className="bg-blue-600 text-white p-2.5 rounded-full font-semibold hover:bg-blue-700 transition-colors shadow-sm disabled:bg-blue-300 disabled:cursor-not-allowed flex-shrink-0"
                disabled={isLoading || (!input.trim() && !imageFile)}
                aria-label="Send message"
              >
                <Send size={20} />
              </button>
            </form>
    )}

    {/* โหมด Listening (มี Check) */}
    {isListening && (
      <div className="flex items-center space-x-2 bg-blue-50 border border-blue-300 rounded-full p-2 shadow-sm transition-all w-full">
        <div className="text-blue-600 animate-pulse font-semibold px-2">Listening...</div>
        <button
          type="button"
          onClick={handleStopListening}
          className="ml-auto bg-green-600 text-white p-2 rounded-full font-semibold hover:bg-green-700 transition"
          aria-label="Stop and transcribe"
        >
          <Check size={24} />
        </button>
      </div>
    )}

    {/* โหมด Transcribing */}
    {isTranscribing && (
      <div className="flex items-center space-x-2 bg-gray-50 border border-gray-300 rounded-full p-2 shadow-sm w-full">
        <div className="flex-1 flex items-center gap-2 px-2 text-gray-600">
          <span className="animate-spin">🔄</span>
          Transcribing...
        </div>
        <button
          type="button"
          className="ml-auto bg-gray-300 text-gray-500 p-2 rounded-full cursor-not-allowed"
          disabled
        >
          <Check size={24} />
        </button>
      </div>
    )}
  </div>
</footer>
      </div>
    </div>
  );
}

export default App;
