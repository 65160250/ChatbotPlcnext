# ChatbotPlcnext 
# 🚀 เริ่มต้นใช้งาน PLCnext Chatbot (Initial Setup Guide)

โปรเจกต์นี้คือระบบแชทบอทที่ใช้เทคโนโลยี **LLM + Langchain + PostgreSQL + React** ซึ่งสามารถฝังเอกสารเป็นเวกเตอร์ และตอบคำถามแบบออฟไลน์ได้ด้วยโมเดลจาก Ollama (เช่น LLaMA 3.2)

---

## ✅ สิ่งที่ต้องติดตั้งก่อนเริ่ม

### 🐳 1. Docker & Docker Compose
- ดาวน์โหลด Docker: [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)
- หลังติดตั้งเสร็จ ให้ตรวจสอบด้วย:
```bash
docker --version
docker compose version
```

## start container
```bash
docker compose up --build
```

