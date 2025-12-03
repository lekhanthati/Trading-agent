# Trading-agent

## ✨ Objective

To build an autonomous, conversational trading agent that can understand natural-language prompts, analyze live forex market data, generate short-term price predictions, and execute real-time trades through an integrated Binance MCP server

## 📂 Project Structure
```
├── main.py            # Main application 
├── predict.py         # Market prediction tool
├── scaler.pkl         # Scaler for prediction tool
├── model.pth          # Transformer Model for prediction tool
├── requirements.txt   # Dependencies      
└── README.md         
```
## 🛠️ Tech Stack

| Technology    | Purpose                          |
|---------------|----------------------------------|
| **LangChain** | Framework to create agents       |
| **FastMCP**   | Library to build MCP servers     |
| **pandas**    | Data manipulation and analysis   |
| **numpy**     | Numerical computing operations   |
| **Streamlit** | Web Application Framework        |
| **torch**     | Deep learning model building     |

## 🔑 Environment Setup

Before running the app, make sure to set up your environment:

1. Create & activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install binance-mcp-server 
   ```
3. Run the app:
   ```bash
   fastmcp run predict.py:mcp --transport http --port 8001
   binance-mcp-server --api-key $BINANCE_API_KEY --api-secret $BINANCE_API_SECRET --binance-testnet
   streamlit run main.py
   ```
