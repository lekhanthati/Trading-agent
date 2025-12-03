import joblib
import ccxt
import pandas as pd
import joblib
import torch
import torch.nn as nn
import numpy as np
from fastmcp import FastMCP

mcp = FastMCP("Server-2")


exchange = ccxt.binance({'enableRateLimit': True})

def fetch_last_200_candles_ist_naive(symbol='BTCUSDT', timeframe='1m'):
    data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=200)

    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    # Convert to datetime (UTC)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

    # Convert to IST
    df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Kolkata')

    # Remove timezone → make timestamp naive (no +05:30)
    df['timestamp'] = df['timestamp'].dt.tz_localize(None)

    # Set index
    df.set_index('timestamp', inplace=True)

    return df



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class TransformerRegressor(nn.Module):
    def __init__(self, num_features=5, d_model=64, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=0.2, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        self.regressor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 32), 
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = x[:, -1, :]  # last timestep
        out = self.regressor(x)
        return out.squeeze(-1)



@mcp.tool()
def predict(symbol : str = 'BTCUSDT', timeframe :  str = '1m') -> dict:
    """
    Predict the next candle magnitude and direction for the given symbol and timeframe.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT', 'ETHUSDT')
        timeframe: Timeframe for candlestick data (e.g., '1m', '5m', '1h')
        
    Returns:
        Dictionary containing the magnitude and direction of the predicted candle.
    """
    # Load data
    df = fetch_last_200_candles_ist_naive(symbol, timeframe)
    # Load scaler
    scaler = joblib.load('scaler.pkl')

    # Instantiate model
    model = TransformerRegressor()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)


    checkpoint = torch.load("model2.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    seq_scaled = scaler.transform(df[['open','high','low','close','volume']])

        # ---- 2) Convert to tensor shape (1, seq_len, num_features) ----
    x = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0).to(device)

        # ---- 3) Make prediction ----
    model.eval()
    with torch.no_grad():
        prediction = model(x).item()

    direction = 'positive' if prediction > 0 else 'negative'
    time = str(df.iloc[-1].name)

    return {'magnitude' : prediction, 'direction' : direction, 'time' : time}

if __name__ == "__main__":

    mcp.run()
