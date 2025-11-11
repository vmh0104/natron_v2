//+------------------------------------------------------------------+
//|                                              NatronEA.mq5        |
//|                        Natron Transformer MQL5 Integration       |
//+------------------------------------------------------------------+
#property copyright "Natron V2"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>

//--- Input parameters
input string   ServerHost = "127.0.0.1";      // Socket server host
input int      ServerPort = 8888;             // Socket server port
input double   LotSize = 0.01;                // Lot size
input int      MagicNumber = 123456;          // Magic number
input int      StopLoss = 50;                 // Stop loss (points)
input int      TakeProfit = 100;              // Take profit (points)
input double   BuyThreshold = 0.6;           // Buy probability threshold
input double   SellThreshold = 0.6;           // Sell probability threshold
input int      SequenceLength = 96;           // Required candles for prediction

//--- Global variables
int socketHandle = INVALID_HANDLE;
CTrade trade;
datetime lastBarTime = 0;
double ohlcvBuffer[][5];  // [time][open,high,low,close,volume]

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   trade.SetExpertMagicNumber(MagicNumber);
   trade.SetDeviationInPoints(10);
   trade.SetTypeFilling(ORDER_FILLING_FOK);
   
   Print("🚀 Natron EA initialized");
   Print("   Server: ", ServerHost, ":", ServerPort);
   
   // Connect to socket server
   if(!ConnectToServer())
   {
      Print("❌ Failed to connect to Natron server");
      return INIT_FAILED;
   }
   
   Print("✅ Connected to Natron server");
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(socketHandle != INVALID_HANDLE)
   {
      SocketClose(socketHandle);
   }
   Print("🔌 Natron EA disconnected");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // Check for new bar
   datetime currentBarTime = iTime(_Symbol, PERIOD_CURRENT, 0);
   if(currentBarTime == lastBarTime)
      return;
   
   lastBarTime = currentBarTime;
   
   // Collect OHLCV data
   if(!CollectOHLCVData())
   {
      Print("⚠️ Insufficient data for prediction");
      return;
   }
   
   // Get prediction from Natron server
   double buyProb = 0.0;
   double sellProb = 0.0;
   string regime = "";
   
   if(!GetPrediction(buyProb, sellProb, regime))
   {
      Print("❌ Failed to get prediction");
      return;
   }
   
   Print("📊 Prediction: Buy=", buyProb, ", Sell=", sellProb, ", Regime=", regime);
   
   // Trading logic
   ManagePositions(buyProb, sellProb);
}

//+------------------------------------------------------------------+
//| Connect to socket server                                         |
//+------------------------------------------------------------------+
bool ConnectToServer()
{
   socketHandle = SocketCreate();
   if(socketHandle == INVALID_HANDLE)
   {
      Print("❌ Failed to create socket");
      return false;
   }
   
   if(!SocketConnect(socketHandle, ServerHost, ServerPort, 1000))
   {
      Print("❌ Failed to connect to ", ServerHost, ":", ServerPort);
      SocketClose(socketHandle);
      socketHandle = INVALID_HANDLE;
      return false;
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Collect OHLCV data for prediction                                |
//+------------------------------------------------------------------+
bool CollectOHLCVData()
{
   int bars = iBars(_Symbol, PERIOD_CURRENT);
   if(bars < SequenceLength)
      return false;
   
   ArrayResize(ohlcvBuffer, SequenceLength);
   
   for(int i = 0; i < SequenceLength; i++)
   {
      int idx = SequenceLength - 1 - i;
      ohlcvBuffer[i][0] = iOpen(_Symbol, PERIOD_CURRENT, idx);
      ohlcvBuffer[i][1] = iHigh(_Symbol, PERIOD_CURRENT, idx);
      ohlcvBuffer[i][2] = iLow(_Symbol, PERIOD_CURRENT, idx);
      ohlcvBuffer[i][3] = iClose(_Symbol, PERIOD_CURRENT, idx);
      ohlcvBuffer[i][4] = (double)iVolume(_Symbol, PERIOD_CURRENT, idx);
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Get prediction from Natron server                                |
//+------------------------------------------------------------------+
bool GetPrediction(double &buyProb, double &sellProb, string &regime)
{
   if(socketHandle == INVALID_HANDLE)
   {
      if(!ConnectToServer())
         return false;
   }
   
   // Build JSON request
   string json = BuildJSONRequest();
   
   // Send request
   uchar request[];
   StringToCharArray(json, request, 0, StringLen(json));
   
   if(SocketSend(socketHandle, request) <= 0)
   {
      Print("❌ Failed to send request");
      SocketClose(socketHandle);
      socketHandle = INVALID_HANDLE;
      return false;
   }
   
   // Receive response
   uchar response[];
   int timeout = 1000; // 1 second timeout
   int received = SocketRead(socketHandle, response, timeout);
   
   if(received <= 0)
   {
      Print("❌ No response from server");
      return false;
   }
   
   // Parse JSON response
   string responseStr = CharArrayToString(response);
   return ParseJSONResponse(responseStr, buyProb, sellProb, regime);
}

//+------------------------------------------------------------------+
//| Build JSON request                                               |
//+------------------------------------------------------------------+
string BuildJSONRequest()
{
   string json = "{\"action\":\"predict\",\"ohlcv\":[";
   
   for(int i = 0; i < SequenceLength; i++)
   {
      datetime time = iTime(_Symbol, PERIOD_CURRENT, SequenceLength - 1 - i);
      string timeStr = TimeToString(time, TIME_DATE|TIME_MINUTES);
      
      if(i > 0) json += ",";
      json += "{";
      json += "\"time\":\"" + timeStr + "\",";
      json += "\"open\":" + DoubleToString(ohlcvBuffer[i][0], 5) + ",";
      json += "\"high\":" + DoubleToString(ohlcvBuffer[i][1], 5) + ",";
      json += "\"low\":" + DoubleToString(ohlcvBuffer[i][2], 5) + ",";
      json += "\"close\":" + DoubleToString(ohlcvBuffer[i][3], 5) + ",";
      json += "\"volume\":" + DoubleToString(ohlcvBuffer[i][4], 0);
      json += "}";
   }
   
   json += "]}";
   return json;
}

//+------------------------------------------------------------------+
//| Parse JSON response                                              |
//+------------------------------------------------------------------+
bool ParseJSONResponse(string response, double &buyProb, double &sellProb, string &regime)
{
   // Simple JSON parsing (for production, use a proper JSON library)
   int buyIdx = StringFind(response, "\"buy_prob\":");
   int sellIdx = StringFind(response, "\"sell_prob\":");
   int regimeIdx = StringFind(response, "\"regime\":");
   
   if(buyIdx < 0 || sellIdx < 0 || regimeIdx < 0)
   {
      Print("❌ Invalid JSON response");
      return false;
   }
   
   // Extract buy_prob
   string buyStr = StringSubstr(response, buyIdx + 11, 10);
   buyProb = StringToDouble(buyStr);
   
   // Extract sell_prob
   string sellStr = StringSubstr(response, sellIdx + 12, 10);
   sellProb = StringToDouble(sellStr);
   
   // Extract regime
   int regimeStart = StringFind(response, "\"regime\":\"", regimeIdx) + 10;
   int regimeEnd = StringFind(response, "\"", regimeStart);
   regime = StringSubstr(response, regimeStart, regimeEnd - regimeStart);
   
   return true;
}

//+------------------------------------------------------------------+
//| Manage trading positions                                         |
//+------------------------------------------------------------------+
void ManagePositions(double buyProb, double sellProb)
{
   // Check existing positions
   bool hasBuy = PositionSelect(_Symbol) && PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY;
   bool hasSell = PositionSelect(_Symbol) && PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL;
   
   // Buy signal
   if(buyProb >= BuyThreshold && !hasBuy)
   {
      if(hasSell)
         trade.PositionClose(_Symbol);
      
      double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      double sl = price - StopLoss * _Point;
      double tp = price + TakeProfit * _Point;
      
      if(trade.Buy(LotSize, _Symbol, price, sl, tp, "Natron Buy"))
         Print("✅ Buy order opened: BuyProb=", buyProb);
   }
   
   // Sell signal
   if(sellProb >= SellThreshold && !hasSell)
   {
      if(hasBuy)
         trade.PositionClose(_Symbol);
      
      double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      double sl = price + StopLoss * _Point;
      double tp = price - TakeProfit * _Point;
      
      if(trade.Sell(LotSize, _Symbol, price, sl, tp, "Natron Sell"))
         Print("✅ Sell order opened: SellProb=", sellProb);
   }
}

//+------------------------------------------------------------------+
