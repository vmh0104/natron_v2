//+------------------------------------------------------------------+
//|                                                    NatronAI.mq5  |
//|                                    Natron Transformer EA v2.0    |
//|                        AI-Powered Multi-Task Trading System      |
//+------------------------------------------------------------------+
#property copyright "Natron AI Team"
#property link      "https://github.com/natron-ai"
#property version   "2.00"
#property strict

//--- Input parameters
input string   ServerHost = "localhost";        // Python Server Host
input int      ServerPort = 9999;               // Python Server Port
input int      MagicNumber = 20241111;          // Magic Number
input double   LotSize = 0.1;                   // Lot Size
input double   BuyThreshold = 0.6;              // Buy Signal Threshold
input double   SellThreshold = 0.6;             // Sell Signal Threshold
input double   StopLoss = 100;                  // Stop Loss (points)
input double   TakeProfit = 200;                // Take Profit (points)
input int      Slippage = 10;                   // Maximum Slippage
input bool     UseTrailingStop = true;          // Use Trailing Stop
input double   TrailingStop = 50;               // Trailing Stop (points)
input bool     EnableRegimeFilter = true;       // Enable Regime Filter
input string   AllowedRegimes = "BULL_STRONG,BULL_WEAK,RANGE"; // Allowed Regimes
input int      SequenceLength = 96;             // Number of candles to send
input int      UpdateInterval = 60;             // Update interval (seconds)

//--- Global variables
int socketHandle = INVALID_HANDLE;
datetime lastUpdateTime = 0;
string symbolName;
ENUM_TIMEFRAMES timeframe = PERIOD_M15;

//--- Structure for AI prediction
struct AIPrediction
{
   double buy_prob;
   double sell_prob;
   double direction_up;
   double direction_down;
   string regime;
   double regime_confidence;
   double confidence;
   double latency_ms;
};

//+------------------------------------------------------------------+
//| Expert initialization function                                     |
//+------------------------------------------------------------------+
int OnInit()
{
   symbolName = Symbol();
   
   Print("╔═══════════════════════════════════════════════════════════╗");
   Print("║          NATRON TRANSFORMER EA v2.0                       ║");
   Print("║          AI-Powered Multi-Task Trading System             ║");
   Print("╚═══════════════════════════════════════════════════════════╝");
   
   Print("Symbol: ", symbolName);
   Print("Timeframe: ", EnumToString(timeframe));
   Print("Server: ", ServerHost, ":", ServerPort);
   Print("Magic Number: ", MagicNumber);
   Print("Lot Size: ", LotSize);
   
   //--- Connect to Python server
   if(!ConnectToServer())
   {
      Print("❌ Failed to connect to Python server");
      Print("   Make sure the socket server is running:");
      Print("   python src/bridge/socket_server.py");
      return(INIT_FAILED);
   }
   
   Print("✅ Connected to Natron AI Server");
   Print("Ready to trade!");
   
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                   |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   DisconnectFromServer();
   Print("Natron EA stopped. Reason: ", reason);
}

//+------------------------------------------------------------------+
//| Expert tick function                                               |
//+------------------------------------------------------------------+
void OnTick()
{
   //--- Check if enough time has passed since last update
   datetime currentTime = TimeCurrent();
   if(currentTime - lastUpdateTime < UpdateInterval)
      return;
   
   lastUpdateTime = currentTime;
   
   //--- Get AI prediction
   AIPrediction prediction;
   if(!GetAIPrediction(prediction))
   {
      Print("⚠️ Failed to get AI prediction");
      return;
   }
   
   //--- Display prediction
   DisplayPrediction(prediction);
   
   //--- Check regime filter
   if(EnableRegimeFilter && !IsRegimeAllowed(prediction.regime))
   {
      Print("⏸️ Regime filter: ", prediction.regime, " not allowed");
      return;
   }
   
   //--- Execute trading logic
   ExecuteTrading(prediction);
   
   //--- Update trailing stop
   if(UseTrailingStop)
      UpdateTrailingStop();
}

//+------------------------------------------------------------------+
//| Connect to Python server                                           |
//+------------------------------------------------------------------+
bool ConnectToServer()
{
   Print("Connecting to ", ServerHost, ":", ServerPort, "...");
   
   socketHandle = SocketCreate();
   if(socketHandle == INVALID_HANDLE)
   {
      Print("Failed to create socket: ", GetLastError());
      return false;
   }
   
   if(!SocketConnect(socketHandle, ServerHost, ServerPort, 5000))
   {
      Print("Failed to connect: ", GetLastError());
      SocketClose(socketHandle);
      socketHandle = INVALID_HANDLE;
      return false;
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Disconnect from server                                             |
//+------------------------------------------------------------------+
void DisconnectFromServer()
{
   if(socketHandle != INVALID_HANDLE)
   {
      SocketClose(socketHandle);
      socketHandle = INVALID_HANDLE;
      Print("Disconnected from server");
   }
}

//+------------------------------------------------------------------+
//| Get AI prediction from server                                      |
//+------------------------------------------------------------------+
bool GetAIPrediction(AIPrediction &prediction)
{
   //--- Reconnect if disconnected
   if(socketHandle == INVALID_HANDLE)
   {
      if(!ConnectToServer())
         return false;
   }
   
   //--- Collect OHLCV data
   string jsonData = CollectOHLCVData(SequenceLength);
   if(jsonData == "")
      return false;
   
   //--- Send request to server
   string request = "{\"action\":\"predict\",\"data\":" + jsonData + "}";
   
   int requestLen = StringLen(request);
   if(SocketSend(socketHandle, request, requestLen) != requestLen)
   {
      Print("Failed to send request: ", GetLastError());
      DisconnectFromServer();
      return false;
   }
   
   //--- Receive response
   char buffer[];
   ArrayResize(buffer, 4096);
   
   int received = SocketReceive(socketHandle, buffer, 4096, 5000);
   if(received <= 0)
   {
      Print("Failed to receive response: ", GetLastError());
      DisconnectFromServer();
      return false;
   }
   
   string response = CharArrayToString(buffer, 0, received);
   
   //--- Parse JSON response
   return ParsePrediction(response, prediction);
}

//+------------------------------------------------------------------+
//| Collect OHLCV data for the last N candles                         |
//+------------------------------------------------------------------+
string CollectOHLCVData(int count)
{
   string json = "[";
   
   for(int i = count - 1; i >= 0; i--)
   {
      datetime time = iTime(symbolName, timeframe, i);
      double open = iOpen(symbolName, timeframe, i);
      double high = iHigh(symbolName, timeframe, i);
      double low = iLow(symbolName, timeframe, i);
      double close = iClose(symbolName, timeframe, i);
      long volume = iVolume(symbolName, timeframe, i);
      
      if(time == 0)
      {
         Print("Failed to get candle data at index ", i);
         return "";
      }
      
      string candle = StringFormat(
         "{\"time\":\"%s\",\"open\":%.5f,\"high\":%.5f,\"low\":%.5f,\"close\":%.5f,\"volume\":%d}",
         TimeToString(time, TIME_DATE|TIME_MINUTES),
         open, high, low, close, volume
      );
      
      json += candle;
      if(i > 0)
         json += ",";
   }
   
   json += "]";
   return json;
}

//+------------------------------------------------------------------+
//| Parse JSON prediction response                                     |
//+------------------------------------------------------------------+
bool ParsePrediction(string json, AIPrediction &prediction)
{
   //--- Simple JSON parsing (for production, use a proper JSON library)
   prediction.buy_prob = ExtractDouble(json, "buy_prob");
   prediction.sell_prob = ExtractDouble(json, "sell_prob");
   prediction.direction_up = ExtractDouble(json, "direction_up");
   prediction.direction_down = ExtractDouble(json, "direction_down");
   prediction.regime = ExtractString(json, "regime");
   prediction.regime_confidence = ExtractDouble(json, "regime_confidence");
   prediction.confidence = ExtractDouble(json, "confidence");
   prediction.latency_ms = ExtractDouble(json, "latency_ms");
   
   return true;
}

//+------------------------------------------------------------------+
//| Extract double value from JSON string                              |
//+------------------------------------------------------------------+
double ExtractDouble(string json, string key)
{
   string searchKey = "\"" + key + "\":";
   int pos = StringFind(json, searchKey);
   if(pos < 0)
      return 0.0;
   
   pos += StringLen(searchKey);
   
   string valueStr = "";
   for(int i = pos; i < StringLen(json); i++)
   {
      string ch = StringSubstr(json, i, 1);
      if(ch == "," || ch == "}" || ch == " ")
         break;
      valueStr += ch;
   }
   
   return StringToDouble(valueStr);
}

//+------------------------------------------------------------------+
//| Extract string value from JSON                                     |
//+------------------------------------------------------------------+
string ExtractString(string json, string key)
{
   string searchKey = "\"" + key + "\":\"";
   int pos = StringFind(json, searchKey);
   if(pos < 0)
      return "";
   
   pos += StringLen(searchKey);
   
   string valueStr = "";
   for(int i = pos; i < StringLen(json); i++)
   {
      string ch = StringSubstr(json, i, 1);
      if(ch == "\"")
         break;
      valueStr += ch;
   }
   
   return valueStr;
}

//+------------------------------------------------------------------+
//| Display prediction on chart                                        |
//+------------------------------------------------------------------+
void DisplayPrediction(AIPrediction &prediction)
{
   string text = StringFormat(
      "🧠 NATRON AI\n" +
      "═══════════════\n" +
      "Buy:  %.2f%%\n" +
      "Sell: %.2f%%\n" +
      "Dir↑: %.2f%%\n" +
      "Dir↓: %.2f%%\n" +
      "───────────────\n" +
      "Regime: %s\n" +
      "Confidence: %.2f%%\n" +
      "Latency: %.1fms",
      prediction.buy_prob * 100,
      prediction.sell_prob * 100,
      prediction.direction_up * 100,
      prediction.direction_down * 100,
      prediction.regime,
      prediction.confidence * 100,
      prediction.latency_ms
   );
   
   Comment(text);
}

//+------------------------------------------------------------------+
//| Check if regime is allowed for trading                             |
//+------------------------------------------------------------------+
bool IsRegimeAllowed(string regime)
{
   if(StringFind(AllowedRegimes, regime) >= 0)
      return true;
   return false;
}

//+------------------------------------------------------------------+
//| Execute trading based on prediction                                |
//+------------------------------------------------------------------+
void ExecuteTrading(AIPrediction &prediction)
{
   //--- Count current positions
   int buyPositions = CountPositions(POSITION_TYPE_BUY);
   int sellPositions = CountPositions(POSITION_TYPE_SELL);
   
   //--- Get current price
   double ask = SymbolInfoDouble(symbolName, SYMBOL_ASK);
   double bid = SymbolInfoDouble(symbolName, SYMBOL_BID);
   
   //--- Buy signal
   if(prediction.buy_prob > BuyThreshold && buyPositions == 0)
   {
      if(sellPositions > 0)
      {
         Print("📊 Closing SELL positions before opening BUY");
         CloseAllPositions(POSITION_TYPE_SELL);
      }
      
      Print("🟢 BUY Signal - Probability: ", prediction.buy_prob * 100, "%");
      OpenPosition(ORDER_TYPE_BUY, ask);
   }
   //--- Sell signal
   else if(prediction.sell_prob > SellThreshold && sellPositions == 0)
   {
      if(buyPositions > 0)
      {
         Print("📊 Closing BUY positions before opening SELL");
         CloseAllPositions(POSITION_TYPE_BUY);
      }
      
      Print("🔴 SELL Signal - Probability: ", prediction.sell_prob * 100, "%");
      OpenPosition(ORDER_TYPE_SELL, bid);
   }
   //--- Close positions if confidence is low
   else if(prediction.confidence < 0.4)
   {
      if(buyPositions > 0 || sellPositions > 0)
      {
         Print("⚠️ Low confidence (", prediction.confidence * 100, "%) - Closing all positions");
         CloseAllPositions(POSITION_TYPE_BUY);
         CloseAllPositions(POSITION_TYPE_SELL);
      }
   }
}

//+------------------------------------------------------------------+
//| Open new position                                                  |
//+------------------------------------------------------------------+
bool OpenPosition(ENUM_ORDER_TYPE orderType, double price)
{
   double sl = 0, tp = 0;
   
   //--- Calculate SL/TP
   double point = SymbolInfoDouble(symbolName, SYMBOL_POINT);
   
   if(orderType == ORDER_TYPE_BUY)
   {
      if(StopLoss > 0)
         sl = price - StopLoss * point;
      if(TakeProfit > 0)
         tp = price + TakeProfit * point;
   }
   else
   {
      if(StopLoss > 0)
         sl = price + StopLoss * point;
      if(TakeProfit > 0)
         tp = price - TakeProfit * point;
   }
   
   //--- Prepare request
   MqlTradeRequest request;
   MqlTradeResult result;
   ZeroMemory(request);
   ZeroMemory(result);
   
   request.action = TRADE_ACTION_DEAL;
   request.symbol = symbolName;
   request.volume = LotSize;
   request.type = orderType;
   request.price = price;
   request.sl = sl;
   request.tp = tp;
   request.deviation = Slippage;
   request.magic = MagicNumber;
   request.comment = "Natron AI";
   
   //--- Send order
   if(!OrderSend(request, result))
   {
      Print("❌ OrderSend failed: ", GetLastError());
      Print("   RetCode: ", result.retcode);
      return false;
   }
   
   if(result.retcode == TRADE_RETCODE_DONE)
   {
      Print("✅ Order executed successfully");
      Print("   Ticket: ", result.order);
      Print("   Price: ", result.price);
      Print("   SL: ", sl);
      Print("   TP: ", tp);
      return true;
   }
   else
   {
      Print("❌ Order failed - RetCode: ", result.retcode);
      return false;
   }
}

//+------------------------------------------------------------------+
//| Count positions by type                                            |
//+------------------------------------------------------------------+
int CountPositions(ENUM_POSITION_TYPE posType)
{
   int count = 0;
   
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket > 0)
      {
         if(PositionGetString(POSITION_SYMBOL) == symbolName &&
            PositionGetInteger(POSITION_MAGIC) == MagicNumber &&
            PositionGetInteger(POSITION_TYPE) == posType)
         {
            count++;
         }
      }
   }
   
   return count;
}

//+------------------------------------------------------------------+
//| Close all positions of specified type                             |
//+------------------------------------------------------------------+
void CloseAllPositions(ENUM_POSITION_TYPE posType)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket > 0)
      {
         if(PositionGetString(POSITION_SYMBOL) == symbolName &&
            PositionGetInteger(POSITION_MAGIC) == MagicNumber &&
            PositionGetInteger(POSITION_TYPE) == posType)
         {
            MqlTradeRequest request;
            MqlTradeResult result;
            ZeroMemory(request);
            ZeroMemory(result);
            
            request.action = TRADE_ACTION_DEAL;
            request.position = ticket;
            request.symbol = symbolName;
            request.volume = PositionGetDouble(POSITION_VOLUME);
            request.type = (posType == POSITION_TYPE_BUY) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
            request.price = (posType == POSITION_TYPE_BUY) ? 
                           SymbolInfoDouble(symbolName, SYMBOL_BID) : 
                           SymbolInfoDouble(symbolName, SYMBOL_ASK);
            request.deviation = Slippage;
            request.magic = MagicNumber;
            
            OrderSend(request, result);
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Update trailing stop for all positions                            |
//+------------------------------------------------------------------+
void UpdateTrailingStop()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket > 0)
      {
         if(PositionGetString(POSITION_SYMBOL) == symbolName &&
            PositionGetInteger(POSITION_MAGIC) == MagicNumber)
         {
            double point = SymbolInfoDouble(symbolName, SYMBOL_POINT);
            double currentSL = PositionGetDouble(POSITION_SL);
            double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
            ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
            
            double newSL = 0;
            
            if(posType == POSITION_TYPE_BUY)
            {
               double bid = SymbolInfoDouble(symbolName, SYMBOL_BID);
               newSL = bid - TrailingStop * point;
               
               if(newSL > currentSL && newSL < bid)
               {
                  ModifyPosition(ticket, newSL, PositionGetDouble(POSITION_TP));
               }
            }
            else
            {
               double ask = SymbolInfoDouble(symbolName, SYMBOL_ASK);
               newSL = ask + TrailingStop * point;
               
               if((currentSL == 0 || newSL < currentSL) && newSL > ask)
               {
                  ModifyPosition(ticket, newSL, PositionGetDouble(POSITION_TP));
               }
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Modify position SL/TP                                              |
//+------------------------------------------------------------------+
bool ModifyPosition(ulong ticket, double sl, double tp)
{
   MqlTradeRequest request;
   MqlTradeResult result;
   ZeroMemory(request);
   ZeroMemory(result);
   
   request.action = TRADE_ACTION_SLTP;
   request.position = ticket;
   request.sl = sl;
   request.tp = tp;
   
   if(OrderSend(request, result))
   {
      if(result.retcode == TRADE_RETCODE_DONE)
         return true;
   }
   
   return false;
}
//+------------------------------------------------------------------+
