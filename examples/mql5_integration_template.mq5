//+------------------------------------------------------------------+
//|                                      Natron AI Trading EA         |
//|                                      MQL5 Integration Template    |
//+------------------------------------------------------------------+
#property copyright "Natron Transformer"
#property version   "2.00"
#property strict

// Socket library for Python communication
#include <Socket.mqh>

//--- Input parameters
input string   PythonHost = "localhost";     // Python server host
input int      PythonPort = 5000;            // Python server port
input int      SequenceLength = 96;          // Number of candles to send
input double   BuyThreshold = 0.70;          // Buy signal threshold
input double   SellThreshold = 0.70;         // Sell signal threshold
input double   ConfidenceMin = 0.75;         // Minimum confidence
input double   LotSize = 0.1;                // Trading lot size
input int      MagicNumber = 123456;         // EA magic number

//--- Global variables
CSocket socket;
datetime lastBarTime = 0;
bool isConnected = false;

//+------------------------------------------------------------------+
//| Expert initialization function                                     |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("Natron AI EA initialized");
   
   // Test connection to Python server
   if(!TestConnection())
   {
      Alert("Failed to connect to Python server at ", PythonHost, ":", PythonPort);
      return(INIT_FAILED);
   }
   
   Print("Connected to Natron API successfully");
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                   |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   socket.Close();
   Print("Natron AI EA deinitialized");
}

//+------------------------------------------------------------------+
//| Expert tick function                                              |
//+------------------------------------------------------------------+
void OnTick()
{
   // Check for new bar
   datetime currentBarTime = iTime(_Symbol, _Period, 0);
   if(currentBarTime == lastBarTime)
      return;
   
   lastBarTime = currentBarTime;
   
   // Get OHLCV data
   MqlRates rates[];
   if(CopyRates(_Symbol, _Period, 0, SequenceLength, rates) != SequenceLength)
   {
      Print("Failed to copy rates");
      return;
   }
   
   // Send to Python API and get prediction
   string jsonData = BuildJSONRequest(rates);
   string response = SendToPythonAPI(jsonData);
   
   if(response == "")
   {
      Print("No response from Python API");
      return;
   }
   
   // Parse prediction
   PredictionResult prediction;
   if(!ParsePrediction(response, prediction))
   {
      Print("Failed to parse prediction");
      return;
   }
   
   // Display prediction
   Comment(StringFormat(
      "Natron AI Prediction\n" +
      "====================\n" +
      "Buy Prob:  %.2f\n" +
      "Sell Prob: %.2f\n" +
      "Direction: %s\n" +
      "Regime:    %s\n" +
      "Confidence: %.2f",
      prediction.buy_prob,
      prediction.sell_prob,
      prediction.direction,
      prediction.regime,
      prediction.confidence
   ));
   
   // Execute trading logic
   ExecuteTradingLogic(prediction);
}

//+------------------------------------------------------------------+
//| Test connection to Python server                                  |
//+------------------------------------------------------------------+
bool TestConnection()
{
   string url = "http://" + PythonHost + ":" + IntegerToString(PythonPort) + "/health";
   
   // Simple HTTP GET request
   char data[];
   string headers;
   char result[];
   
   int res = WebRequest("GET", url, headers, 5000, data, result, headers);
   
   if(res == 200)
   {
      isConnected = true;
      return true;
   }
   
   return false;
}

//+------------------------------------------------------------------+
//| Build JSON request from OHLCV data                               |
//+------------------------------------------------------------------+
string BuildJSONRequest(MqlRates &rates[])
{
   string json = "{\"candles\":[";
   
   for(int i = 0; i < ArraySize(rates); i++)
   {
      if(i > 0)
         json += ",";
      
      json += StringFormat(
         "{\"time\":\"%s\",\"open\":%.5f,\"high\":%.5f,\"low\":%.5f,\"close\":%.5f,\"volume\":%d}",
         TimeToString(rates[i].time, TIME_DATE|TIME_MINUTES),
         rates[i].open,
         rates[i].high,
         rates[i].low,
         rates[i].close,
         rates[i].tick_volume
      );
   }
   
   json += "]}";
   return json;
}

//+------------------------------------------------------------------+
//| Send request to Python API                                        |
//+------------------------------------------------------------------+
string SendToPythonAPI(string jsonData)
{
   string url = "http://" + PythonHost + ":" + IntegerToString(PythonPort) + "/predict";
   
   char data[];
   StringToCharArray(jsonData, data, 0, StringLen(jsonData));
   
   char result[];
   string headers = "Content-Type: application/json\r\n";
   
   int res = WebRequest("POST", url, headers, 5000, data, result, headers);
   
   if(res == 200)
   {
      return CharArrayToString(result);
   }
   
   return "";
}

//+------------------------------------------------------------------+
//| Prediction result structure                                       |
//+------------------------------------------------------------------+
struct PredictionResult
{
   double buy_prob;
   double sell_prob;
   string direction;
   string regime;
   double confidence;
};

//+------------------------------------------------------------------+
//| Parse JSON prediction response                                     |
//+------------------------------------------------------------------+
bool ParsePrediction(string json, PredictionResult &prediction)
{
   // Simple JSON parsing (use proper JSON library for production)
   // This is a simplified example
   
   int pos;
   
   // Extract buy_prob
   pos = StringFind(json, "\"buy_prob\":");
   if(pos >= 0)
   {
      string buyStr = StringSubstr(json, pos + 11, 10);
      prediction.buy_prob = StringToDouble(buyStr);
   }
   
   // Extract sell_prob
   pos = StringFind(json, "\"sell_prob\":");
   if(pos >= 0)
   {
      string sellStr = StringSubstr(json, pos + 12, 10);
      prediction.sell_prob = StringToDouble(sellStr);
   }
   
   // Extract direction
   pos = StringFind(json, "\"direction\":\"");
   if(pos >= 0)
   {
      int endPos = StringFind(json, "\"", pos + 13);
      prediction.direction = StringSubstr(json, pos + 13, endPos - pos - 13);
   }
   
   // Extract regime
   pos = StringFind(json, "\"regime\":\"");
   if(pos >= 0)
   {
      int endPos = StringFind(json, "\"", pos + 10);
      prediction.regime = StringSubstr(json, pos + 10, endPos - pos - 10);
   }
   
   // Extract confidence
   pos = StringFind(json, "\"confidence\":");
   if(pos >= 0)
   {
      string confStr = StringSubstr(json, pos + 13, 10);
      prediction.confidence = StringToDouble(confStr);
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Execute trading logic based on prediction                         |
//+------------------------------------------------------------------+
void ExecuteTradingLogic(PredictionResult &prediction)
{
   // Check if confidence meets minimum threshold
   if(prediction.confidence < ConfidenceMin)
   {
      Print("Confidence too low: ", prediction.confidence);
      return;
   }
   
   // Get current position
   int positions = PositionsTotal();
   
   // BUY SIGNAL
   if(prediction.buy_prob > BuyThreshold && prediction.direction == "UP")
   {
      // Close any sell positions
      CloseAllPositions(ORDER_TYPE_SELL);
      
      // Open buy if no position
      if(positions == 0)
      {
         double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         double sl = ask - 100 * _Point;  // 100 pips SL (adjust as needed)
         double tp = ask + 200 * _Point;  // 200 pips TP
         
         if(OrderSend(_Symbol, ORDER_TYPE_BUY, LotSize, ask, 3, sl, tp, 
                     "Natron AI Buy", MagicNumber) > 0)
         {
            Print("BUY order opened | Prob: ", prediction.buy_prob, 
                  " | Regime: ", prediction.regime);
         }
      }
   }
   
   // SELL SIGNAL
   else if(prediction.sell_prob > SellThreshold && prediction.direction == "DOWN")
   {
      // Close any buy positions
      CloseAllPositions(ORDER_TYPE_BUY);
      
      // Open sell if no position
      if(positions == 0)
      {
         double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         double sl = bid + 100 * _Point;
         double tp = bid - 200 * _Point;
         
         if(OrderSend(_Symbol, ORDER_TYPE_SELL, LotSize, bid, 3, sl, tp, 
                     "Natron AI Sell", MagicNumber) > 0)
         {
            Print("SELL order opened | Prob: ", prediction.sell_prob, 
                  " | Regime: ", prediction.regime);
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Close all positions of specific type                             |
//+------------------------------------------------------------------+
void CloseAllPositions(ENUM_ORDER_TYPE type)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(PositionSelectByTicket(ticket))
      {
         if(PositionGetInteger(POSITION_MAGIC) == MagicNumber &&
            PositionGetInteger(POSITION_TYPE) == type)
         {
            MqlTradeRequest request;
            MqlTradeResult result;
            
            ZeroMemory(request);
            request.action = TRADE_ACTION_DEAL;
            request.position = ticket;
            request.symbol = _Symbol;
            request.volume = PositionGetDouble(POSITION_VOLUME);
            request.type = (type == ORDER_TYPE_BUY) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
            request.price = (type == ORDER_TYPE_BUY) ? 
                           SymbolInfoDouble(_Symbol, SYMBOL_BID) : 
                           SymbolInfoDouble(_Symbol, SYMBOL_ASK);
            request.magic = MagicNumber;
            
            OrderSend(request, result);
         }
      }
   }
}

//+------------------------------------------------------------------+
