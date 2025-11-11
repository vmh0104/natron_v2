//+------------------------------------------------------------------+
//|                                          NatronAI_EA.mq5         |
//|                        Natron Transformer V2 - MQL5 EA           |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "Natron AI Trading System"
#property link      ""
#property version   "2.00"
#property description "Multi-task Transformer model for financial trading"

//--- Input parameters
input string   ServerHost = "localhost";      // Python server host
input int      ServerPort = 8888;             // Python server port
input int      SequenceLength = 96;           // Number of candles for prediction
input double   LotSize = 0.01;                // Lot size
input int      MagicNumber = 123456;          // Magic number
input int      Slippage = 3;                  // Slippage in points
input bool     UseStopLoss = true;            // Use stop loss
input double   StopLoss = 50;                 // Stop loss in points
input bool     UseTakeProfit = true;          // Use take profit
input double   TakeProfit = 100;               // Take profit in points

//--- Global variables
int socketHandle = INVALID_HANDLE;
datetime lastBarTime = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("Natron AI EA initialized");
   
   // Connect to Python socket server
   socketHandle = SocketCreate();
   if(socketHandle == INVALID_HANDLE)
   {
      Print("Failed to create socket");
      return(INIT_FAILED);
   }
   
   if(!SocketConnect(socketHandle, ServerHost, ServerPort, 1000))
   {
      Print("Failed to connect to server: ", ServerHost, ":", ServerPort);
      SocketClose(socketHandle);
      return(INIT_FAILED);
   }
   
   Print("Connected to Natron AI server");
   return(INIT_SUCCEEDED);
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
   Print("Natron AI EA deinitialized");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // Check for new bar
   datetime currentBarTime = iTime(_Symbol, PERIOD_CURRENT, 0);
   if(currentBarTime == lastBarTime)
   {
      return; // No new bar
   }
   lastBarTime = currentBarTime;
   
   // Get prediction from AI model
   double buyProb = 0.0;
   double sellProb = 0.0;
   string regime = "";
   double confidence = 0.0;
   
   if(!GetPrediction(buyProb, sellProb, regime, confidence))
   {
      Print("Failed to get prediction");
      return;
   }
   
   Print("Prediction - Buy: ", buyProb, " Sell: ", sellProb, 
         " Regime: ", regime, " Confidence: ", confidence);
   
   // Trading logic
   if(buyProb > 0.7 && confidence > 0.6)
   {
      OpenBuyOrder();
   }
   else if(sellProb > 0.7 && confidence > 0.6)
   {
      OpenSellOrder();
   }
   
   // Manage existing positions
   ManagePositions();
}

//+------------------------------------------------------------------+
//| Get prediction from Python server                                |
//+------------------------------------------------------------------+
bool GetPrediction(double &buyProb, double &sellProb, string &regime, double &confidence)
{
   // Collect last SequenceLength candles
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   
   if(CopyRates(_Symbol, PERIOD_CURRENT, 0, SequenceLength, rates) < SequenceLength)
   {
      Print("Failed to copy rates");
      return false;
   }
   
   // Prepare JSON message
   string json = "{";
   json += "\"type\":\"predict\",";
   json += "\"candles\":[";
   
   for(int i = SequenceLength - 1; i >= 0; i--)
   {
      if(i < SequenceLength - 1) json += ",";
      json += "{";
      json += "\"time\":" + IntegerToString(rates[i].time) + ",";
      json += "\"open\":" + DoubleToString(rates[i].open, _Digits) + ",";
      json += "\"high\":" + DoubleToString(rates[i].high, _Digits) + ",";
      json += "\"low\":" + DoubleToString(rates[i].low, _Digits) + ",";
      json += "\"close\":" + DoubleToString(rates[i].close, _Digits) + ",";
      json += "\"volume\":" + IntegerToString(rates[i].tick_volume);
      json += "}";
   }
   
   json += "]}";
   
   // Send request
   char request[];
   StringToCharArray(json, request, 0, StringLen(json));
   
   if(SocketSend(socketHandle, request) <= 0)
   {
      Print("Failed to send request");
      return false;
   }
   
   // Receive response
   char response[];
   string responseStr = "";
   uint timeout = 1000; // 1 second timeout
   uint startTime = GetTickCount();
   
   while(GetTickCount() - startTime < timeout)
   {
      uint received = SocketRead(socketHandle, response, 0, 10000);
      if(received > 0)
      {
         responseStr += CharArrayToString(response, 0, received);
         if(StringFind(responseStr, "}") >= 0)
         {
            break; // Complete JSON received
         }
      }
      Sleep(10);
   }
   
   if(StringLen(responseStr) == 0)
   {
      Print("No response from server");
      return false;
   }
   
   // Parse JSON response (simplified - use proper JSON parser in production)
   int statusPos = StringFind(responseStr, "\"status\":\"success\"");
   if(statusPos < 0)
   {
      Print("Error in response: ", responseStr);
      return false;
   }
   
   // Extract prediction values (simplified parsing)
   int buyPos = StringFind(responseStr, "\"buy_prob\":");
   int sellPos = StringFind(responseStr, "\"sell_prob\":");
   int regimePos = StringFind(responseStr, "\"regime\":\"");
   int confPos = StringFind(responseStr, "\"confidence\":");
   
   if(buyPos >= 0)
   {
      string buyStr = StringSubstr(responseStr, buyPos + 10, 10);
      buyProb = StringToDouble(buyStr);
   }
   
   if(sellPos >= 0)
   {
      string sellStr = StringSubstr(responseStr, sellPos + 11, 10);
      sellProb = StringToDouble(sellStr);
   }
   
   if(regimePos >= 0)
   {
      int regimeEnd = StringFind(responseStr, "\"", regimePos + 9);
      regime = StringSubstr(responseStr, regimePos + 9, regimeEnd - regimePos - 9);
   }
   
   if(confPos >= 0)
   {
      string confStr = StringSubstr(responseStr, confPos + 13, 10);
      confidence = StringToDouble(confStr);
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Open buy order                                                   |
//+------------------------------------------------------------------+
void OpenBuyOrder()
{
   // Check if position already exists
   if(PositionSelect(_Symbol))
   {
      if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
      {
         return; // Already long
      }
   }
   
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double sl = UseStopLoss ? ask - StopLoss * _Point : 0;
   double tp = UseTakeProfit ? ask + TakeProfit * _Point : 0;
   
   MqlTradeRequest request = {};
   MqlTradeResult result = {};
   
   request.action = TRADE_ACTION_DEAL;
   request.symbol = _Symbol;
   request.volume = LotSize;
   request.type = ORDER_TYPE_BUY;
   request.price = ask;
   request.sl = sl;
   request.tp = tp;
   request.deviation = Slippage;
   request.magic = MagicNumber;
   request.comment = "Natron AI Buy";
   
   if(!OrderSend(request, result))
   {
      Print("Buy order failed: ", result.retcode);
   }
   else
   {
      Print("Buy order opened: ", result.order);
   }
}

//+------------------------------------------------------------------+
//| Open sell order                                                  |
//+------------------------------------------------------------------+
void OpenSellOrder()
{
   // Check if position already exists
   if(PositionSelect(_Symbol))
   {
      if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL)
      {
         return; // Already short
      }
   }
   
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double sl = UseStopLoss ? bid + StopLoss * _Point : 0;
   double tp = UseTakeProfit ? bid - TakeProfit * _Point : 0;
   
   MqlTradeRequest request = {};
   MqlTradeResult result = {};
   
   request.action = TRADE_ACTION_DEAL;
   request.symbol = _Symbol;
   request.volume = LotSize;
   request.type = ORDER_TYPE_SELL;
   request.price = bid;
   request.sl = sl;
   request.tp = tp;
   request.deviation = Slippage;
   request.magic = MagicNumber;
   request.comment = "Natron AI Sell";
   
   if(!OrderSend(request, result))
   {
      Print("Sell order failed: ", result.retcode);
   }
   else
   {
      Print("Sell order opened: ", result.order);
   }
}

//+------------------------------------------------------------------+
//| Manage existing positions                                         |
//+------------------------------------------------------------------+
void ManagePositions()
{
   // Add position management logic here
   // e.g., trailing stop, exit conditions, etc.
}

//+------------------------------------------------------------------+
