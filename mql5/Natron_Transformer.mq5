//+------------------------------------------------------------------+
//|                                          Natron_Transformer.mq5 |
//|                        Natron Transformer V2 - MQL5 EA           |
//+------------------------------------------------------------------+
#property copyright "Natron Transformer V2"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>

//--- Input parameters
input string   ServerHost = "127.0.0.1";      // Python Server Host
input int      ServerPort = 8888;              // Python Server Port
input double   LotSize = 0.01;                 // Lot Size
input int      MagicNumber = 123456;           // Magic Number
input int      Slippage = 10;                  // Slippage
input double   BuyThreshold = 0.6;             // Buy Signal Threshold
input double   SellThreshold = 0.6;            // Sell Signal Threshold
input int      SequenceLength = 96;            // Required Candles

//--- Global variables
int socket_handle = INVALID_HANDLE;
CTrade trade;
datetime last_bar_time = 0;
double buy_prob = 0.0;
double sell_prob = 0.0;
string direction = "";
string regime = "";
double confidence = 0.0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   trade.SetExpertMagicNumber(MagicNumber);
   trade.SetDeviationInPoints(Slippage);
   trade.SetTypeFilling(ORDER_FILLING_FOK);
   
   // Connect to Python server
   if(!ConnectToServer())
   {
      Print("❌ Failed to connect to Python server");
      return(INIT_FAILED);
   }
   
   Print("✅ Connected to Natron Transformer server");
   Print("   Host: ", ServerHost, " Port: ", ServerPort);
   
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(socket_handle != INVALID_HANDLE)
   {
      SocketClose(socket_handle);
   }
   Print("🔌 Disconnected from server");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // Check for new bar
   datetime current_bar_time = iTime(_Symbol, PERIOD_CURRENT, 0);
   if(current_bar_time == last_bar_time)
      return;
   
   last_bar_time = current_bar_time;
   
   // Collect candles
   MqlRates rates[];
   int copied = CopyRates(_Symbol, PERIOD_CURRENT, 0, SequenceLength, rates);
   
   if(copied < SequenceLength)
   {
      Print("⚠️  Not enough candles: ", copied, " / ", SequenceLength);
      return;
   }
   
   // Send candles to server and get prediction
   if(!GetPrediction(rates))
   {
      Print("❌ Failed to get prediction");
      return;
   }
   
   // Execute trading logic
   ExecuteTradingLogic();
}

//+------------------------------------------------------------------+
//| Connect to Python server                                         |
//+------------------------------------------------------------------+
bool ConnectToServer()
{
   socket_handle = SocketCreate();
   if(socket_handle == INVALID_HANDLE)
   {
      Print("❌ Socket creation failed");
      return false;
   }
   
   if(!SocketConnect(socket_handle, ServerHost, ServerPort, 1000))
   {
      Print("❌ Socket connection failed");
      SocketClose(socket_handle);
      socket_handle = INVALID_HANDLE;
      return false;
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Get prediction from server                                        |
//+------------------------------------------------------------------+
bool GetPrediction(MqlRates &rates[])
{
   // Prepare JSON message
   string json = "{";
   json += "\"type\":\"predict\",";
   json += "\"candles\":[";
   
   for(int i = 0; i < ArraySize(rates); i++)
   {
      if(i > 0) json += ",";
      json += "{";
      json += "\"time\":\"" + TimeToString(rates[i].time) + "\",";
      json += "\"open\":" + DoubleToString(rates[i].open, 8) + ",";
      json += "\"high\":" + DoubleToString(rates[i].high, 8) + ",";
      json += "\"low\":" + DoubleToString(rates[i].low, 8) + ",";
      json += "\"close\":" + DoubleToString(rates[i].close, 8) + ",";
      json += "\"volume\":" + DoubleToString(rates[i].tick_volume, 0);
      json += "}";
   }
   
   json += "]}";
   
   // Send request
   uchar request[];
   StringToCharArray(json, request);
   
   if(SocketSend(socket_handle, request) <= 0)
   {
      Print("❌ Send failed");
      return false;
   }
   
   // Receive response
   uchar response[];
   string response_str = "";
   int timeout = 1000; // 1 second timeout
   uint start_time = GetTickCount();
   
   while(GetTickCount() - start_time < timeout)
   {
      int received = SocketRead(socket_handle, response, 4096);
      if(received > 0)
      {
         response_str += CharArrayToString(response, 0, received);
         
         // Try to parse JSON
         if(StringFind(response_str, "\"status\":\"ok\"") >= 0)
         {
            // Parse predictions
            ParsePrediction(response_str);
            return true;
         }
      }
      
      Sleep(10);
   }
   
   Print("❌ Timeout waiting for response");
   return false;
}

//+------------------------------------------------------------------+
//| Parse prediction JSON                                            |
//+------------------------------------------------------------------+
void ParsePrediction(string json)
{
   // Simple JSON parsing (for production, use proper JSON library)
   int buy_idx = StringFind(json, "\"buy_prob\":");
   int sell_idx = StringFind(json, "\"sell_prob\":");
   int dir_idx = StringFind(json, "\"direction\":\"");
   int reg_idx = StringFind(json, "\"regime\":\"");
   int conf_idx = StringFind(json, "\"confidence\":");
   
   if(buy_idx >= 0)
   {
      string buy_str = StringSubstr(json, buy_idx + 11, 10);
      buy_prob = StringToDouble(buy_str);
   }
   
   if(sell_idx >= 0)
   {
      string sell_str = StringSubstr(json, sell_idx + 12, 10);
      sell_prob = StringToDouble(sell_str);
   }
   
   if(dir_idx >= 0)
   {
      int dir_end = StringFind(json, "\"", dir_idx + 13);
      direction = StringSubstr(json, dir_idx + 13, dir_end - dir_idx - 13);
   }
   
   if(reg_idx >= 0)
   {
      int reg_end = StringFind(json, "\"", reg_idx + 10);
      regime = StringSubstr(json, reg_idx + 10, reg_end - reg_idx - 10);
   }
   
   if(conf_idx >= 0)
   {
      string conf_str = StringSubstr(json, conf_idx + 14, 10);
      confidence = StringToDouble(conf_str);
   }
   
   Print("📊 Prediction: Buy=", buy_prob, " Sell=", sell_prob, 
         " Direction=", direction, " Regime=", regime, " Confidence=", confidence);
}

//+------------------------------------------------------------------+
//| Execute trading logic                                            |
//+------------------------------------------------------------------+
void ExecuteTradingLogic()
{
   // Close existing positions if signal changes
   if(PositionSelect(_Symbol))
   {
      long position_type = PositionGetInteger(POSITION_TYPE);
      
      // Close buy position if sell signal
      if(position_type == POSITION_TYPE_BUY && sell_prob > SellThreshold)
      {
         trade.PositionClose(_Symbol);
         Print("🔴 Closed BUY position (Sell signal)");
      }
      
      // Close sell position if buy signal
      if(position_type == POSITION_TYPE_SELL && buy_prob > BuyThreshold)
      {
         trade.PositionClose(_Symbol);
         Print("🟢 Closed SELL position (Buy signal)");
      }
   }
   
   // Open new positions
   if(!PositionSelect(_Symbol))
   {
      // Buy signal
      if(buy_prob > BuyThreshold && confidence > 0.7 && direction == "Up")
      {
         double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         if(trade.Buy(LotSize, _Symbol, ask, 0, 0, "Natron Buy Signal"))
         {
            Print("🟢 Opened BUY position at ", ask);
         }
      }
      
      // Sell signal
      if(sell_prob > SellThreshold && confidence > 0.7 && direction == "Down")
      {
         double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         if(trade.Sell(LotSize, _Symbol, bid, 0, 0, "Natron Sell Signal"))
         {
            Print("🔴 Opened SELL position at ", bid);
         }
      }
   }
}

//+------------------------------------------------------------------+
