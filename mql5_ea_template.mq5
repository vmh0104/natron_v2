"""
Sample MQL5 Expert Advisor (EA) Template
Connects to Natron Socket Server for real-time AI predictions
"""
//+------------------------------------------------------------------+
//|                                              Natron_EA_Template.mq5 |
//|                                  Natron Transformer Integration |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "Natron AI"
#property version   "1.00"
#property strict

//--- Input parameters
input string   ServerHost = "localhost";      // Socket server host
input int      ServerPort = 8888;            // Socket server port
input int      SequenceLength = 96;          // Number of candles for prediction
input double   BuyThreshold = 0.6;          // Minimum buy probability
input double   SellThreshold = 0.6;          // Minimum sell probability
input double   LotSize = 0.01;               // Trading lot size
input int      MagicNumber = 123456;         // Magic number for orders
input bool     UseRegimeFilter = true;      // Filter trades by regime

//--- Global variables
int socket_handle = INVALID_HANDLE;
datetime last_prediction_time = 0;
int prediction_interval = 60; // Predict every 60 seconds

//+------------------------------------------------------------------+
//| Expert initialization function                                     |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("🧠 Natron EA Initializing...");
   
   // Connect to socket server
   if(!ConnectToServer())
   {
      Print("❌ Failed to connect to Natron server");
      return(INIT_FAILED);
   }
   
   Print("✅ Connected to Natron server");
   Print("   Host: ", ServerHost);
   Print("   Port: ", ServerPort);
   
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
   Print("🔌 Disconnected from Natron server");
}

//+------------------------------------------------------------------+
//| Expert tick function                                               |
//+------------------------------------------------------------------+
void OnTick()
{
   // Check if it's time for a new prediction
   datetime current_time = TimeCurrent();
   if(current_time - last_prediction_time < prediction_interval)
      return;
   
   // Get prediction from Natron AI
   double buy_prob = 0.0;
   double sell_prob = 0.0;
   string regime = "";
   double confidence = 0.0;
   
   if(GetNatronPrediction(buy_prob, sell_prob, regime, confidence))
   {
      Print("📊 Natron Prediction:");
      Print("   Buy Prob: ", buy_prob);
      Print("   Sell Prob: ", sell_prob);
      Print("   Regime: ", regime);
      Print("   Confidence: ", confidence);
      
      // Execute trading logic
      ExecuteTradingLogic(buy_prob, sell_prob, regime, confidence);
      
      last_prediction_time = current_time;
   }
   else
   {
      Print("⚠️ Failed to get prediction from Natron server");
   }
}

//+------------------------------------------------------------------+
//| Connect to socket server                                          |
//+------------------------------------------------------------------+
bool ConnectToServer()
{
   socket_handle = SocketCreate();
   if(socket_handle == INVALID_HANDLE)
   {
      Print("❌ Failed to create socket");
      return false;
   }
   
   if(!SocketConnect(socket_handle, ServerHost, ServerPort, 5000))
   {
      Print("❌ Failed to connect to ", ServerHost, ":", ServerPort);
      SocketClose(socket_handle);
      socket_handle = INVALID_HANDLE;
      return false;
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Get prediction from Natron server                                 |
//+------------------------------------------------------------------+
bool GetNatronPrediction(double &buy_prob, double &sell_prob, string &regime, double &confidence)
{
   if(socket_handle == INVALID_HANDLE)
   {
      if(!ConnectToServer())
         return false;
   }
   
   // Prepare candles data
   string json_data = PrepareCandlesJSON();
   
   // Send request
   string request = "{\"type\":\"predict\",\"candles\":" + json_data + "}";
   
   if(!SocketSend(socket_handle, request))
   {
      Print("❌ Failed to send request");
      SocketClose(socket_handle);
      socket_handle = INVALID_HANDLE;
      return false;
   }
   
   // Receive response
   uchar response[];
   int timeout = 1000; // 1 second timeout
   int received = SocketRead(socket_handle, response, timeout);
   
   if(received <= 0)
   {
      Print("❌ No response from server");
      return false;
   }
   
   // Parse JSON response
   string response_str = CharArrayToString(response);
   return ParsePredictionResponse(response_str, buy_prob, sell_prob, regime, confidence);
}

//+------------------------------------------------------------------+
//| Prepare candles JSON for Natron server                           |
//+------------------------------------------------------------------+
string PrepareCandlesJSON()
{
   string json = "[";
   
   for(int i = SequenceLength - 1; i >= 0; i--)
   {
      if(i < SequenceLength - 1)
         json += ",";
      
      datetime time = iTime(_Symbol, PERIOD_CURRENT, i);
      double open = iOpen(_Symbol, PERIOD_CURRENT, i);
      double high = iHigh(_Symbol, PERIOD_CURRENT, i);
      double low = iLow(_Symbol, PERIOD_CURRENT, i);
      double close = iClose(_Symbol, PERIOD_CURRENT, i);
      long volume = iVolume(_Symbol, PERIOD_CURRENT, i);
      
      json += "{";
      json += "\"time\":\"" + TimeToString(time) + "\",";
      json += "\"open\":" + DoubleToString(open, 8) + ",";
      json += "\"high\":" + DoubleToString(high, 8) + ",";
      json += "\"low\":" + DoubleToString(low, 8) + ",";
      json += "\"close\":" + DoubleToString(close, 8) + ",";
      json += "\"volume\":" + IntegerToString(volume);
      json += "}";
   }
   
   json += "]";
   return json;
}

//+------------------------------------------------------------------+
//| Parse prediction response from JSON                               |
//+------------------------------------------------------------------+
bool ParsePredictionResponse(string json_str, double &buy_prob, double &sell_prob, string &regime, double &confidence)
{
   // Simple JSON parsing (for production, use a proper JSON library)
   // This is a simplified version
   
   int buy_pos = StringFind(json_str, "\"buy_prob\":");
   int sell_pos = StringFind(json_str, "\"sell_prob\":");
   int regime_pos = StringFind(json_str, "\"regime\":");
   int conf_pos = StringFind(json_str, "\"confidence\":");
   
   if(buy_pos < 0 || sell_pos < 0 || regime_pos < 0 || conf_pos < 0)
      return false;
   
   // Extract values (simplified - use proper JSON parser in production)
   buy_prob = StringToDouble(ExtractJSONValue(json_str, "buy_prob"));
   sell_prob = StringToDouble(ExtractJSONValue(json_str, "sell_prob"));
   regime = ExtractJSONStringValue(json_str, "regime");
   confidence = StringToDouble(ExtractJSONValue(json_str, "confidence"));
   
   return true;
}

//+------------------------------------------------------------------+
//| Extract JSON value (simplified helper)                           |
//+------------------------------------------------------------------+
string ExtractJSONValue(string json, string key)
{
   string search = "\"" + key + "\":";
   int pos = StringFind(json, search);
   if(pos < 0)
      return "0";
   
   pos += StringLen(search);
   int end_pos = StringFind(json, ",", pos);
   if(end_pos < 0)
      end_pos = StringFind(json, "}", pos);
   
   if(end_pos < 0)
      return "0";
   
   return StringSubstr(json, pos, end_pos - pos);
}

//+------------------------------------------------------------------+
//| Extract JSON string value (simplified helper)                   |
//+------------------------------------------------------------------+
string ExtractJSONStringValue(string json, string key)
{
   string search = "\"" + key + "\":\"";
   int pos = StringFind(json, search);
   if(pos < 0)
      return "";
   
   pos += StringLen(search);
   int end_pos = StringFind(json, "\"", pos);
   
   if(end_pos < 0)
      return "";
   
   return StringSubstr(json, pos, end_pos - pos);
}

//+------------------------------------------------------------------+
//| Execute trading logic based on predictions                        |
//+------------------------------------------------------------------+
void ExecuteTradingLogic(double buy_prob, double sell_prob, string regime, double confidence)
{
   // Regime filter
   if(UseRegimeFilter)
   {
      if(regime == "BEAR_STRONG" || regime == "VOLATILE")
      {
         Print("⚠️ Regime filter: Skipping trade (", regime, ")");
         return;
      }
   }
   
   // Check confidence threshold
   if(confidence < 0.5)
   {
      Print("⚠️ Low confidence: ", confidence);
      return;
   }
   
   // Close existing positions if signal changes
   if(buy_prob > BuyThreshold && sell_prob < SellThreshold)
   {
      CloseSellPositions();
      if(!HasBuyPosition())
      {
         OpenBuyOrder();
      }
   }
   else if(sell_prob > SellThreshold && buy_prob < BuyThreshold)
   {
      CloseBuyPositions();
      if(!HasSellPosition())
      {
         OpenSellOrder();
      }
   }
}

//+------------------------------------------------------------------+
//| Check if buy position exists                                      |
//+------------------------------------------------------------------+
bool HasBuyPosition()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket > 0)
      {
         if(PositionGetString(POSITION_SYMBOL) == _Symbol &&
            PositionGetInteger(POSITION_MAGIC) == MagicNumber &&
            PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
         {
            return true;
         }
      }
   }
   return false;
}

//+------------------------------------------------------------------+
//| Check if sell position exists                                     |
//+------------------------------------------------------------------+
bool HasSellPosition()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket > 0)
      {
         if(PositionGetString(POSITION_SYMBOL) == _Symbol &&
            PositionGetInteger(POSITION_MAGIC) == MagicNumber &&
            PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL)
         {
            return true;
         }
      }
   }
   return false;
}

//+------------------------------------------------------------------+
//| Open buy order                                                     |
//+------------------------------------------------------------------+
void OpenBuyOrder()
{
   MqlTradeRequest request = {};
   MqlTradeResult result = {};
   
   request.action = TRADE_ACTION_DEAL;
   request.symbol = _Symbol;
   request.volume = LotSize;
   request.type = ORDER_TYPE_BUY;
   request.price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   request.deviation = 10;
   request.magic = MagicNumber;
   request.comment = "Natron AI Buy Signal";
   
   if(OrderSend(request, result))
   {
      Print("✅ Buy order opened: Ticket ", result.order);
   }
   else
   {
      Print("❌ Buy order failed: ", result.retcode);
   }
}

//+------------------------------------------------------------------+
//| Open sell order                                                    |
//+------------------------------------------------------------------+
void OpenSellOrder()
{
   MqlTradeRequest request = {};
   MqlTradeResult result = {};
   
   request.action = TRADE_ACTION_DEAL;
   request.symbol = _Symbol;
   request.volume = LotSize;
   request.type = ORDER_TYPE_SELL;
   request.price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   request.deviation = 10;
   request.magic = MagicNumber;
   request.comment = "Natron AI Sell Signal";
   
   if(OrderSend(request, result))
   {
      Print("✅ Sell order opened: Ticket ", result.order);
   }
   else
   {
      Print("❌ Sell order failed: ", result.retcode);
   }
}

//+------------------------------------------------------------------+
//| Close buy positions                                                |
//+------------------------------------------------------------------+
void CloseBuyPositions()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket > 0)
      {
         if(PositionGetString(POSITION_SYMBOL) == _Symbol &&
            PositionGetInteger(POSITION_MAGIC) == MagicNumber &&
            PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
         {
            MqlTradeRequest request = {};
            MqlTradeResult result = {};
            
            request.action = TRADE_ACTION_DEAL;
            request.position = ticket;
            request.symbol = _Symbol;
            request.volume = PositionGetDouble(POSITION_VOLUME);
            request.type = ORDER_TYPE_SELL;
            request.price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
            request.deviation = 10;
            request.magic = MagicNumber;
            
            OrderSend(request, result);
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Close sell positions                                               |
//+------------------------------------------------------------------+
void CloseSellPositions()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket > 0)
      {
         if(PositionGetString(POSITION_SYMBOL) == _Symbol &&
            PositionGetInteger(POSITION_MAGIC) == MagicNumber &&
            PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL)
         {
            MqlTradeRequest request = {};
            MqlTradeResult result = {};
            
            request.action = TRADE_ACTION_DEAL;
            request.position = ticket;
            request.symbol = _Symbol;
            request.volume = PositionGetDouble(POSITION_VOLUME);
            request.type = ORDER_TYPE_BUY;
            request.price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
            request.deviation = 10;
            request.magic = MagicNumber;
            
            OrderSend(request, result);
         }
      }
   }
}
//+------------------------------------------------------------------+
