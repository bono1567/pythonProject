import csv
import threading
from datetime import datetime

import heapq
from collections import defaultdict

class Order:
    def __init__(self, order_id, ticker, timestamp, seq_num, side, size, price):
        self.order_id = order_id      
        self.ticker = ticker          
        self.timestamp = timestamp   
        self.seq_num = seq_num
        self.side = side
        self.size = size
        self.remaining = size  
        self.price = price

class OrderBook:
    def __init__(self):
        self.buy_orders = defaultdict(list)
        self.sell_orders = defaultdict(list)
        
        self.buy_heap = []
        self.sell_heap = []

        self.valid_orders = set()

        self.id_map = {}

    def add_limit_order(self, order: Order):
        # Add to order map
        self.id_map[order.order_id] = order
        self.valid_orders.add(order.order_id)
        
        # Add to price level (Heap handles priority)

        if order.side == 0:  # Buy
            self.buy_orders[order.price].append(order)
            # Buy heap: max heap
            priority_tuple = (-order.price, -order.size, order.seq_num, order.order_id)
            heapq.heappush(self.buy_heap, priority_tuple)
        else:  # Sell
            self.sell_orders[order.price].append(order)
            # Sell heap: min heap
            priority_tuple = (order.price, -order.size, order.seq_num, order.order_id)
            heapq.heappush(self.sell_heap, priority_tuple)

    def cancel_order(self, order_id):
        if order_id not in self.id_map:
            # Cancel orders which do not exist
            return
        
        self.id_map.pop(order_id)
        self.valid_orders.discard(order_id)
        
        # While fetching best order this condition is checked.
        # # Remove from price level
        # price_orders = self.buy_orders[order.price] if order.side == 0 else self.sell_orders[order.price]
        # try:
        #     price_orders.remove(order)
        # except ValueError:
        #     pass
        
        # # Clean up empty price levels
        # if not price_orders:
        #     if order.side == 0:
        #         del self.buy_orders[order.price]
        #     else:
        #         del self.sell_orders[order.price]

    def get_best_order(self, side):
        heap = self.buy_heap if side == 0 else self.sell_heap
        price_orders = self.buy_orders if side == 0 else self.sell_orders
        
        # You can put a check for self trading as well if we had data such as client ID
        while heap:
            priority_tuple = heap[0]
            order_id = priority_tuple[3]
            
            # Check if order is still valid
            if order_id not in self.valid_orders:
                heapq.heappop(heap)
                continue
                
            # Get the actual order
            order = self.id_map[order_id]
            
            # Check if order still has remaining quantity
            if order.remaining <= 0:
                heapq.heappop(heap)
                self.valid_orders.discard(order_id)
                self.id_map.pop(order_id, None)
                continue
                
            # Verify the order still exists in the price level
            price_level = price_orders.get(order.price, [])
            if not price_level or order not in price_level:
                heapq.heappop(heap)
                continue
                
            return order
            
        return None

    def match_order(self, incoming, exec_logs, vwap_acc):
        opposite_side = 1 - incoming.side
        
        while incoming.remaining > 0:
            best_available = self.get_best_order(opposite_side) 
            
            if not best_available:
                break
                
            # Check if prices cross
            if incoming.side == 0:  # Incoming buy
                if best_available.price > incoming.price:
                    break
            else:  # Incoming sell
                if best_available.price < incoming.price:
                    break
            
            # Execute trade
            trade_qty = min(incoming.remaining, best_available.remaining)
            exec_px = best_available.price
            
            ts = incoming.timestamp.strftime("%Y-%m-%dT%H:%M:%S")
            buyer, seller = (incoming, best_available) if incoming.side == 0 else (best_available, incoming)
            
            # Log execution (buy side first)
            exec_logs.append((buyer.order_id, buyer.ticker, ts, '0', str(trade_qty), f"{exec_px:.2f}"))
            exec_logs.append((seller.order_id, seller.ticker, ts, '1', str(trade_qty), f"{exec_px:.2f}"))

            # Update VWAP
            key = (incoming.ticker, incoming.timestamp.date().isoformat())
            vwap_acc.setdefault(key, [0.0, 0])
            vwap_acc[key][0] += exec_px * trade_qty
            vwap_acc[key][1] += trade_qty
            
            # Update remaining quantities
            incoming.remaining -= trade_qty
            best_available.remaining -= trade_qty
            
            
            # This bit is redundant
            if best_available.remaining == 0:
                price_orders = self.buy_orders if best_available.side == 0 else self.sell_orders
                price_level = price_orders[best_available.price]
                price_level.remove(best_available)  # Remove the specific order object // O(n) worse case time complexity
                
                self.valid_orders.discard(best_available.order_id)
                self.id_map.pop(best_available.order_id, None)
                
                # Clean up empty price level
                if not price_level:
                    del price_orders[best_available.price]
        
        # Add remaining quantity to book if any
        if incoming.remaining > 0:
            self.add_limit_order(incoming)



def process_ticker_orders(ticker, orders, exec_logs, vwap_acc):
    print(f"Processing orders for ticker: {ticker}")
    book = OrderBook()
    for (ts, seq_num, order_type, side, size, price, order_id) in orders:
        if order_type == 'L':
            order = Order(order_id, ticker, ts, seq_num, side, size, price)
            # print(f"Adding order: {order_id}, Type: {order_type}, Side: {'Buy' if side == 0 else 'Sell'}, Size: {size}, Price: {price}")
            book.match_order(order, exec_logs[ticker], vwap_acc)
        elif order_type == 'C':
            book.cancel_order(order_id)


def process_orders(input_files):
    orders_by_ticker = {'CUBI': [], 'SYST': [], 'STRT': []}
    
    # Read and parse each file
    for fname in input_files:
        try:
            with open(fname, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    try:
                        order_id = row[0].strip()
                        ticker = row[1].strip()
                        ts = datetime.fromisoformat(row[2].strip())
                        seq_num = int(row[3])
                        order_type = row[4].strip()
                        side = int(row[5])
                        size = int(row[6])
                        price = float(row[7])
                        orders_by_ticker[ticker].append(
                            (ts, seq_num, order_type, side, size, price, order_id)
                        )
                    except (IndexError, ValueError) as e:
                        print(f"Error processing order: {row}, Error: {e}")
        except FileNotFoundError:
            print(f"File not found: {fname}")
            continue
    
    # Sorted by timestamp now, sort by seq_num within each ticker
    for ticker, orders in orders_by_ticker.items():
        orders.sort(key=lambda x:  x[1])

    # Initialize execution logs and order books
    exec_logs = {'CUBI': [], 'SYST': [], 'STRT': []}
    vwap_acc = {}

    # Execute each ticker in different threads (could use async but might end up chaing the implementation do the functions)
    threads = []
    for ticker, orders in orders_by_ticker.items():
        t = threading.Thread(target=process_ticker_orders, args=(ticker, orders, exec_logs, vwap_acc))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
    
    return exec_logs, vwap_acc


def write_exec_logs(exec_logs):
    for ticker in ['CUBI', 'SYST', 'STRT']:
        filename = ticker.lower() + '_exec_v1.csv'
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['order_id', 'ticker', 'exec_time', 'order_side', 'exec_qty', 'exec_px'])
            for row in exec_logs[ticker]:
                writer.writerow(row)


def write_vwap(vwap_acc):
    rows = []
    for (ticker, date), (sum_pq, sum_q) in vwap_acc.items():
        if sum_q > 0:
            vwap = sum_pq / sum_q
        else:
            vwap = 0.0
        # Format VWAP to two decimals
        rows.append((date, ticker, f"{vwap:.2f}"))
    
    # Sort by date, then ticker
    rows.sort(key=lambda x: (x[0], x[1]))
    
    with open('ticker_vwap.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['date', 'ticker', 'vwap'])
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    input_files = ["./data/orders_2018-11-20T00-00-00.txt", "./data/orders_2018-11-20T01-00-00.txt"]
    exec_logs, vwap_acc = process_orders(input_files) # exec_logs = {'CUBI': [], ..}, vwap_acc = {('CUBI', '2023-03-04') = [], ...}
    write_exec_logs(exec_logs)
    write_vwap(vwap_acc)