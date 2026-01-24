import heapq
import csv
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading

class Order:
    def __init__(self, order_id, tick, ts, seq_num, order_type, side, size, price):
        self.order_id = order_id
        self.tick = tick
        self.ts = ts
        self.seq_num = seq_num
        self.order_type = order_type
        self.side = side
        self.size = size # For priority queue implementation since original size needs to be maintained
        self.remaining = size
        self.price = price

class OrderBook:
    def __init__(self):
        self.buy_heap = []
        self.sell_heap = []
        self.valid_orders = set()
        self.order_map = {}
    
    def delete_order(self, order_id):
        self.valid_orders.discard(order_id)
        self.order_map.pop(order_id, None)
    
    def get_best_order(self, side):
        price_heap = self.buy_heap if side == 0 else self.sell_heap

        while price_heap:
            priority_tuple = price_heap[0]
            order_id = priority_tuple[3]  # order_id is at index 3

            if order_id not in self.valid_orders:
                heapq.heappop(price_heap)
                continue

            original_order = self.order_map[order_id]

            if original_order.remaining <= 0:
                self.valid_orders.discard(order_id)
                self.order_map.pop(order_id)
                heapq.heappop(price_heap)
                continue
            
            return original_order
        return None
    
    def add_limit_order(self, order: Order):
        self.order_map[order.order_id] = order
        self.valid_orders.add(order.order_id)

        if order.side == 0:  # Buy
            # Buy heap: max heap (negative price for max heap behavior)
            priority_tuple = (-order.price, -order.size, order.seq_num, order.order_id)
            heapq.heappush(self.buy_heap, priority_tuple)
        else:  # Sell
            # Sell heap: min heap
            priority_tuple = (order.price, -order.size, order.seq_num, order.order_id)
            heapq.heappush(self.sell_heap, priority_tuple)

    def process_ticker_order(self, order: Order, exec_log, vwap, lock):
        opposite_side = 1 - order.side

        while order.remaining > 0:
            best_available = self.get_best_order(opposite_side)

            if not best_available:
                break

            # Check if trade is possible (buy price >= sell price)
            if order.side == 0:  # incoming buy order
                if order.price < best_available.price:
                    break
            else:  # incoming sell order
                if order.price > best_available.price:
                    break

            trade_qty = min(order.remaining, best_available.remaining)
            exec_px = best_available.price

            ts = order.ts.strftime("%Y-%m-%dT%H:%M:%S")
            buyer, seller = (order, best_available) if order.side == 0 else (best_available, order)

            # Use lock for thread-safe access to shared data
            with lock:
                exec_log.append((buyer.order_id, buyer.tick, ts, '0', str(trade_qty), f"{exec_px:.2f}"))
                exec_log.append((seller.order_id, seller.tick, ts, '1', str(trade_qty), f"{exec_px:.2f}"))

                key = (order.tick, order.ts.date().isoformat())
                vwap.setdefault(key, [0.0, 0])
                vwap[key][0] += exec_px * trade_qty
                vwap[key][1] += trade_qty

            order.remaining -= trade_qty
            best_available.remaining -= trade_qty

            if best_available.remaining <= 0:
                self.valid_orders.discard(best_available.order_id)
                self.order_map.pop(best_available.order_id, None)

        if order.remaining > 0:
            self.add_limit_order(order)

def process_order(ticker: str, orders: list, execution_book: dict, vwap: dict, lock):
    print(f"Processing orders for ticker: {ticker}")
    book = OrderBook()
    for (order_id, tick, ts, seq_num, order_type, side, size, price) in orders:
        if order_type == 'L':
            order = Order(order_id, tick, ts, seq_num, order_type, side, size, price)
            book.process_ticker_order(order, execution_book[ticker], vwap, lock)
        else:
            book.delete_order(order_id)

def write_exec_logs(exec_logs):
    for ticker in ['CUBI', 'SYST', 'STRT']:
        filename = ticker.lower() + '_exec_v3.csv'
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
        rows.append((date, ticker, f"{vwap:.2f}"))
    
    # Sort by date, then ticker
    rows.sort(key=lambda x: (x[0], x[1]))
    
    with open('ticker_vwap.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['date', 'ticker', 'vwap'])
        for row in rows:
            writer.writerow(row)

# Main execution
all_files = ['./data/orders_1.txt', './data/orders_2.txt']
all_orders = {'CUBI': [], 'SYST': [], 'STRT': []}

# Read all files first
for file_data in all_files:
    with open(file_data, 'r') as file:
        for line in file:
            fields = line.strip().split(',')
            if len(fields) < 8:
                continue  # Skip malformed lines
                
            order_id = fields[0].strip()
            ticker = fields[1].strip()
            ts = datetime.fromisoformat(fields[2].strip())
            seq_num = int(fields[3])
            order_type = fields[4].strip()
            side = int(fields[5])
            size = int(fields[6])
            price = float(fields[7])
            # fields[8] appears to be exchange/venue info, ignoring for now

            if ticker in all_orders:  # Only process known tickers
                all_orders[ticker].append((order_id, ticker, ts, seq_num, order_type, side, size, price))

# Sort orders by sequence number for each ticker
for ticker, orders in all_orders.items():
    orders.sort(key=lambda x: x[3])

# Initialize data structures
execution_book = {'CUBI': [], 'SYST': [], 'STRT': []}
vwap = {}
lock = threading.Lock()

# Process all orders using ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=3) as pool:
    futures = [pool.submit(process_order, ticker, orders, execution_book, vwap, lock) 
               for ticker, orders in all_orders.items()]
    for future in futures:
        future.result()

# Write results to files
write_exec_logs(execution_book)
write_vwap(vwap)