"""
CAVEX (Caveman Stock Exchange) Level-2 Order Book Matching Engine

This script implements a latency-sensitive limit-order book and matching engine
for a simplified exchange (CAVEX) trading only three symbols: CUBI, SYST, STRT.
The exchange has second-level timestamp precision and processes orders in
the sequence they arrive (increasing timestamp, with seq_num ordering within each timestamp).
This design follows the CAVEX programming challenge specification.

Key features:
- Supports limit orders (buy/sell) and cancel orders (by order_id).
- Order priority: price → original size → time (seq_num), per price level:contentReference[oaicite:0]{index=0}.
  * Buy side: highest prices first; Sell side: lowest prices first.
  * At the same price, larger size orders queue before smaller ones; ties broken by earlier seq_num.
- Execution logging: for each trade, outputs two rows (buy side first) with columns:
  (order_id, ticker, exec_time, order_side, exec_qty, exec_px).
- VWAP calculation: daily volume-weighted average price per ticker (date,ticker,vwap).

Data Input:
Two flat files (covering sequential hours) contain order events (limit or cancel).
Each line format (CSV) is assumed as:
    order_id, ticker, timestamp, seq_num, order_type, order_side, size, limit_price, exchange
where:
- order_type: 'L' = limit order, 'C' = cancel order (limit_price will be 0).
- order_side: 0 = buy, 1 = sell:contentReference[oaicite:1]{index=1}.
- For limit orders: 'size' is original quantity, 'limit_price' > 0.
- For cancel orders: 'size' equals the remaining quantity being cancelled, and 'limit_price' = 0:contentReference[oaicite:2]{index=2}.
- Timestamp is ISO format (yyyy-mm-ddTHH:MM:SS).
- Each order is uniquely (ticker, seq_num) and sequence numbers increase with arrival:contentReference[oaicite:3]{index=3}.
  However, within the same timestamp (second), seq_nums may not appear sorted in the feed.
  We sort them by seq_num within each timestamp block.

Outputs:
- One CSV file per ticker (cubi_exec.csv, syst_exec.csv, strt_exec.csv) listing executions in order of occurrence.
- A CSV ticker_vwap.csv with daily VWAP per ticker (date,ticker,vwap).

The code below reads input files, processes orders for each ticker separately (since no cross-symbol matching),
maintains the order book, logs executions, and computes VWAP.
"""
import csv
from datetime import datetime

# Data structures for the order book and orders:

class Order:
    """Represents an order in the book or incoming."""
    def __init__(self, order_id, ticker, timestamp, seq_num, side, size, price):
        self.order_id = order_id      # Unique identifier string
        self.ticker = ticker          # 'CUBI', 'SYST', or 'STRT'
        self.timestamp = timestamp    # datetime object of order arrival
        self.seq_num = seq_num        # sequence number (int)
        self.side = side              # 0=buy, 1=sell
        self.size = size              # original order quantity
        self.remaining = size         # remaining quantity yet to be filled/cancelled
        self.price = price            # limit price (0 for cancel, >0 for limit)
        # Original size is needed for priority; 'remaining' updated as trades occur.

class OrderBook:
    """Level-2 order book for one ticker symbol, supporting price-size-time priority."""
    def __init__(self):
        # Maps price -> list of Order objects at that price
        self.buy_book = {}   # buy side: key=price, value=list of orders (sorted by size desc, time asc)
        self.sell_book = {}  # sell side: same structure
        # Sorted lists of price levels for fast best-price lookup
        self.buy_prices = []  # descending sorted list of buy prices (best = highest)
        self.sell_prices = [] # ascending sorted list of sell prices (best = lowest)
        # Map order_id to Order object (for quick cancel lookup)
        self.id_map = {}

    def _insert_price_level(self, price_list, price, descending=False):
        """
        Insert price into sorted price_list if not present.
        descending=True for buy prices (list sorted high->low).
        """
        # If price already in list, no action.
        if price in price_list:
            return
        # Insert maintaining order
        import bisect
        if not price_list:
            price_list.append(price)
        else:
            if descending:
                # list sorted descending: insert to keep descending
                neg_list = [-p for p in price_list]
                pos = bisect.bisect_left(neg_list, -price)
                price_list.insert(pos, price)
            else:
                # ascending
                pos = bisect.bisect_left(price_list, price)
                price_list.insert(pos, price)

    def _remove_price_level(self, price_list, price):
        """Remove price from price_list if present."""
        if price in price_list:
            price_list.remove(price)

    def add_limit_order(self, order):
        """
        Add a new limit order to the book (after matching as needed).
        Inserts into book with correct priority (size desc, seq asc) within its price level.
        """
        book = self.buy_book if order.side == 0 else self.sell_book
        price_list = self.buy_prices if order.side == 0 else self.sell_prices
        # Ensure price level exists
        if order.price not in book:
            book[order.price] = []
            # Insert price into price list sorted properly
            # Buys: descending; Sells: ascending.
            if order.side == 0:
                self._insert_price_level(self.buy_prices, order.price, descending=True)
            else:
                self._insert_price_level(self.sell_prices, order.price, descending=False)
        # Insert order into the list at this price by size-time priority
        orders_at_price = book[order.price]
        # Find position: larger original size => earlier; if tie, smaller seq_num => earlier
        i = 0
        while i < len(orders_at_price):
            o = orders_at_price[i]
            # If existing order has larger size, skip
            if o.size > order.size:
                i += 1
                continue
            # If same size, order by seq (earlier sequence first)
            if o.size == order.size and o.seq_num <= order.seq_num:
                i += 1
                continue
            break
        orders_at_price.insert(i, order)
        # Track in id_map for cancelation
        self.id_map[order.order_id] = order

    def cancel_order(self, order_id):
        """
        Remove an order by order_id (cancel full remaining qty) from the book.
        """
        if order_id not in self.id_map:
            # Order not found or already gone; ignore
            return
        order = self.id_map.pop(order_id)
        # Identify side and price level
        book = self.buy_book if order.side == 0 else self.sell_book
        price_list = self.buy_prices if order.side == 0 else self.sell_prices
        price_level = order.price
        if price_level in book:
            orders_at_price = book[price_level]
            # Remove the order from list
            orders_at_price = [o for o in orders_at_price if o.order_id != order_id]
            # If list now empty, remove price level
            if not orders_at_price:
                del book[price_level]
                self._remove_price_level(price_list, price_level)
            else:
                book[price_level] = orders_at_price
        # No execution occurs; cancel just removes order.

    def match_order(self, incoming_order, exec_logs, vwap_acc):
        """
        Match an incoming limit order against the book, generating executions.
        - incoming_order.remaining is reduced as trades happen.
        - Pop and update resting orders as they fill.
        - exec_logs: append tuples for each side of executions.
        - vwap_acc: dict to accumulate (ticker, date) -> [sum_price_qty, sum_qty].
        """
        # Determine book sides
        if incoming_order.side == 0:
            # Buy order matches against best sell prices (lowest).
            opposite_book = self.sell_book
            opposite_prices = self.sell_prices
            price_cmp = lambda best_price: best_price <= incoming_order.price
        else:
            # Sell order matches against best buy prices (highest).
            opposite_book = self.buy_book
            opposite_prices = self.buy_prices
            price_cmp = lambda best_price: best_price >= incoming_order.price

        # While incoming still has qty and there is a matchable price level
        while incoming_order.remaining > 0 and opposite_prices:
            best_price = opposite_prices[0]
            # Check if best_price is matchable under limit
            if not price_cmp(best_price):
                break  # no more matchable prices
            # There is a match:
            orders_at_price = opposite_book[best_price]
            # Take the first order (highest priority at this price)
            resting_order = orders_at_price[0]
            trade_qty = min(incoming_order.remaining, resting_order.remaining)
            # Execution price is the resting order's price (order book price)
            exec_px = resting_order.price
            # Record executions: buy side first, then sell side
            if incoming_order.side == 0:
                buy_order, sell_order = incoming_order, resting_order
            else:
                buy_order, sell_order = resting_order, incoming_order
            exec_time_str = incoming_order.timestamp.strftime("%Y-%m-%dT%H:%M:%S")
            # Append buy side execution
            exec_logs.append((
                buy_order.order_id,
                buy_order.ticker,
                exec_time_str,
                '0',            # buy side code
                str(trade_qty),
                f"{exec_px:.2f}"
            ))
            # Append sell side execution
            exec_logs.append((
                sell_order.order_id,
                sell_order.ticker,
                exec_time_str,
                '1',            # sell side code
                str(trade_qty),
                f"{exec_px:.2f}"
            ))
            # Accumulate for VWAP (count the trade once)
            trade_date = incoming_order.timestamp.date().isoformat()
            key = (incoming_order.ticker, trade_date)
            if key not in vwap_acc:
                vwap_acc[key] = [0.0, 0]
            vwap_acc[key][0] += exec_px * trade_qty
            vwap_acc[key][1] += trade_qty
            # Update remaining quantities
            incoming_order.remaining -= trade_qty
            resting_order.remaining -= trade_qty
            # Remove resting order if fully filled
            if resting_order.remaining == 0:
                orders_at_price.pop(0)
                self.id_map.pop(resting_order.order_id, None)
                if not orders_at_price:
                    del opposite_book[best_price]
                    opposite_prices.pop(0)
            # If incoming order fully matched, break
            if incoming_order.remaining == 0:
                break
            # Otherwise, continue matching same incoming against next best price
        # If incoming order still has remaining qty after matches, insert into book
        if incoming_order.remaining > 0:
            self.add_limit_order(incoming_order)


def process_orders(input_files):
    """
    Main processing function:
    - Read input files and aggregate orders by ticker.
    - Sort orders by timestamp (and seq_num within timestamp).
    - Process orders for each ticker through the matching engine.
    - Return execution logs (per ticker) and VWAP accumulators.
    """
    orders_by_ticker = {'CUBI': [], 'SYST': [], 'STRT': []}
    # Read and parse each file
    for fname in input_files:
        try:
            with open(fname, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    # Skip empty or header lines if any
                    if not row or row[0].startswith('#') or row[0] == 'order_id':
                        continue
                    # Expected columns:
                    # 0: order_id, 1: ticker, 2: timestamp, 3: seq_num,
                    # 4: order_type, 5: order_side, 6: size, 7: limit_price, 8: exchange
                    order_id = row[0].strip()
                    ticker = row[1].strip()
                    # Parse timestamp
                    ts = datetime.fromisoformat(row[2].strip())
                    seq_num = int(row[3])
                    order_type = row[4].strip()
                    side = int(row[5])
                    size = int(row[6])
                    price = float(row[7])
                    # Only process known tickers
                    if ticker not in orders_by_ticker:
                        continue
                    # Append a tuple for later processing
                    orders_by_ticker[ticker].append(
                        (ts, seq_num, order_type, side, size, price, order_id)
                    )
        except FileNotFoundError:
            print(f"File not found: {fname}")
            continue

    # Sort orders by timestamp, then seq_num within each ticker
    for ticker, orders in orders_by_ticker.items():
        orders.sort(key=lambda x: (x[0], x[1]))

    # Initialize execution logs and order books
    exec_logs = {'CUBI': [], 'SYST': [], 'STRT': []}
    vwap_acc = {}  # key (ticker, date) -> [sum_price_qty, sum_qty]

    # Process each ticker separately (no cross-ticker matching needed)
    for ticker, orders in orders_by_ticker.items():
        book = OrderBook()
        for (ts, seq_num, order_type, side, size, price, order_id) in orders:
            if order_type == 'L':
                # Limit order: create Order and attempt match
                order = Order(order_id, ticker, ts, seq_num, side, size, price)
                book.match_order(order, exec_logs[ticker], vwap_acc)
            elif order_type == 'C':
                # Cancel order: remove existing order by ID
                book.cancel_order(order_id)
            else:
                # Unknown order type; ignore
                continue

    return exec_logs, vwap_acc

def write_exec_logs(exec_logs):
    """
    Write execution logs to CSV files per ticker.
    Each exec_log[ticker] is a list of tuples corresponding to CSV rows.
    """
    for ticker in ['CUBI', 'SYST', 'STRT']:
        filename = ticker.lower() + '_exec_v2.csv'
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['order_id', 'ticker', 'exec_time', 'order_side', 'exec_qty', 'exec_px'])
            for row in exec_logs[ticker]:
                writer.writerow(row)

def write_vwap(vwap_acc):
    """
    Write daily VWAP to ticker_vwap.csv.
    vwap_acc keys: (ticker, date) -> [sum_price_qty, sum_qty].
    """
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
    # List of input files (modify these names as needed)
    input_files = ["./data/orders_1.txt", "./data/orders_2.txt"]
    exec_logs, vwap_acc = process_orders(input_files)
    write_exec_logs(exec_logs)
    write_vwap(vwap_acc)

    # Bonus: Potential optimizations in C++/Java implementation
    # --------------------------------------------------------
    # In a lower-level language like C++ or Java, we could optimize for latency and throughput:
    # - Memory layout and custom allocators: tightly-packed structs and pre-allocated memory pools
    #   would reduce garbage collection overhead and allocations for orders and nodes.
    # - Faster data structures: e.g., using std::map or TreeMap for price levels gives log-time best-price retrieval,
    #   or a custom tree/heap. We could also use specialized priority queues or skip lists for quick access.
    # - Lock-free concurrency: languages like C++/Java allow fine-grained concurrency (lock-free queues, atomic ops)
    #   to handle high-order throughput with minimal latency.
    # - Avoid dynamic typing/boxing: using primitive types (int, double) directly improves speed vs Python objects.
    # - Inlining and compiler optimizations: the matching loop could be inlined and optimized; branch predictions are more efficient.
    # - Batch processing and low-level I/O: using buffered I/O and efficient parsing can speed data ingestion.
    # Overall, C++/Java implementations allow more control over memory and CPU, yielding faster order matching
    # under heavy load and lower latencies.
