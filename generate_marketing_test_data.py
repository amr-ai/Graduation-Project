"""
Generate a large, rich e-commerce dataset for testing the Marketing Agent.

Creates data/marketing_test_data.csv with ~5000 rows and 20 columns.
"""

import csv
import os
import random
import hashlib
from datetime import datetime, timedelta

random.seed(2024)

NUM_ORDERS = 5000
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "data", "marketing_test_data.csv")

# ── Product catalog (28 products, 6 categories) ───────────────────────

PRODUCTS = {
    "Electronics": [
        ("Wireless Mouse", 25.99, 12.00),
        ("USB-C Hub", 49.99, 22.00),
        ("Mechanical Keyboard", 89.99, 38.00),
        ("27\" Monitor", 349.99, 180.00),
        ("Webcam HD", 59.99, 25.00),
        ("Bluetooth Speaker", 39.99, 16.00),
        ("Portable Charger", 29.99, 11.00),
        ("Noise Cancelling Headphones", 199.99, 85.00),
        ("Smart Watch", 249.99, 110.00),
        ("Wireless Earbuds", 79.99, 30.00),
    ],
    "Clothing": [
        ("Cotton T-Shirt", 19.99, 5.50),
        ("Denim Jeans", 49.99, 15.00),
        ("Winter Jacket", 129.99, 45.00),
        ("Running Shoes", 89.99, 32.00),
        ("Wool Sweater", 59.99, 20.00),
        ("Casual Shorts", 29.99, 9.00),
        ("Silk Scarf", 39.99, 12.00),
        ("Leather Belt", 34.99, 10.00),
    ],
    "Home & Kitchen": [
        ("Coffee Maker", 79.99, 35.00),
        ("Air Fryer", 119.99, 50.00),
        ("Bed Sheets Set", 44.99, 14.00),
        ("LED Desk Lamp", 34.99, 12.00),
        ("Vacuum Cleaner", 199.99, 90.00),
        ("Blender Pro", 69.99, 28.00),
        ("Cutting Board Set", 24.99, 8.00),
    ],
    "Books & Media": [
        ("Bestseller Novel", 14.99, 4.00),
        ("Cookbook", 24.99, 8.00),
        ("Programming Guide", 39.99, 12.00),
        ("Art Supplies Kit", 34.99, 15.00),
        ("Language Course DVD", 29.99, 7.00),
    ],
    "Sports & Outdoors": [
        ("Yoga Mat", 29.99, 8.00),
        ("Dumbbell Set", 59.99, 25.00),
        ("Camping Tent", 149.99, 60.00),
        ("Water Bottle", 14.99, 3.50),
        ("Fitness Tracker", 79.99, 30.00),
        ("Hiking Backpack", 89.99, 35.00),
    ],
    "Beauty & Personal Care": [
        ("Face Moisturizer", 24.99, 7.00),
        ("Hair Dryer", 44.99, 18.00),
        ("Perfume Set", 69.99, 22.00),
        ("Electric Toothbrush", 39.99, 14.00),
        ("Skincare Bundle", 54.99, 16.00),
    ],
}

REGIONS = ["North", "South", "East", "West", "Central"]
CITIES = {
    "North": ["Cairo", "Alexandria", "Tanta"],
    "South": ["Aswan", "Luxor", "Qena"],
    "East": ["Suez", "Ismailia", "Port Said"],
    "West": ["Marsa Matrouh", "Siwa", "Alamein"],
    "Central": ["Giza", "Fayoum", "Beni Suef"],
}
PAYMENT_METHODS = ["Credit Card", "PayPal", "Bank Transfer", "Cash on Delivery", "Mobile Wallet"]
CHANNELS = ["Website", "Mobile App", "In-Store", "Social Media", "Marketplace"]
COUPON_CODES = [None, None, None, None, "SAVE10", "WELCOME15", "SUMMER20", "VIP25", "FLASH30", "LOYALTY5"]
GENDERS = ["Male", "Female"]
AGE_GROUPS = ["18-24", "25-34", "35-44", "45-54", "55+"]
DEVICES = ["Desktop", "Mobile", "Tablet"]
SATISFACTION = [1, 2, 3, 4, 5]

# ── Generate 200 customers ────────────────────────────────────────────

FIRST_NAMES_M = [
    "Ahmed", "Mohamed", "Omar", "Youssef", "Ali", "Hassan", "Khaled",
    "Tarek", "Amir", "Karim", "John", "David", "James", "Robert",
    "William", "Lucas", "Noah", "Liam", "Ethan", "Mason",
]
FIRST_NAMES_F = [
    "Sara", "Fatima", "Layla", "Nour", "Hana", "Mona", "Dina",
    "Reem", "Salma", "Yasmin", "Emma", "Sophie", "Olivia", "Ava",
    "Mia", "Ella", "Zoe", "Lily", "Chloe", "Grace",
]
LAST_NAMES = [
    "Ibrahim", "Hassan", "Ali", "Mohamed", "Ahmed", "Salem", "Nasser",
    "Farouk", "Mansour", "Khalil", "Smith", "Johnson", "Williams",
    "Brown", "Jones", "Garcia", "Miller", "Davis", "Wilson", "Taylor",
]

CUSTOMERS = []
for i in range(200):
    gender = random.choice(GENDERS)
    if gender == "Male":
        first = random.choice(FIRST_NAMES_M)
    else:
        first = random.choice(FIRST_NAMES_F)
    last = random.choice(LAST_NAMES)
    cid = f"CUST-{1000 + i}"
    age_group = random.choice(AGE_GROUPS)
    region = random.choice(REGIONS)
    CUSTOMERS.append({
        "id": cid,
        "name": f"{first} {last}",
        "email": f"{first.lower()}.{last.lower()}{random.randint(1,99)}@{'gmail.com' if random.random() > 0.4 else 'yahoo.com'}",
        "gender": gender,
        "age_group": age_group,
        "home_region": region,
    })


def random_date(start, end):
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


def generate_rows():
    rows = []
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)

    category_weights = {
        "Electronics": 0.30,
        "Clothing": 0.22,
        "Home & Kitchen": 0.18,
        "Sports & Outdoors": 0.12,
        "Beauty & Personal Care": 0.10,
        "Books & Media": 0.08,
    }
    categories = list(category_weights.keys())
    weights = list(category_weights.values())

    # Customer tiers
    vip_custs = CUSTOMERS[:20]
    regular_custs = CUSTOMERS[20:80]
    at_risk_custs = CUSTOMERS[80:130]
    new_custs = CUSTOMERS[130:]

    order_counter = 10000

    def make_order(cust, date_start, date_end, qty_range=(1, 6)):
        nonlocal order_counter
        order_counter += 1
        cat = random.choices(categories, weights=weights, k=1)[0]
        product_name, base_price, cost = random.choice(PRODUCTS[cat])
        qty = random.randint(*qty_range)
        unit_price = round(base_price * random.uniform(0.90, 1.10), 2)
        discount_pct = 0.0
        coupon = random.choice(COUPON_CODES)
        if coupon:
            discount_pct = int(coupon.replace("SAVE", "").replace("WELCOME", "").replace("SUMMER", "")
                               .replace("VIP", "").replace("FLASH", "").replace("LOYALTY", ""))
        subtotal = round(qty * unit_price, 2)
        discount_amt = round(subtotal * discount_pct / 100, 2)
        total = round(subtotal - discount_amt, 2)
        cost_total = round(cost * qty * random.uniform(0.95, 1.05), 2)
        profit = round(total - cost_total, 2)

        order_date = random_date(date_start, date_end)
        ship_date = order_date + timedelta(days=random.randint(1, 10))
        delivery_date = ship_date + timedelta(days=random.randint(1, 7))

        region = cust["home_region"] if random.random() > 0.2 else random.choice(REGIONS)
        city = random.choice(CITIES[region])
        channel = random.choice(CHANNELS)
        device = random.choice(DEVICES) if channel in ("Website", "Mobile App") else "N/A"
        rating = random.choices(SATISFACTION, weights=[5, 10, 20, 35, 30], k=1)[0]
        returned = random.random() < 0.05

        return {
            "order_id": f"ORD-{order_counter}",
            "customer_id": cust["id"],
            "customer_name": cust["name"],
            "email": cust["email"],
            "gender": cust["gender"],
            "age_group": cust["age_group"],
            "product": product_name,
            "category": cat,
            "quantity": qty,
            "unit_price": unit_price,
            "cost_price": round(cost * random.uniform(0.95, 1.05), 2),
            "discount_pct": discount_pct,
            "coupon_code": coupon if coupon else "",
            "subtotal": subtotal,
            "discount_amount": discount_amt,
            "total_price": total,
            "profit": profit,
            "order_date": order_date.strftime("%Y-%m-%d"),
            "ship_date": ship_date.strftime("%Y-%m-%d"),
            "delivery_date": delivery_date.strftime("%Y-%m-%d"),
            "region": region,
            "city": city,
            "payment_method": random.choice(PAYMENT_METHODS),
            "sales_channel": channel,
            "device": device,
            "satisfaction_rating": rating,
            "returned": returned,
        }

    # VIP — 20 customers, 30-60 orders each (spread across 2 years)
    for cust in vip_custs:
        for _ in range(random.randint(30, 60)):
            rows.append(make_order(cust, start_date, end_date, (1, 8)))

    # Regular — 60 customers, 8-20 orders each
    for cust in regular_custs:
        for _ in range(random.randint(8, 20)):
            rows.append(make_order(cust, start_date, end_date, (1, 5)))

    # At-risk — 50 customers, bought 3-8 times in 2023, then stopped
    at_risk_end = datetime(2024, 3, 31)
    for cust in at_risk_custs:
        for _ in range(random.randint(3, 8)):
            rows.append(make_order(cust, start_date, at_risk_end, (1, 4)))

    # New — 70 customers, 1-3 purchases in last 3 months of 2024
    new_start = datetime(2024, 10, 1)
    for cust in new_custs:
        for _ in range(random.randint(1, 3)):
            rows.append(make_order(cust, new_start, end_date, (1, 3)))

    random.shuffle(rows)
    return rows


def main():
    rows = generate_rows()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    fieldnames = [
        "order_id", "customer_id", "customer_name", "email",
        "gender", "age_group", "product", "category",
        "quantity", "unit_price", "cost_price", "discount_pct",
        "coupon_code", "subtotal", "discount_amount", "total_price",
        "profit", "order_date", "ship_date", "delivery_date",
        "region", "city", "payment_method", "sales_channel",
        "device", "satisfaction_rating", "returned",
    ]

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} rows x {len(fieldnames)} columns -> {OUTPUT_PATH}")

    from collections import Counter
    cats = Counter(r["category"] for r in rows)
    custs = len(set(r["customer_id"] for r in rows))
    print(f"\nCategories: {dict(cats)}")
    print(f"Unique customers: {custs}")
    print(f"Unique products: {len(set(r['product'] for r in rows))}")
    print(f"Date range: {min(r['order_date'] for r in rows)} to {max(r['order_date'] for r in rows)}")
    print(f"Columns: {fieldnames}")


if __name__ == "__main__":
    main()
