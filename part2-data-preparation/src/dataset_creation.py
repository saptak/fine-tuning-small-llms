#!/usr/bin/env python3
"""
Dataset Creation Utilities
From: Fine-Tuning Small LLMs with Docker Desktop - Part 2
"""

import pandas as pd
import json
import argparse
from typing import List, Dict
from pathlib import Path

class SQLDatasetCreator:
    def __init__(self, output_dir: str = "./data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.examples = []
    
    def add_example(self, instruction: str, table_schema: str, sql_query: str, 
                   explanation: str = "", difficulty: str = "medium"):
        """Add a training example to the dataset"""
        
        example = {
            "instruction": instruction,
            "input": f"Table Schema: {table_schema}",
            "output": sql_query,
            "explanation": explanation,
            "difficulty": difficulty,
            "id": len(self.examples)
        }
        
        self.examples.append(example)
        return example
    
    def create_basic_examples(self):
        """Create fundamental SQL examples"""
        
        # Basic SELECT operations
        self.add_example(
            instruction="Select all columns from the users table",
            table_schema="users (id, name, email, created_at)",
            sql_query="SELECT * FROM users;",
            explanation="Basic SELECT statement to retrieve all columns and rows",
            difficulty="easy"
        )
        
        self.add_example(
            instruction="Find all users who registered in the last 30 days",
            table_schema="users (id, name, email, created_at)",
            sql_query="SELECT * FROM users WHERE created_at >= DATE_SUB(CURDATE(), INTERVAL 30 DAY);",
            explanation="Uses DATE_SUB function to filter recent registrations",
            difficulty="medium"
        )
        
        # Aggregation queries
        self.add_example(
            instruction="Count the total number of orders per customer",
            table_schema="orders (id, customer_id, amount, order_date)",
            sql_query="SELECT customer_id, COUNT(*) as order_count FROM orders GROUP BY customer_id;",
            explanation="Groups by customer and counts orders using COUNT(*)",
            difficulty="medium"
        )
        
        self.add_example(
            instruction="Find the average order amount per month",
            table_schema="orders (id, customer_id, amount, order_date)",
            sql_query="SELECT DATE_FORMAT(order_date, '%Y-%m') as month, AVG(amount) as avg_amount FROM orders GROUP BY DATE_FORMAT(order_date, '%Y-%m');",
            explanation="Uses DATE_FORMAT to group by month and AVG for average calculation",
            difficulty="medium"
        )
        
        # JOIN operations
        self.add_example(
            instruction="Show customer names with their total order amounts",
            table_schema="customers (id, name, email), orders (id, customer_id, amount, order_date)",
            sql_query="SELECT c.name, SUM(o.amount) as total_amount FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id, c.name;",
            explanation="INNER JOIN between customers and orders with SUM aggregation",
            difficulty="hard"
        )
        
        # Complex analytical queries
        self.add_example(
            instruction="Find customers who have spent more than the average customer spending",
            table_schema="customers (id, name, email), orders (id, customer_id, amount)",
            sql_query="""SELECT c.name, SUM(o.amount) as total_spent 
FROM customers c 
JOIN orders o ON c.id = o.customer_id 
GROUP BY c.id, c.name 
HAVING SUM(o.amount) > (
    SELECT AVG(customer_total) 
    FROM (
        SELECT SUM(amount) as customer_total 
        FROM orders 
        GROUP BY customer_id
    ) as customer_totals
);""",
            explanation="Complex query with subquery to find above-average spenders",
            difficulty="expert"
        )
    
    def create_advanced_examples(self):
        """Create advanced SQL examples"""
        
        # Window functions
        self.add_example(
            instruction="Rank customers by their total spending within each region",
            table_schema="customers (id, name, region), orders (id, customer_id, amount)",
            sql_query="""SELECT 
    c.name, 
    c.region,
    SUM(o.amount) as total_spent,
    RANK() OVER (PARTITION BY c.region ORDER BY SUM(o.amount) DESC) as spending_rank
FROM customers c
JOIN orders o ON c.id = o.customer_id
GROUP BY c.id, c.name, c.region;""",
            explanation="Uses window function RANK() with PARTITION BY for regional rankings",
            difficulty="expert"
        )
        
        # Common Table Expressions (CTEs)
        self.add_example(
            instruction="Find the second highest order amount for each customer",
            table_schema="orders (id, customer_id, amount, order_date)",
            sql_query="""WITH ranked_orders AS (
    SELECT 
        customer_id,
        amount,
        ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY amount DESC) as rn
    FROM orders
)
SELECT customer_id, amount as second_highest_amount
FROM ranked_orders 
WHERE rn = 2;""",
            explanation="Uses CTE with ROW_NUMBER() to find second highest values",
            difficulty="expert"
        )
    
    def format_for_training(self, format_type: str = "alpaca") -> List[Dict]:
        """Format examples for different training approaches"""
        
        formatted_examples = []
        
        for example in self.examples:
            if format_type == "alpaca":
                formatted = {
                    "instruction": example["instruction"],
                    "input": example["input"],
                    "output": example["output"]
                }
            
            elif format_type == "chat":
                formatted = {
                    "messages": [
                        {"role": "system", "content": "You are an expert SQL developer who generates accurate and efficient SQL queries."},
                        {"role": "user", "content": f"{example['instruction']}\n\n{example['input']}"},
                        {"role": "assistant", "content": example["output"]}
                    ]
                }
            
            elif format_type == "completion":
                formatted = {
                    "prompt": f"### SQL Request:\n{example['instruction']}\n\n{example['input']}\n\n### SQL Query:\n",
                    "completion": example["output"]
                }
            
            formatted_examples.append(formatted)
        
        return formatted_examples
    
    def save_dataset(self, filename: str = "sql_training_data", format_type: str = "alpaca"):
        """Save dataset in specified format"""
        
        formatted_data = self.format_for_training(format_type)
        
        # Save as JSON
        json_path = self.output_dir / f"{filename}_{format_type}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, indent=2, ensure_ascii=False)
        
        # Save as CSV (for Alpaca format)
        if format_type == "alpaca":
            df = pd.DataFrame(formatted_data)
            csv_path = self.output_dir / f"{filename}_alpaca.csv"
            df.to_csv(csv_path, index=False)
        
        print(f"Dataset saved: {json_path}")
        print(f"Total examples: {len(formatted_data)}")
        
        return json_path

def create_sql_dataset(output_dir: str, format_type: str = "alpaca"):
    """Create comprehensive SQL dataset"""
    
    creator = SQLDatasetCreator(output_dir=output_dir)
    
    # Add examples
    creator.create_basic_examples()
    creator.create_advanced_examples()
    
    # Add domain-specific examples
    creator.add_example(
        instruction="Create a query to find the top 10 products by sales volume",
        table_schema="products (id, name, category), order_items (id, product_id, quantity, order_id)",
        sql_query="SELECT p.name, SUM(oi.quantity) as total_sold FROM products p JOIN order_items oi ON p.id = oi.product_id GROUP BY p.id, p.name ORDER BY total_sold DESC LIMIT 10;",
        difficulty="medium"
    )
    
    creator.add_example(
        instruction="Calculate monthly revenue growth rate",
        table_schema="orders (id, amount, order_date)",
        sql_query="""WITH monthly_revenue AS (
    SELECT 
        DATE_FORMAT(order_date, '%Y-%m') as month,
        SUM(amount) as revenue
    FROM orders 
    GROUP BY DATE_FORMAT(order_date, '%Y-%m')
),
revenue_with_lag AS (
    SELECT 
        month,
        revenue,
        LAG(revenue) OVER (ORDER BY month) as prev_revenue
    FROM monthly_revenue
)
SELECT 
    month,
    revenue,
    ROUND(((revenue - prev_revenue) / prev_revenue) * 100, 2) as growth_rate_percent
FROM revenue_with_lag
WHERE prev_revenue IS NOT NULL;""",
        difficulty="expert"
    )
    
    # Save in specified format
    return creator.save_dataset("sql_dataset", format_type)

def main():
    """Main function with CLI interface"""
    
    parser = argparse.ArgumentParser(description="Create training datasets for LLM fine-tuning")
    parser.add_argument("--output-dir", default="./data/datasets", help="Output directory for datasets")
    parser.add_argument("--format", choices=["alpaca", "chat", "completion"], default="alpaca", help="Dataset format")
    parser.add_argument("--dataset-type", choices=["sql", "code", "support"], default="sql", help="Type of dataset to create")
    
    args = parser.parse_args()
    
    print(f"🚀 Creating {args.dataset_type} dataset in {args.format} format")
    print(f"Output directory: {args.output_dir}")
    
    if args.dataset_type == "sql":
        dataset_path = create_sql_dataset(args.output_dir, args.format)
        print(f"✅ SQL dataset created: {dataset_path}")
    else:
        print(f"❌ Dataset type '{args.dataset_type}' not implemented yet")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
