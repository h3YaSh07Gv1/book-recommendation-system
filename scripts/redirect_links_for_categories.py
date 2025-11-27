import pandas as pd
import re

# Load your CSV file
df = pd.read_csv("books_with_categories.csv")  # change filename if needed

# We're using the correct column: 'thumbnail'
column_name = "thumbnail"

def transform_link(old_link):
    if isinstance(old_link, str) and "id=" in old_link:
        match = re.search(r"id=([^&]+)", old_link)
        if match:
            book_id = match.group(1)
            return f"https://www.google.co.in/books/edition/_/{book_id}?hl=en&kptab=overview"
    return None  # or old_link if you want to keep originals

# Apply transformation
df["redirect_link"] = df[column_name].apply(transform_link)

# Print sample to confirm
print(df[["thumbnail", "redirect_link"]].head())

# Save to new CSV
df.to_csv("books_with_redirect_links.csv", index=False)

print("✅ Done! Created books_with_redirect_links.csv with redirection links.")
