import gradio as gr
import pandas as pd
from src.data.loader import data_loader
from src.engine.recommender import Recommender
from src.data.feedback_store import feedback_store

# Initialize Recommender (will load model once)
# We need to load documents first
print("Loading data...")
documents = data_loader.load_documents("data/tagged_description.txt")
books = data_loader.load_books("data/books_with_redirect_links(emotions).csv")
recommender = Recommender(documents)
print("Data loaded and Recommender initialized.")

def handle_feedback(query, book_title, feedback_type):
    feedback_store.log_feedback(query, book_title, feedback_type)
    return f"Thanks for your feedback on '{book_title}'!"

def format_book_html(row, query, explain_text=None):
    description = row["description"]
    truncated_desc_split = description.split()
    truncated_description = " ".join(truncated_desc_split[:20]) + "..."
    
    if pd.isna(row["authors"]):
        authors_str = "Unknown Author"
    else:
        authors_split = str(row["authors"]).split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])} and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

    redirect_link = row["redirect_link"] if "redirect_link" in row and pd.notna(row["redirect_link"]) else "#"
    
    explain_html = ""
    if explain_text:
        explain_html = f"<div class='book-explain'>💡 <b>Why:</b> \"{explain_text}\"</div>"

    # We can't easily put Gradio buttons INSIDE the HTML string that trigger Python functions directly 
    # without using some JS/Gradio trickery or creating components in a loop.
    # For simplicity in this version, we will just show the books. 
    # To implement REAL feedback buttons for EACH book in a gallery, we need to use gr.Gallery or a list of components.
    # But gr.Gallery doesn't support custom buttons per item easily.
    # So we will stick to HTML for display, but maybe add a "Feedback" section below where user can select a book?
    # OR, we can use a workaround: Make the book title a button? No.
    
    # BETTER APPROACH: Use a separate "Rate Recommendations" section.
    return f"""
    <div class="book-item">
        <a href="{redirect_link}" target="_blank">
            <img src="{row['large_thumbnail']}" alt="{row['title']}" />
            <div class="book-caption">
                <div class="book-title">{row['title']}</div>
                <div class="book-author">by {authors_str}</div>
                <div class="book-desc">{truncated_description}</div>
                {explain_html}
            </div>
        </a>
    </div>
    """

def search_books(query, category, tone):
    if not query:
        return "Please enter a query.", []
    
    try:
        recs = recommender.hybrid_search(query)
        filtered_books = recommender.filter_books(books, recs, query, category, tone)
        
        # Filter out bad covers
        filtered_books = filtered_books[
            (filtered_books["large_thumbnail"].notna()) & 
            (filtered_books["large_thumbnail"] != "cover-not-found.jpg")
        ]
        
        if filtered_books.empty:
            return "<div class='error-msg'>No books found matching your criteria. Try a different query or filters.</div>", []
        
        html_output = """<div class="book-gallery">"""
        
        # For the top result, add explanation
        first = True
        book_titles = []
        for _, row in filtered_books.iterrows():
            explain_text = None
            if first:
                try:
                    explain_text = recommender.explain_match(query, row["description"])
                except Exception as e:
                    print(f"Error explaining match: {e}")
                    explain_text = "Could not generate explanation."
                first = False
                
            html_output += format_book_html(row, query, explain_text)
            book_titles.append(row["title"])
        
        html_output += "</div>"
        
        # Return HTML and list of choices for feedback dropdown
        return html_output, gr.update(choices=book_titles, value=None)
        
    except Exception as e:
        print(f"Search Error: {e}")
        return f"<div class='error-msg'>An error occurred during search: {str(e)}</div>", []

def mood_journey(start_mood, end_mood):
    try:
        journey_books = recommender.get_mood_journey(books, start_mood, end_mood)
        
        if not journey_books:
            return "Could not generate a journey. Please try different moods."
            
        html_output = """<div class="journey-container">"""
        
        steps = ["Start: " + start_mood, "Bridge", "End: " + end_mood]
        
        for i, book in enumerate(journey_books):
            row = book
            description = row["description"]
            truncated_desc_split = description.split()
            truncated_description = " ".join(truncated_desc_split[:15]) + "..."
            
            if pd.isna(row["authors"]):
                authors_str = "Unknown Author"
            else:
                authors_split = str(row["authors"]).split(";")
                authors_str = authors_split[0] 
            
            redirect_link = row["redirect_link"] if "redirect_link" in row and pd.notna(row["redirect_link"]) else "#"

            html_output += f"""
            <div class="journey-step">
                <div class="step-label">{steps[i]}</div>
                <div class="book-item">
                    <a href="{redirect_link}" target="_blank">
                        <img src="{row['large_thumbnail']}" alt="{row['title']}" />
                        <div class="book-caption">
                            <div class="book-title">{row['title']}</div>
                            <div class="book-author">{authors_str}</div>
                        </div>
                    </a>
                </div>
            </div>
            """
            if i < 2:
                html_output += "<div class='arrow'>➡️</div>"
                
        html_output += "</div>"
        return html_output
    except Exception as e:
        print(f"Mood Journey Error: {e}")
        return f"<div class='error-msg'>An error occurred generating the journey: {str(e)}</div>"

# CSS (Unchanged)
css = """
.book-gallery {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 20px;
    padding: 20px;
}
.book-item {
    text-align: center;
    transition: transform 0.2s;
    background: rgba(255, 255, 255, 0.05);
    padding: 10px;
    border-radius: 10px;
}
.book-item:hover {
    transform: scale(1.05);
    background: rgba(255, 255, 255, 0.1);
}
.book-item a {
    text-decoration: none;
    color: inherit;
    display: block;
}
.book-item img {
    width: 100%;
    height: 200px;
    object-fit: cover;
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.book-caption {
    margin-top: 10px;
    font-size: 14px;
    line-height: 1.4;
}
.book-title {
    font-weight: bold;
    margin-bottom: 5px;
    color: #e0e0e0;
}
.book-author {
    color: #aaa;
    margin-bottom: 5px;
    font-size: 12px;
}
.book-desc {
    font-size: 12px;
    color: #888;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.book-explain {
    margin-top: 8px;
    font-size: 11px;
    color: #76baff;
    background: rgba(118, 186, 255, 0.1);
    padding: 4px;
    border-radius: 4px;
}
.journey-container {
    display: flex;
    align_items: center;
    justify-content: center;
    gap: 20px;
    padding: 40px;
    overflow-x: auto;
}
.journey-step {
    text-align: center;
    min-width: 160px;
}
.step-label {
    font-weight: bold;
    margin-bottom: 10px;
    color: #ffb74d;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.arrow {
    font-size: 24px;
    color: #666;
}
.error-msg {
    color: #ff6b6b;
    background: rgba(255, 107, 107, 0.1);
    padding: 15px;
    border-radius: 8px;
    text-align: center;
    margin: 20px 0;
    border: 1px solid #ff6b6b;
}
"""

def create_dashboard():
    categories = ["All"] + sorted(books["simple_category"].unique())
    tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

    with gr.Blocks(theme=gr.themes.Glass(), css=css) as dashboard:
        gr.Markdown("# 📚 NextGen Book Recommender")
        
        with gr.Tabs():
            with gr.TabItem("🔍 Semantic Search"):
                with gr.Row():
                    user_query = gr.Textbox(
                        label="Describe your ideal book", 
                        placeholder="e.g., A sci-fi novel about artificial intelligence and ethics"
                    )
                with gr.Row():
                    category_dropdown = gr.Dropdown(choices=categories, label="Category", value="All")
                    tone_dropdown = gr.Dropdown(choices=tones, label="Tone", value="All")
                
                search_btn = gr.Button("Find Books", variant="primary")
                search_output = gr.HTML(label="Recommendations")
                
                # Feedback Section
                gr.Markdown("### 📝 Rate Recommendations")
                with gr.Row():
                    feedback_book = gr.Dropdown(label="Select Book to Rate", choices=[])
                    feedback_type = gr.Radio(["Like", "Dislike"], label="Your Feedback")
                    feedback_btn = gr.Button("Submit Feedback")
                
                feedback_msg = gr.Markdown()

                search_btn.click(
                    fn=search_books,
                    inputs=[user_query, category_dropdown, tone_dropdown],
                    outputs=[search_output, feedback_book]
                )
                
                feedback_btn.click(
                    fn=handle_feedback,
                    inputs=[user_query, feedback_book, feedback_type],
                    outputs=feedback_msg
                )
                
            with gr.TabItem("🌊 Mood Journey"):
                gr.Markdown("### Take an emotional journey through reading.")
                gr.Markdown("Select a starting mood and an ending mood. We'll curate a 3-book path for you.")
                
                with gr.Row():
                    start_mood = gr.Dropdown(choices=tones[1:], label="Start Mood", value="Sad")
                    end_mood = gr.Dropdown(choices=tones[1:], label="End Mood", value="Happy")
                
                journey_btn = gr.Button("Generate Journey", variant="primary")
                journey_output = gr.HTML(label="Your Journey")
                
                journey_btn.click(
                    fn=mood_journey,
                    inputs=[start_mood, end_mood],
                    outputs=journey_output
                )

    return dashboard
