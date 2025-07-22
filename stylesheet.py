custom_stylesheet = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&family=Playfair+Display:wght@600&display=swap');

    * {
        font-family: 'Poppins', sans-serif;
    }
    .stApp {
        background-color: #F5F1E3;
        padding: 20px;
    }
    .header {
        text-align: center;
        padding: 20px;
        background: #A7C7A4;
        color: #2E5733;
        font-size: min(8vw, 36px);
        font-weight: 600;
        font-family: 'Playfair Display', serif;
        border-radius: 12px;
        margin-bottom: 30px;
        word-wrap: break-word;
    }
    .product-container {
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
        justify-content: center;
        padding: 10px;
    }
    .product-card {
        margin:auto;
        background: #FFFFFF;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 2px 4px 10px rgba(0,0,0,0.1);
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
        width: 100%;
        max-width: 280px;
        margin-bottom: 15px;
    }
    .product-card:hover {
        transform: translateY(-5px);
        box-shadow: 4px 6px 15px rgba(0,0,0,0.15);
    }
    .product-card img {
        width: 100%;
        max-width: 160px;
        height: auto;
        display: block;
        margin: 10px auto;
        border-radius: 10px;
    }
    .product-title {
        font-size: 18px;
        font-weight: 600;
        color: #3B5D3A;
        margin-bottom: 5px;
    }
    .product-category {
        font-size: 14px;
        color: #6B6B6B;
        margin-bottom: 10px;
    }
    .eco-score {
        font-size: 14px;
        font-weight: bold;
        color: #2E5733;
        background-color: #E6EFE9;
        padding: 5px 10px;
        border-radius: 8px;
        display: inline-block;
        margin-top: 5px;
    }
    .sidebar .block-container {
        background: #E6EFE9;
        padding: 15px;
        border-radius: 12px;
    }
    .stButton>button {
        background-color: #4E7D57;
        color: white;
        padding: 10px 16px;
        border-radius: 8px;
        font-size: 16px;
        font-weight: 600;
        transition: all 0.3s ease-in-out;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #3A5C40;
        transform: scale(1.05);
    }
    .stMarkdown p {
        color: #2E5733; /* Ensure visibility */
    }
    @media (max-width: 768px) {
        .header {
            font-size: min(6vw, 28px);
            padding: 15px;
        }
        .product-container {
            flex-direction: column;
            align-items: center;
        }
        .product-card {
            width: 95%;
            max-width: 320px;
        }
        .stButton>button {
            font-size: 14px;
            padding: 8px;
        }
    }
    @media (max-width: 480px) {
        .header {
            font-size: min(5vw, 24px);
            padding: 10px;
        }
        .product-card {
            width: 95%;
        }
        .stButton>button {
            font-size: 14px;
            padding: 6px 12px;
        }
    }
    </style>
"""