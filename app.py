from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import random
from flask_sqlalchemy import SQLAlchemy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# load files===========================================================================================================
trending_products = pd.read_csv("models/trending_products.csv")
train_data = pd.read_csv("models/clean_data.csv")

# database configuration---------------------------------------
app.secret_key = "alskdjfwoeieiurlskdjfslkdjf"
# Using MySQL with environment variables for containerization
mysql_host = os.environ.get('MYSQL_HOST', 'localhost')
mysql_user = os.environ.get('MYSQL_USER', 'root')
mysql_password = os.environ.get('MYSQL_PASSWORD', 'root')
mysql_db = os.environ.get('MYSQL_DB', 'ecom')

app.config['SQLALCHEMY_DATABASE_URI'] = f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


# Define your model class for the 'signup' table
class Signup(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)

# Define your model class for the 'signup' table
class Signin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)


# Recommendations functions============================================================================================
# Function to truncate product name
def truncate(text, length):
    if len(text) > length:
        return text[:length] + "..."
    else:
        return text


def content_based_recommendations(train_data, item_name, top_n=10):
    # Check if the item name exists in the training data
    if item_name not in train_data['Name'].values:
        print(f"Item '{item_name}' not found in the training data.")
        return pd.DataFrame()

    # Create a TF-IDF vectorizer for item descriptions
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    # Apply TF-IDF vectorization to item descriptions
    tfidf_matrix_content = tfidf_vectorizer.fit_transform(train_data['Tags'])

    # Calculate cosine similarity between items based on descriptions
    cosine_similarities_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)

    # Find the index of the item
    item_index = train_data[train_data['Name'] == item_name].index[0]

    # Get the cosine similarity scores for the item
    similar_items = list(enumerate(cosine_similarities_content[item_index]))

    # Sort similar items by similarity score in descending order
    similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)

    # Get the top N most similar items (excluding the item itself)
    top_similar_items = similar_items[1:top_n+1]

    # Get the indices of the top similar items
    recommended_item_indices = [x[0] for x in top_similar_items]

    # Get the details of the top similar items
    recommended_items_details = train_data.iloc[recommended_item_indices][['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']]

    return recommended_items_details

def rating_based_recommendations(train_data, top_n=10):
    """
    Generate recommendations based on ratings (trending products)
    """
    # Group items by name, review count, brand, and image URL, and calculate average rating
    average_ratings = train_data.groupby(['Name', 'ReviewCount', 'Brand', 'ImageURL'])['Rating'].mean().reset_index()
    
    # Sort items by rating in descending order
    top_rated_items = average_ratings.sort_values(by='Rating', ascending=False)
    
    # Return the top N items
    return top_rated_items.head(top_n)

def collaborative_filtering_recommendations(train_data, target_user_id, top_n=10):
    """
    Generate recommendations using collaborative filtering
    """
    # Handle non-int values in user ID
    try:
        target_user_id = int(target_user_id)
    except ValueError:
        # If it can't be converted to int, use a default value
        target_user_id = 4
    
    # Create the user-item matrix
    user_item_matrix = train_data.pivot_table(index='ID', columns='ProdID', values='Rating', aggfunc='mean').fillna(0)
    
    # Check if target user exists in matrix
    if target_user_id not in user_item_matrix.index:
        # Return empty dataframe if user not found
        return pd.DataFrame()
    
    try:
        # Calculate the user similarity matrix using cosine similarity
        user_similarity = cosine_similarity(user_item_matrix)
        
        # Find the index of the target user in the matrix
        target_user_index = user_item_matrix.index.get_loc(target_user_id)
        
        # Get the similarity scores for the target user
        user_similarities = user_similarity[target_user_index]
        
        # Sort the users by similarity in descending order (excluding the target user)
        similar_users_indices = user_similarities.argsort()[::-1][1:]
        
        # Generate recommendations based on similar users
        recommended_items = []
        
        for user_index in similar_users_indices[:5]:  # Use top 5 similar users
            # Get items rated by the similar user but not by the target user
            rated_by_similar_user = user_item_matrix.iloc[user_index]
            not_rated_by_target_user = (rated_by_similar_user > 0) & (user_item_matrix.iloc[target_user_index] == 0)
            
            # Extract the item IDs of recommended items
            recommended_items.extend(user_item_matrix.columns[not_rated_by_target_user][:top_n])
            
            if len(recommended_items) >= top_n:
                break
                
        # Get the details of recommended items
        if recommended_items:
            recommended_items_details = train_data[train_data['ProdID'].isin(recommended_items)][['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']]
            return recommended_items_details.head(top_n)
        else:
            # Fallback to rating-based recommendations if no collaborative recommendations found
            return rating_based_recommendations(train_data, top_n)
    except Exception as e:
        print(f"Error in collaborative filtering: {e}")
        # Fallback to rating-based recommendations in case of error
        return rating_based_recommendations(train_data, top_n)

def hybrid_recommendations(train_data, target_user_id, item_name, top_n=10):
    """
    Generate recommendations using a hybrid approach (content-based + collaborative)
    """
    # Get content-based recommendations
    content_based_rec = content_based_recommendations(train_data, item_name, top_n)
    
    # Get collaborative filtering recommendations
    collaborative_rec = collaborative_filtering_recommendations(train_data, target_user_id, top_n)
    
    # Merge and deduplicate the recommendations
    hybrid_rec = pd.concat([content_based_rec, collaborative_rec]).drop_duplicates()
    
    return hybrid_rec.head(top_n)

# routes===============================================================================
# List of predefined image URLs
random_image_urls = [
    "static/img/img_1.png",
    "static/img/img_2.png",
    "static/img/img_3.png",
    "static/img/img_4.png",
    "static/img/img_5.png",
    "static/img/img_6.png",
    "static/img/img_7.png",
    "static/img/img_8.png",
]


@app.route("/")
def index():
    # Create a list of random image URLs for each product
    random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
    price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
    return render_template('index.html',trending_products=trending_products.head(8),truncate = truncate,
                           random_product_image_urls=random_product_image_urls,
                           random_price = random.choice(price))

@app.route("/main")
def main():
    # Initialize an empty DataFrame for content_based_rec
    # Also provide other variables that might be used in the template
    random_product_image_urls = [random.choice(random_image_urls) for _ in range(8)]
    price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
    return render_template('main.html', 
                          content_based_rec=pd.DataFrame(), 
                          truncate=truncate,
                          random_product_image_urls=random_product_image_urls,
                          random_price=random.choice(price))

# routes
@app.route("/index")
def indexredirect():
    # Create a list of random image URLs for each product
    random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
    price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
    return render_template('index.html', trending_products=trending_products.head(8), truncate=truncate,
                           random_product_image_urls=random_product_image_urls,
                           random_price=random.choice(price))

@app.route("/signup", methods=['POST','GET'])
def signup():
    if request.method=='POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        new_signup = Signup(username=username, email=email, password=password)
        db.session.add(new_signup)
        db.session.commit()

        # Create a list of random image URLs for each product
        random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
        price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
        return render_template('index.html', trending_products=trending_products.head(8), truncate=truncate,
                               random_product_image_urls=random_product_image_urls, random_price=random.choice(price),
                               signup_message='User signed up successfully!'
                               )

# Route for signup page
@app.route('/signin', methods=['POST', 'GET'])
def signin():
    if request.method == 'POST':
        username = request.form['signinUsername']
        password = request.form['signinPassword']
        new_signup = Signin(username=username,password=password)
        db.session.add(new_signup)
        db.session.commit()

        # Create a list of random image URLs for each product
        random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
        price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
        return render_template('index.html', trending_products=trending_products.head(8), truncate=truncate,
                               random_product_image_urls=random_product_image_urls, random_price=random.choice(price),
                               signup_message='User signed in successfully!'
                               )
                               
# Content-based recommendations page
@app.route("/content_based", methods=['GET', 'POST'])
def content_based():
    random_product_image_urls = [random.choice(random_image_urls) for _ in range(8)]
    price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
    
    # Default recommendation
    if request.method == 'GET':
        return render_template('content_based.html', 
                            content_based_rec=pd.DataFrame(), 
                            truncate=truncate,
                            random_product_image_urls=random_product_image_urls,
                            random_price=random.choice(price))
    else:  # POST request
        prod = request.form.get('prod')
        nbr = int(request.form.get('nbr', 10))
        content_based_rec = content_based_recommendations(train_data, prod, top_n=nbr)

        if content_based_rec.empty:
            message = "No recommendations available for this product."
            return render_template('content_based.html', 
                                message=message,
                                content_based_rec=pd.DataFrame(),
                                truncate=truncate,
                                random_product_image_urls=random_product_image_urls,
                                random_price=random.choice(price))
        else:
            return render_template('content_based.html', 
                                content_based_rec=content_based_rec, 
                                truncate=truncate,
                                random_product_image_urls=random_product_image_urls,
                                random_price=random.choice(price),
                                search_item=prod)

# Collaborative filtering recommendations page
@app.route("/collaborative", methods=['GET', 'POST'])
def collaborative():
    random_product_image_urls = [random.choice(random_image_urls) for _ in range(8)]
    price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
    
    # Default recommendation
    if request.method == 'GET':
        return render_template('collaborative.html', 
                            collaborative_rec=pd.DataFrame(), 
                            truncate=truncate,
                            random_product_image_urls=random_product_image_urls,
                            random_price=random.choice(price))
    else:  # POST request
        user_id = request.form.get('user_id', 4)
        nbr = int(request.form.get('nbr', 10))
        collaborative_rec = collaborative_filtering_recommendations(train_data, user_id, top_n=nbr)

        if collaborative_rec.empty:
            message = "No recommendations available for this user."
            return render_template('collaborative.html', 
                                message=message,
                                collaborative_rec=pd.DataFrame(),
                                truncate=truncate,
                                random_product_image_urls=random_product_image_urls,
                                random_price=random.choice(price))
        else:
            return render_template('collaborative.html', 
                                collaborative_rec=collaborative_rec, 
                                truncate=truncate,
                                random_product_image_urls=random_product_image_urls,
                                random_price=random.choice(price),
                                user_id=user_id)

# Rating-based recommendations page
@app.route("/rating_based", methods=['GET'])
def rating_based():
    random_product_image_urls = [random.choice(random_image_urls) for _ in range(8)]
    price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
    
    # Get top 10 rated products
    nbr = int(request.args.get('nbr', 10))
    rating_based_rec = rating_based_recommendations(train_data, top_n=nbr)
    
    return render_template('rating_based.html', 
                          rating_based_rec=rating_based_rec, 
                          truncate=truncate,
                          random_product_image_urls=random_product_image_urls,
                          random_price=random.choice(price))

# Hybrid recommendations page
@app.route("/hybrid", methods=['GET', 'POST'])
def hybrid():
    random_product_image_urls = [random.choice(random_image_urls) for _ in range(8)]
    price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
    
    # Default recommendation
    if request.method == 'GET':
        return render_template('hybrid.html', 
                            hybrid_rec=pd.DataFrame(), 
                            truncate=truncate,
                            random_product_image_urls=random_product_image_urls,
                            random_price=random.choice(price))
    else:  # POST request
        user_id = request.form.get('user_id', 4)
        prod = request.form.get('prod')
        nbr = int(request.form.get('nbr', 10))
        
        if not prod:
            message = "Please specify a product name."
            return render_template('hybrid.html', 
                                message=message,
                                hybrid_rec=pd.DataFrame(),
                                truncate=truncate,
                                random_product_image_urls=random_product_image_urls,
                                random_price=random.choice(price))
                                
        hybrid_rec = hybrid_recommendations(train_data, user_id, prod, top_n=nbr)

        if hybrid_rec.empty:
            message = "No recommendations available for this combination."
            return render_template('hybrid.html', 
                                message=message,
                                hybrid_rec=pd.DataFrame(),
                                truncate=truncate,
                                random_product_image_urls=random_product_image_urls,
                                random_price=random.choice(price))
        else:
            return render_template('hybrid.html', 
                                hybrid_rec=hybrid_rec, 
                                truncate=truncate,
                                random_product_image_urls=random_product_image_urls,
                                random_price=random.choice(price),
                                search_item=prod,
                                user_id=user_id)

@app.route("/recommendations", methods=['POST', 'GET'])
def recommendations():
    if request.method == 'POST':
        prod = request.form.get('prod')
        nbr = int(request.form.get('nbr'))
        content_based_rec = content_based_recommendations(train_data, prod, top_n=nbr)

        if content_based_rec.empty:
            message = "No recommendations available for this product."
            # Initialize an empty DataFrame for content_based_rec to avoid UndefinedError
            random_product_image_urls = [random.choice(random_image_urls) for _ in range(8)]
            price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
            return render_template('main.html', 
                                 message=message,
                                 content_based_rec=pd.DataFrame(),
                                 truncate=truncate,
                                 random_product_image_urls=random_product_image_urls,
                                 random_price=random.choice(price))
        else:
            # Create a list of random image URLs for each recommended product
            random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
            price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
            return render_template('main.html', content_based_rec=content_based_rec, truncate=truncate,
                                   random_product_image_urls=random_product_image_urls,
                                   random_price=random.choice(price))


if __name__=='__main__':
    with app.app_context():
        db.create_all()  # Create database tables if they don't exist
    app.run(debug=True, host='0.0.0.0', port=8000)