from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('Combined_Influencers_Dataset_Final.csv')

# Select relevant features
features = ['Followers', 'Engagement Rate', 'Campaign ROI', 'Ad Cost (USD)']

# Normalize the features
scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)

# Add categorical features
df_normalized['Platform'] = df['Platform']
df_normalized['Location'] = df['Location']
df_normalized['Industry'] = df['Industry']

# Create dummy variables for categorical features
df_encoded = pd.get_dummies(df_normalized, columns=['Platform', 'Location', 'Industry'])

# Calculate cosine similarity
cosine_sim = cosine_similarity(df_encoded)

# Function to get recommendations with platform, location, and industry filtering
def get_recommendations(followers, engagement_rate, campaign_roi, ad_cost, platform, location, industry, top_n=5):
    df_filtered = df[(df['Platform'] == platform) & 
                     (df['Location'] == location) & 
                     (df['Industry'] == industry)]
    
    if df_filtered.empty:
        return pd.DataFrame()  # Return an empty DataFrame if no matches are found

    df_encoded_filtered = df_encoded[(df['Platform'] == platform) & 
                                     (df['Location'] == location) & 
                                     (df['Industry'] == industry)]

    client_preferences = {
        'Followers': followers,
        'Engagement Rate': engagement_rate, 
        'Campaign ROI': campaign_roi,
        'Ad Cost (USD)': ad_cost,
        'Platform': platform,
        'Location': location,
        'Industry': industry
    }

    client_df = pd.DataFrame([client_preferences])

    client_normalized = pd.DataFrame(scaler.transform(client_df[features]), columns=features)

    client_normalized['Platform'] = client_preferences['Platform']
    client_normalized['Location'] = client_preferences['Location']
    client_normalized['Industry'] = client_preferences['Industry']

    client_encoded = pd.get_dummies(client_normalized, columns=['Platform', 'Location', 'Industry'])

    for col in df_encoded_filtered.columns:
        if col not in client_encoded.columns:
            client_encoded[col] = 0

    client_encoded = client_encoded[df_encoded_filtered.columns]

    client_similarity = cosine_similarity(client_encoded, df_encoded_filtered)

    similar_indices = client_similarity[0].argsort()[::-1][:top_n]
    return df_filtered.iloc[similar_indices]

@app.route('/')
def landing_page():
    return render_template('index.html')  # Landing page

@app.route('/login')
def login_page():
    return render_template('login.html')  # Login page

@app.route('/signup')
def signup_page():
    return render_template('signup 2.html')  # Signup page

@app.route('/mvp')
def mvp_page():
    platforms = df['Platform'].unique()
    locations = df['Location'].unique()
    industries = df['Industry'].unique()
    return render_template('mvp.html', platforms=platforms, locations=locations, industries=industries)  # MVP form page

@app.route('/recommend', methods=['POST'])
def recommend():
    followers = int(request.form['followers'])
    engagement_rate = float(request.form['engagement_rate'])
    campaign_roi = float(request.form['campaign_roi'])
    ad_cost = int(request.form['ad_cost'])
    platform = request.form['platform']
    location = request.form['location']
    industry = request.form['industry']

    recommendations = get_recommendations(followers, engagement_rate, campaign_roi, ad_cost, platform, location, industry)
    
    if recommendations is None or recommendations.empty:
        recommendations = pd.DataFrame(columns=['Username', 'Platform', 'Followers', 'Engagement Rate', 'Location', 'Industry', 'Campaign ROI', 'Ad Cost (USD)'])

    return render_template('recommendations.html', recommendations=recommendations.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
