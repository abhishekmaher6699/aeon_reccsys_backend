
# End-to-End Aeon Essay Recommendation System  

## Backend  

A robust, Dockerized FastAPI backend hosted on AWS EC2 powers the application by providing two core functionalities:  

- **Recommend Articles by URL**: Suggests essays similar to the one identified by its URL.  
- **Fetch Articles by User Prompt**: Retrieves essays tailored to a user’s input or query.  

---

## Features  

- **Dynamic Recommendations**: Always up-to-date with the latest Aeon essays, thanks to on-demand web scraping.  
- **Scalable Architecture**: Dockerized backend ensures easy scaling and redeployment on AWS.  
- **Automated Updates**: Automatic model updates and redeployments with CI/CD pipelines triggered by database changes.  
- **Optimized Performance**: Leveraging PCA and cosine similarity for efficient and accurate recommendations.  
- **User-Centric Design**: Tailored recommendations via content-based filtering and prompt matching.  

---

## How It Works  

### **Step 1: Data Ingestion**  
Essay data is ingested from a PostgreSQL database and exported into a `data.csv` file. Before further processing, the data undergoes thorough validation to ensure consistency and accuracy.  

### **Step 2: Data Transformation**  
The ingested data undergoes several pre-processing transformations to enhance model performance, including:  
- Expanding contractions (e.g., "isn't" to "is not").  
- Removing stopwords, special characters, and punctuation.  
- Converting text to lowercase for uniformity.  

### **Step 3: Object Creation**  
1. **Vectorization**: Essay text is transformed into numerical data using **TF-IDF** (Term Frequency-Inverse Document Frequency), resulting in a high-dimensional dataset (10,000+ dimensions).  
2. **Dimensionality Reduction**: To manage computational complexity and improve performance, **PCA** (Principal Component Analysis) reduces the data's dimensionality to 100 features.  
3. **Cosine Similarity Matrix**: A similarity matrix is computed from the PCA-reduced vectors to identify relationships between essays.  
4. **Serialization**: The TF-IDF vectorizer, PCA model, and cosine similarity matrix are serialized and stored in pickle files for efficient reuse in the application.  

### **Step 4: API Development and Model Integration**  
Using the **FastAPI framework**, we create two API endpoints powered by the serialized objects and specialized models:  

- **Recommend Articles by URL**:  
  This endpoint leverages a content-based filtering model to recommend essays similar to the provided URL. If a recently published Aeon essay is not yet available in the database, the backend dynamically scrapes its content to generate recommendations in real-time. This ensures an up-to-date and seamless user experience.  

- **Fetch Articles by User Prompt**:  
  The user's query undergoes the same pre-processing pipeline as the essays. The prompt is vectorized using the stored TF-IDF model, and a similarity matrix is generated on-the-fly to deliver personalized results.  

### **Step 5: Containerization**  
The FastAPI application is packaged into a Docker container and uploaded to **AWS Elastic Container Registry (ECR)**.  

### **Step 6: Deployment**  
The Docker image is pulled from the ECR to an AWS EC2 instance and deployed, making the backend fully operational and scalable.  

---

## CI/CD Pipeline with GitHub Actions  

All steps are orchestrated via a GitHub Actions CI/CD pipeline, which ensures smooth and automated workflows:  
1. **Trigger on New Data**: A Python script monitors the database for changes and triggers the pipeline whenever new data is added.  
2. **Rebuilding and Redeploying**: The pipeline automatically updates the data, regenerates models, and redeploys the containerized application without manual intervention.  
3. **Testing and Validation**: Every update is rigorously tested for accuracy and stability before deployment.  

---

## Tech Stack

- Application: Python, FastAPI
- Deployment: Docker, AWS ECR, AWS EC2
- Database: PostgreSQL
- CI/CD: Github Actions.

---


This comprehensive pipeline and backend design ensure a seamless, fast, and personalized recommendation experience for Aeon essay readers.  
