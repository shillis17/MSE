# Music Similarity Recommendation Engine  
**Full Project Proposal**

---

## A. Problem Statement & Motivation

Discovering new music that aligns with individual taste is increasingly difficult due to the overwhelming volume of digital content. While commercial platforms employ proprietary recommendation algorithms, these systems lack transparency and often prioritize mainstream content. This limits exploration, particularly for listeners interested in independent or non-commercial music.

The primary beneficiaries of this project are music enthusiasts and researchers interested in music discovery beyond mainstream platforms. Using the Free Music Archive (FMA), a large, openly available dataset of independent music, this project aims to develop and evaluate multiple music similarity models to enable meaningful discovery of new tracks and artists.

Existing approaches include traditional content-based recommendation systems using cosine similarity over audio features, which will serve as our baseline. While effective for simple similarity detection, these methods struggle to capture more nuanced relationships. More recent approaches, such as Siamese neural networks, learn richer representations but require careful tuning and significant data preparation.

This project distinguishes itself by directly comparing traditional audio-feature-based models with modern generative AI approaches, including embedding-based similarity using Large Language Models (LLMs). By evaluating these methods side-by-side, we aim to assess whether semantic embeddings can outperform classical audio-based similarity measures in music recommendation tasks.

---

## B. Data Strategy

### Data Source
We will use the publicly available **Free Music Archive (FMA)** dataset.

Repository: https://github.com/mdeff/fma  
Date Retrieved: February 2026

| File Name | Rows | Columns | Size |
|----------|------|---------|------|
| raw_tracks.csv | 109,727 | 39 | ~119 KB |
| raw_echonest.csv | 14,512 | 250 | ~47 KB |
| raw_artists.csv | 16,916 | 25 | ~13 KB |
| raw_albums.csv | 15,234 | 19 | ~23 KB |
| raw_genres.csv | 164 | 5 | ~6 KB |

### Features & Data Types
The primary features are numerical audio descriptors from `raw_echonest.csv`, including:
- acousticness
- danceability
- energy
- instrumentalness
- liveness
- speechiness
- tempo
- valence

Additional categorical metadata (track title, artist name, genre) will be used for labeling and textual embedding generation.

There is no predefined target variable. The objective is to compute similarity scores or ranked recommendations between tracks.

### Preprocessing Plan
- **Data Merging:** Combine all CSV files into a unified dataset keyed by track ID.
- **Missing Values:** Restrict modeling to the 14,512 tracks with complete Echonest features.
- **Normalization:** Apply Z-score normalization to all numerical audio features.
- **Categorical Encoding:** One-hot encode genre labels for traditional models.
- **Text Construction:** Generate textual descriptions for LLM embeddings using track metadata.
- **Data Splitting:** Use an 80/20 train-test split for evaluation consistency.

---

## C. Technical Methodology

### Data Pipeline
1. Load and merge raw datasets
2. Clean and normalize features
3. Generate feature matrices and text embeddings
4. Train similarity models
5. Evaluate and compare results
6. Deploy selected models

### Baseline Models
- Cosine similarity over normalized audio features
- K-Nearest Neighbors (KNN)
- Matrix Factorization

### Advanced Models
- Siamese Neural Network trained in PyTorch
- Embedding-based similarity using LLMs (OpenAI API)

### Technology Stack
- Python, Pandas, NumPy
- Scikit-learn
- PyTorch
- OpenAI API
- AWS (EC2, S3)

---

## D. Evaluation & Metrics

### Model Performance Metrics
- **Top-K Precision and Recall:** Measures recommendation relevance
- **Mean Reciprocal Rank (MRR):** Evaluates ranking quality
- **Cosine Similarity Agreement:** Comparison against baseline similarity scores

These metrics were chosen to reflect ranking quality rather than classification accuracy, which is more appropriate for recommendation systems.

### Project Success Criteria
- Advanced models outperform baseline methods on ranking metrics
- System generates consistent, interpretable recommendations
- Deployed application successfully returns recommendations in real time

A qualitative listening evaluation will be conducted as a secondary validation method.

---

## E. Project Management

### Timeline

### Timeline

| Sprint | Focus Area | Key Tasks |
|------|-----------|-----------|
| **Sprint 1** | Foundation & Traditional ML | Environment setup and data ingestion<br>Preprocessing pipeline<br>Baseline, KNN, and matrix factorization models |
| **Sprint 2** | Advanced Models & Evaluation | Train Siamese Network<br>Generate LLM embeddings<br>Evaluation framework and comparative analysis<br>Begin application development |
| **Sprint 3** | Deployment & Documentation | Backend API and frontend development<br>AWS deployment<br>Load testing and optimization<br>Final report and presentation |


### Risks & Contingency Plans
1. **API Rate Limits or Cost Constraints**  
   *Mitigation:* Cache embeddings locally and limit API calls during training.

2. **Subjectivity of Music Similarity**  
   *Mitigation:* Rely on quantitative ranking metrics and use qualitative evaluation only as supplemental analysis.
