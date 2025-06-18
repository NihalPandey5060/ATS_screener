import streamlit as st
import fitz
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
import io
import re
from typing import List, Tuple, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResumeScreener:
    """AI-powered Resume Screening System"""
    
    def __init__(self):
        """Initialize the resume screener with the sentence transformer model"""
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.resumes_data = []
        self.job_description = ""
        self.similarity_scores = []
        
    def extract_text_from_pdf(self, pdf_file) -> str:
        """
        Extract text from a PDF file using PyMuPDF
        
        Args:
            pdf_file: Uploaded PDF file object
            
        Returns:
            str: Extracted text from the PDF
        """
        try:
            # Read the PDF file
            pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
            text = ""
            
            # Extract text from each page
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                text += page.get_text()
            
            pdf_document.close()
            
            # Clean the extracted text
            text = self.clean_text(text)
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess extracted text
        
        Args:
            text: Raw text from PDF
            
        Returns:
            str: Cleaned text
        """
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', '', text)
        # Strip leading/trailing whitespace
        text = text.strip()
        return text
    
    def extract_skills(self, text: str) -> List[str]:
        """
        Extract potential skills from text using keyword matching
        
        Args:
            text: Text to extract skills from
            
        Returns:
            List[str]: List of potential skills
        """
        # Common technical skills (you can expand this list)
        skill_keywords = [
            'python', 'java', 'javascript', 'react', 'angular', 'vue', 'node.js',
            'sql', 'mongodb', 'postgresql', 'mysql', 'aws', 'azure', 'docker',
            'kubernetes', 'git', 'jenkins', 'agile', 'scrum', 'machine learning',
            'deep learning', 'tensorflow', 'pytorch', 'scikit-learn', 'pandas',
            'numpy', 'matplotlib', 'seaborn', 'flask', 'django', 'fastapi',
            'html', 'css', 'bootstrap', 'tailwind', 'typescript', 'php',
            'c++', 'c#', '.net', 'spring', 'hibernate', 'junit', 'selenium',
            'jira', 'confluence', 'figma', 'adobe', 'photoshop', 'illustrator',
            'excel', 'powerpoint', 'word', 'power bi', 'tableau', 'spark',
            'hadoop', 'kafka', 'redis', 'elasticsearch', 'nginx', 'apache'
        ]
        
        text_lower = text.lower()
        found_skills = []
        
        for skill in skill_keywords:
            if skill in text_lower:
                found_skills.append(skill.title())
        
        return list(set(found_skills))  # Remove duplicates
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts using the sentence transformer model
        
        Args:
            texts: List of text strings
            
        Returns:
            np.ndarray: Array of embeddings
        """
        try:
            embeddings = self.model.encode(texts, show_progress_bar=True)
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return np.array([])
    
    def calculate_similarity_scores(self, resume_embeddings: np.ndarray, 
                                  job_embedding: np.ndarray) -> List[float]:
        """
        Calculate cosine similarity scores between resumes and job description
        
        Args:
            resume_embeddings: Array of resume embeddings
            job_embedding: Job description embedding
            
        Returns:
            List[float]: List of similarity scores
        """
        try:
            # Reshape job embedding for cosine similarity calculation
            job_embedding_reshaped = job_embedding.reshape(1, -1)
            
            # Calculate cosine similarity
            similarities = cosine_similarity(resume_embeddings, job_embedding_reshaped)
            
            # Flatten the result and convert to list
            return similarities.flatten().tolist()
            
        except Exception as e:
            logger.error(f"Error calculating similarity scores: {e}")
            return []
    
    def process_resumes(self, uploaded_files: List, job_description: str) -> pd.DataFrame:
        """
        Process uploaded resumes and job description to generate rankings
        
        Args:
            uploaded_files: List of uploaded PDF files
            job_description: Job description text
            
        Returns:
            pd.DataFrame: DataFrame with resume rankings and scores
        """
        if not uploaded_files or not job_description.strip():
            return pd.DataFrame()
        
        # Extract text from resumes
        resume_texts = []
        resume_names = []
        resume_skills = []
        
        for uploaded_file in uploaded_files:
            if uploaded_file.name.lower().endswith('.pdf'):
                text = self.extract_text_from_pdf(uploaded_file)
                if text:
                    resume_texts.append(text)
                    resume_names.append(uploaded_file.name)
                    skills = self.extract_skills(text)
                    resume_skills.append(', '.join(skills) if skills else 'No skills detected')
        
        if not resume_texts:
            return pd.DataFrame()
        
        # Generate embeddings
        with st.spinner("Generating embeddings for resumes and job description..."):
            resume_embeddings = self.generate_embeddings(resume_texts)
            job_embedding = self.generate_embeddings([job_description])
        
        if resume_embeddings.size == 0 or job_embedding.size == 0:
            return pd.DataFrame()
        
        # Calculate similarity scores
        similarity_scores = self.calculate_similarity_scores(resume_embeddings, job_embedding)
        
        # Create results DataFrame
        results_data = {
            'Resume Name': resume_names,
            'Similarity Score': [round(score * 100, 2) for score in similarity_scores],
            'Skills Detected': resume_skills,
            'Text Length': [len(text) for text in resume_texts]
        }
        
        df = pd.DataFrame(results_data)
        
        # Sort by similarity score in descending order
        df = df.sort_values('Similarity Score', ascending=False).reset_index(drop=True)
        
        return df
    
    def create_visualizations(self, df: pd.DataFrame):
        """
        Create visualizations for the results
        
        Args:
            df: DataFrame with resume rankings
        """
        if df.empty:
            return
        
        # Create a bar chart of similarity scores
        fig_bar = px.bar(
            df, 
            x='Resume Name', 
            y='Similarity Score',
            title='Resume Similarity Scores',
            color='Similarity Score',
            color_continuous_scale='viridis'
        )
        fig_bar.update_layout(
            xaxis_tickangle=-45,
            height=400
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Create a scatter plot of similarity vs text length
        fig_scatter = px.scatter(
            df,
            x='Text Length',
            y='Similarity Score',
            title='Similarity Score vs Text Length',
            hover_data=['Resume Name'],
            size='Similarity Score',
            color='Similarity Score',
            color_continuous_scale='plasma'
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    def export_to_csv(self, df: pd.DataFrame) -> bytes:
        """
        Export results DataFrame to CSV format
        
        Args:
            df: DataFrame to export
            
        Returns:
            bytes: CSV data as bytes
        """
        output = io.StringIO()
        df.to_csv(output, index=False)
        return output.getvalue().encode('utf-8')

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="AI Resume Screener",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: none;
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .metric-card h4 {
        margin: 0 0 0.5rem 0;
        font-size: 0.9rem;
        opacity: 0.9;
        font-weight: 500;
    }
    .metric-card h2 {
        margin: 0;
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-card-secondary {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: none;
        transition: transform 0.2s ease;
    }
    .metric-card-secondary:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .metric-card-success {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: none;
        transition: transform 0.2s ease;
    }
    .metric-card-success:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .metric-card-warning {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: none;
        transition: transform 0.2s ease;
    }
    .metric-card-warning:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .progress-container {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 0.5rem;
        padding: 0.5rem;
        margin-top: 0.5rem;
    }
    .progress-bar {
        background: rgba(255, 255, 255, 0.3);
        border-radius: 0.25rem;
        height: 0.5rem;
        position: relative;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        border-radius: 0.25rem;
        transition: width 0.3s ease;
    }
    .stats-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.5rem;
        margin-top: 1rem;
    }
    .stat-item {
        background: rgba(255, 255, 255, 0.1);
        padding: 0.5rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stat-label {
        font-size: 0.7rem;
        opacity: 0.8;
        margin-bottom: 0.25rem;
    }
    .stat-value {
        font-size: 1.1rem;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🤖 AI Resume Screening System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize the resume screener
    if 'screener' not in st.session_state:
        st.session_state.screener = ResumeScreener()
    
    screener = st.session_state.screener
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("📤 Upload Files")
        
        # File upload for resumes
        uploaded_resumes = st.file_uploader(
            "Upload Resume PDFs",
            type=['pdf'],
            accept_multiple_files=True,
            help="Select multiple PDF files containing resumes"
        )
        
        st.markdown("---")
        
        # Job description input
        st.header("📝 Job Description")
        job_description = st.text_area(
            "Enter the job description",
            height=300,
            placeholder="Paste the job description here...",
            help="Enter the complete job description to match against resumes"
        )
        
        st.markdown("---")
        
        # Process button
        process_button = st.button(
            "🚀 Process Resumes",
            type="primary",
            use_container_width=True
        )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Results")
        
        if process_button and uploaded_resumes and job_description.strip():
            with st.spinner("Processing resumes..."):
                # Process the resumes
                results_df = screener.process_resumes(uploaded_resumes, job_description)
                
                if not results_df.empty:
                    st.session_state.results_df = results_df
                    st.success(f"✅ Successfully processed {len(results_df)} resumes!")
                    
                    # Display results table
                    st.subheader("📋 Resume Rankings")
                    st.dataframe(
                        results_df,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Create visualizations
                    st.subheader("📈 Visualizations")
                    screener.create_visualizations(results_df)
                    
                else:
                    st.error("❌ No valid resumes found or processing failed.")
        
        elif 'results_df' in st.session_state:
            # Display previous results if available
            st.subheader("📋 Resume Rankings")
            st.dataframe(
                st.session_state.results_df,
                use_container_width=True,
                hide_index=True
            )
            
            st.subheader("📈 Visualizations")
            screener.create_visualizations(st.session_state.results_df)
    
    with col2:
        st.header("📈 Recruitment Analytics")
        
        if 'results_df' in st.session_state and not st.session_state.results_df.empty:
            df = st.session_state.results_df
            
            # Calculate comprehensive metrics
            avg_score = df['Similarity Score'].mean()
            max_score = df['Similarity Score'].max()
            min_score = df['Similarity Score'].min()
            median_score = df['Similarity Score'].median()
            std_score = df['Similarity Score'].std()
            total_candidates = len(df)
            
            # Calculate candidate quality distribution using HR terminology
            top_tier_candidates = len(df[df['Similarity Score'] >= 85])
            strong_candidates = len(df[(df['Similarity Score'] >= 70) & (df['Similarity Score'] < 85)])
            qualified_candidates = len(df[(df['Similarity Score'] >= 50) & (df['Similarity Score'] < 70)])
            underqualified_candidates = len(df[df['Similarity Score'] < 50])
            
            # Calculate skill match analysis
            all_skills = []
            for skills_str in df['Skills Detected']:
                if skills_str != 'No skills detected':
                    skills = [skill.strip() for skill in skills_str.split(',')]
                    all_skills.extend(skills)
            
            unique_skills_found = len(set(all_skills)) if all_skills else 0
            avg_skills_per_candidate = len(all_skills) / total_candidates if total_candidates > 0 else 0
            
            # Most in-demand skills analysis
            skill_counts = {}
            for skills_str in df['Skills Detected']:
                if skills_str != 'No skills detected':
                    skills = [skill.strip() for skill in skills_str.split(',')]
                    for skill in skills:
                        skill_counts[skill] = skill_counts.get(skill, 0) + 1
            
            most_common_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Calculate candidate pool quality metrics
            shortlist_ready = len(df[df['Similarity Score'] >= 75])  # Ready for interview
            needs_review = len(df[(df['Similarity Score'] >= 60) & (df['Similarity Score'] < 75)])  # May need further review
            rejection_candidates = len(df[df['Similarity Score'] < 60])  # Likely to reject
            
            # Display main metrics with HR terminology
            st.markdown(f"""
            <div class="metric-card">
                <h4>🎯 Overall Candidate Match Rate</h4>
                <h2>{avg_score:.1f}%</h2>
                <div class="progress-container">
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {avg_score}%; background: linear-gradient(90deg, #4CAF50, #8BC34A);"></div>
                    </div>
                </div>
                <small style="opacity: 0.8;">Average job-candidate fit across all applications</small>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card-secondary">
                <h4>🏆 Best Match Score</h4>
                <h2>{max_score:.1f}%</h2>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-label">Lowest Match</div>
                        <div class="stat-value">{min_score:.1f}%</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Median Match</div>
                        <div class="stat-value">{median_score:.1f}%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Candidate quality breakdown
            st.markdown(f"""
            <div class="metric-card-success">
                <h4>👥 Candidate Quality Distribution</h4>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-label">Top Tier (85%+)</div>
                        <div class="stat-value">{top_tier_candidates}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Strong (70-84%)</div>
                        <div class="stat-value">{strong_candidates}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Qualified (50-69%)</div>
                        <div class="stat-value">{qualified_candidates}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Underqualified (<50%)</div>
                        <div class="stat-value">{underqualified_candidates}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Shortlisting recommendations
            st.markdown(f"""
            <div class="metric-card-warning">
                <h4>📋 Shortlisting Recommendations</h4>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-label">Interview Ready</div>
                        <div class="stat-value">{shortlist_ready}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Needs Review</div>
                        <div class="stat-value">{needs_review}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Likely Reject</div>
                        <div class="stat-value">{rejection_candidates}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Total Pool</div>
                        <div class="stat-value">{total_candidates}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Skills analysis
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);">
                <h4>🔍 Skills Gap Analysis</h4>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-label">Unique Skills Found</div>
                        <div class="stat-value">{unique_skills_found}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Avg Skills/Candidate</div>
                        <div class="stat-value">{avg_skills_per_candidate:.1f}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Most common skills display
            if most_common_skills:
                st.markdown("**🔥 Most In-Demand Skills in Pool:**")
                for skill, count in most_common_skills:
                    percentage = (count / total_candidates) * 100
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.1); padding: 0.5rem; border-radius: 0.5rem; margin: 0.25rem 0;">
                        <span style="font-weight: bold;">{skill}</span>
                        <span style="float: right; opacity: 0.8;">{count} candidates ({percentage:.0f}%)</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Additional recruitment metrics
            st.markdown("---")
            st.subheader("📊 Recruitment Insights")
            
            col_stats1, col_stats2 = st.columns(2)
            
            with col_stats1:
                st.metric("Match Consistency", f"{std_score:.2f}", 
                         help="Lower values indicate more consistent candidate quality")
                st.metric("Total Applications", total_candidates)
                st.metric("Shortlist Rate", f"{(shortlist_ready/total_candidates*100):.1f}%",
                         help="Percentage of candidates ready for interview")
                
            with col_stats2:
                st.metric("Quality Range", f"{max_score - min_score:.1f}",
                         help="Spread between best and worst matches")
                st.metric("Skilled Candidates", f"{len(df[df['Skills Detected'] != 'No skills detected'])}",
                         help="Candidates with detectable technical skills")
                st.metric("Review Rate", f"{(needs_review/total_candidates*100):.1f}%",
                         help="Percentage needing manual review")
            
            # Export functionality
            st.markdown("---")
            st.subheader("💾 Export Results")
            
            if st.button("📥 Download Candidate Report", use_container_width=True):
                csv_data = screener.export_to_csv(df)
                st.download_button(
                    label="📄 Download CSV Report",
                    data=csv_data,
                    file_name="candidate_screening_report.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        else:
            st.info("📋 Upload resumes and job description to see recruitment analytics")
            
            # Show placeholder metrics for better UX
            st.markdown("""
            <div class="metric-card" style="opacity: 0.6;">
                <h4>🎯 Overall Candidate Match Rate</h4>
                <h2>--</h2>
                <div class="progress-container">
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: 0%; background: linear-gradient(90deg, #4CAF50, #8BC34A);"></div>
                    </div>
                </div>
                <small style="opacity: 0.8;">Average job-candidate fit across all applications</small>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="metric-card-secondary" style="opacity: 0.6;">
                <h4>🏆 Best Match Score</h4>
                <h2>--</h2>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-label">Lowest Match</div>
                        <div class="stat-value">--</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Median Match</div>
                        <div class="stat-value">--</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with ❤️ using Streamlit, Sentence Transformers, and PyMuPDF</p>
        <p>AI-powered resume screening for better hiring decisions</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 