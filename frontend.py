import streamlit as st
import pickle
import pandas as pd
import plotly.express as px

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="AI Email Intelligence",
    page_icon="📧",
    layout="wide"
)

# ----------------------------
# Modern UI Styling
# ----------------------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #eef2ff, #e0f7fa);
}

/* Glass Card */
.card {
    background: rgba(255, 255, 255, 0.85);
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.08);
    backdrop-filter: blur(6px);
    margin-bottom: 20px;
}

/* KPI Cards */
.kpi1 {background: linear-gradient(135deg,#4e73df,#224abe); color:white;}
.kpi2 {background: linear-gradient(135deg,#1cc88a,#13855c); color:white;}
.kpi3 {background: linear-gradient(135deg,#f6c23e,#dda20a); color:white;}

.kpi-card {
    padding: 25px;
    border-radius: 16px;
    text-align: center;
    font-weight: 600;
    box-shadow: 0 6px 20px rgba(0,0,0,0.12);
}

/* Titles */
.section-title {
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 15px;
    color: #333;
}

.sidebar .sidebar-content {
    background: linear-gradient(180deg,#4e73df,#224abe);
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# Load Models
# ----------------------------
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
cat_model = pickle.load(open("category_model.pkl", "rb"))
urg_model = pickle.load(open("urgency_model.pkl", "rb"))

# ----------------------------
# Session Storage
# ----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------
# Sidebar Navigation
# ----------------------------
st.sidebar.title("📊 AI Email System")
page = st.sidebar.radio("Navigate", ["📨 Email Classifier", "📈 Analytics Dashboard"])

# =====================================================
# EMAIL CLASSIFIER PAGE
# =====================================================
if page == "📨 Email Classifier":

    st.title("📧 AI Email Classification System")
    st.write("Smart categorization and urgency detection powered by NLP.")

    col1, col2 = st.columns([2,1])

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Enter Email Details</div>', unsafe_allow_html=True)

        email_id = st.text_input("Email ID")
        subject = st.text_input("Subject")
        body = st.text_area("Body", height=180)

        if st.button("🚀 Analyze Email"):

            if subject.strip()=="" or body.strip()=="":
                st.warning("Please enter subject and body.")
            else:
                full_text = subject + " " + body
                email_vec = vectorizer.transform([full_text])

                category = cat_model.predict(email_vec)[0]
                urgency = urg_model.predict(email_vec)[0]

                cat_conf = max(cat_model.predict_proba(email_vec)[0])*100
                urg_conf = max(urg_model.predict_proba(email_vec)[0])*100

                st.session_state.history.append({
                    "Email ID": email_id,
                    "Category": category,
                    "Urgency": urgency,
                    "Category Confidence": round(cat_conf,2),
                    "Urgency Confidence": round(urg_conf,2)
                })

                st.success("Email Classified Successfully!")

                st.subheader("Prediction Results")

                st.metric("Category", category, f"{cat_conf:.2f}% confidence")
                st.metric("Urgency", urgency, f"{urg_conf:.2f}% confidence")

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Classification Guide</div>', unsafe_allow_html=True)

        st.markdown("""
**📌 Categories**
- Complaint  
- Request  
- Feedback  
- Spam  

**🚦 Urgency Levels**
- High  
- Medium  
- Low  
""")

        st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# ANALYTICS DASHBOARD PAGE
# =====================================================
if page == "📈 Analytics Dashboard":

    st.title("📊 Email Analytics Dashboard")

    if len(st.session_state.history)==0:
        st.info("No emails classified yet.")
    else:
        df = pd.DataFrame(st.session_state.history)

        # KPI Cards
        col1, col2, col3 = st.columns(3)

        col1.markdown(f"""
        <div class="kpi-card kpi1">
            Total Emails<br><h2>{len(df)}</h2>
        </div>
        """, unsafe_allow_html=True)

        col2.markdown(f"""
        <div class="kpi-card kpi2">
            High Priority<br><h2>{len(df[df['Urgency']=='High'])}</h2>
        </div>
        """, unsafe_allow_html=True)

        col3.markdown(f"""
        <div class="kpi-card kpi3">
            Spam Emails<br><h2>{len(df[df['Category']=='Spam'])}</h2>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        col4, col5 = st.columns(2)

        with col4:
            fig1 = px.pie(df, names="Category", title="Category Distribution",
                          color_discrete_sequence=px.colors.qualitative.Bold)
            st.plotly_chart(fig1, use_container_width=True)

        with col5:
            fig2 = px.bar(df, x="Urgency", color="Urgency",
                          title="Urgency Distribution",
                          color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        st.subheader("📋 Classification History")
        st.dataframe(df, use_container_width=True)