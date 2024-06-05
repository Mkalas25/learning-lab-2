
import requests
from streamlit_lottie import st_lottie
import streamlit as st
import pandas as pd
from datetime import datetime


st.set_page_config(page_title="Welcome to Camaco Learning Lab")

# 1.to add gif
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# Function to check login credentials
def check_login(username, password):
    # Dictionary storing username: password pairs
    user_database = {
        "Jparent": "a1234",
        "Mkalas": "1234",
        "Gopal": "Password1",
        "Jpais": "Pass@123",
        "asdf" : "1234" 
    }
    # Check if the username exists in the dictionary and if the password matches
    return user_database.get(username) == password

# Function to show the login page
def show_login_page():
    lottie_coding3 = load_lottieurl("https://lottie.host/278504bd-a8e6-4aa1-9659-429473146744/FbR81V5ilh.json")
    st_lottie(lottie_coding3,height=250,width=700,key="intern")
    st.image("logo2.png")
   
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if check_login(username, password):
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.rerun()
        else:
            st.error("Invalid username or password")
  

# Function to show the home page after login
def show_home_page():
    lottie_coding4 = load_lottieurl("https://lottie.host/a75942d5-bf23-4cc1-a6b1-dbd0749b605d/xCKJuDroS7.json")
    st_lottie(lottie_coding4,height=250,width=700,key="home1")

    # Access the username from session state
    username = st.session_state.get('username', 'Guest')
    st.title(f"Hi {username}")
    st.write(f"Welcome, {username}! Welcome to Camaco Learning Lab")
    st.title("courses 📚")
    st.write("Please select a course:")

    if st.button("Course 1-GD&T-Geometric dimensioning and tolerancing"):
        st.session_state.page = "course_1"
        st.session_state.history.append("home")
        st.rerun()      
    if st.button("Course 2-Bending Guidelines"):
        st.session_state.page = "course_2"
        st.session_state.history.append("home")
        st.rerun()

    # Add a horizontal line using HTML
    st.markdown("---")
    st.title("Tests and Evaluavtion 📄")
    if st.button("Test"):
        st.session_state.page = "test"
        st.session_state.history.append("home")
        st.rerun()

    # Add a horizontal line using HTML
    st.markdown("---")
    st.title("Feedback 📌")
    if st.button("Feedback"):
        st.session_state.page = "Feedback"
        st.session_state.history.append("home")
        st.rerun()
    # Add a horizontal line using HTML
    st.markdown("---")

def chatbot():
    import streamlit as st
    from PyPDF2 import PdfReader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    import os
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.chains.question_answering import load_qa_chain
    from langchain.prompts import PromptTemplate
    from dotenv import load_dotenv
    from langchain_community.vectorstores import FAISS
    from langchain.document_loaders import PyPDFLoader


    load_dotenv()
    os.getenv("GOOGLE_API_KEY")

    def get_pdf_text():
        path = "weld.pdf"
        loader = PyPDFLoader(path)
        text = loader.load()
        text="\n".join([page.page_content for page in text])
        return text



    def get_text_chunks(text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
        return chunks


    def get_vector_store(text_chunks):
        embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")


    def get_conversational_chain():

        prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """

        model = ChatGoogleGenerativeAI(model="gemini-pro",
                                temperature=0.3)

        prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

        return chain



    def user_input(user_question):
        embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain()

        
        response = chain(
            {"input_documents":docs, "question": user_question}
            , return_only_outputs=True)

        print(response)
        st.write("Reply: ", response["output_text"])

    def main():
        st.header("Get you answers...🙋‍♂️")

        user_question = st.text_input("Ask a Question...")

        if user_question:
            user_input(user_question)
            raw_text = get_pdf_text()
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)

    if __name__ == "__main__":
        main()

# Functions to show course pages
def show_course_1_page():
    st.title("Welcome to Course 1")
    st.title("GD&T-Geometric dimensioning and tolerancing")
    scenes = [
    ("""Geometric dimensioning and tolerancing (GD&T) is a system for defining and communicating engineering tolerances via a symbolic language on 
    engineering drawings and computer-generated 3D models that describes a physical object's nominal geometry and the permissible variation thereof. 
    GD&T is used to define the nominal (theoretically perfect) geometry of parts and assemblies, the allowable variation in size, form, orientation,
    and location of individual features, and how features may vary in relation to one another such that a component is considered satisfactory for its intended use..""", "rule1.png"),
    ("""GD&T Flatness is very straight forward. It is a common symbol that references how flat a surface is regardless of any other datums or features. 
    It comes in useful if a feature is to be defined on a drawing that needs to be uniformly flat without tightening any other dimensions on the drawing. 
    The flatness tolerance references two parallel planes (parallel to the surface that it is called out on) that define a zone where the entire reference surface must lie. 
    Flatness tolerance is always less than the dimensional tolerance associated with it.""", ["flat1.png","flat2.png"]),
    ("""Description:
Parallelism is a fairly common symbol that describes a parallel orientation of one referenced feature to a datum surface or line. 
It can reference a 2D line referenced to another element, but more commonly it relates the orientation of one surface plane parallel to another datum plane in a 3-Dimensional tolerance zone.
The tolerance indirectly controls the 0° angle between the parts by controlling where the surface can lie based on the datum. See the tolerance zone below for more details.
Note: Parallelism does not control the angle of the referenced feature, but only creates an envelope in which the feature must lie.
It is important to determine what the reference feature is (surface or axis) and then what is acting as the datum (surface or axis) to determine how the parallelism is to be controlled.""", "pall.png"),
    ("""True Position, or just Position as the ASME Y14.5 standard calls it, is defined as the total permissible variation that a feature can have from its “true” position. The “True Position” is the exact coordinate,
    or location defined by basic dimensions or other means that represents the nominal value. In other words, the Geometric Dimensioning and Tolerancing “Position” tolerance is how far your feature’s location can vary 
    from its “True Position”.""",["loc.png","pos.png"]),
    ("""Rule 5 case 1- Non-Clam shell design: Maximum allowable gap at the weld joint by tolerance analysis:
   For > 1.4mm thick parts:
     ≤0.8mm gap by tolerance analysis.
      For < 1.4mm thick parts:
       ≤0.5mm gap by tolerance analysis. 
 For > 1.4mm thick parts:
joint to be designed with touch condition.
Max 0.4 trimline / surface profile tolerance on each part or max total 0.8 mm (both parts) trimline / surface profile tolerance.
For < 1.4mm thick parts:
Joint to be design with touch condition.
Max 0.25 trimline / surface profile tolerance on each part or max total 0.5 mm (both parts) trimline / surface profile tolerance.""",["rule 5 case 1.1.png","rule 5 case 1.2.png"]),
("""Rule 5 case 2- Clam shell design: 1.Joint design with gap is required for clam shell joints for ease of assembly.
Min 0.1mm gap / per side to be protected for the joint for ease of assembly. 

For > 1.4mm thick parts:
0.45 mm Nominal Gap per side.
Max 0.35 trimline / surface profile tolerance on each part or max total 0.7 mm (both parts) trimline / surface profile tolerance.
For < 1.4mm thick parts:
0.3 mm Nominal Gap per side.
Max 0.2 trimline / surface profile tolerance on each part or max total 0.4 mm (both parts) trimline / surface profile tolerance.
2. When ever possible Design with 2 L bracket.

For > 1.4mm thick parts:
Joint to be designed with touch condition.
Max 0.4 trimline / surface profile tolerance on each part or max total 0.8 mm (both parts) trimline / surface profile tolerance.

For < 1.4mm thick parts:
Joint to be design with touch condition.
Max 0.25 trimline / surface profile tolerance on each part or max total 0.5 mm (both parts) trimline / surface profile tolerance.""",["rule 5 case 2.1.png","Rule 5_2.2.png"]),
("""Rule 5 case 3- Slip Joint design: If Max allowed weld gap can not be achieved by part design, then slip assembly joints to be designed to allow part motion during assembly process. Parts to be designed with 2 locating slots. Slots direction to be inline with the movement of the part.

For > 1.4mm thick parts:
Joint to be designed with touch condition.
Max 0.4 trimline / surface profile tolerance on each part or max total 0.8 mm (both parts) trimline / surface profile tolerance.

For < 1.4mm thick parts:
Joint to be design with touch condition.
Max 0.25 trimline / surface profile tolerance on each part or max total 0.5 mm (both parts) trimline / surface profile tolerance.""","rule 5 case 3.png"),
("""Rule 5 case 4- For Large parts: For parts where it is difficult to control the entire surface / trimline profile to required tolerance mentioned above,
For surface add local emboss / coning with profile tolerance mentioned above at the weld joint.
For trim edge, add local standoff with profile tolerance mentioned above at the weld joint.

For > 1.4mm thick parts:
≤0.8mm gap by tolerance analysis.
Joint to be designed with touch condition.
Max 0.4 trimline / surface profile tolerance on each part or max total 0.8 mm (both parts) trimline / surface profile tolerance.

For < 1.4mm thick parts:
≤0.5mm gap by tolerance analysis.
Joint to be design with touch condition.
Max 0.25 trimline / surface profile tolerance on each part or max total 0.5 mm (both parts) trimline / surface profile tolerance.""","rule 5 case 4.png")]

    for index, (text, image) in enumerate(scenes, start=1):
        with st.expander(f" {index}",expanded=True):
            st.markdown(text)
            st.image(image)

    st.title("Videos")
    st.video("vid1.mp4")
    st.video("vid2.mp4")

    # Define custom CSS for the button
    button_css = """
    <style>
    .stButton > button {
        background-color: #4CAF50; /* Green */
        color: white;
    }

    .stButton > button:hover {
        background-color: #45a049; /* Darker green */
    }
    </style>
    """

    # Inject CSS into the Streamlit app
    st.markdown(button_css, unsafe_allow_html=True)

    # Create a "Back" button
    if st.button("Back"):
        if 'history' in st.session_state and st.session_state.history:
            st.session_state.page = st.session_state.history.pop()
            st.rerun()

    chatbot()


def show_course_2_page():
    st.title("Course 2")
    st.write("Welcome to Course 2")
    # Define custom CSS for the button
    button_css = """
    <style>
    .stButton > button {
        background-color: #4CAF50; /* Green */
        color: white;
    }

    .stButton > button:hover {
        background-color: #45a049; /* Darker green */
    }
    </style>
    """

    # Inject CSS into the Streamlit app
    st.markdown(button_css, unsafe_allow_html=True)

    # Create a "Back" button
    if st.button("Back"):
        if 'history' in st.session_state and st.session_state.history:
            st.session_state.page = st.session_state.history.pop()
            st.rerun()

def test():
    # Define the questions and options
    questions = [
        "What is your favorite color?",
        "What is your favorite animal?",
        "What is your favorite food?",
        "What is your favorite hobby?",
        "What is your favorite season?"
    ]

    options = [
        ["Red", "Blue", "Green", "Yellow"],
        ["Cat", "Dog", "Bird", "Fish"],
        ["Pizza", "Burger", "Pasta", "Salad"],
        ["Reading", "Traveling", "Gaming", "Cooking"],
        ["Spring", "Summer", "Fall", "Winter"]
    ]

    # Streamlit app title
    st.title("Lets test your knowledge...")

    # Get the username
    username = st.session_state.get('username', 'Guest')

    # Dictionary to store the responses
    responses = {}

    # Loop through the questions and display them with radio buttons
    for i, question in enumerate(questions):
        responses[question] = st.radio(question, options[i])

    # Save the responses to a CSV file when the submit button is clicked
    if st.button("Submit"):
        # df=pd.DataFrame()
        # df=pd.read_csv("responses.csv")
        # username_list=df["Username"].to_list()
        # if username in username_list:
        #     st.error("Rsponse already exist 🚧")
        #     pass                
        # else:       
        if username:
            # Ensure no empty values
            if all(value for value in responses.values()):
                responses["Username"] = username
                responses["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                df = pd.DataFrame([responses])
                # Save to CSV
                csv_file = "responses.csv"

                right_answers=["Red","Dog","Burger","Reading","Winter"]
                answer_by_user=df.loc[0].to_list()[0:5]
                right_answer_count=0
                if right_answers[0] ==answer_by_user[0]:
                    right_answer_count=right_answer_count+1
                if right_answers[1] ==answer_by_user[1]:
                    right_answer_count=right_answer_count+1
                if right_answers[2] ==answer_by_user[2]:
                    right_answer_count=right_answer_count+1
                if right_answers[3] ==answer_by_user[3]:
                    right_answer_count=right_answer_count+1
                if right_answers[4] ==answer_by_user[4]:
                    right_answer_count=right_answer_count+1

                if right_answer_count>=3:
                    result="PASS"
                    # df["Result"]=[result]
                    st.success(f"{username}, You pass the test 🎯")
                else:
                    result="FAIL"
                    # df["Result"]=[result]
                    st.error(f"{username}, Better luck next time ⌛")
                    st.title("Relearn the course")                     
                
                
                # try:
                #     existing_df = pd.read_csv(csv_file)
                #     df = pd.concat([existing_df, df], ignore_index=True)
                    
                # except FileNotFoundError:
                #     pass
                
                # df.to_csv(csv_file, index=False)
                
                st.success("Responses saved successfully!")
            else:
                st.error("Please answer all the questions.")
        else:
            st.error("Please enter your username before submitting.")
    
    # Define custom CSS for the button
    button_css = """
    <style>
    .stButton > button {
        background-color: #4CAF50; /* Green */
        color: white;
    }

    .stButton > button:hover {
        background-color: #45a049; /* Darker green */
    }
    </style>
    """

    # Inject CSS into the Streamlit app
    st.markdown(button_css, unsafe_allow_html=True)

    # Create a "Back" button
    if st.button("Back"):
        if 'history' in st.session_state and st.session_state.history:
            st.session_state.page = st.session_state.history.pop()
            st.rerun()

def show_course_3_page():
    st.title("Feedback")
    st.write("Please provide your feedback:")

    # Define feedback options
    feedback_options = ["Excellent", "Good", "Average", "Poor"]
    feedback = st.radio("How would you rate the course?", feedback_options)

    # Submit button to submit feedback
    if st.button("Submit"):
        st.success(f"Thank you for your feedback! You rated the course as: {feedback}")
        st.success("Your feedback has been submitted successfully!")

    # Define custom CSS for the button
    button_css = """
    <style>
    .stButton > button {
        background-color: #4CAF50; /* Green */
        color: white;
    }

    .stButton > button:hover {
        background-color: #45a049; /* Darker green */
    }
    </style>
    """

    # Inject CSS into the Streamlit app
    st.markdown(button_css, unsafe_allow_html=True)

    # Create a "Back" button
    if st.button("Back"):
        if 'history' in st.session_state and st.session_state.history:
            st.session_state.page = st.session_state.history.pop()
            st.rerun()

# Main logic to control page flow
def main():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    
    if 'history' not in st.session_state:
        st.session_state.history = []

    if st.session_state['logged_in']:
        if 'page' not in st.session_state:
            st.session_state.page = 'home'

        if st.session_state.page == 'home':
            show_home_page()
        elif st.session_state.page == 'course_1':
            show_course_1_page()
        elif st.session_state.page == 'course_2':
            show_course_2_page()
        elif st.session_state.page == 'test':
            test()
        elif st.session_state.page == 'Feedback':
            show_course_3_page()
        # elif st.session_state.page == 'Return_course_1':
        #     show_course_1_page()
    else:
        show_login_page()

if __name__ == "__main__":
    main()



