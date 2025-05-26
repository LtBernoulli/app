import streamlit as st
import pandas as pd
import plotly.express as px
import sqlite3  # Assuming you're using SQLite for your database

# Establish database connection
connection = sqlite3.connect('your_database.db')  # Replace with your actual database name
cursor = connection.cursor()

# Function to create a download link for the dataframe
def get_table_download_link(df, filename, text):
    """Generates a link allowing the data in a given pandas dataframe to be downloaded"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def run():
    ad_user = st.text_input("Username")
    ad_password = st.text_input("Password", type='password')
    if st.button('Login'):
        if ad_user == 'briit' and ad_password == 'briit123':
            st.success("Welcome Dr Briit !")
            # Display Data
            cursor.execute('''SELECT * FROM user_data''')
            data = cursor.fetchall()
            st.header("**User's Data**")
            df = pd.DataFrame(data, columns=['ID', 'Name', 'Email', 'Resume Score', 'Timestamp', 'Total Page',
                                             'Predicted Field', 'User Level', 'Actual Skills', 'Recommended Skills',
                                             'Recommended Course'])
            st.dataframe(df)
            st.markdown(get_table_download_link(df,'User_Data.csv','Download Report'), unsafe_allow_html=True)
            ## Admin Side Data
            query = 'select * from user_data;'
            plot_data = pd.read_sql(query, connection)

            ## Pie chart for predicted field recommendations
            labels = plot_data.Predicted_Field.unique()
            print(labels)
            values = plot_data.Predicted_Field.value_counts()
            print(values)
            st.subheader("**Pie-Chart for Predicted Field Recommendation**")
            fig = px.pie(df, values=values, names=labels, title='Predicted Field according to the Skills')
            st.plotly_chart(fig)

            ### Pie chart for User's👨‍💻 Experienced Level
            labels = plot_data.User_level.unique()
            values = plot_data.User_level.value_counts()
            st.subheader("**Pie-Chart for User's Experienced Level**")
            fig = px.pie(df, values=values, names=labels, title="Pie-Chart📈 for User's👨‍💻 Experienced Level")
            st.plotly_chart(fig)
        else:
            st.error("Wrong ID & Password Provided")

# Remove the following line as it's a terminal command, not Python code
# streamlit run C:\Users\CSC\anaconda3\Lib\site-packages\ipykernel_launcher.py [ARGUMENTS]
# Correct way to execute another Python script from inside your script
 