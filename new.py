import  streamlit as st
import numpy as np
import joblib  
import re 

model=joblib.load(open(r'C:\Users\Admin\Desktop\encoders\best_model_catboost.pkl','rb'))
features=[
    "What is your current academic level?",
            'What is your current CGPA?',
            'Are you aware of your university’s career services?',
            'How many internships have you completed during your studies?',
            'Which types of internships have you completed during your studies, and how many for each type?  [Virtual/Remote Internship]',
            'Which types of internships have you completed during your studies, and how many for each type?  [Industry/Corporate Internship]',
            'Which types of internships have you completed during your studies, and how many for each type?  [Government Internship]',
            'How many hours per week do you spend on extracurricular ( Non-academic / Supplementary )activities?',
            'On average, how many hours per week do you spend on self-learning (such as online courses, tutorials, or independent study)?',
            'Have you attended any career-related workshops or training sessions?',
            'Do you have a part-time job while studying? If yes, how many hours per week do you work?',
            'On a scale of 0 to 5, how confident are you in your ability to secure a job after graduation, considering your skills, experience, and job market conditions?',
            'In the past 6 months, how many job applications have you submitted?',
            'Have you received any job offers before graduation?',
            'On a scale of 1-5, how important do you think networking is in securing a job?',
            'How many certificates have you achieved so far?'
]


cat_features=[
    'What is your gender?',
    'What is your age?',
    'What is your current academic level?',
    'Are you aware of your university’s career services?',
    'What is your department?',
    'How many internships have you completed during your studies?',
    'On average, how many hours per week do you spend on self-learning (such as online courses, tutorials, or independent study)?',
    'Have you attended any career-related workshops or training sessions?',
    'Do you have a part-time job while studying? If yes, how many hours per week do you work?',
    'In the past 6 months, how many job applications have you submitted?',
    'How long do you expect it will take to secure a job after graduation?',
    'Have you received any job offers before graduation?',
    'What is your expected starting salary after graduation? Please enter the amount in USD ($).',
    'To what extent do you think your university’s career services have helped you prepare for the job market?',
    'Which of the following skills do you think employers value the most in your field?',
    'How would you rate the amount of hands-on training provided by your university?',
    'How many certificates have you achieved so far?',
    'Which of the following career paths do you prefer?',
    'Reflecting on your studies, which technical skills do you feel most comfortable using or applying?',
    'Which elective course have you found most helpful for your career preparation and readiness?',
    'Have you taken any courses outside of the university that you found particularly impactful that should be added to the university curriculum?',
    'Which professional or technical skills do you feel are missing from your university education?',
    'What specific improvements would you like to see in your university’s career preparation programs?'   
]
def regex(name):
    return re.sub(r'[<>:"/\|?*]', '_', name)


def load_encoder(feature):
    feature_name=regex(feature)
    return joblib.load(open(f'C:/Users/Admin/Desktop/encoders/{feature_name}_encoder.pkl','rb'))

def main ():
    st.set_page_config(layout='wide')
    st.title('Career Readiness Predictions')
    inputs={}
    for feature in features:
        if feature in cat_features:
            enc=load_encoder(feature)
            if hasattr(enc,'classes_'):
                options=enc.classes_.tolist()
                inputs[feature]=st.selectbox(feature,options)
            elif hasattr(enc,'categories_'):
                options=enc.categories_.tolist()
                inputs[feature]=st.selectbox(feature,options)
        else : 
            inputs[feature]=st.text_input(feature)
    

    if st.button('hit me'):
        new_input=[]
        for feature in features:
            value=inputs[feature]
            if feature in cat_features:
                enc=load_encoder(feature)
                value=enc.transform(np.array([[value]]))[0]
            new_input.append(value)    
        new_input=np.array(new_input,dtype='object').reshape(1,-1)    
        y_pred=model.predict(new_input)
        if y_pred ==1:
            st.success('Your so Ready Keep Going')
        else :
            st.error('You are not Enough , You are not Ready')    


if __name__=='__main__':
    main()




            



