import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt



def main():
    page_style()
    prediction_user(user_in())
    importsnce_plot()


def page_style():
    # image = Image.open('avia_model.jpeg')
    # icon_image = Image.open('icon_avia.jpg')
    #
    # st.set_page_config(
    #     layout="wide",
    #     initial_sidebar_state="auto",
    #     page_title="avia_clients_model",
    #     page_icon=icon_image,
    # )

    st.title('_:blue[Предсказание удовлетворенности клиента авиакомпании полетом.]_')

    # st.image(image)


def user_in():
    st.write("## Выберите параметры, по которым будет строиться предсказание")

    customer_type = st.radio('Тип клиента (лоялен ли компании)', ['Лояльный', 'Нелояльный'])
    type_of_travel = st.radio('Тип поездки', ['Бизнес-поездка', 'Частная поездка'])
    class_type = st.radio('Класс обслуживания', ['Эконом', 'Эконом-плюс', 'Бизнес'])
    departure_delay = st.slider('Задержка отправления рейса в минутах', 0, 100, 0)
    arrival_delay = st.slider('Задержка прибытия рейса в минутах', 0, 100, 0)
    inflight_wifi = st.slider('Удовлетворенность услугами wifi на борту (где 1 - это отсуствие данной услуги, 5 - отличный сервис)', 1, 5, 0)
    online_boarding = st.slider('Удовлетворенность услугами онлайн-регистрации на рейс (где 1 - это отсуствие данной услуги, 5 - отличный сервис)', 1, 5, 0)

    translatetion = {
        'Лояльный': 'Loyal Customer',
        'Нелояльный': 'disloyal Customer',
        'Бизнес-поездка': 'Business travel',
        'Частная поездка': 'Personal Travel',
        'Эконом': 'Eco',
        'Эконом-плюс': 'Eco Plus',
        'Бизнес': 'Business'
    }

    user_input = pd.DataFrame({
        'Customer Type': [translatetion[customer_type]],
        'Type of Travel': [translatetion[type_of_travel]],
        'Class': [translatetion[class_type]],
        'Departure Delay in Minutes': [departure_delay],
        'Arrival Delay in Minutes': [arrival_delay],
        'Inflight wifi service': [inflight_wifi],
        'Online boarding': [online_boarding]
    })

    return user_input


def prediction_user(user_input):
    with open('model.pickle', 'rb') as f_1:
        model = pickle.load(f_1)
    with open('ohe.pickle', 'rb') as f_2:
        ohe = pickle.load(f_2)
    with open('scaler.pickle', 'rb') as f_3:
        scaler = pickle.load(f_3)

    categorical_features = ['Customer Type', 'Type of Travel', 'Class']
    numeric_features = ['Departure Delay in Minutes', 'Arrival Delay in Minutes', 'Inflight wifi service', 'Online boarding']

    input_data_ohe = pd.DataFrame(ohe.transform(user_input[categorical_features]).toarray(),
                                  columns=ohe.get_feature_names_out(categorical_features))
    input_data_scaled = pd.DataFrame(scaler.transform(user_input[numeric_features]), columns=numeric_features)

    input_data_processed = pd.concat([input_data_ohe, input_data_scaled], axis=1)

    prediction = model.predict(input_data_processed)[0]
    prediction_proba = model.predict_proba(input_data_processed)

    st.subheader('Предсказание:')

    if prediction == 0:
        prob_percent = prediction_proba[0][0] * 100
        st.subheader(f'Клиент будет неудовлетворен полетом c вероятностью: {round(prob_percent)}%')
    else:
        prob_percent = prediction_proba[0][1] * 100
        st.subheader(f'Клиент будет удовлетворен полетом c вероятностью: {round(prob_percent)}%')


def importsnce_plot():
    with open('model.pickle', 'rb') as f:
        model = pickle.load(f)

    importance = model.feature_importances_
    importance = np.delete(importance, np.where(importance < 0.0005))

    fig, ax = plt.subplots(figsize=(8, 5), dpi=80)
    ax.bar(range(len(importance)), importance * 100)
    ax.set_xticks(range(len(importance)))
    ax.set_xticklabels(['Тип клиента_Нелояльный', 'Тип поездки_Частная поездка',
       'Класс_Эконом', 'Класс_Эконом-плюс',
       'Задержка отправления в минутах', 'Задержка прибытия в минутах',
       'WiFi сервис', 'Онлайн-регистрация'], rotation='vertical', fontsize=8)
    ax.set_ylabel('Важность признака в процентах')
    ax.set_title('График важности признаков', fontsize=12)
    st.pyplot(fig)



if __name__ == '__main__':
    main()
