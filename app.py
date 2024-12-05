import json
import streamlit as st
import pandas as pd
from pycaret.clustering import load_model, predict_model  # type: ignore
import plotly.express as px  # type: ignore
import plotly.graph_objects as go

st.set_page_config(layout="wide")

MODEL_NAME = 'welcome_survey_clustering_pipeline_v2'

DATA = 'welcome_survey_simple_v2.csv'

CLUSTER_NAMES_AND_DESCRIPTIONS = 'welcome_survey_cluster_names_and_descriptions_v2.json'

@st.cache_data
def get_model():
    return load_model(MODEL_NAME)

@st.cache_data
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding='utf-8') as f:
        return json.loads(f.read())

@st.cache_data
def get_all_participants():
    model = get_model()
    all_df = pd.read_csv(DATA, sep=';')
    df_with_clusters = predict_model(model, data=all_df)

    return df_with_clusters

with st.sidebar:
    st.header("Powiedź nam coś o sobie")
    st.markdown("Pomożemy Ci znaleźć osoby, które mają podobne zainteresowania")
    age = st.selectbox("Wiek", ['>18', '25-34', '45-54', '35-44', '>=65', '55-64', 'unknown'])
    edu_level = st.selectbox('wykształcenie', ['Podstawowe', 'Średnie', 'Wyższe'])
    fav_animals = st.selectbox('Ulubione zwierzęta', ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'])
    fav_place = st.selectbox('Ulubione miejsce', ['Nad wodą', 'W lesie', 'W górach', 'Inne'])
    gender = st.radio('Płeć', ['Mężczyzna', 'Kobieta', 'Wszyscy'])

    person_df = pd.DataFrame([
        {
            'age': age,
            'edu_level': edu_level,
            'fav_animals': fav_animals,
            'fav_place': fav_place,
            'gender': gender
        }
    ])

model = get_model()
all_df = get_all_participants()
cluster_names_and_descriptions = get_cluster_names_and_descriptions()

predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]
predicted_cluster_data = cluster_names_and_descriptions[predicted_cluster_id]

st.markdown(
    f"""
    <h1 style="text-align: center;">
        Najbliżej Ci do grupy: <br> {predicted_cluster_data['name']}
    </h1>
    """, unsafe_allow_html=True
)

st.markdown("<br><br>", unsafe_allow_html=True)

c0, c1 = st.columns([2, 1])
with c0:
    st.markdown(
    f"""
    <div style="font-size: 30px;"><br><br>
        {predicted_cluster_data['description']}
    </div>
    """, unsafe_allow_html=True
)
with c1:
    cluster_name = predicted_cluster_data['name']
    safe_cluster_name = cluster_name.lower().replace(' ', '_') \
        .replace('ó', 'o') \
        .replace('ą', 'a') \
        .replace('ę', 'e') \
        .replace('ś', 's') \
        .replace('ć', 'c') \
        .replace('ń', 'n') \
        .replace('ł', 'l') \
        .replace('ż', 'z') \
        .replace('ź', 'z') \
        .replace('ü', 'u') 
    
    image_url = f"https://raw.githubusercontent.com/agaho8/zad_dom_modul_7/main/{safe_cluster_name}_image.webp"
    
    # Wyświetlenie obrazu w kolumnie
    st.image(image_url, use_container_width=True)

st.markdown("<br><br>", unsafe_allow_html=True)

same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]
filtered_cluster_df = same_cluster_df[same_cluster_df["gender"] == gender]
if gender != "Wszyscy":
    filtered_cluster_df = same_cluster_df[same_cluster_df["gender"] == gender]
else:
    filtered_cluster_df = same_cluster_df

male_count = len(same_cluster_df[same_cluster_df["gender"] == "Mężczyzna"])
female_count = len(same_cluster_df[same_cluster_df["gender"] == "Kobieta"])
total_count = len(same_cluster_df)

st.markdown(
    """
    <div style="text-align: center; font-size: 24px;">
        <strong>Liczba Twoich znajomych</strong><br>
        <span style="font-size: 32px;">{}</span>
    </div>
    """.format(total_count),
    unsafe_allow_html=True
)
col1, col2 = st.columns(2)
with col1:
    st.markdown(
        """
        <div style="text-align: center;">
            <strong>Mężczyźni</strong><br>
            <span style="font-size: 24px;">{}</span>
        </div>
        """.format(male_count),
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        """
        <div style="text-align: center;">
            <strong>Kobiety</strong><br>
            <span style="font-size: 24px;">{}</span>
        </div>
        """.format(female_count),
        unsafe_allow_html=True
    )

same_cluster_df['gender'] = same_cluster_df['gender'].cat.add_categories('Nie wskazano płci')
same_cluster_df['gender'] = same_cluster_df['gender'].fillna('Nie wskazano płci')

st.header("Osoby z grupy")

gender_age_df = same_cluster_df.groupby(['age', 'gender']).size().reset_index(name='count')
# Filtrowanie danych w zależności od wyboru płci
if gender != "Wszyscy":
    filtered_df_age = gender_age_df[gender_age_df['gender'] == gender]
else:
    filtered_df_age = gender_age_df  # Pokaż wszystkie dane, jeśli wybrano "Wszyscy"

# Tworzymy wykres kolumnowy
fig = px.bar(filtered_df_age, x='age', y='count', color='age',
             title="Rozkład wieku w grupie",
             labels={"age": "Wiek", "count": "Liczba osób"},
             color_discrete_sequence=["#1f77b4", "#ff7f0e", "#2ca02c"])  # Kolory

if gender == 'Wszyscy':
    # Linia dla mężczyzn
    fig.add_trace(go.Scatter(
        x=gender_age_df[gender_age_df['gender'] == 'Mężczyzna']['age'],
        y=gender_age_df[gender_age_df['gender'] == 'Mężczyzna']['count'],
        mode='lines+markers', name="Mężczyźni", line=dict(color='blue')
    ))

    # Linia dla kobiet
    fig.add_trace(go.Scatter(
        x=gender_age_df[gender_age_df['gender'] == 'Kobieta']['age'],
        y=gender_age_df[gender_age_df['gender'] == 'Kobieta']['count'],
        mode='lines+markers', name="Kobiety", line=dict(color='orange')
    ))

# Dostosowanie wykresu
fig.update_layout(
    showlegend=True if gender == "Wszyscy" else False,  # Pokazuje legendę tylko dla "Wszyscy"
    legend_title="Wiek",  # Ustawienie tytułu legendy
    legend=dict(
        traceorder="normal",
        itemsizing='constant',  # Skaluje elementy legendy
        title="Płeć",  # Tylko 'Płeć' w tytule legendy
        itemclick="toggleothers",  # Pozwala na klikanie, by wyświetlić tylko wybraną płeć
    )
)
# Usunięcie legendy zwierząt i pokazanie tylko legendy płci
fig.for_each_trace(lambda trace: trace.update(showlegend=False) if trace.name not in ['Mężczyźni', 'Kobiety'] else ())

st.plotly_chart(fig)

import plotly.express as px

# Grupowanie danych po 'edu_level' (wykształcenie) i 'gender' (płeć)
gender_education_df = filtered_cluster_df.groupby(['edu_level', 'gender']).size().reset_index(name='count')

# Tworzymy wykres słupkowy, z podziałem na płeć w ramach wykształcenia
fig = px.bar(gender_education_df, 
             x='edu_level',  # Na osi X wykształcenie
             y='count',  # Liczba osób
             color='gender',  # Kolorowanie według płci
             title="Rozkład wykształcenia i płci w grupie",
             labels={"edu_level": "Wykształcenie", "count": "Liczba osób", "gender": "Płeć"},
             barmode='stack')  # Wykorzystanie barmode='stack', by słupki były nakładające się

# Dostosowanie wykresu
fig.update_layout(
    xaxis_title="Wykształcenie",  # Tytuł osi X
    yaxis_title="Liczba osób",  # Tytuł osi Y
    showlegend=True  # Pokażemy legendę, aby wskazać płeć
)

# Wyświetlanie wykresu
st.plotly_chart(fig)


gender_animal_df = same_cluster_df.groupby(['fav_animals', 'gender']).size().reset_index(name='count')

# Filtrowanie danych w zależności od wyboru płci
if gender != "Wszyscy":
    filtered_df = gender_animal_df[gender_animal_df['gender'] == gender]
else:
    filtered_df = gender_animal_df  # Pokaż wszystkie dane, jeśli wybrano "Wszyscy"

# Tworzymy wykres kolumnowy
fig = px.bar(filtered_df, x='fav_animals', y='count', color='fav_animals',
             title="Rozkład ulubionych zwierząt w grupie",
             labels={"fav_animals": "Ulubione zwierzęta", "count": "Liczba osób"},
             color_discrete_sequence=["#1f77b4", "#ff7f0e", "#2ca02c"])  # Kolory

# Jeśli wybrano "Wszyscy", dodajemy linie
if gender == "Wszyscy":
    # Linia dla mężczyzn
    fig.add_trace(go.Scatter(
        x=gender_animal_df[gender_animal_df['gender'] == 'Mężczyzna']['fav_animals'],
        y=gender_animal_df[gender_animal_df['gender'] == 'Mężczyzna']['count'],
        mode='lines+markers', name="Mężczyźni", line=dict(color='blue')
    ))

    # Linia dla kobiet
    fig.add_trace(go.Scatter(
        x=gender_animal_df[gender_animal_df['gender'] == 'Kobieta']['fav_animals'],
        y=gender_animal_df[gender_animal_df['gender'] == 'Kobieta']['count'],
        mode='lines+markers', name="Kobiety", line=dict(color='orange')
    ))

# Dostosowanie wykresu
fig.update_layout(
    showlegend=True if gender == "Wszyscy" else False,  # Pokazuje legendę tylko dla "Wszyscy"
    legend_title="Płeć",  # Ustawienie tytułu legendy
    legend=dict(
        traceorder="normal",
        itemsizing='constant',  # Skaluje elementy legendy
        title="Płeć",  # Tylko 'Płeć' w tytule legendy
        itemclick="toggleothers",  # Pozwala na klikanie, by wyświetlić tylko wybraną płeć
    )
)

# Usunięcie legendy zwierząt i pokazanie tylko legendy płci
fig.for_each_trace(lambda trace: trace.update(showlegend=False) if trace.name not in ['Mężczyźni', 'Kobiety'] else ())


# Wyświetlenie wykresu
st.plotly_chart(fig)

# Dodajemy kategorię 'Nie wskazano miejsca' do kolumny, jeśli jest typu kategorycznego
if filtered_cluster_df['fav_place'].dtype.name == 'category':
    filtered_cluster_df['fav_place'] = filtered_cluster_df['fav_place'].cat.add_categories('Nie wskazano miejsca')

# Zastępujemy null na 'Nie wskazano miejsca'
filtered_cluster_df['fav_place'] = filtered_cluster_df['fav_place'].fillna('Nie wskazano miejsca')

# Tworzymy wykres kołowy
fig = px.pie(filtered_cluster_df, names="fav_place", title="Rozkład ulubionych miejsc w grupie", 
             labels={"fav_place": "Ulubione miejsce"},)  

# Wyświetlanie wykresu
st.plotly_chart(fig)


fig = px.pie(same_cluster_df, names="gender", title="Rozkład płci w grupie", 
             labels={"gender": "Płeć"}, 
             hole=0.3)  # hole=0.3 tworzy wykres kołowy z dziurą w środku (jak wykres donut)

fig.update_traces(textinfo="percent+label", pull=[0.1, 0.1])  # Procenty i etykiety na wykresie

st.plotly_chart(fig)