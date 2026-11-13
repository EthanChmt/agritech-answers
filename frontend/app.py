"""Front-end Streamlit : aucune logique de Machine Learning, uniquement des appels à l'API FastAPI."""

import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT_S = 10

CROPS = {"Barley": "Orge", "Cotton": "Coton", "Maize": "Maïs", "Rice": "Riz", "Soybean": "Soja", "Wheat": "Blé"}
REGIONS = {"North": "Nord", "South": "Sud", "East": "Est", "West": "Ouest"}
SOILS = {
    "Chalky": "Crayeux", "Clay": "Argileux", "Loam": "Franc (loam)",
    "Peaty": "Tourbeux", "Sandy": "Sableux", "Silt": "Limoneux (silt)",
}
WEATHER = {"Sunny": "Ensoleillé", "Cloudy": "Nuageux", "Rainy": "Pluvieux"}

st.set_page_config(page_title="Rendement des cultures", page_icon="🌾", layout="wide")


# ---------------------------------------------------------------- appels API
def call_api(endpoint: str, payload: dict) -> tuple[dict | None, str | None]:
    """Retourne (réponse JSON, None) ou (None, message d'erreur lisible)."""
    try:
        response = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=TIMEOUT_S)
    except requests.exceptions.ConnectionError:
        return None, f"Impossible de joindre l'API ({API_URL}). Vérifiez qu'elle est démarrée."
    except requests.exceptions.Timeout:
        return None, "L'API met trop de temps à répondre. Réessayez dans un instant."
    if response.status_code == 422:
        return None, "Certaines valeurs sont hors des plages acceptées par le modèle."
    if response.status_code == 503:
        return None, "Le modèle n'est pas chargé côté serveur."
    if not response.ok:
        detail = response.json().get("detail", response.text) if response.content else response.reason
        return None, f"Erreur {response.status_code} : {detail}"
    return response.json(), None


@st.cache_data(ttl=15, show_spinner=False)
def api_status() -> str:
    try:
        body = requests.get(f"{API_URL}/health", timeout=3).json()
        return "ok" if body.get("model_loaded") else "no_model"
    except (requests.exceptions.RequestException, ValueError):
        return "down"


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Mode")
    mode = st.radio("Que voulez-vous faire ?", ["Prédiction", "Recommandation"], label_visibility="collapsed")
    st.caption(
        "**Prédiction** : estimer le rendement d'une culture que vous avez choisie.  \n"
        "**Recommandation** : comparer les 6 cultures sur votre parcelle."
    )
    status = api_status()
    if status == "ok":
        st.success("API disponible")
    elif status == "no_model":
        st.error("API joignable, modèle non chargé")
    else:
        st.error("API injoignable")
        st.caption(f"Adresse utilisée : `{API_URL}`")

# ------------------------------------------------------------------- saisie
st.title("Rendement de ma parcelle")
st.write("Renseignez les conditions de votre parcelle. Le rendement est exprimé en tonnes par hectare (t/ha).")

col_left, col_right = st.columns(2, gap="large")
with col_left:
    rainfall = st.slider("Pluviométrie (mm)", 100, 1000, 550, step=10)
    temperature = st.slider("Température moyenne (°C)", 15.0, 40.0, 27.5, step=0.5)
with col_right:
    fertilizer = st.toggle("Engrais utilisé", value=True)
    irrigation = st.toggle("Irrigation utilisée", value=True)
    crop = st.selectbox("Culture", list(CROPS), format_func=CROPS.get) if mode == "Prédiction" else None

with st.expander("Autres paramètres (effet non mesurable sur le rendement dans nos données)"):
    c1, c2, c3, c4 = st.columns(4)
    region = c1.selectbox("Région", list(REGIONS), format_func=REGIONS.get)
    soil = c2.selectbox("Type de sol", list(SOILS), index=2, format_func=SOILS.get)
    weather = c3.selectbox("Météo", list(WEATHER), format_func=WEATHER.get)
    days = c4.number_input("Jours avant récolte", 60, 150, 105)

context = {
    "rainfall_mm": float(rainfall),
    "temperature_celsius": float(temperature),
    "fertilizer_used": fertilizer,
    "irrigation_used": irrigation,
    "region": region,
    "soil_type": soil,
    "weather_condition": weather,
    "days_to_harvest": int(days),
}

# ------------------------------------------------------------------- calcul
if st.button(f"Lancer la {mode.lower()}", type="primary"):
    st.session_state.pop("result", None)
    with st.spinner("Calcul en cours…"):
        if mode == "Prédiction":
            data, error = call_api("/predict", {**context, "crop": crop})
            result = {"mode": mode, "data": data}
        else:
            data, error = call_api("/recommend", context)
            scenarios = []
            if data:
                top_crop = data["ranking"][0]["crop"]
                for label, f, i in [
                    ("Ni engrais ni irrigation", False, False),
                    ("Engrais seul", True, False),
                    ("Irrigation seule", False, True),
                    ("Engrais + irrigation", True, True),
                ]:
                    s, _ = call_api("/predict", {**context, "fertilizer_used": f, "irrigation_used": i, "crop": top_crop})
                    if s:
                        scenarios.append({"Scénario": label, "Rendement (t/ha)": s["predicted_yield"]})
            result = {"mode": mode, "data": data, "scenarios": scenarios}
    if error:
        st.error(error)
    else:
        st.session_state["result"] = result

# ----------------------------------------------------------------- résultats
result = st.session_state.get("result")
if result and result["mode"] == mode:
    data = result["data"]
    st.divider()

    if mode == "Prédiction":
        st.subheader(f"Rendement estimé : {CROPS[data['crop']]}")
        m1, m2 = st.columns([1, 2])
        m1.metric("Rendement attendu", f"{data['predicted_yield']:.2f} t/ha")
        m2.write(
            f"Fourchette indicative : **{data['lower_bound']:.2f} à {data['upper_bound']:.2f} t/ha**. "
            f"Le modèle se trompe en moyenne de ± {data['rmse']:.2f} t/ha sur des parcelles qu'il n'a jamais vues."
        )

    else:
        st.subheader("Comparaison des cultures")
        if data["uncertainty_warning"]:
            st.warning(
                f"L'écart entre la meilleure et la moins bonne culture ({data['spread']:.3f} t/ha) est bien inférieur "
                f"à l'incertitude du modèle (± {data['rmse']:.2f} t/ha). Dans nos données, la culture ne change pas "
                "le rendement : ce classement ne permet pas de désigner une culture optimale."
            )
        else:
            st.success(data["message"])

        table = pd.DataFrame(data["ranking"])
        table["Culture"] = table["crop"].map(CROPS)
        order = table["Culture"].tolist()
        base = alt.Chart(table).encode(x=alt.X("Culture:N", sort=order, axis=alt.Axis(labelAngle=0)))
        bars = base.mark_bar(color="#6b8f47").encode(
            y=alt.Y("predicted_yield:Q", title="Rendement prédit (t/ha)", scale=alt.Scale(domainMin=0)),
            tooltip=["Culture", alt.Tooltip("predicted_yield:Q", title="t/ha")],
        )
        band = base.mark_errorbar(color="#3a3a35").encode(y="lower_bound:Q", y2="upper_bound:Q")
        st.altair_chart(bars + band, width="stretch")
        st.caption("Les traits noirs indiquent la fourchette d'incertitude du modèle.")

        st.dataframe(
            table.rename(columns={
                "rank": "Rang", "predicted_yield": "Rendement prédit (t/ha)",
                "lower_bound": "Borne basse (t/ha)", "upper_bound": "Borne haute (t/ha)",
            })[["Rang", "Culture", "Rendement prédit (t/ha)", "Borne basse (t/ha)", "Borne haute (t/ha)"]],
            hide_index=True, width="stretch",
        )

        if result.get("scenarios"):
            st.subheader("Ce qui change vraiment le rendement")
            sc = pd.DataFrame(result["scenarios"])
            st.altair_chart(
                alt.Chart(sc).mark_bar(color="#8a6d3b").encode(
                    x=alt.X("Scénario:N", sort=sc["Scénario"].tolist(), axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Rendement (t/ha):Q", scale=alt.Scale(domainMin=0)),
                    tooltip=["Scénario", "Rendement (t/ha)"],
                ),
                width="stretch",
            )
            gain = sc["Rendement (t/ha)"].iloc[-1] - sc["Rendement (t/ha)"].iloc[0]
            st.caption(
                f"Sur vos conditions de pluie et de température, passer de « ni engrais ni irrigation » à "
                f"« engrais + irrigation » ajoute environ {gain:.1f} t/ha, quelle que soit la culture."
            )