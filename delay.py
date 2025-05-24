import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

feature_weights = {
    'дз_размер': 0.9,
    'погода': 0.7,
    'прическа': 0.3,
    'предупреждал': 0.6,
    'конста': 0.8
}

@st.cache_data
def load_data():
    df = pd.read_csv("data_danya.csv")
    required_columns = ['дз_размер', 'погода', 'прическа', 'предупреждал', 'конста', 'опоздание_минуты']
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"Отсутствуют колонки: {', '.join(missing)}")

    encoders = {
        'дз_размер': LabelEncoder().fit(['легко', 'тяжело']),
        'погода': LabelEncoder().fit(['солнечно', 'снег', 'дождь']),
        'прическа': LabelEncoder().fit(['длинная', 'короткая']),
        'предупреждал': LabelEncoder().fit(['нет', 'да']),
        'конста': LabelEncoder().fit(['нет', 'да'])
    }

    for col in encoders:
        df[col] = encoders[col].transform(df[col])

    return df, encoders

@st.cache_resource
def train_model(df):
    X = df[['дз_размер', 'погода', 'прическа', 'предупреждал', 'конста']]
    y = df['опоздание_минуты']
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X, y)
    return model

def main():
    try:
        df, encoders = load_data()
        model = train_model(df)
    except Exception as e:
        st.error(f"Ошибка: {str(e)}")
        return

    st.title("📚 Прогноз опоздания Дани")
    st.markdown("---")
    st.sidebar.header("⚖️ Настройка важности факторов")
    weights = {
        'дз_размер': st.sidebar.slider("Размер ДЗ", 0.5, 2.0, feature_weights['дз_размер']),
        'погода': st.sidebar.slider("Погода", 0.3, 1.5, feature_weights['погода']),
        'прическа': st.sidebar.slider("Прическа", 0.1, 1.0, feature_weights['прическа']),
        'предупреждал': st.sidebar.slider("Предупреждение", 0.4, 1.8, feature_weights['предупреждал']),
        'конста': st.sidebar.slider("Конста", 0.5, 2.2, feature_weights['конста'])
    }

    # Ввод данных
    col1, col2 = st.columns(2)
    with col1:
        dz_size = st.selectbox("📝 Размер последнего ДЗ", encoders['дз_размер'].classes_)
        weather = st.selectbox("🌤️ Погода", encoders['погода'].classes_)
    with col2:
        hair = st.selectbox("💇 Прическа", encoders['прическа'].classes_)
        warned = st.radio("📢 Предупреждал?", encoders['предупреждал'].classes_)
        consta = st.radio("🎓 Конста?", encoders['конста'].classes_)

    if st.button("🔮 Рассчитать вероятность"):
        try:
            features = [
                encoders['дз_размер'].transform([dz_size])[0] * weights['дз_размер'],
                encoders['погода'].transform([weather])[0] * weights['погода'],
                encoders['прическа'].transform([hair])[0] * weights['прическа'],
                encoders['предупреждал'].transform([warned])[0] * weights['предупреждал'],
                encoders['конста'].transform([consta])[0] * weights['конста']
            ]
            
            features_df = pd.DataFrame(
                [features], 
                columns=model.feature_names_in_
            )

            proba = model.predict_proba(features_df)[0]
            st.markdown("### 📊 Вероятности опоздания")
            results = pd.DataFrame({
                'Минуты': model.classes_,
                'Вероятность': proba
            }).sort_values('Минуты', ascending=False)

            for _, row in results.iterrows():
                st.markdown(f"**{row['Минуты']} мин**: {row['Вероятность']*100:.1f}%")
                st.progress(row['Вероятность'])

            avg_delay = np.dot(proba, model.classes_)
            st.markdown(f"**Среднее ожидаемое опоздание:** {avg_delay:.1f} минут")

        except Exception as e:
            st.error(f"Ошибка: {str(e)}")

if __name__ == "__main__":
    main()